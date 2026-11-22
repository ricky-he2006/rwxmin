"""Record independent validator calls. Reads only the pinned SQLGlot checkout."""
import zlib
import pickle
import sys
import time
import unittest
from pathlib import Path

import runner
import validatorcalls

root = Path(runner.REPO)
files = sorted((root / "tests/dialects").glob("test_*.py"))
modules = [runner.module_name_for(str(f.relative_to(root))) for f in files]
runner.zygote_import(modules)
from tests.dialects.test_dialect import Validator
from sqlglot.dialects.dialect import Dialect
from sqlglot import exp, generator
from dispatch import generator_classes, name as class_name

class Setting:
    def __init__(self, owner, name, value):
        self.key = (owner, name)
        self.value = value

    def __get__(self, instance, owner):
        if active:
            settings[self.key] = settings.get(self.key, 0) | active
        return self.value

monitor = sys.monitoring
tool = 4
monitor.use_tool_id(tool, "validator-calls")
repo_prefix = str(root) + "/"
prefixes = tuple(str(root / "sqlglot" / folder) + "/" for folder in ("parsers", "generators", "dialects"))
core = {str(root / "sqlglot" / file) for file in ("parser.py", "generator.py")}
coverage = {}
active = 0
calls = []
settings = {}
tracked = set()
dialects = set(Dialect.classes.values()) | {Dialect}
classes = dialects | {base for d in dialects for base in d.generator_class.__mro__
                      if base.__module__.startswith("sqlglot.")}
for cls in classes:
    owner = cls.__module__ + ":" + cls.__qualname__
    for name, value in list(vars(cls).items()):
        if name.isupper() and (value is None or type(value) in (bool, str, int, float) or isinstance(value, exp.Expr)):
            tracked.add((owner, name))
            setattr(cls, name, Setting(owner, name, value))


known_generators = generator_classes()
original_dispatch = generator._build_dispatch
class ObservedDispatch(dict):
    def get(self, expression, default=None):
        if active:
            key = self.owner, class_name(expression)
            routes[key] = routes.get(key, 0) | active
        return super().get(expression, default)


def observe_dispatch(cls):
    table = ObservedDispatch(original_dispatch(cls))
    table.owner = class_name(cls) if cls in known_generators else None
    return table


generator._build_dispatch = observe_dispatch
generator._DISPATCH_CACHE.clear()


def line(code, number):
    path = code.co_filename
    if active and (path in core or path.startswith(prefixes)):
        by_line = coverage.setdefault(path[len(repo_prefix):], {})
        by_line[number] = by_line.get(number, 0) | active
    return monitor.DISABLE

monitor.register_callback(tool, monitor.events.LINE, line)
def traced(name, method):
    def run(*args, **kwargs):
        global active
        if active:
            return method(*args, **kwargs)
        active = 1 << len(calls)
        calls.append(name)
        monitor.restart_events()
        start = time.process_time_ns()
        try:
            return method(*args, **kwargs)
        finally:
            costs.append(time.process_time_ns() - start)
            active = 0
    return run

for name in ("validate_all", "validate_identity", "validate_transpile"):
    setattr(Validator, name, traced(name, getattr(Validator, name)))

data = {}
for mod in modules:
    for cls in vars(sys.modules[mod]).values():
        if not isinstance(cls, type) or cls.__module__ != mod or not issubclass(cls, Validator):
            continue
        for method in unittest.defaultTestLoader.getTestCaseNames(cls):
            names = validatorcalls.statements(getattr(cls, method))
            if (not names or cls.parse_one is not Validator.parse_one
                    or any(getattr(cls, n) is not getattr(Validator, n) for n in set(names))):
                continue
            case = cls(method)
            calls = []
            coverage, settings, routes, costs = {}, {}, {}, []
            monitor.set_events(tool, monitor.events.LINE)
            try:
                result = unittest.TestSuite([case]).run(unittest.TestResult())
            finally:
                monitor.set_events(tool, 0)
            if result.wasSuccessful() and calls == names:
                tid = mod.replace(".", "/") + ".py::" + cls.__name__ + "::" + method
                data[tid] = names, coverage, settings, routes, costs
index = {}
with open(sys.argv[1] + ".bin", "wb") as fh:
    for tid, entry in data.items():
        blob = zlib.compress(pickle.dumps(entry, protocol=4), level=3)
        index[tid] = (fh.tell(), len(blob))
        fh.write(blob)
with open(sys.argv[1], "wb") as fh:
    pickle.dump((tracked, index), fh, protocol=4)
print("validator methods", len(data), "independent calls", sum(len(v[0]) for v in data.values()))
