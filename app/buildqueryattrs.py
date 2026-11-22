"""Record which TPC fixture reads each scalar dialect setting on the pinned checkout."""
import gzip
import json
import os
import pickle
import sys

import runner
sys.path.insert(0, runner.REPO)
from sqlglot import exp
from sqlglot.dialects.dialect import Dialect


class Setting:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __get__(self, obj, owner):
        reads[self.key] = reads.get(self.key, 0) | current
        if optimizing:
            in_opt[self.key] = in_opt.get(self.key, 0) | current
        if depth == 0:
            outside[self.key] = outside.get(self.key, 0) | current
        if not initial_parse:
            later[self.key] = later.get(self.key, 0) | current
        return self.value


def switch(index, at_parse=False):
    global current, initial_parse
    initial_parse = at_parse
    current = 1 if index is None else 2 << index


ids = ["tests/test_executor.py::TestExecutor::test_execute_" + suffix for suffix in ("tpcds", "tpch")]
runner.zygote_import({"tests.test_executor"})
# Force dialect registration before installing descriptors on the concrete owners.
classes = set(Dialect.classes.values()) | {Dialect}
generators = {base for dialect in classes for base in dialect.generator_class.__mro__
              if base.__module__.startswith("sqlglot.")}
classes.update(generators)
tracked = set()
for cls in classes:
    for name, value in list(vars(cls).items()):
        if name.isupper() and (value is None or type(value) in (bool, str, int, float) or isinstance(value, exp.Expr)):
            key = (f"{cls.__module__}:{cls.__qualname__}", name)
            setattr(cls, name, Setting(key, value))
            tracked.add(key)

runner.SUBMIT_LAZY = {0, 1}
runner.MEMO_IDS = dict(enumerate(ids))
runner.MEMO_DUCK = True
runner.QUERY_TRACE = switch
# Monitor the same boundaries as buildmemo's runtime-line map. Reads in the
# optimizer, initial parse and reference transpiler are verified separately.
import sqlglot.executor as executor
import tests.test_executor as test_executor

depth = 0
optimizing = 0

def boundary(fn, optimizer=False):
    def call(*args, **kwargs):
        global depth, optimizing
        depth += 1
        optimizing += optimizer
        try:
            return fn(*args, **kwargs)
        finally:
            depth -= 1
            optimizing -= optimizer
    call._br_memo = True
    return call

runner.install_memo()
executor.optimize = boundary(executor.optimize, optimizer=True)
test_executor.transpile = boundary(test_executor.transpile)
test_executor.parse_one = boundary(test_executor.parse_one)
out = {}
for i, tid in enumerate(ids):
    # Independent class setup and cold test caches for each test.
    path = sys.argv[1] + ".part"
    pid = os.fork()
    if pid == 0:
        reads, outside, later, in_opt = {}, {}, {}, {}
        switch(None)
        failed = runner.run_tests([i], ids, timeouts=[300.0])
        if not failed:
            with open(path, "wb") as fh:
                pickle.dump((reads, outside, later, in_opt), fh)
        os._exit(int(bool(failed)))
    _, status = os.waitpid(pid, 0)
    if status == 0:
        with open(path, "rb") as fh:
            out[tid] = pickle.load(fh)
        os.unlink(path)
# The optimizer suite loops over independent SQL fixture pairs. Trace scalar
# settings in that loop too, including reads after it, when futures are checked.
here = os.path.dirname(os.path.realpath(__file__))
with gzip.open(os.path.join(here, "dispatch.json.gz"), "rt") as fh:
    test_ids = json.load(fh)["ids"]
with gzip.open(os.path.join(here, "iter.json.gz"), "rt") as fh:
    candidates = json.load(fh)
iteration_reads = {}
runner.SUBMIT_LAZY = set()
runner.MEMO_IDS = {}
runner.MEMO_DUCK = False
runner.QUERY_TRACE = None
for number, record in candidates.items():
    tid = test_ids[int(number)]
    if not tid.startswith("tests/test_optimizer.py::"):
        continue
    pid = os.fork()
    if pid == 0:
        reads, outside, later, in_opt = {}, {}, {}, {}
        switch(None)
        runner.zygote_import({"tests.test_optimizer"})
        counts = {}
        runner.ITER_HOOK = (switch, counts)
        runner.install_iter_wrappers()
        failed = runner.run_tests([0], [tid], timeouts=[180.0])
        if not failed and counts.get(0) == record["n"]:
            with open(path, "wb") as fh:
                pickle.dump(reads, fh)
            os._exit(0)
        os._exit(1)
    _, status = os.waitpid(pid, 0)
    if status == 0:
        with open(path, "rb") as fh:
            iteration_reads[tid] = pickle.load(fh)
        os.unlink(path)
print("optimizer fixture traces", len(iteration_reads))
with open(sys.argv[1], "wb") as fh:
    pickle.dump((tracked, out, iteration_reads), fh)
print("TPC scalar-setting trace", len(tracked), {k: len(v[0]) for k, v in out.items()})
