"""Line coverage for independent SQL assertions in the pinned dialect tests."""
import ast
import functools
import zlib
import os
import pickle


@functools.lru_cache(maxsize=1)
def load():
    path = os.path.join(os.path.dirname(__file__), "validatorcalls.idx")
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        return set(), {}


@functools.lru_cache(maxsize=1)
def method_nodes(filename):
    with open(filename, encoding="utf-8") as fh:
        return {n.lineno: n for n in ast.walk(ast.parse(fh.read())) if isinstance(n, ast.FunctionDef)}


def statements(method):
    """Only standalone validator calls with literal inputs have independent results."""
    code = method.__code__
    tree = method_nodes(code.co_filename).get(code.co_firstlineno)
    if tree is None or tree.decorator_list:
        return None
    names = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return None
        call = stmt.value
        fn = call.func
        if (not isinstance(fn, ast.Attribute) or not isinstance(fn.value, ast.Name) or fn.value.id != "self"
                or fn.attr not in ("validate_identity", "validate_transpile", "validate_all")):
            return None
        try:
            for arg in (*call.args, *(kw.value for kw in call.keywords)):
                ast.literal_eval(arg)
        except (ValueError, SyntaxError):
            return None
        names.append(fn.attr)
    return names if len(names) >= 4 else None


@functools.lru_cache(maxsize=256)
def entry(offset, length):
    path = os.path.join(os.path.dirname(__file__), "validatorcalls.idx.bin")
    with open(path, "rb") as fh:
        fh.seek(offset)
        return pickle.loads(zlib.decompress(fh.read(length)))


def masks(lines, delta, candidates, routes=()):
    tracked, data = load()
    if not tracked.issuperset(delta):
        return {}
    retained = {}
    for tid in candidates:
        if tid not in data:
            continue
        names, table, settings, dispatch, costs = entry(*data[tid])
        if routes and any(owner is None for owner, _ in dispatch):
            continue
        affected = 0
        for key in delta:
            affected |= settings.get(key, 0)
        for key in routes:
            affected |= dispatch.get(key, 0)
        for path, numbers in lines.items():
            coverage = table.get(path, {})
            for n in numbers:
                affected |= coverage.get(n, 0)
        if affected.bit_count() < len(names):
            fraction = sum(cost for n, cost in enumerate(costs) if affected & (1 << n)) / max(1, sum(costs))
            retained[tid] = names, affected, fraction
    return retained


def apply(case, plan):
    names, affected, _ = plan
    index = 0
    for name in set(names):
        original = getattr(case, name)
        def filtered(*args, _original=original, _name=name, **kwargs):
            nonlocal index
            current = index
            index += 1
            if current >= len(names) or names[current] != _name or affected & (1 << current):
                return _original(*args, **kwargs)
            return None
        setattr(case, name, filtered)
