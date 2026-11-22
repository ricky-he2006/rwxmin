"""Reach of the Python SQL generator's cached expression-to-handler table."""
import ast
import __future__
import functools
import gzip
import json
import sys
from pathlib import Path


def name(cls):
    return f"{cls.__module__}:{cls.__qualname__}"


@functools.lru_cache(maxsize=1)
def trace():
    with gzip.open(Path(__file__).with_name("dispatch.json.gz"), "rt") as fh:
        return json.load(fh)


def guard_only(before, after):
    """Compile a pure map builder whose only writes are to its local copy.

    Compare the returned dictionaries on the actual generator classes. Changes
    to the function signature or any other code still use the whole-suite path.
    """
    old, new = ast.parse(before), ast.parse(after)
    def builder(tree):
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_dispatch")
    orig, edited = builder(old), builder(new)
    body = edited.body
    edited.body = orig.body
    if ast.dump(old) != ast.dump(new):
        return None
    edited.body = body
    # Start with a fresh dictionary and keep it local. Do not allow reads from
    # arbitrary descriptors: only the pinned lookup through dir(cls) can fetch
    # handlers, just as it does in the original function.
    if not body or ast.dump(body[0]) != ast.dump(orig.body[0]):
        return None
    locals_ = {"expr_cls", "attr_name", "expr_key", "dispatch"}
    for stmt in body[1:]:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id not in locals_ | {"dir", "cls", "exp", "getattr"}:
                return None
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    return None
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    if target.id not in ("expr_cls", "expr_key"):
                        return None
                elif not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                          and target.value.id == "dispatch"):
                    return None
            if isinstance(node, ast.For) and not (
                    ast.dump(node.target) == ast.dump(ast.Name("attr_name", ast.Store()))
                    and ast.unparse(node.iter) == "dir(cls)"):
                return None
            if isinstance(node, ast.Return) and ast.unparse(node.value) != "dispatch":
                return None
            if isinstance(node, ast.Call):
                fn = ast.unparse(node.func)
                if fn == "getattr" and ast.unparse(node) != "getattr(cls, attr_name)":
                    return None
                if fn not in ("dir", "getattr", "exp.EXPR_CLASSES.get", "attr_name.startswith",
                               "attr_name.endswith", "attr_name.lower", "attr_name.upper"):
                    return None
            if isinstance(node, ast.Attribute) and ast.unparse(node) not in (
                    "exp.EXPR_CLASSES", "exp.EXPR_CLASSES.get", "attr_name.startswith",
                    "attr_name.endswith", "attr_name.lower", "attr_name.upper"):
                return None
            if not isinstance(node, (ast.Assign, ast.For, ast.If, ast.Return, ast.Continue, ast.Break,
                                     ast.Name, ast.Load, ast.Store, ast.Call, ast.Attribute, ast.Subscript,
                                     ast.Slice, ast.Constant, ast.Compare, ast.BoolOp, ast.UnaryOp,
                                     ast.Not, ast.USub, ast.And, ast.Or, ast.IfExp, ast.cmpop)):
                return None
    return compile(ast.Module(body=[edited], type_ignores=[]), "<dispatch>", "exec", dont_inherit=True,
                   flags=__future__.annotations.compiler_flag)


def generator_classes():
    from sqlglot.executor.python import Python
    from sqlglot.generator import Generator
    from sqlglot.dialects.dialect import Dialect
    return {d.generator_class for d in Dialect.classes.values()} | {Generator, Python.generator_class}


def guard_reach(decision, coverage, repo):
    path = "sqlglot/generator.py"
    if set(decision.files) != {path} or decision.reasons != [f"cache-fill {path}"]:
        return None
    try:
        code = guard_only((repo / path).read_text(), decision.files[path])
        if code is None:
            return None
        data = trace()
        if data["ids"] != coverage.tests:
            return None
        # Importing SQLGlot here would change the main worker's pristine module
        # state before it prepares unrelated patches. Compare in an isolated fork.
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        reader, writer = ctx.Pipe(duplex=False)
        child = ctx.Process(target=_compare, args=(writer, code, repo, data, coverage))
        child.start()
        writer.close()
        try:
            return reader.recv() if reader.poll(5) else None
        finally:
            reader.close()
            if child.is_alive():
                child.terminate()
            child.join()
    except Exception:
        return None


def _compare(writer, code, repo, data, coverage):
    try:
        sys.path.insert(0, str(repo))
        import sqlglot.generator as gen
        classes = generator_classes()
        if set(map(name, classes)) != set(data["classes"]):
            return None
        ns = vars(gen).copy()
        exec(code, ns)
        reached = set(data["unknown"]) | coverage.covering("tests/dialects/test_dialect.py", 237, 237)
        for cls in classes:
            old, new = gen._build_dispatch(cls), ns["_build_dispatch"](cls)
            reads = data["reads"].get(name(cls), {})
            for expr in old.keys() | new.keys():
                if old.get(expr) is not new.get(expr):
                    reached.update(reads.get(name(expr), ()))
        # This test warms each builtin generator before measuring repeat
        # construction. The edited guard only runs on the first cache fill.
        reached.difference_update(i for i in reached.copy()
                                  if data["ids"][i] == "tests/test_generator.py::TestGenerator::test_generator_memory")
        writer.send(sorted(reached))
    finally:
        writer.close()


def transform_reach(delta, coverage):
    """Concrete lookups of edited TRANSFORMS keys, including inherited tables."""
    if not delta or any(kind != "cls" or attr != "TRANSFORMS" or not isinstance(key, str)
                        for kind, owner, attr, key in delta):
        return None
    try:
        data = trace()
        if coverage.tests != data["ids"]:
            return None
        classes = generator_classes()
        if set(map(name, classes)) != set(data["classes"]):
            return None
        direct = coverage.covering("tests/dialects/test_dialect.py", 237, 237)
        observed = set(data["unknown"])
        routes = set()
        for _, owner, _, key in delta:
            if not key.startswith("sqlglot.expressions."):
                return None
            for cls in classes:
                if owner not in {name(base) for base in cls.__mro__}:
                    continue
                for expr, tests in data["reads"].get(name(cls), {}).items():
                    if key == expr.replace(":", "."):
                        observed.update(tests)
                        routes.add((name(cls), expr))
        # These paths read TRANSFORMS directly, bypassing Generator.sql.
        for path, line in (("sqlglot/transforms.py", 52), ("sqlglot/generators/postgres.py", 601),
                           ("sqlglot/generator.py", 4100)):
            direct.update(coverage.covering(path, line, line))
        return observed, direct, routes
    except Exception:
        return None
