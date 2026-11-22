"""Image build time: baseline memo of sqlglot.optimizer.optimize at the pinned commit.

usage: python buildopt.py <out prefix>      (writes <prefix>.idx and <prefix>.bin)

optimize() runs a fixed pipeline of pure rules. While the optimizer and executor tests
run once on the unmodified checkout, every optimize() call with the default rules is
recorded: a digest and a pickle of the tree before each rule and of the result. Line
tracing records which sqlglot lines each rule (and the preamble) ever runs.

At selection time, for a patch that only changes function bodies, a rule that never
runs a changed line maps a baseline input to its baseline output, so it need not run:
only the affected rules run, on their baseline inputs, and if their output equals the
baseline the stored final result is returned.
"""
import hashlib
import inspect
import os
import pickle
import sys
import unittest
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.environ.get("BR_REPO", "/opt/sqlglot")
sys.path.insert(0, REPO)
os.environ.setdefault("SKIP_INTEGRATION", "0")
os.environ["PYTHONPATH"] = REPO + os.pathsep + os.environ.get("PYTHONPATH", "")

import concurrent.futures as _cf  # noqa: E402
import concurrent.futures.process as _cfp  # noqa: E402

import optmemo  # noqa: E402
from canon import Undumpable, dump_tree  # noqa: E402


class _Inline:
    """Pools run their work in this process so that it can be recorded."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        if initializer:
            initializer(*initargs)

    def submit(self, fn, *a, **k):
        f = _cf.Future()
        try:
            f.set_result(fn(*a, **k))
        except BaseException as e:  # noqa
            f.set_exception(e)
        return f

    def map(self, fn, *its, **kw):
        return [fn(*args) for args in zip(*its)]

    def shutdown(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


_cf.ProcessPoolExecutor = _Inline
_cfp.ProcessPoolExecutor = _Inline

import sqlglot  # noqa: E402,F401
import sqlglot.optimizer.optimizer as O  # noqa: E402

RULES = O.RULES
NCTX = len(RULES) + 1          # 0 = preamble, i + 1 = rule i
PREFIX = os.path.realpath(REPO) + os.sep + "sqlglot" + os.sep
ROOT = os.path.realpath(REPO) + os.sep
mon = sys.monitoring
TOOL = 4
mon.use_tool_id(TOOL, "bropt")
cov = {}                       # (file, line) -> bitmask of contexts
ctx = [None]
# Track per-input coverage only in leaf optimizer rules. Other modules include
# process caches, dynamic dispatch and setup code shared by unrelated calls.
LEAVES = {
    "sqlglot/optimizer/" + stem + ".py" for stem in (
        "eliminate_joins", "eliminate_ctes", "eliminate_subqueries", "merge_subqueries",
        "optimize_joins", "pushdown_predicates", "pushdown_projections", "unnest_subqueries",
    )
}
line_ids = {}
per_key = {}
active = [None]



def on_line(code, line):
    fn = code.co_filename
    if not fn.startswith(PREFIX):
        return mon.DISABLE
    c = ctx[0]
    if c is None:
        return None            # not inside a recorded optimize(): keep listening
    k = (fn[len(ROOT):], line)
    cov[k] = cov.get(k, 0) | (1 << c)
    if active[0] is not None and k[0] in LEAVES:
        index = line_ids.setdefault(k, len(line_ids))
        active[0][c] |= 1 << index
    return mon.DISABLE


def switch(c):
    ctx[0] = c
    mon.restart_events()


mon.register_callback(TOOL, mon.events.LINE, on_line)
mon.set_events(TOOL, mon.events.LINE)

keys = {}                      # call key -> tuple of stage digests (len NCTX: before each rule, final)
blobs = {}                     # digest -> compressed pickle
orig = O.optimize
depth = [0]
stats = {"calls": 0, "recorded": 0, "unkeyable": 0, "raised": 0}


def recorder(expression, schema=None, db=None, catalog=None, dialect=None, rules=None, sql=None,
             copy=True, **kwargs):
    if rules is None:
        rules = RULES
    if depth[0] or rules is not RULES or not copy:
        return orig(expression, schema, db, catalog, dialect, rules, sql, copy, **kwargs)
    stats["calls"] += 1
    try:
        key = optmemo.call_key(expression, schema, db, catalog, dialect, sql, kwargs)
    except optmemo.Unkeyable:
        stats["unkeyable"] += 1
        return orig(expression, schema, db, catalog, dialect, rules, sql, copy, **kwargs)
    repeated = key in keys
    visits = [0] * NCTX
    stages = []

    def mark(i):
        def rec(expression):
            d = optmemo.digest(expression)
            if not repeated and d is not None and d not in blobs:
                try:
                    blobs[d] = zlib.compress(pickle.dumps(dump_tree(expression), protocol=4), 6)
                except (Undumpable, RecursionError):
                    d = None
            stages.append(d)
            switch(i + 1 if i < len(RULES) else None)
            return expression
        return rec

    seq = []
    for i, r in enumerate(RULES):
        seq += [mark(i), r]
    seq.append(mark(len(RULES)))
    depth[0] += 1
    active[0] = visits
    switch(0)
    try:
        out = orig(expression, schema, db, catalog, dialect, tuple(seq), sql, copy, **kwargs)
    except BaseException:
        stats["raised"] += 1
        raise
    finally:
        depth[0] -= 1
        active[0] = None
        switch(None)
    if len(stages) == NCTX and None not in stages:
        if repeated and tuple(stages) != keys[key]:
            per_key.pop(key, None)
        elif not repeated:
            per_key[key] = visits
        elif key in per_key:
            per_key[key] = [a | b for a, b in zip(per_key[key], visits)]
        keys[key] = tuple(stages)
        stats["recorded"] += 1
    return out


optmemo.replace_everywhere(orig, recorder)
suite = unittest.TestSuite()
for name in ("tests.test_optimizer", "tests.test_executor", "tests.test_lineage"):
    try:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    except Exception as e:  # noqa
        print("cannot load", name, repr(e), file=sys.stderr)
res = unittest.TestResult()
suite.run(res)
mon.set_events(TOOL, 0)
print("tests", res.testsRun, "failures", len(res.failures), "errors", len(res.errors), stats, file=sys.stderr)
if res.failures or res.errors:
    for _, tb in (res.failures + res.errors)[:3]:
        print(tb[-600:], file=sys.stderr)

used = {d for st in keys.values() for d in st}
index, off = {}, 0
with open(sys.argv[1] + ".bin", "wb") as fh:
    for d in sorted(used):
        b = blobs[d]
        fh.write(b)
        index[d] = (off, len(b))
        off += len(b)
lines = {}
for (f, l), m in cov.items():
    lines.setdefault(f, {})[l] = m
lineid = {}
for (path, line), number in line_ids.items():
    lineid.setdefault(path, {})[line] = number
sets = {}
kcov = {key: tuple(sets.setdefault(mask, len(sets)) for mask in stages) for key, stages in per_key.items()}
with open(sys.argv[1] + ".idx", "wb") as fh:
    pickle.dump({"keys": keys, "blobs": index, "lineid": lineid, "ksets": tuple(sets), "kcov": kcov, "cov": lines, "rules": [r.__name__ for r in RULES],
                 "args": [inspect.getfullargspec(r).args for r in RULES]}, fh, protocol=4)
print("keys", len(keys), "trees", len(index), "bytes", off, file=sys.stderr)
