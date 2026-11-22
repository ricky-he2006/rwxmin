"""Image build time: baseline memo for the TPC executor and DuckDB optimizer tests.

usage: python buildmemo.py <out.pkl>
Records, while running the tests once on the unmodified checkout:
  - the duckdb reference dataframe of every reference query (stored lazily in <out>.bin)
  - a digest of every query's optimized expression
  - which sqlglot lines run outside parse/optimize/transpile ("rlines")
  - baseline (sql, optimized) -> (duckdb_sql_a, duckdb_sql_b) for test_optimized_tpch
  - baseline (setup_digest, query_sql) -> rows for DuckDB-backed TestOptimizer tests
"""
import hashlib
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import runner  # noqa: E402

IDS = [
    "tests/test_executor.py::TestExecutor::test_execute_tpcds",
    "tests/test_executor.py::TestExecutor::test_execute_tpch",
    "tests/test_executor.py::TestExecutor::test_optimized_tpch",
]
OPT_DUCK_IDS = [
    "tests/test_optimizer.py::TestOptimizer::test_optimize_in_nulls",
    "tests/test_optimizer.py::TestOptimizer::test_optimize_correlated_all",
    "tests/test_optimizer.py::TestOptimizer::test_optimize_scalar_subquery_predicates",
    "tests/test_optimizer.py::TestOptimizer::test_qualify_semi_anti_join_alias_shadowing_column",
]

runner.zygote_import({"tests.test_executor", "tests.test_optimizer"})
import duckdb  # noqa: E402
import sqlglot.executor as E  # noqa: E402
import tests.test_executor as T  # noqa: E402
import tests.test_optimizer as TO  # noqa: E402

REPO = os.path.realpath(runner.REPO) + os.sep
rec = {
    "pool_results": {},
    "duck": {},
    "dsql": {},
    "opt": {},
    "rlines": {},
    "tpch": {},
    "tpch_opt_pairs": {},
    "duck_rows": {},
}
runner.MEMO_REC = rec
runner.MEMO_IDS = dict(enumerate(IDS))
runner.SUBMIT_LAZY = {0, 1}

depth = [0]


def front(f):
    def w(*a, **k):
        depth[0] += 1
        try:
            return f(*a, **k)
        finally:
            depth[0] -= 1
    return w


runner.install_memo()
E.optimize = front(E.optimize)
E.optimize._br_memo = True
T.transpile = front(T.transpile)
T.parse_one = front(T.parse_one)

mon = sys.monitoring
TOOL = 4
mon.use_tool_id(TOOL, "brmemo")
cur = {}


def on_line(code, line):
    fn = code.co_filename
    if not fn.startswith(REPO) or "/tests/" in fn[len(REPO) - 1:]:
        return mon.DISABLE
    if depth[0]:
        return None
    cur.setdefault(fn[len(REPO):], set()).add(line)
    return mon.DISABLE


mon.register_callback(TOOL, mon.events.LINE, on_line)
# Keep a second monitor so the existing outside-phase trace keeps its semantics.
QUERY_TOOL = 3
mon.use_tool_id(QUERY_TOOL, "brquery")
query_maps = {}


def switch_query(index, at_parse=False):
    global current_query
    current_query = contexts.setdefault((index, at_parse), {})
    mon.restart_events()


def query_line(code, line):
    path = code.co_filename
    if path.startswith(REPO + "sqlglot/"):
        current_query.setdefault(path[len(REPO):], set()).add(line)
    return mon.DISABLE


mon.register_callback(QUERY_TOOL, mon.events.LINE, query_line)
for i, tid in enumerate(IDS):
    contexts = {}
    if i < 2:
        switch_query(None)
        runner.QUERY_TRACE = switch_query
        mon.set_events(QUERY_TOOL, mon.events.LINE)
    cur = {}
    rec["tpch"][tid] = "tpch" in tid
    mon.restart_events()
    mon.set_events(TOOL, mon.events.LINE)
    failed = runner.run_tests([i], IDS, timeouts=[3000.0])
    mon.set_events(TOOL, 0)
    mon.set_events(QUERY_TOOL, 0)
    runner.QUERY_TRACE = None
    if i < 2:
        query_maps[tid] = contexts
    if failed:
        print("baseline test failed:", tid, file=sys.stderr)
        sys.exit(1)
    rec["rlines"][tid] = {f: frozenset(v) for f, v in cur.items()}
    print(tid, "queries", len(rec["opt"].get(tid, {})), "rlines", sum(len(v) for v in cur.values()), file=sys.stderr)

# Record baseline duckdb transpilation pairs for test_optimized_tpch
for _, sql, optimized in T.TestExecutor.tpch_sqls:
    qa = T.transpile(sql, write="duckdb")[0]
    qb = T.transpile(optimized, write="duckdb")[0]
    rec["tpch_opt_pairs"][(sql, optimized)] = (qa, qb)

# Record baseline DuckDB query results for TestOptimizer tests
class _RecResult:
    def __init__(self, real_res, key, sink):
        self._res = real_res
        self._key = key
        self._sink = sink

    def fetchall(self):
        rows = self._res.fetchall()
        self._sink[self._key] = rows
        return rows

    def __getattr__(self, name):
        return getattr(self._res, name)


class _RecConn:
    def __init__(self, real_conn, sink):
        self._real = real_conn
        self._digest = b""
        self._sink = sink
        self._dirty = False

    def execute(self, sql, *a, **k):
        res = self._real.execute(sql, *a, **k)
        if a or k or self._dirty or not isinstance(sql, str):
            self._dirty = True
            return res
        s = sql.lstrip().upper()
        if s.startswith(("CREATE ", "INSERT ")):
            self._digest = hashlib.sha256(self._digest + b"\0" + sql.encode("utf-8")).digest()
            return res
        return _RecResult(res, (self._digest, sql), self._sink)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._real.close()

    def close(self):
        self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


class _RecDuckMod:
    def __init__(self, real_mod, sink):
        self._real = real_mod
        self._sink = sink

    def connect(self, *a, **k):
        c = self._real.connect(*a, **k)
        if not a and not k:
            return _RecConn(c, self._sink)
        return c

    def __getattr__(self, name):
        return getattr(self._real, name)


orig_duck = TO.duckdb
TO.duckdb = _RecDuckMod(duckdb, rec["duck_rows"])
try:
    failed = runner.run_tests(list(range(len(OPT_DUCK_IDS))), OPT_DUCK_IDS, timeouts=[300.0] * len(OPT_DUCK_IDS))
    if failed:
        print("baseline optimizer duckdb tests failed:", [OPT_DUCK_IDS[i] for i in failed], file=sys.stderr)
        sys.exit(1)
finally:
    TO.duckdb = orig_duck

print("duck frames", len(rec["duck"]), "tpch_opt_pairs", len(rec["tpch_opt_pairs"]),
      "duck_rows", len(rec["duck_rows"]), file=sys.stderr)

# Write heavy DataFrames and pool blobs to an indexed binary file so memo.pkl
# contains zero pandas objects and loads in ~4 ms without importing pandas.
out_pkl = sys.argv[1]
out_bin = out_pkl[:-4] + ".bin" if out_pkl.endswith(".pkl") else out_pkl + ".bin"
duck_idx = {}
pool_idx = {}
with open(out_bin, "wb") as bfh:
    off = 0
    for k, df in rec["duck"].items():
        blob = pickle.dumps(df, protocol=4)
        bfh.write(blob)
        duck_idx[k] = (off, len(blob))
        off += len(blob)
    for k, raw in rec["pool_results"].items():
        bfh.write(raw)
        pool_idx[k] = (off, len(raw))
        off += len(raw)

base_sqls = {
    True: frozenset((sql, repr(sorted(args.items()))) for args, sql, _ in T.TestExecutor.tpch_sqls),
    False: frozenset((sql, repr(sorted(args.items()))) for args, sql, _ in T.TestExecutor.tpcds_sqls),
}
meta = {
    "parsed": rec.get("parsed", {}),
    "executor_keys": rec.get("executor_keys", {}),
    "dsql": rec["dsql"],
    "opt": rec["opt"],
    "rlines": rec["rlines"],
    "tpch": rec["tpch"],
    "tpch_opt_pairs": rec["tpch_opt_pairs"],
    "base_sqls": base_sqls,
    "duck_rows": rec["duck_rows"],
    "duck_idx": duck_idx,
    "pool_idx": pool_idx,
}
with open(out_pkl, "wb") as fh:
    pickle.dump(meta, fh, protocol=4)
# setUpClass only ran for the first test. Its outside-query lines apply to both.
shared = {}
query_index = {}
for tid, contexts in query_maps.items():
    for file, lines in contexts.pop((None, False)).items():
        shared.setdefault(file, set()).update(lines)
    # One integer per covered line. Bit i is set when fixture i hits it.
    by_phase = [{}, {}]
    indices = set()
    for (i, at_parse), files in contexts.items():
        indices.add(i)
        for file, lines in files.items():
            masks = by_phase[at_parse].setdefault(file, {})
            for line in lines:
                masks[line] = masks.get(line, 0) | (1 << i)
    if indices == set(range(len(indices))):
        query_index[tid] = (len(indices), by_phase)
lite = {
    "query_shared": shared,
    "query_index": query_index,
    "opt": {k: bool(v) for k, v in rec["opt"].items()},
    "rlines": rec["rlines"],
    "tpch": rec["tpch"],
    "tpch_opt_tid": "tests/test_executor.py::TestExecutor::test_optimized_tpch",
    "duck_opt_tids": OPT_DUCK_IDS,
}
with open(out_pkl.replace(".pkl", "_lite.pkl"), "wb") as fh:
    pickle.dump(lite, fh, protocol=4)
