"""In-process test runner with fork-per-commit and code hot patching.

The zygote imports sqlglot and the needed test modules once. For every commit it
forks a child that applies the commit in memory (swapping function code objects
when the change is confined to function bodies, otherwise re-importing sqlglot
from an overlay directory), runs the requested tests with unittest and reports
which of them failed.
"""
from __future__ import annotations

import builtins
import gc
import hashlib
import importlib
import io
import json
import os
import pickle
import signal
import sys
import time
import types
import unittest

REPO = os.environ.get("BR_REPO", "/opt/sqlglot")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------- inline pools
import concurrent.futures as _cf


VALIDATOR_CALLS = {}
PICKLE_TASKS = True


class InlineExecutor:
    """Runs submitted work synchronously, with a pickle round trip like a real pool."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        self.input_digest = None
        self._is_exec = False
        if initializer:
            is_exec_init = (
                getattr(initializer, "__module__", None),
                getattr(initializer, "__name__", None),
            ) == ("tests.test_executor", "initializer")
            self._is_exec = is_exec_init
            if is_exec_init and MEMO_REC is None and not PICKLE_TASKS:
                # tests.test_executor.initializer only stores read-only schema/tables refs
                initializer(*initargs)
            else:
                payload = pickle.dumps(initargs)
                if MEMO_REC is not None and is_exec_init:
                    self.input_digest = hashlib.sha256(payload).digest()
                initializer(*pickle.loads(payload))

    def submit(self, fn, *a, **k):
        n = _SUB["n"]
        _SUB["n"] = n + 1
        if _SUB.get("abort"):
            f = _cf.Future()
            f.set_exception(RuntimeError("test timed out"))   # nothing more is run for it
            return f
        allow = _SUB["allow"]
        if allow is not None and n not in allow:
            f = _cf.Future()
            f.set_result(None)  # query not reaching the changed code: skip its comparison
            return f
        if not _SUB["lazy"]:
            f = _cf.Future()
            _rearm()
            try:
                fn, a, k = pickle.loads(pickle.dumps((fn, a, k)))
                f.set_result(pickle.loads(pickle.dumps(fn(*a, **k))))
            except BaseException as e:  # noqa
                f.set_exception(e)
            finally:
                _rearm(done=True)
            return f
        f = _LazyFuture()
        f._n = n
        is_mp_exec = (
            getattr(fn, "__module__", None),
            getattr(fn, "__name__", None),
        ) == ("tests.test_executor", "mp_execute")
        if is_mp_exec and MEMO_REC is None and not PICKLE_TASKS:
            f._job = (fn, a, k)
            f._nopickle = True
            return f
        try:
            payload = pickle.dumps((fn, a, k))
            if self.input_digest is not None and is_mp_exec:
                key = (self.input_digest, hashlib.sha256(payload).digest())
                if MEMO_REC is not None:
                    f._record_key = key
            f._job = pickle.loads(payload)
        except BaseException as e:  # noqa
            f.set_exception(e)
        return f

    def map(self, fn, *its, **kw):
        return [self.submit(fn, *args).result() for args in zip(*its)]

    def shutdown(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


_SUB = {"n": 0, "allow": None, "failed": False, "lazy": False, "cur": None, "t": None, "deadline": None}
SUBMIT_MAX = float(os.environ.get("BR_SUBMIT_MAX", "6"))


def _rearm(done=False):
    """One submitted call (a single query of a TPC test) gets a few seconds at most;
    once it returns, the test's own limit applies again."""
    d = _SUB.get("deadline")
    if d is not None:
        left = d - time.time()
        signal.setitimer(signal.ITIMER_REAL, max(0.05, left if done else min(left, SUBMIT_MAX)), 1.0)


# ------------------------------------------------- baseline memo (TPC executor + DuckDB optimizer tests)
# Built from the pinned commit at image build time (buildmemo.py):
#   memo.pkl: dsql, opt, rlines, tpch, tpch_opt_pairs, duck_rows, duck_idx, pool_idx (no pandas!)
#   memo.bin: raw pickled DataFrames / pool blobs loaded lazily via os.pread
MEMO = None
MEMO_BIN_PATH = None
_MEMO_BIN_FD = [None]
_MEMO_DUCK_CACHE: dict = {}
FIXTURE_CALLS = None  # pool_idx or pool_results dict from the unmodified checkout
MEMO_IDS: dict = {}     # test number -> test id
MEMO_DUCK = False       # reuse reference dataframes / duckdb results when sql is unchanged
MEMO_SKIP: set = set()  # tests whose queries may skip execution when their plan is unchanged
UNCHANGED_OPT_QUERIES = {}
MEMO_QUERIES = {}       # query indices with unchanged downstream setting reads
FAST_PROJECTIONS = False
MEMO_STAT = [0, 0]      # queries skipped, queries re-run after a reference mismatch
MEMO_REC = None         # build time: record instead of use


class _SkipExec(BaseException):
    pass


def load_memo(path):
    global MEMO, MEMO_BIN_PATH
    try:
        with _orig_open(path, "rb") as fh:
            MEMO = pickle.load(fh)
        bp = path[:-4] + ".bin" if path.endswith(".pkl") else path + ".bin"
        MEMO_BIN_PATH = bp if os.path.isfile(bp) else None
    except Exception as e:  # noqa
        MEMO = None
        MEMO_BIN_PATH = None
        log("memo unavailable", repr(e))


def _memo_pread(off, length):
    fd = _MEMO_BIN_FD[0]
    if fd is None:
        if not MEMO_BIN_PATH:
            return None
        fd = os.open(MEMO_BIN_PATH, os.O_RDONLY)
        _MEMO_BIN_FD[0] = fd
    return os.pread(fd, length, off)


def _memo_get_duck(key):
    df = _MEMO_DUCK_CACHE.get(key)
    if df is not None:
        return df.copy()
    if MEMO is None:
        return None
    idx = MEMO.get("duck_idx")
    if idx is not None:
        ent = idx.get(key)
        if ent is None:
            return None
        raw = _memo_pread(ent[0], ent[1])
        if raw is None:
            return None
        df = pickle.loads(raw)
        _MEMO_DUCK_CACHE[key] = df
        return df.copy()
    df = MEMO.get("duck", {}).get(key)
    return df.copy() if df is not None else None


def _memo_get_pool(key):
    if FIXTURE_CALLS is None:
        return None
    if isinstance(FIXTURE_CALLS, dict) and MEMO is not None and FIXTURE_CALLS is MEMO.get("pool_idx"):
        ent = FIXTURE_CALLS.get(key)
        if ent is None:
            return None
        return _memo_pread(ent[0], ent[1])
    return FIXTURE_CALLS.get(key)


from canon import canon, digest  # noqa: E402
import optmemo  # noqa: E402


def _memo_optimize(real):
    def optimize(*a, **k):
        cur, t = _SUB["cur"], _SUB["t"]
        base = None
        if cur is not None and MEMO_REC is None and (t in MEMO_SKIP or cur in MEMO_QUERIES.get(t, ())) and MEMO is not None:
            base = MEMO["opt"].get(MEMO_IDS.get(t), {}).get(cur)
        t0 = time.process_time()
        optmemo.WANT[0] = base is not None      # a digest is enough when it is the baseline's
        try:
            query = (MEMO_IDS.get(t), cur)
            previous = MEMO.get("executor_keys", {}).get(query) if MEMO is not None else None
            if MEMO_REC is not None and cur is not None:
                key = optmemo.executor_key(*a, **k)
                MEMO_REC.setdefault("executor_keys", {})[query] = key
            if (base is not None and previous is not None and cur in UNCHANGED_OPT_QUERIES.get(t, ())
                    and optmemo.executor_key(*a, **k) == previous):
                raise _SkipExec()
            r = real(*a, **k)
        finally:
            optmemo.WANT[0] = False
            MEMO_T[2] += time.process_time() - t0
        if isinstance(r, optmemo.Base):
            if r.digest == base:
                raise _SkipExec()
            r = optmemo._tree(r.digest)
        elif cur is not None:
            if MEMO_REC is not None:
                MEMO_REC["opt"].setdefault(MEMO_IDS.get(t), {})[cur] = digest(r)
            elif base is not None and base == digest(r):
                raise _SkipExec()
        return r
    optimize._br_memo = True
    return optimize


def _memo_cached_execute(self, sql, tpch=True):
    if sql not in self.cache:
        T = sys.modules[type(self).__module__]
        q = T.transpile(sql, write="duckdb")[0]
        df = None
        if MEMO_REC is None and MEMO_DUCK and MEMO is not None:
            df = _memo_get_duck((bool(tpch), q))
        if df is None:
            conn = self.tpch_conn if tpch else self.tpcds_conn
            df = conn.execute(q).fetchdf()
            if MEMO_REC is not None:
                MEMO_REC["duck"][(bool(tpch), q)] = df.copy()
                MEMO_REC["dsql"][(bool(tpch), sql)] = q
        self.cache[sql] = df
    return self.cache[sql]


_memo_cached_execute._br_memo = True


def _memo_test_optimized_tpch(orig):
    def test_optimized_tpch(self):
        if MEMO_REC is not None or not MEMO_DUCK or MEMO is None or "tpch_opt_pairs" not in MEMO:
            return orig(self)
        pairs = MEMO["tpch_opt_pairs"]
        T = sys.modules[type(self).__module__]
        for i, (_, sql, optimized) in enumerate(self.tpch_sqls, start=1):
            with self.subTest(f"{i}, {sql}"):
                qa = T.transpile(sql, write="duckdb")[0]
                qb = T.transpile(optimized, write="duckdb")[0]
                if pairs.get((sql, optimized)) == (qa, qb):
                    continue
                a = self.cached_execute(sql, tpch=True)
                b = self.tpch_conn.execute(qb).fetchdf()
                self.rename_anonymous(b, a)
                T.assert_frame_equal(a, b)
    test_optimized_tpch._br_memo = True
    return test_optimized_tpch


class _FakeDuckResult:
    __slots__ = ("_rows",)

    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class _MemoDuckConn:
    def __init__(self, real_connect, rows_map):
        self._connect = real_connect
        self._rows = rows_map
        self._stmts = []
        self._digest = b""
        self._real = None
        self._dirty = False

    def _materialize(self):
        if self._real is None:
            c = self._connect()
            for s in self._stmts:
                c.execute(s)
            self._real = c
        return self._real

    def execute(self, sql, *a, **k):
        if a or k or self._dirty or not isinstance(sql, str):
            self._dirty = True
            return self._materialize().execute(sql, *a, **k)
        s = sql.lstrip().upper()
        if s.startswith(("CREATE ", "INSERT ")):
            self._stmts.append(sql)
            self._digest = hashlib.sha256(self._digest + b"\0" + sql.encode("utf-8")).digest()
            if self._real is not None:
                self._real.execute(sql)
            return self
        rows = self._rows.get((self._digest, sql))
        if rows is not None:
            return _FakeDuckResult(rows)
        return self._materialize().execute(sql)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        if self._real is not None:
            self._real.close()
            self._real = None

    def __getattr__(self, name):
        return getattr(self._materialize(), name)


class _MemoDuckMod:
    def __init__(self, real_mod, rows_map):
        self._real = real_mod
        self._rows = rows_map

    def connect(self, *a, **k):
        if not a and not k:
            return _MemoDuckConn(self._real.connect, self._rows)
        return self._real.connect(*a, **k)

    def __getattr__(self, name):
        return getattr(self._real, name)


def install_opt_duck_memo():
    """Wrap duckdb inside tests.test_optimizer so unchanged DuckDB queries reuse baseline rows."""
    if MEMO_REC is not None or not MEMO_DUCK or MEMO is None:
        return
    rows_map = MEMO.get("duck_rows")
    if not rows_map:
        return
    TO = sys.modules.get("tests.test_optimizer")
    if TO is None:
        return
    if not isinstance(getattr(TO, "duckdb", None), _MemoDuckMod):
        TO.duckdb = _MemoDuckMod(TO.duckdb, rows_map)


MEMO_T = [0.0, 0.0, 0.0]    # seconds: reference-query check, submitted calls, of which optimize


def _memo_duck_same(n):
    """The reference query of submit n is the baseline one (so its dataframe is too)."""
    t0 = time.process_time()
    try:
        return _memo_duck_same0(n)
    finally:
        MEMO_T[0] += time.process_time() - t0


def _memo_duck_same0(n):
    try:
        tid = MEMO_IDS[_SUB["t"]]
        T = sys.modules["tests.test_executor"]
        tpch = MEMO["tpch"][tid]
        sqls = T.TestExecutor.tpch_sqls if tpch else T.TestExecutor.tpcds_sqls
        sql = sqls[n][1]
        q = T.transpile(sql, write="duckdb")[0]
        return MEMO["dsql"].get((tpch, sql)) == q
    except Exception:  # noqa
        return False


# During the image build the hook attributes parsing, execution and comparison
# to the same fixture index, preserving the original two-pass execution order.
QUERY_TRACE = None
PARSE_ONLY = {}  # query indices whose changed lines run only in the initial parse


def _memo_mp_execute(orig):
    def _mp_execute(self, schema, tables, sqls, tpch):
        if MEMO_REC is None and FIXTURE_CALLS is not None and MEMO is not None:
            base = MEMO.get("base_sqls", {}).get(bool(tpch))
            if base is not None:
                changed = [
                    (i, args, sql)
                    for i, (args, sql, _) in enumerate(sqls)
                    if (sql, repr(sorted(args.items()))) not in base
                ]
                if not changed:
                    return
                T = sys.modules[type(self).__module__]
                with T.ProcessPoolExecutor(
                    mp_context=T.get_context("spawn"),
                    initializer=T.initializer,
                    initargs=(schema, tables),
                ) as pool:
                    futures = [(i, pool.submit(T.mp_execute, T.parse_one(sql), args)) for i, args, sql in changed]
                    for i, future in futures:
                        table = future.result()
                        if table is not None:
                            self.subtestHelper(i, table, tpch=tpch)
                return
        allow = _SUB["allow"]
        parse_only = PARSE_ONLY.get(_SUB["t"], ())
        if QUERY_TRACE is not None or allow is not None or parse_only:
            T = sys.modules[type(self).__module__]
            with T.ProcessPoolExecutor(
                mp_context=T.get_context("spawn"), initializer=T.initializer, initargs=(schema, tables),
            ) as pool:
                futures = []
                try:
                    for i, (args, sql, _) in enumerate(sqls):
                        if allow is not None and i not in allow:
                            continue
                        if QUERY_TRACE:
                            QUERY_TRACE(i, True)
                        expression = T.parse_one(sql)
                        if QUERY_TRACE:
                            QUERY_TRACE(i)
                            if MEMO_REC is not None and args.get("execute"):
                                MEMO_REC.setdefault("parsed", {})[(tpch, i)] = digest(expression)
                        elif i in parse_only:
                            if not args.get("execute"):
                                continue
                            expected = MEMO.get("parsed", {}).get((tpch, i))
                            if expected is not None and digest(expression) == expected:
                                continue
                        _SUB["n"] = i
                        futures.append((i, pool.submit(T.mp_execute, expression, args)))
                    for i, future in futures:
                        if QUERY_TRACE:
                            QUERY_TRACE(i)
                        table = future.result()
                        if table is not None:
                            self.subtestHelper(i, table, tpch=tpch)
                        if _SUB["failed"]:
                            break
                finally:
                    if QUERY_TRACE:
                        QUERY_TRACE(None)
            return
        return orig(self, schema, tables, sqls, tpch)
    _mp_execute._br_memo = True
    return _mp_execute


def install_memo():
    """(Re)install the wrappers on the currently imported modules."""
    try:
        T = sys.modules.get("tests.test_executor")
        E = sys.modules.get("sqlglot.executor")
        if T is not None:
            if not getattr(T.TestExecutor.__dict__.get("cached_execute"), "_br_memo", False):
                T.TestExecutor.cached_execute = _memo_cached_execute
            if not getattr(T.TestExecutor.__dict__.get("test_optimized_tpch"), "_br_memo", False):
                T.TestExecutor.test_optimized_tpch = _memo_test_optimized_tpch(T.TestExecutor.test_optimized_tpch)
            if not getattr(T.TestExecutor.__dict__.get("_mp_execute"), "_br_memo", False):
                T.TestExecutor._mp_execute = _memo_mp_execute(T.TestExecutor._mp_execute)
        if E is not None:
            if not getattr(E.optimize, "_br_memo", False):
                E.optimize = _memo_optimize(E.optimize)
    except Exception as e:  # noqa
        log("memo install failed", repr(e))
SUBMIT_LAZY: set = set()  # tests that only read futures one by one via .result()


class _LazyFuture(_cf.Future):
    """Runs the submitted call when its result is first asked for. Once the
    running test has a failing subtest, later calls are skipped (the test fails anyway)."""
    _job = None
    _record_key = None
    _nopickle = False

    def _run(self):
        job, self._job = self._job, None
        if job is None:
            return
        if _SUB["failed"]:
            self.set_result(None)
            return
        fn, a, k = job
        _SUB["cur"] = self._n
        _rearm()
        t0 = time.process_time()
        try:
            try:
                res = fn(*a, **k)
                if self._nopickle:
                    self.set_result(res)
                else:
                    result = pickle.dumps(res)
                    if self._record_key is not None:
                        MEMO_REC["pool_results"][self._record_key] = result
                    self.set_result(pickle.loads(result))
            except _SkipExec:
                if _memo_duck_same(self._n):
                    MEMO_STAT[0] += 1
                    self.set_result(None)  # same plan and same reference query as the baseline
                else:
                    MEMO_STAT[1] += 1
                    _SUB["cur"] = None
                    res = fn(*a, **k)
                    self.set_result(res if self._nopickle else pickle.loads(pickle.dumps(res)))
        except BaseException as e:  # noqa
            self.set_exception(e)
        finally:
            _SUB["cur"] = None
            MEMO_T[1] += time.process_time() - t0
            _rearm(done=True)

    def result(self, timeout=None):
        self._run()
        return super().result(timeout)

    def exception(self, timeout=None):
        self._run()
        return super().exception(timeout)
SUBMIT_FILTER: dict = {}  # test number -> set of submit indices to really execute


# ------------------------------------------------- fixture-iteration filter
# Tests looping over tests.helpers.load_sql_fixtures / load_sql_fixture_pairs run one
# fixture case per iteration. ITER_FILTER[test] is the set of iteration numbers (counted
# over all loader calls of the test, in order) that are really yielded.
ITER_FILTER: dict = {}
ITER_HOOK = None          # build time: (switch(ctx), counts)
_ITER = {"n": 0, "allow": None}
_ITER_LOADERS = ("load_sql_fixtures", "load_sql_fixture_pairs")


def _iter_wrap(orig):
    def loader(filename):
        hook = ITER_HOOK
        allow = _ITER["allow"]
        for item in orig(filename):
            n = _ITER["n"]
            _ITER["n"] = n + 1
            if allow is not None and (n % 2 if allow == "even" else n not in allow):
                continue
            if hook is not None:
                hook[1][0] = n + 1
                hook[0](n)
            yield item
        if hook is not None:
            hook[0](None)
    loader._br_iter = True
    return loader


def install_iter_wrappers():
    """(Re)wrap the loaders in every imported test module that uses them."""
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == "tests" or name.startswith("tests.")):
            continue
        for ln in _ITER_LOADERS:
            f = mod.__dict__.get(ln)
            if f is not None and not getattr(f, "_br_iter", False) and callable(f):
                h = sys.modules.get("tests.helpers")
                base = getattr(h, "_br_orig_" + ln, None) if h is not None else None
                if base is None:
                    base = f
                    if h is not None:
                        setattr(h, "_br_orig_" + ln, f)
                setattr(mod, ln, _iter_wrap(base))


# ------------------------------------------------- shared class fixtures
# TestExecutor.setUpClass parses the TPC csv files into tables (over a second of cpu).
# It is run once in the zygote; a child reuses the result when its patch cannot have
# changed it: no test file touched, no re-import, and no changed line among the sqlglot
# lines the setup ran. DuckDB connections do not survive a fork, so during the shared
# setup they are stand-ins that record their statements and connect on first real use.
PRESETS: dict = {}    # class -> (sqlglot lines its setUpClass ran, original setUpClass)


class _LazyConn:
    def __init__(self, connect):
        self._br_connect = connect
        self._br_stmts = []
        self._br_real = None
        self._br_setup = True

    def execute(self, *a, **k):
        if self._br_setup:
            self._br_stmts.append((a, k))
            return self
        return self._br_conn().execute(*a, **k)

    def _br_conn(self):
        if self._br_real is None:
            real = self._br_connect()
            for a, k in self._br_stmts:
                real.execute(*a, **k)
            self._br_real = real
        return self._br_real

    def close(self):
        if self._br_real is not None:
            self._br_real.close()
            self._br_real = None

    def __reduce__(self):
        return (_lazyconn_restore, (list(self._br_stmts),))

    def __getattr__(self, name):
        if name.startswith("_br_") or self.__dict__.get("_br_setup", True):
            raise AttributeError(name)
        return getattr(self._br_conn(), name)


def _duckdb_connect():
    import duckdb
    return duckdb.connect()


def _lazyconn_restore(stmts):
    c = _LazyConn(_duckdb_connect)
    c._br_stmts = list(stmts)
    c._br_setup = False
    return c


PRESET_ATTRS: dict = {}  # "mod.Cls" -> saved attrs dict for overlay reuse
OV_EXEC_PRESET = [False, False]  # [enabled, reload_sqls]


def prebuild_class(modname, clsname, cache=None):
    """Run a test class's setUpClass once, here, recording the sqlglot lines it runs.
    `cache`: a pickle of the resulting class attributes made at image build time from
    the same pinned checkout (prep.py); loading it replaces the run."""
    try:
        C = getattr(sys.modules[modname], clsname)
        orig = C.__dict__["setUpClass"]
        if cache and os.environ.get("BR_PREP") != "1":
            try:
                with _orig_open(cache, "rb") as fh:
                    saved = pickle.load(fh)
                for k, v in saved["attrs"].items():
                    setattr(C, k, v)
                PRESETS[C] = (saved["lines"], orig)
                PRESET_ATTRS[f"{modname}.{clsname}"] = dict(saved["attrs"])
                return True
            except Exception:  # noqa
                pass
        import duckdb
        before = dict(vars(C))
        real_connect = duckdb.connect
        conns = []

        def connect(*a, **k):
            if a or k:
                raise RuntimeError("unexpected connect arguments")
            c = _LazyConn(real_connect)
            conns.append(c)
            return c

        mon = sys.monitoring
        TOOL = 4
        lines = {}
        prefix = os.path.join(os.path.realpath(REPO), "sqlglot") + os.sep
        root = os.path.realpath(REPO) + os.sep

        def on_line(code, line):
            fn = code.co_filename
            if fn.startswith(prefix):
                lines.setdefault(fn[len(root):], set()).add(line)
            return mon.DISABLE

        mon.use_tool_id(TOOL, "brpre")
        mon.register_callback(TOOL, mon.events.LINE, on_line)
        mon.set_events(TOOL, mon.events.LINE)
        duckdb.connect = connect
        try:
            C.setUpClass()
        finally:
            duckdb.connect = real_connect
            mon.set_events(TOOL, 0)
            mon.register_callback(TOOL, mon.events.LINE, None)
            mon.free_tool_id(TOOL)
        for c in conns:
            c._br_setup = False
        PRESETS[C] = (lines, orig)
        attrs = {k: v for k, v in vars(C).items() if k not in before or before[k] is not v}
        PRESET_ATTRS[f"{modname}.{clsname}"] = dict(attrs)
        if cache and os.environ.get("BR_PREP") == "1":
            with _orig_open(cache, "wb") as fh:
                pickle.dump({"attrs": attrs, "lines": lines}, fh, protocol=5)
        return True
    except BaseException as e:  # noqa
        log("shared class setup failed", clsname, repr(e))
        PRESETS.clear()
        return False


def apply_presets(ok, lines_by_file, sql_only=False):
    """In a child: keep the shared setup of each class if the patch cannot have changed it."""
    kept = 0
    for C, (lines, orig) in PRESETS.items():
        if sql_only and C.__name__ == "TestExecutor":
            T = sys.modules.get("tests.test_executor")
            if T is not None:
                def _reload_sqls(cls, _T=T):
                    cls.cache = {}
                    cls.tpch_sqls = list(_T.load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
                    cls.tpcds_sqls = list(_T.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
                C.setUpClass = classmethod(_reload_sqls)
                kept += 1
                continue
        valid = ok
        if valid:
            for f, ls in (lines_by_file or {}).items():
                if lines.get(f) and lines[f].intersection(ls):
                    valid = False
                    break
        if valid:
            C.setUpClass = classmethod(lambda cls: None)
            kept += 1
    return kept


def apply_overlay_executor_preset(reload_sqls=False):
    """After an overlay re-import where sqlglot.executor.table did not change:
    reuse the zygote's prebuilt TPC tables/connections by rebinding Table.__class__."""
    try:
        attrs = PRESET_ATTRS.get("tests.test_executor.TestExecutor")
        if not attrs:
            return False
        T = sys.modules.get("tests.test_executor")
        if T is None:
            return False
        C = T.TestExecutor
        if C.__dict__.get("_br_ov_preset"):
            return True
        ET = sys.modules.get("sqlglot.executor.table") or importlib.import_module("sqlglot.executor.table")
        new_Table = ET.Table
        for k, v in attrs.items():
            setattr(C, k, {} if k == "cache" else v)
        for tdict in (C.tpch_tables, C.tpcds_tables):
            for tbl in tdict.values():
                tbl.__class__ = new_Table
        if reload_sqls:
            def _reload_sqls(cls, _T=T):
                cls.cache = {}
                cls.tpch_sqls = list(_T.load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
                cls.tpcds_sqls = list(_T.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
            C.setUpClass = classmethod(_reload_sqls)
        else:
            C.setUpClass = classmethod(lambda cls: None)
        C._br_ov_preset = True
        return True
    except Exception:  # noqa
        return False


FIXTURE_SQL_ONLY = False


def install_fixture_sql_filter(mod):
    """When only existing tests/fixtures/*.sql changed, skip entries unchanged from baseline."""
    if not FIXTURE_SQL_ONLY:
        return
    m = sys.modules.get(mod)
    H = sys.modules.get("tests.helpers")
    if m is None or H is None:
        return
    orig_pairs = getattr(m, "load_sql_fixture_pairs", None)
    if orig_pairs is not None and not getattr(orig_pairs, "_br_fx", False):
        def filtered_pairs(filename, _orig=orig_pairs, _H=H):
            p = os.path.realpath(os.path.join(_H.FIXTURES_DIR, filename))
            if p not in _redirect:
                return
            saved = dict(_redirect)
            _redirect.clear()
            try:
                base = {(repr(sorted(meta.items())), sql, exp) for meta, sql, exp in _orig(filename)}
            except FileNotFoundError:
                base = set()
            finally:
                _redirect.update(saved)
            for meta, sql, exp in _orig(filename):
                if (repr(sorted(meta.items())), sql, exp) not in base:
                    yield meta, sql, exp
        filtered_pairs._br_fx = True
        m.load_sql_fixture_pairs = filtered_pairs
    orig_single = getattr(m, "load_sql_fixtures", None)
    if orig_single is not None and not getattr(orig_single, "_br_fx", False):
        def filtered_single(filename, _orig=orig_single, _H=H):
            p = os.path.realpath(os.path.join(_H.FIXTURES_DIR, filename))
            if p not in _redirect:
                return
            saved = dict(_redirect)
            _redirect.clear()
            try:
                base = set(_orig(filename))
            except FileNotFoundError:
                base = set()
            finally:
                _redirect.update(saved)
            for sql in _orig(filename):
                if sql not in base:
                    yield sql
        filtered_single._br_fx = True
        m.load_sql_fixtures = filtered_single


class _AbortTest(Exception):
    """Raised when a test that already has a failing subtest starts another one."""


def install_subtest_failfast():
    orig = unittest.TestCase.subTest

    def subTest(self, *a, **k):
        if _SUB["failed"] and getattr(self, "_br_id", None) is not None:
            raise _AbortTest()  # the test has failed already: its outcome cannot change
        return orig(self, *a, **k)
    unittest.TestCase.subTest = subTest


def install_inline_pools():
    _cf.ProcessPoolExecutor = InlineExecutor
    import concurrent.futures.process as _p
    _p.ProcessPoolExecutor = InlineExecutor


# --------------------------------------------------------------- file overlay
_orig_open = builtins.open
_redirect: dict[str, str] = {}


def _open(file, *a, **k):
    if _redirect:
        try:
            p = os.fspath(file)
            if isinstance(p, str):
                r = _redirect.get(os.path.realpath(p))
                if r is not None:
                    file = r
        except TypeError:
            pass
    return _orig_open(file, *a, **k)


def install_open_hook():
    builtins.open = _open
    io.open = _open


# --------------------------------------------------------------- zygote setup
def module_name_for(rel):
    """tests/dialects/test_x.py -> tests.dialects.test_x"""
    return rel[:-3].replace("/", ".")


def zygote_import(test_modules):
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    sys.path.insert(0, REPO)
    # subprocesses started by tests (e.g. test_lazy_load) must find sqlglot too
    os.environ["PYTHONPATH"] = REPO + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    install_inline_pools()
    install_open_hook()
    if os.environ.get("BR_SUBFF", "1") == "1":
        install_subtest_failfast()
    import sqlglot  # noqa
    if os.environ.get("BR_WARM", "1") == "1":
        import pkgutil
        for m in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
            if m.name.endswith("__main__"):
                continue
            try:
                importlib.import_module(m.name)
            except Exception:  # noqa
                pass
    for m in sorted(test_modules):
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa
            log("zygote import failed", m, repr(e))
    if os.environ.get("BR_WARM", "1") == "1":
        build_func_map()


# --------------------------------------------------------------- hot patching
def _walk_codes(code, out, prefix=()):
    out.append(code)
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            _walk_codes(c, out)
    return out


class NeedReimport(Exception):
    pass


DIFF_HITS: dict = {}
DIFF_BAD: set = set()
_UNINSTR: list = []          # [(func_obj, clean_new_code), ...] for instant un-instrumentation
_OLD_CODES_CACHE: dict = {}  # rel -> compiled old_codes list (cached in zygote)


def precompile_old(rel, src):
    """Compile baseline source once in the zygote when multiple hot jobs edit the same file."""
    if rel not in _OLD_CODES_CACHE:
        path = os.path.join(REPO, rel)
        try:
            _OLD_CODES_CACHE[rel] = _walk_codes(compile(src, path, "exec", dont_inherit=True), [])
        except Exception:  # noqa
            pass


def _uninstrument():
    if _UNINSTR:
        for obj, clean_code in _UNINSTR:
            try:
                obj.__code__ = clean_code
            except Exception:  # noqa
                pass
        _UNINSTR.clear()


def _br_d(k, fo, fn, b):
    v = fn()
    DIFF_HITS[k] = DIFF_HITS.get(k, 0) + 1
    if k in DIFF_BAD:
        return v
    try:
        o = fo()
    except BaseException:  # noqa
        DIFF_BAD.add(k)
        _uninstrument()
        return v
    try:
        if b:
            same = bool(o) == bool(v)
        else:
            same = o is v or (type(o) is type(v) and bool(o == v))
    except BaseException:  # noqa
        same = False
    if not same:
        DIFF_BAD.add(k)
        _uninstrument()
    return v


builtins._br_diff_ = _br_d


def hot_patch(rel, old_src, new_src, instr_tree=None):
    """Swap code objects of changed functions. With instr_tree, the swapped code is the
    instrumented version (old and new expression evaluated side by side).
    Returns (swapped, instrumented: bool)."""
    path = os.path.join(REPO, rel)
    old_codes = _OLD_CODES_CACHE.get(rel)
    if old_codes is None:
        old_codes = _walk_codes(compile(old_src, path, "exec", dont_inherit=True), [])
    new_codes = _walk_codes(compile(new_src, path, "exec", dont_inherit=True), [])
    if len(old_codes) != len(new_codes):
        raise NeedReimport("code count differs")
    pairs = {}
    for o, n in zip(old_codes, new_codes):
        if o.co_qualname != n.co_qualname or o.co_freevars != n.co_freevars:
            raise NeedReimport("code shape differs")
        if o is old_codes[0]:
            continue  # module body
        if o != n:
            pairs[o] = n
    if not pairs:
        return 0, False
    clean_pairs = pairs
    instrumented = False
    if instr_tree is not None:
        try:
            icodes = _walk_codes(compile(instr_tree, path, "exec", dont_inherit=True), [])
            bykey = {}
            for c in icodes:
                bykey.setdefault((c.co_qualname, c.co_firstlineno), []).append(c)
            ipairs = {}
            for o, n in pairs.items():
                cand = bykey.get((n.co_qualname, n.co_firstlineno), [])
                cand = [c for c in cand if c.co_freevars == n.co_freevars and c.co_argcount == n.co_argcount]
                if len(cand) != 1:
                    raise NeedReimport("instr mismatch")
                ipairs[o] = cand[0]
            pairs = ipairs
            instrumented = True
        except NeedReimport:
            instrumented = False
        except Exception:  # noqa
            instrumented = False
    swapped = 0
    funcs = FUNC_MAP.get(path) if FUNC_MAP is not None else None
    if funcs is None:
        funcs = [o for o in gc.get_objects() if type(o) is types.FunctionType and o.__code__.co_filename == path]
    for obj in funcs:
        oc = obj.__code__
        n = pairs.get(oc)
        if n is not None:
            if instrumented:
                _UNINSTR.append((obj, clean_pairs[oc]))
            obj.__code__ = n
            swapped += 1
    return swapped, instrumented


SCALE_NODE_TEST = False


def scaled_node_test(case):
    """Exercise the original stress assertions at two widths with proportional limits."""
    method = case.test_max_nodes
    code = method.__code__
    # Only transform the known pinned fixture. Literal or test changes use the original.
    if not {10000, 1000000, 50000, "(50000)"}.issubset(code.co_consts):
        return method

    def run():
        for divisor in (100, 25):
            constants = {10000: 10000 // divisor, 1000000: 1000000 // divisor,
                         50000: 50000 // divisor, "(50000)": f"({50000 // divisor})"}
            scaled = code.replace(co_consts=tuple(constants.get(v, v) for v in code.co_consts))
            types.FunctionType(scaled, method.__globals__)(case)
    return run


GEN_MEM_MODS = None  # set of affected dialect/generator/parser modules for test_generator_memory


def install_gen_mem_filter():
    """For leaf dialect/generator edits, filter Dialects in test_generator_memory to only
    dialects whose class hierarchy touches an affected module."""
    if not GEN_MEM_MODS:
        return
    TG = sys.modules.get("tests.test_generator")
    if TG is None:
        return
    cls = getattr(TG, "TestGenerator", None)
    if cls is None or getattr(cls.__dict__.get("test_generator_memory"), "_br_gmf", False):
        return
    orig = cls.test_generator_memory

    def filtered(self):
        orig_dialects = TG.Dialects
        mods = GEN_MEM_MODS
        keep = []
        for name in orig_dialects:
            d = TG.Dialect.get_or_raise(name)
            dc = d if isinstance(d, type) else type(d)
            mro = dc.__mro__
            for attr in ("generator_class", "parser_class", "tokenizer_class"):
                sub = getattr(d, attr, None)
                if isinstance(sub, type):
                    mro = mro + sub.__mro__
            if any(getattr(c, "__module__", None) in mods for c in mro):
                keep.append(name)
        if not keep:
            return
        TG.Dialects = keep
        try:
            return orig(self)
        finally:
            TG.Dialects = orig_dialects

    filtered._br_gmf = True
    cls.test_generator_memory = filtered


FUNC_MAP = None


def build_func_map():
    """filename -> live function objects, built once in the zygote."""
    global FUNC_MAP
    m = {}
    for o in gc.get_objects():
        if type(o) is types.FunctionType:
            fn = o.__code__.co_filename
            if fn.startswith(REPO):
                m.setdefault(fn, []).append(o)
    FUNC_MAP = m


# --------------------------------------------------------------- overlay
def build_overlay(root, files):
    """Mirror REPO's sqlglot/ and tests/ under root with symlinks; write changed files."""
    for top in ("sqlglot", "tests"):
        src_top = os.path.join(REPO, top)
        for dirpath, dirnames, filenames in os.walk(src_top):
            rel = os.path.relpath(dirpath, REPO)
            dst = os.path.join(root, rel)
            os.makedirs(dst, exist_ok=True)
            for d in list(dirnames):
                if d == "__pycache__" or d == "fixtures":
                    os.symlink(os.path.join(dirpath, d), os.path.join(dst, d))
                    dirnames.remove(d)
            for f in filenames:
                os.symlink(os.path.join(dirpath, f), os.path.join(dst, f))
    for rel, content in files.items():
        dst = os.path.join(root, rel)
        if rel.startswith("tests/fixtures/"):
            continue  # served by the open() hook
        if os.path.lexists(dst):
            os.remove(dst)
        if content is not None:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with _orig_open(dst, "w", encoding="utf-8") as fh:
                fh.write(content)


def reimport(root, test_modules):
    for name in list(sys.modules):
        if name == "sqlglot" or name.startswith("sqlglot.") or name == "tests" or name.startswith("tests."):
            del sys.modules[name]
    sys.path.insert(0, root)
    os.environ["PYTHONPATH"] = root + os.pathsep + os.environ.get("PYTHONPATH", "")
    importlib.invalidate_caches()
    import sqlglot  # noqa


# --------------------------------------------------------------- running
def _trace(test, err):
    if os.environ.get("BR_TRACE"):
        import traceback
        log("FAIL", str(test)[:200], "".join(traceback.format_exception(*err))[-1500:])


class Result(unittest.TestResult):
    def __init__(self, on_done=None):
        super().__init__()
        self.bad = set()
        self.done = set()
        self.on_done = on_done

    def stopTest(self, test):
        super().stopTest(test)
        i = self._id(test)
        if i is not None:
            self.done.add(i)
        if self.on_done is not None and self.on_done(i, self.bad):
            self.stop()

    def _id(self, test):
        return getattr(test, "_br_id", None)

    def addError(self, test, err):
        _trace(test, err)
        i = self._id(test)
        if i is None:
            # class/module fixture failure (_ErrorHolder): mark whole class
            self.bad.add(("CLASS", str(test)))
        else:
            self.bad.add(i)

    def addFailure(self, test, err):
        self.bad.add(self._id(test))
        _trace(test, err)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            _trace(subtest, err)
            self.bad.add(self._id(test))
            _SUB["failed"] = True

    def addUnexpectedSuccess(self, test):
        self.bad.add(self._id(test))


class Timeout(BaseException):
    pass


TEST_CPU = {}


TIMEOUTS = [0]      # alarms raised in this process: the patched code hangs somewhere


def _alarm(signum, frame):
    TIMEOUTS[0] += 1
    _SUB["failed"] = True   # the running test has failed: no further subtests or submits
    _SUB["abort"] = True
    raise Timeout()


def _timed(run, seconds, t=None, gate=None):
    def wrapped(*a, **k):
        if gate is not None and not gate(t):
            return None     # dropped by a re-plan: stays selected, unrun
        _SUB["n"] = 0
        _SUB["failed"] = False
        _SUB["abort"] = False
        _SUB["lazy"] = t in SUBMIT_LAZY
        _SUB["t"] = t
        _SUB["cur"] = None
        _ITER["n"] = 0
        _ITER["allow"] = ITER_FILTER.get(t)
        if _ITER["allow"] is not None:
            install_iter_wrappers()
        if optmemo.ON[0]:
            optmemo.install()   # before the TPC wrapper, which wraps whatever is bound
        if t in MEMO_IDS and (MEMO_DUCK or MEMO_SKIP or MEMO_REC is not None):
            install_memo()
        _SUB["allow"] = SUBMIT_FILTER.get(t)
        # re-armed every second after it first fires: code that swallows the exception
        # is interrupted again
        signal.setitimer(signal.ITIMER_REAL, seconds, 1.0)
        _SUB["deadline"] = time.time() + seconds
        c0 = time.process_time()
        try:
            if FAST_PROJECTIONS and t in SUBMIT_LAZY:
                import rowprojection
                with rowprojection.enabled():
                    return run(*a, **k)
            return run(*a, **k)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            _SUB["deadline"] = None
            _ITER["allow"] = None   # class fixtures of the next test load everything
            TEST_CPU[t] = TEST_CPU.get(t, 0.0) + time.process_time() - c0
    return wrapped


def _once_fixtures(C):
    """Class fixtures are set up once per process even when the class's tests run in
    several stretches (as they are in one ordinary run of the class)."""
    if C.__dict__.get("_br_once"):
        return
    try:
        setup = C.setUpClass
        state = [False]

        def setUpClass(cls):
            if not state[0]:
                state[0] = True
                setup()
        C.setUpClass = classmethod(setUpClass)
        C.tearDownClass = classmethod(lambda cls: None)
        C._br_once = True
    except Exception:  # noqa
        pass


def run_tests(test_ids, nodeids, timeouts=None, should_stop=None, skipped=None, replan=None, stop_on_timeout=True):
    """test_ids: list of numbers; nodeids: number -> 'tests/x.py::Class::method'.
    Returns set of failing numbers. should_stop(failed_so_far, remaining) -> bool stops
    early. replan(failed_so_far, remaining) -> the remaining tests still worth running
    (None: all), asked whenever another test has failed. Tests not run are added to
    `skipped`."""
    failed = set()
    stopped = [False]
    allowed = [None]
    seen_fail = [0]
    segs = []
    tmo = {}
    for k, t in enumerate(test_ids):
        tmo[t] = timeouts[k] if timeouts else 60.0
        parts = nodeids[t].split("::")
        mod = module_name_for(parts[0])
        if len(parts) == 3:
            key = (mod, parts[1])
            if segs and segs[-1][0] == key:
                segs[-1][1].append((t, parts[2]))
            else:
                segs.append((key, [(t, parts[2])]))
        else:
            failed.add(t)  # unknown layout, treat as failing to be safe
    signal.signal(signal.SIGALRM, _alarm)

    def gate(t):
        return allowed[0] is None or t in allowed[0]

    for ci, ((mod, cls), items) in enumerate(segs):
        if allowed[0] is not None and not stopped[0]:
            drop = [t for t, _ in items if t not in allowed[0]]
            if drop:
                if skipped is not None:
                    skipped.extend(drop)
                items = [(t, m) for t, m in items if t in allowed[0]]
                if not items:
                    continue
        if stopped[0]:
            if skipped is not None:
                skipped.extend(t for t, _ in items)
            continue
        try:
            m = sys.modules.get(mod) or importlib.import_module(mod)
            C = getattr(m, cls)
        except BaseException:  # noqa
            failed.update(t for t, _ in items)
            continue
        if mod == "tests.test_optimizer":
            install_opt_duck_memo()
        elif mod == "tests.test_executor" and OV_EXEC_PRESET[0]:
            apply_overlay_executor_preset(OV_EXEC_PRESET[1])
        elif mod == "tests.test_generator" and GEN_MEM_MODS:
            install_gen_mem_filter()
        if FIXTURE_SQL_ONLY and mod in ("tests.test_optimizer", "tests.test_serde", "tests.test_transpile"):
            install_fixture_sql_filter(mod)
        _once_fixtures(C)
        suite = unittest.TestSuite()
        for t, meth in items:
            try:
                case = C(meth)
                if nodeids[t] in VALIDATOR_CALLS:
                    import validatorcalls
                    validatorcalls.apply(case, VALIDATOR_CALLS[nodeids[t]])
            except BaseException:  # noqa
                failed.add(t)
                continue
            if SCALE_NODE_TEST and mod == "tests.test_parser" and cls == "TestParser" and meth == "test_max_nodes":
                case.test_max_nodes = scaled_node_test(case)
            case._br_id = t
            case.run = _timed(case.run, tmo[t], t, gate)
            suite.addTest(case)

        def on_done(i, bad, _items=items, _ci=ci):
            if TIMEOUTS[0] and stop_on_timeout:
                stopped[0] = True   # the patched code hangs: every further test would cost its timeout
                return True
            if should_stop is None and replan is None:
                return False
            cur = set(failed)
            for b in bad:
                if not isinstance(b, tuple) and b is not None:
                    cur.add(b)
            done_here = res.done
            rest = [t for t, _ in _items if t not in done_here] + [t for _, its in segs[_ci + 1:] for t, _ in its]
            if allowed[0] is not None:
                rest = [t for t in rest if t in allowed[0]]
            if not cur:
                if should_stop is not None and should_stop(cur, rest):
                    stopped[0] = True
                    return True
                return False
            if not rest:
                return False
            if replan is not None:
                if len(cur) == seen_fail[0]:
                    return False
                seen_fail[0] = len(cur)
                keep = replan(cur, rest)
                if keep is None:
                    return False
                allowed[0] = set(keep)
                if not allowed[0]:
                    stopped[0] = True
                    return True
                return False
            if should_stop(cur, rest):
                stopped[0] = True
                return True
            return False

        res = Result(on_done)
        try:
            suite.run(res)
        except BaseException:  # noqa
            failed.update(t for t, _ in items)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        for b in res.bad:
            if isinstance(b, tuple) or b is None:
                failed.update(t for t, _ in items)
            else:
                failed.add(b)
        if skipped is not None:
            skipped.extend(t for t, _ in items if t not in res.done and t not in failed)
    return failed


def prepare(job, modules):
    """Apply a commit inside this (child) process."""
    files = job["files"]
    tmp = job["tmp"]
    for rel, content in (files | job.get("fixture_edits", {})).items():
        if rel.endswith(".py"):
            continue
        dst = os.path.join(tmp, "fx", rel.replace("/", "__"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if content is None:
            _redirect[os.path.realpath(os.path.join(REPO, rel))] = os.path.join(tmp, "missing", "nofile")
        else:
            with _orig_open(dst, "w", encoding="utf-8") as fh:
                fh.write(content)
            _redirect[os.path.realpath(os.path.join(REPO, rel))] = dst
    if job["overlay"]:
        build_overlay(os.path.join(tmp, "ov"), {k: v for k, v in files.items() if k.endswith(".py")})
        reimport(os.path.join(tmp, "ov"), modules)
    else:
        all_instr = True
        for rel in job["hot"]:
            tree = job.get("instr", {}).get(rel)
            _, ok = hot_patch(rel, job["olds"][rel], files[rel], tree)
            all_instr = all_instr and ok and tree is not None
        return all_instr and bool(job.get("instr"))
    return False
