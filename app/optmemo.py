"""Baseline memo for sqlglot.optimizer.optimize (built by buildopt.py)."""
from __future__ import annotations

import enum
import hashlib
import operator
import os
import pickle
import sys
import zlib

from canon import digest, load_tree  # noqa: F401

_Enum = enum.Enum
_sha1 = hashlib.sha1
_item0 = operator.itemgetter(0)


class Unkeyable(Exception):
    pass


def _strict(e, depth=0):
    """Canonical form of a call argument; raises for anything it cannot describe exactly."""
    if depth > 400:
        raise Unkeyable()
    t = type(e)
    if e is None or t is str or t is int or t is bool or t is float:
        return e
    if t is list or t is tuple:
        d1 = depth + 1
        return ["L" if t is list else "U"] + [_strict(x, d1) for x in e]
    if hasattr(e, "_meta") and hasattr(e, "_type"):
        args = getattr(e, "args", None)
        if type(args) is dict:
            d1 = depth + 1
            items = [(k, _strict(v, d1)) for k, v in args.items() if v is not None and v != []]
            items.sort(key=_item0)
            et = e._type
            ec = e.comments
            em = e._meta
            return ("X", t.__qualname__, items,
                    None if et is None else _strict(et, d1),
                    None if not ec else _strict(ec, d1),
                    None if not em else _strict(em, d1))
    if isinstance(e, (str, int, float, bool)):
        return e
    if isinstance(e, _Enum):
        return ("E", t.__qualname__, e.name)
    if isinstance(e, type):
        return ("T", e.__module__, e.__qualname__)
    if isinstance(e, (list, tuple)):
        d1 = depth + 1
        return ["L" if isinstance(e, list) else "U"] + [_strict(x, d1) for x in e]
    if t is dict or isinstance(e, dict):
        d1 = depth + 1
        items = [(repr(_strict(k, d1)), _strict(v, d1)) for k, v in e.items()]
        items.sort(key=_item0)
        return ("D", items)
    if isinstance(e, (set, frozenset)):
        d1 = depth + 1
        return ("S", sorted(repr(_strict(x, d1)) for x in e))
    args = getattr(e, "args", None)
    if isinstance(args, dict) and hasattr(e, "_meta") and hasattr(e, "_type"):
        d1 = depth + 1
        items = [(k, _strict(v, d1)) for k, v in args.items() if v is not None and v != []]
        items.sort(key=_item0)
        return ("X", t.__qualname__, items,
                _strict(e._type, d1), _strict(e.comments or None, d1), _strict(e._meta or None, d1))
    mod = t.__module__
    if mod.startswith("sqlglot.dialects") and hasattr(e, "settings") and hasattr(e, "version"):
        d1 = depth + 1
        return ("DIALECT", t.__qualname__, _strict(tuple(e.version), d1), _strict(e.normalization_strategy, d1),
                _strict(dict(e.settings), d1))
    if mod == "sqlglot.schema" and t.__qualname__ == "MappingSchema":
        d1 = depth + 1
        items = [(k, _strict(v, d1)) for k, v in vars(e).items()]
        items.sort(key=_item0)
        return ("SCHEMA", items)
    raise Unkeyable()


def call_key(expression, schema, db, catalog, dialect, sql, kwargs):
    try:
        parts = (_strict(expression), _strict(schema), _strict(db), _strict(catalog), _strict(dialect),
                 _strict(sql), _strict(kwargs))
        return _sha1(repr(parts).encode("utf-8", "replace")).digest()
    except RecursionError:
        raise Unkeyable()


def replace_everywhere(old, new):
    """Rebind every module global that is `old` (modules import optimize by name)."""
    n = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == "sqlglot" or name.startswith("sqlglot.") or name.startswith("tests")):
            continue
        d = getattr(mod, "__dict__", None)
        if not d:
            continue
        for k, v in list(d.items()):
            if v is old:
                d[k] = new
                n += 1
    return n


# ---------------------------------------------------------------- run time
IDX = None
_FD = [None]
ON = [False]
AFFECTED: list = []     # contexts (0 = preamble, i + 1 = rule i) that may run a changed line
STAT = [0, 0, 0, 0]     # served from memo, short-circuited, diverged, passed through


TIMES = [0.0] * 6      # key, tree loads, digests (inside rules span), affected rules span, pass-through, after divergence
_pt = __import__("time").process_time


class _Final(BaseException):
    pass


class _Diverged(BaseException):
    def __init__(self, i, tree):
        self.i = i
        self.tree = tree


def load(prefix):
    global IDX
    import gc
    try:
        gc.disable()
        with open(prefix + ".idx", "rb") as fh:
            IDX = pickle.load(fh)
        IDX["_bin"] = prefix + ".bin"
    except Exception:  # noqa
        IDX = None
    finally:
        gc.enable()
    return IDX


def affected(lines_by_file):
    """Contexts whose recorded runs touch any of the given lines."""
    if IDX is None:
        return None
    m = 0
    for f, ls in lines_by_file.items():
        d = IDX["cov"].get(f)
        if d:
            for l in ls:
                m |= d.get(l, 0)
    return [i for i in range(len(IDX["rules"]) + 1) if (m >> i) & 1]


_TREE_RAW: dict = {}
_RUN_SAME: set = set()
_CACHE_OK = [False]


def _tree(d):
    raw = _TREE_RAW.get(d)
    if raw is None:
        if _FD[0] is None:
            _FD[0] = os.open(IDX["_bin"], os.O_RDONLY)
        off, ln = IDX["blobs"][d]
        raw = pickle.loads(zlib.decompress(os.pread(_FD[0], ln, off)))
        if len(_TREE_RAW) < 1024:
            _TREE_RAW[d] = raw
    return load_tree(raw)


_ORIG = [None, None]    # the real optimize and its module
CH_MASK = [None]        # changed lines as a bit set over IDX["lineid"] (None: per-call filter off)
WANT = [False]          # the caller only needs to know whether the result is the baseline one


class Base:
    """Returned instead of a tree when WANT is set and the result is the recorded one."""
    __slots__ = ("digest",)

    def __init__(self, d):
        self.digest = d


def executor_key(expression, schema=None, db=None, catalog=None, dialect=None, rules=None, sql=None,
                 copy=True, **kwargs):
    if rules is not None or not copy:
        return None
    try:
        return call_key(expression, schema, db, catalog, dialect, sql, kwargs)
    except Unkeyable:
        return None


def set_changed(lines_by_file):
    """Tell the stand-in which lines the patch changed (function bodies only)."""
    _RUN_SAME.clear()
    if IDX is None or not IDX.get("lineid") or not lines_by_file or not set(lines_by_file) <= IDX["lineid"].keys():
        CH_MASK[0] = None
        _CACHE_OK[0] = False
        return
    m = 0
    for f, ls in lines_by_file.items():
        d = IDX["lineid"].get(f)
        if d:
            for l in ls:
                i = d.get(l)
                if i is not None:
                    m |= 1 << i
    # a changed line that fills a process-level cache or depends on the hash seed is not
    # tied to the calls that happened to run it: no per-call judgement then
    cold = bool(m & IDX.get("cold", 0))
    _CACHE_OK[0] = not cold
    CH_MASK[0] = None if cold or not IDX.get("kcov") else m


def optimize(expression, schema=None, db=None, catalog=None, dialect=None, rules=None, sql=None,
             copy=True, **kwargs):
    """Stand-in for sqlglot.optimizer.optimizer.optimize (module level: tests pickle it)."""
    want, WANT[0] = WANT[0], False
    orig, mod = _ORIG
    RULES = mod.RULES
    if rules is None:
        rules = RULES
    if not ON[0] or IDX is None or rules is not RULES or not copy:
        STAT[3] += 1
        return orig(expression, schema, db, catalog, dialect, rules, sql, copy, **kwargs)
    mask = CH_MASK[0]
    if mask is None and 0 in AFFECTED:
        STAT[3] += 1
        return orig(expression, schema, db, catalog, dialect, rules, sql, copy, **kwargs)
    t0 = _pt()
    try:
        key = call_key(expression, schema, db, catalog, dialect, sql, kwargs)
        st = IDX["keys"].get(key)
    except Unkeyable:
        key = st = None
    TIMES[0] += _pt() - t0
    if st is None or len(RULES) + 1 != len(st):
        STAT[3] += 1
        t0 = _pt()
        try:
            return orig(expression, schema, db, catalog, dialect, rules, sql, copy, **kwargs)
        finally:
            TIMES[4] += _pt() - t0
    if _CACHE_OK[0] and key in _RUN_SAME:
        STAT[0] += 1
        if want:
            return Base(st[-1])
        t0 = _pt()
        try:
            return _tree(st[-1])
        finally:
            TIMES[1] += _pt() - t0
    kc = IDX["kcov"].get(key) if mask is not None and "kcov" in IDX else None
    if kc is not None:
        # this very call, at the pinned commit: which stages ran a changed line?
        sets = IDX["ksets"]
        if sets[kc[0]] & mask:
            STAT[3] += 1
            return orig(expression, schema, db, catalog, dialect, rules, sql, copy, **kwargs)
        aff = [i for i in range(len(RULES)) if sets[kc[i + 1]] & mask]
    elif 0 in AFFECTED:
        STAT[3] += 1
        return orig(expression, schema, db, catalog, dialect, rules, sql, copy, **kwargs)
    else:
        aff = [c - 1 for c in AFFECTED]
    if not aff:
        STAT[0] += 1
        if want:
            return Base(st[-1])
        t0 = _pt()
        try:
            return _tree(st[-1])
        finally:
            TIMES[1] += _pt() - t0
    seq = []
    for n, i in enumerate(aff):
        nxt = aff[n + 1] if n + 1 < len(aff) else None

        def chk(expression, i=i, nxt=nxt):
            t0 = _pt()
            try:
                if digest(expression) != st[i + 1]:
                    raise _Diverged(i, expression)
            finally:
                TIMES[2] += _pt() - t0
            if nxt is None:
                raise _Final()
            t0 = _pt()
            try:
                return _tree(st[nxt])
            finally:
                TIMES[1] += _pt() - t0
        seq += [RULES[i], chk]
    t0 = _pt()
    first = _tree(st[aff[0]])
    TIMES[1] += _pt() - t0
    t0 = _pt()
    try:
        return orig(first, schema, db, catalog, dialect, tuple(seq), sql, False, **kwargs)
    except _Final:
        STAT[1] += 1
        if _CACHE_OK[0]:
            _RUN_SAME.add(key)
        TIMES[3] += _pt() - t0
        if want:
            return Base(st[-1])
        t0 = _pt()
        try:
            return _tree(st[-1])
        finally:
            TIMES[1] += _pt() - t0
    except _Diverged as e:
        STAT[2] += 1
        TIMES[3] += _pt() - t0
        t0 = _pt()
        try:
            return orig(e.tree, schema, db, catalog, dialect, tuple(RULES[e.i + 1:]), sql, False, **kwargs)
        finally:
            TIMES[5] += _pt() - t0


def install():
    """(Re)bind optimize in the currently imported modules to the stand-in."""
    mod = sys.modules.get("sqlglot.optimizer.optimizer")
    if mod is None or IDX is None:
        return
    cur = mod.__dict__.get("optimize")
    if cur is None or cur is optimize:
        return
    _ORIG[0], _ORIG[1] = cur, mod
    replace_everywhere(cur, optimize)
