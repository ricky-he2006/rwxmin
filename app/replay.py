"""Parse-phase record / replay.

A "phase" is one outermost call of Dialect.parse / Dialect.parse_into: tokenize + parse.
At the pinned commit every test's phase calls are recorded as (input key, output digest).
For a patch that only changes function bodies, a test that runs the changed lines only
inside such phases cannot change its outcome unless one of its recorded inputs now gives
a different output; replaying the inputs under the patch finds exactly those tests.

The output digest covers everything a caller can observe: the returned trees (classes,
args, comments, types, meta), a raised exception (type, message, structured errors) and
the log records emitted through the "sqlglot" logger.
"""
from __future__ import annotations

import enum
import hashlib
import importlib
import json
import logging
import sys
import time

import os
import pickle
import zlib

import runner
from canon import dump_tree, load_tree

PH = {"depth": 0}
REC: list = []        # (key string, digest, seconds) of the running test
UNREC = [0]           # phase calls that could not be recorded (they count as out of phase)


class Unrecordable(Exception):
    pass


def _enc(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return ["v", v]
    if isinstance(v, enum.Enum):
        return ["e", type(v).__module__, type(v).__qualname__, v.name]
    if isinstance(v, type):
        if "<locals>" in v.__qualname__:
            raise Unrecordable()
        return ["t", v.__module__, v.__qualname__]
    if isinstance(v, (list, tuple)):
        return ["l" if isinstance(v, list) else "u", [_enc(x) for x in v]]
    raise Unrecordable()


def _resolve(mod, qual):
    obj = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def _dec(e):
    k = e[0]
    if k == "v":
        return e[1]
    if k == "e":
        return _resolve(e[1], e[2])[e[3]]
    if k == "t":
        return _resolve(e[1], e[2])
    if k == "l":
        return [_dec(x) for x in e[1]]
    if k == "u":
        return tuple(_dec(x) for x in e[1])
    raise ValueError(k)


MAX_SQL = 20000


def shrink(sql, opts):
    """Scaled-down stand-ins for a huge input that is one unit repeated n times
    ("SELECT u,u,u,..."): the same unit repeated far fewer times, with a max_nodes
    limit scaled by the same ratio. Returns [(sql, opts)] or [] if it has no such shape."""
    parts = sql.split(",")
    n = len(parts)
    if n < 2000:
        return []
    unit = parts[1]
    if not unit or any(p != unit for p in parts[1:]) or not parts[0].endswith(unit):
        return []
    head = parts[0][: len(parts[0]) - len(unit)]
    out = []
    for m in (100, 400):
        o = dict(opts)
        mn = o.get("max_nodes")
        if isinstance(mn, int) and not isinstance(mn, bool) and mn > 0:
            o["max_nodes"] = max(1, mn * m // n)
        out.append((head + ",".join([unit] * m), o))
    return out


def make_key(kind, dialect, sql, opts, into=None, long_ok=False):
    cls = type(dialect)
    if "<locals>" in cls.__qualname__ or not isinstance(sql, str) or (len(sql) > MAX_SQL and not long_ok):
        raise Unrecordable()
    settings = {}
    for k, v in dialect.settings.items():
        if not (v is None or isinstance(v, (bool, int, float, str))):
            raise Unrecordable()
        settings[k] = v
    version = list(dialect.version)
    if version == [sys.maxsize, 0, 0]:
        version = None
    ns = dialect.normalization_strategy
    norm = None if ns is cls.NORMALIZATION_STRATEGY else ns.value
    return [kind, cls.__module__, cls.__qualname__, version, norm, settings, sql,
            [[k, _enc(v)] for k, v in sorted(opts.items())], None if into is None else _enc(into)]


_DIALECTS: dict = {}


def _dialect(key):
    s = key[5]
    d = (key[1], key[2], tuple(key[3]) if key[3] is not None else None, key[4],
         tuple(sorted(s.items())) if s else ())
    inst = _DIALECTS.get(d)
    if inst is None:
        cls = _resolve(key[1], key[2])
        kw = dict(s)
        if key[3] is not None:
            kw["version"] = ".".join(str(x) for x in key[3])
        if key[4] is not None:
            kw["normalization_strategy"] = key[4]
        inst = _DIALECTS[d] = cls(**kw)
    return inst


class _Logs(logging.Handler):
    def __init__(self):
        super().__init__(0)
        self.records = []

    def emit(self, record):
        try:
            self.records.append((record.levelname, record.getMessage()))
        except Exception:  # noqa
            self.records.append((record.levelname, "?"))


_LOGGER = logging.getLogger("sqlglot")
_BATCH_LOG = [None]
_sha1 = hashlib.sha1


def begin_batch():
    h = _Logs()
    _LOGGER.addHandler(h)
    _BATCH_LOG[0] = h


def end_batch():
    h = _BATCH_LOG[0]
    _BATCH_LOG[0] = None
    if h is not None:
        _LOGGER.removeHandler(h)


def observe(fn):
    """Run fn() and digest everything observable about it. Returns (digest, result, exc)."""
    bh = _BATCH_LOG[0]
    if bh is not None:
        h = bh
        h.records.clear()
    else:
        h = _Logs()
        _LOGGER.addHandler(h)
    res = exc = None
    try:
        res = fn()
        out = ("OK", runner.canon(res))
    except RecursionError as e:
        exc = e
        out = ("REC",)
    except Exception as e:  # noqa
        exc = e
        # several errors are reported in set-iteration order: compare them as a bag
        errs = getattr(e, "errors", None)
        out = ("EXC", type(e).__qualname__, sorted(str(e).split("\n\n")),
               sorted(repr(runner.canon(x)) for x in errs) if isinstance(errs, list) else runner.canon(errs))
    finally:
        if bh is None:
            _LOGGER.removeHandler(h)
    try:
        dig = _sha1(repr((out, h.records)).encode("utf-8", "replace")).hexdigest()[:16]
    except RecursionError:
        dig = None
    return dig, res, exc


_PCACHE: dict = {}   # parse results reused by the generate inputs recorded after them


def _freeze(x):
    if type(x) is list:
        return tuple(_freeze(i) for i in x)
    if type(x) is dict:
        return tuple(sorted((k, _freeze(v)) for k, v in x.items()))
    return x


def _pkey(key):
    s = key[5]
    opts = key[7]
    into = key[8]
    return (key[0], key[1], key[2], tuple(key[3]) if key[3] is not None else None, key[4],
            tuple(sorted(s.items())) if s else (), key[6],
            _freeze(opts) if opts else (), _freeze(into) if into is not None else None)


def _parse_key(key):
    d = _dialect(key)
    k7 = key[7]
    if k7:
        opts = {k: _dec(v) for k, v in k7}
        if key[0] == "p":
            return d.parse(key[6], **opts)
        return d.parse_into(_dec(key[8]), key[6], **opts)
    if key[0] == "p":
        return d.parse(key[6])
    return d.parse_into(_dec(key[8]), key[6])


def run_key(key, cache_parse=True):
    """Replay one recorded phase call under the current code; returns its digest."""
    if key[0] == "g":
        src, idx = key[6]
        copy = bool(key[8])
        if src == "s":
            try:
                tree = snapshot(idx)
            except Exception:  # noqa
                return None
        else:
            sk = _pkey(src)
            res = _PCACHE.get(sk) if copy else None
            if res is None:
                try:
                    res = _parse_key(src)
                except Exception:  # noqa
                    return None        # its input can no longer be produced
                if copy:
                    if len(_PCACHE) > 512:
                        _PCACHE.clear()
                    _PCACHE[sk] = res
            try:
                tree = res[idx]
            except Exception:  # noqa
                return None
        d = _dialect(key)
        k7 = key[7]
        opts = {k: _dec(v) for k, v in k7} if k7 else {}
        return observe(lambda: ("G", d.generate(tree, copy=copy, **opts), None if copy else runner.canon(tree)))[0]
    bh = _BATCH_LOG[0]
    if bh is not None:
        h = bh
        h.records.clear()
    else:
        h = _Logs()
        _LOGGER.addHandler(h)
    res = exc = None
    try:
        res = _parse_key(key)
        out = ("OK", runner.canon(res))
    except RecursionError as e:
        exc = e
        out = ("REC",)
    except Exception as e:  # noqa
        exc = e
        errs = getattr(e, "errors", None)
        out = ("EXC", type(e).__qualname__, sorted(str(e).split("\n\n")),
               sorted(repr(runner.canon(x)) for x in errs) if isinstance(errs, list) else runner.canon(errs))
    finally:
        if bh is None:
            _LOGGER.removeHandler(h)
    try:
        dig = _sha1(repr((out, h.records)).encode("utf-8", "replace")).hexdigest()[:16]
    except RecursionError:
        dig = None
    if exc is None and cache_parse:
        if len(_PCACHE) > 512:
            _PCACHE.clear()
        _PCACHE[_pkey(key)] = res
    return dig


SNAPS: dict = {}     # build time: digest hex -> compressed snapshot of a generate input
SNAP_INDEX: dict = {}   # run time: digest hex -> (offset, length) in the snapshot file
SNAP_PATH = [None]
_SNAP_FD = [None]


def snapshot(sha):
    """Rebuild a recorded generate input (an exact copy of the baseline tree)."""
    if sha in SNAPS:
        blob = SNAPS[sha]
    else:
        if _SNAP_FD[0] is None:
            _SNAP_FD[0] = os.open(SNAP_PATH[0], os.O_RDONLY)
        off, ln = SNAP_INDEX[sha]
        blob = os.pread(_SNAP_FD[0], ln, off)
    return load_tree(pickle.loads(zlib.decompress(blob)))


TREES: dict = {}     # build time: digest of a parsed tree -> (parse key, index in its result)
UNREC_G = [0]


def install_recorder():
    """Build time: wrap the phase entry points to record their calls."""
    from sqlglot.dialects.dialect import Dialect
    orig_parse, orig_into = Dialect.parse, Dialect.parse_into

    def _surrogates(self, sql, opts):
        """Record scaled-down stand-ins for a huge repetitive input. True if recorded."""
        try:
            small = shrink(sql, opts)
            if not small:
                return False
            recs = []
            for ssql, sopts in small:
                key = json.dumps(make_key("p", self, ssql, sopts, long_ok=True))
                PH["depth"] += 1
                t = time.perf_counter()
                try:
                    dig = observe(lambda: orig_parse(self, ssql, **sopts))[0]
                finally:
                    PH["depth"] -= 1
                recs.append((key, dig, time.perf_counter() - t))
            REC.extend(recs)
            return True
        except Exception:  # noqa
            return False

    def _phase(key_fn, call, big=None):
        if PH["depth"]:
            return call()
        try:
            key = json.dumps(key_fn())
        except Exception:  # noqa
            if big is None or not _surrogates(*big):
                UNREC[0] += 1
                return call()
            # the real call still runs as a phase; the stand-ins are what gets replayed
            PH["depth"] += 1
            try:
                return call()
            finally:
                PH["depth"] -= 1
        PH["depth"] += 1
        t = time.perf_counter()
        try:
            dig, res, exc = observe(call)
        finally:
            PH["depth"] -= 1
        REC.append((key, dig, time.perf_counter() - t))
        if exc is not None:
            raise exc
        if type(res) is list:
            for i, tree in enumerate(res):
                if tree is not None:
                    d = runner.digest(tree)
                    if d is not None:
                        TREES.setdefault(d, (key, i))
        return res

    orig_generate = Dialect.generate

    def generate(self, expression, copy=True, **opts):
        if PH["depth"]:
            return orig_generate(self, expression, copy=copy, **opts)
        try:
            d = runner.digest(expression)
            src = TREES.get(d)
            k = make_key("g", self, "", opts)
            if src is not None:
                k[6] = [json.loads(src[0]), src[1]]
            else:
                # not the direct result of a recorded parse: keep an exact snapshot of it
                sha = d.hex()
                if sha not in SNAPS:
                    SNAPS[sha] = zlib.compress(pickle.dumps(dump_tree(expression), protocol=4), 6)
                k[6] = ["s", sha]
            k[8] = bool(copy)
            key = json.dumps(k)
        except Exception:  # noqa
            UNREC_G[0] += 1      # not replayable: it simply runs outside any phase
            return orig_generate(self, expression, copy=copy, **opts)
        PH["depth"] += 1
        t = time.perf_counter()
        try:
            dig, res, exc = observe(lambda: ("G", orig_generate(self, expression, copy=copy, **opts),
                                             None if copy else runner.canon(expression)))
        finally:
            PH["depth"] -= 1
        REC.append((key, dig, time.perf_counter() - t))
        if exc is not None:
            raise exc
        return res[1]

    Dialect.generate = generate

    def parse(self, sql, **opts):
        big = (self, sql, opts) if isinstance(sql, str) and len(sql) > MAX_SQL else None
        return _phase(lambda: make_key("p", self, sql, opts), lambda: orig_parse(self, sql, **opts), big)

    def parse_into(self, expression_type, sql, **opts):
        return _phase(lambda: make_key("i", self, sql, opts, expression_type),
                      lambda: orig_into(self, expression_type, sql, **opts))

    Dialect.parse = parse
    Dialect.parse_into = parse_into
