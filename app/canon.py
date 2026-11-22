"""Seed-independent structural digests of sqlglot expression trees."""
from __future__ import annotations

import enum
import hashlib
import importlib
import operator

_Enum = enum.Enum
_sha1 = hashlib.sha1
_item0 = operator.itemgetter(0)
_new = object.__new__


def canon(e, depth=0):
    t = type(e)
    if e is None or t is str or t is int or t is bool or t is float:
        return e
    if t is list or t is tuple:
        return [canon(x) for x in e]
    if hasattr(e, "_meta"):
        args = getattr(e, "args", None)
        if type(args) is dict:
            # arg insertion order follows set iteration in places, i.e. the process hash seed
            items = [(k, canon(v)) for k, v in args.items() if v is not None and v != []]
            items.sort(key=_item0)
            et = e._type
            ec = e.comments
            em = e._meta
            return (t.__qualname__, items,
                    None if et is None else canon(et),
                    None if not ec else canon(ec),
                    None if not em else canon(em))
    if isinstance(e, (str, int, float, bool)):
        return e
    if isinstance(e, _Enum):
        return ("E", e.name)
    if isinstance(e, (list, tuple)):
        return [canon(x) for x in e]
    if t is dict or isinstance(e, dict):
        return {str(k): canon(v) for k, v in e.items()}
    if isinstance(e, (set, frozenset)):
        # iteration order depends on the process hash seed
        return ("S", sorted(repr(canon(x)) for x in e))
    args = getattr(e, "args", None)
    if isinstance(args, dict) and hasattr(e, "_meta"):
        items = [(k, canon(v)) for k, v in args.items() if v is not None and v != []]
        items.sort(key=_item0)
        return (t.__qualname__, items,
                canon(e._type), canon(e.comments or None), canon(e._meta or None))
    r = repr(e)
    if " at 0x" in r or " object at " in r:
        r = ""   # default reprs carry an address
    return ("O", t.__qualname__, r)


def digest(e):
    try:
        return _sha1(repr(canon(e)).encode("utf-8", "replace")).digest()
    except RecursionError:
        return None


# ---------------------------------------------------------------- exact tree snapshots
# Expression pickling goes through sqlglot.serde, which is lossy (lazy types, empty
# comments) and is itself code a patch may change. These snapshots copy the raw slots and
# rebuild a tree without calling any sqlglot code besides object allocation.
class Undumpable(Exception):
    pass


def _is_expr(e):
    return isinstance(getattr(e, "args", None), dict) and hasattr(e, "_meta") and hasattr(e, "arg_key")


def dump_tree(v):
    t = type(v)
    if v is None or t is str or t is int or t is bool or t is float:
        return v
    if _is_expr(v):
        return ("X", t.__module__, t.__qualname__, [(k, dump_tree(x)) for k, x in v.args.items()],
                dump_tree(v._type), dump_tree(v.comments), dump_tree(v._meta))
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, _Enum):
        return ("E", t.__module__, t.__qualname__, v.name)
    if isinstance(v, list):
        return ("L", [dump_tree(x) for x in v])
    if isinstance(v, tuple):
        return ("U", [dump_tree(x) for x in v])
    if isinstance(v, dict):
        return ("D", [(dump_tree(k), dump_tree(x)) for k, x in v.items()])
    if isinstance(v, frozenset):
        return ("F", [dump_tree(x) for x in v])
    if isinstance(v, set):
        return ("S", [dump_tree(x) for x in v])
    raise Undumpable(t.__qualname__)


_CLS: dict = {}


def _cls(mod, qual):
    c = _CLS.get((mod, qual))
    if c is None:
        c = importlib.import_module(mod)
        for part in qual.split("."):
            c = getattr(c, part)
        _CLS[(mod, qual)] = c
    return c


def load_tree(d):
    if type(d) is not tuple:
        return d
    tag = d[0]
    if tag == "X":
        key = (d[1], d[2])
        cls = _CLS.get(key)
        if cls is None:
            cls = _cls(d[1], d[2])
        node = _new(cls)
        args = {}
        for k, x in d[3]:
            if type(x) is tuple:
                xt = x[0]
                v = load_tree(x)
                if xt == "X":
                    v.parent = node
                    v.arg_key = k
                    v.index = None
                elif xt == "L":
                    for i, (cx, c) in enumerate(zip(x[1], v)):
                        if type(cx) is tuple and cx[0] == "X":
                            c.parent = node
                            c.arg_key = k
                            c.index = i
            else:
                v = x
            args[k] = v
        node.args = args
        node.parent = None
        node.arg_key = None
        node.index = None
        d4 = d[4]
        d5 = d[5]
        d6 = d[6]
        node._type = load_tree(d4) if type(d4) is tuple else d4
        node.comments = load_tree(d5) if type(d5) is tuple else d5
        node._meta = load_tree(d6) if type(d6) is tuple else d6
        node._hash = None
        return node
    if tag == "E":
        key = (d[1], d[2])
        cls = _CLS.get(key)
        if cls is None:
            cls = _cls(d[1], d[2])
        return cls[d[3]]
    if tag == "L":
        return [load_tree(x) if type(x) is tuple else x for x in d[1]]
    if tag == "U":
        return tuple(load_tree(x) if type(x) is tuple else x for x in d[1])
    if tag == "D":
        return {(load_tree(k) if type(k) is tuple else k): (load_tree(x) if type(x) is tuple else x) for k, x in d[1]}
    if tag == "F":
        return frozenset(load_tree(x) if type(x) is tuple else x for x in d[1])
    if tag == "S":
        return {load_tree(x) if type(x) is tuple else x for x in d[1]}
    raise ValueError(tag)
