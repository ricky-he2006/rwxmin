"""Structural diff of import-time state between two imports of sqlglot.

Old modules (kept from the zygote) and new modules (re-imported from the
overlay) are compared attribute by attribute. The result names what changed:
  ("cls", "module:Qual.Name", attr, key-or-None)   class attribute (dict key level)
  ("mod", "module", name, key-or-None)             module global
Objects defined by sqlglot are matched across the two imports by qualified name.
"""
from __future__ import annotations

import enum
import re
import types

_MAX_DEPTH = 80     # deeper than this counts as different (never as equal)


class _Cmp:
    def __init__(self):
        self.memo = {}
        self.budget = 3_000_000

    def same(self, a, b, depth=0):
        self.budget -= 1
        if self.budget < 0:
            raise RuntimeError("diff budget exhausted")
        if a is b:
            return True
        ta, tb = type(a), type(b)
        if ta is not tb and (ta.__qualname__ != tb.__qualname__ or ta.__module__ != tb.__module__):
            return False
        key = (id(a), id(b))
        m = self.memo.get(key)
        if m is not None:
            return m
        self.memo[key] = True  # assume equal on cycles
        r = self._same(a, b, depth)
        self.memo[key] = r
        return r

    def _same(self, a, b, depth):
        if depth > _MAX_DEPTH:
            # e.g. a parsed template expression kept on a class: its leaves sit deep. What
            # cannot be compared is reported as changed.
            return False
        ta = type(a)
        if ta is str or ta is int or ta is bool or a is None:
            return a == b
        if isinstance(a, (bytes, float, complex, range)):
            return a == b
        if isinstance(a, enum.Enum):
            return a.name == b.name and type(a).__qualname__ == type(b).__qualname__
        if isinstance(a, type):
            return a.__qualname__ == b.__qualname__ and a.__module__ == b.__module__
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(self.same(x, y, depth + 1) for x, y in zip(a, b))
        if isinstance(a, dict):
            if len(a) != len(b):
                return False
            if a.keys() == b.keys():
                return all(self.same(a[k], b[k], depth + 1) for k in a)
            ka = {self.key(k): v for k, v in a.items()}
            kb = {self.key(k): v for k, v in b.items()}
            if ka.keys() != kb.keys():
                return False
            return all(self.same(ka[k], kb[k], depth + 1) for k in ka)
        if isinstance(a, (set, frozenset)):
            return len(a) == len(b) and {self.key(x) for x in a} == {self.key(x) for x in b}
        if isinstance(a, types.FunctionType):
            return self.same_code(a.__code__, b.__code__) and self.same(a.__defaults__, b.__defaults__, depth + 1) \
                and self.same(a.__kwdefaults__, b.__kwdefaults__, depth + 1) and self.same_closure(a, b, depth)
        if isinstance(a, (types.BuiltinFunctionType, types.MethodWrapperType)):
            return getattr(a, "__qualname__", None) == getattr(b, "__qualname__", None)
        if isinstance(a, types.MethodType):
            return self.same(a.__func__, b.__func__, depth + 1) and self.same(a.__self__, b.__self__, depth + 1)
        if isinstance(a, (staticmethod, classmethod)):
            return self.same(a.__func__, b.__func__, depth + 1)
        if isinstance(a, property):
            return self.same(a.fget, b.fget, depth + 1) and self.same(a.fset, b.fset, depth + 1)
        if isinstance(a, (types.MemberDescriptorType, types.GetSetDescriptorType, types.WrapperDescriptorType,
                          types.MethodDescriptorType, types.ClassMethodDescriptorType)):
            return a.__name__ == b.__name__ and a.__objclass__.__qualname__ == b.__objclass__.__qualname__
        if type(a).__name__ in ("TypeVar", "ParamSpec", "TypeVarTuple"):
            return a.__name__ == b.__name__
        if type(a).__name__ in ("lock", "RLock", "_RLock", "Lock"):
            return True
        if isinstance(a, re.Pattern):
            return a.pattern == b.pattern and a.flags == b.flags
        tn = type(a).__name__
        if tn in ("dict_values", "dict_keys", "dict_items"):
            # views compare by identity; what they show is what matters
            return self.same(list(a), list(b), depth + 1)
        if tn == "_tuplegetter":
            # a NamedTuple field: (index, doc)
            try:
                return a.__reduce__()[1] == b.__reduce__()[1]
            except Exception:  # noqa
                return False
        if tn == "_abc_data":
            return True
        if isinstance(a, types.ModuleType):
            return a.__name__ == b.__name__
        try:
            import functools
            if isinstance(a, functools.partial):
                return self.same(a.func, b.func, depth + 1) and self.same(a.args, b.args, depth + 1) \
                    and self.same(a.keywords, b.keywords, depth + 1)
        except Exception:  # noqa
            pass
        # instance state: the __dict__ and every slot of the class hierarchy (a subclass
        # without __slots__ has both)
        da = getattr(a, "__dict__", None)
        db = getattr(b, "__dict__", None)
        slots = []
        for klass in type(a).__mro__:
            sa = klass.__dict__.get("__slots__")
            if sa:
                slots.extend([sa] if isinstance(sa, str) else sa)
        if da is None and db is None and not slots:
            try:
                return bool(a == b)
            except Exception:  # noqa
                return repr(a) == repr(b)
        if (da is None) != (db is None):
            return False
        if da is not None and not self.same(da, db, depth + 1):
            return False
        for s in slots:
            if s in ("__dict__", "__weakref__", "_hash"):
                continue
            if not self.same(getattr(a, s, None), getattr(b, s, None), depth + 1):
                return False
        return True

    def key(self, k):
        tk = type(k)
        if tk is str or tk is int or tk is bool or k is None:
            return k
        if isinstance(k, enum.Enum):
            return ("E", tk.__qualname__, k.name)
        if isinstance(k, type):
            return ("T", k.__module__, k.__qualname__)
        if isinstance(k, tuple):
            return tuple(self.key(x) for x in k)
        if isinstance(k, frozenset):
            return ("FS", frozenset(self.key(x) for x in k))
        if isinstance(k, (float, bytes)):
            return k
        return ("O", tk.__qualname__, repr(k))

    def same_code(self, a, b):
        if a.co_code != b.co_code or a.co_names != b.co_names or a.co_varnames != b.co_varnames \
                or a.co_freevars != b.co_freevars or len(a.co_consts) != len(b.co_consts):
            return False
        for x, y in zip(a.co_consts, b.co_consts):
            if isinstance(x, types.CodeType):
                if not isinstance(y, types.CodeType) or not self.same_code(x, y):
                    return False
            elif type(x) is not type(y) or x != y:
                return False
        return True

    def same_closure(self, a, b, depth):
        ca, cb = a.__closure__ or (), b.__closure__ or ()
        if len(ca) != len(cb):
            return False
        for x, y in zip(ca, cb):
            try:
                vx, vy = x.cell_contents, y.cell_contents
            except ValueError:
                continue
            if not self.same(vx, vy, depth + 1):
                return False
        return True


def _classes_of(mod):
    out = []
    seen = set()

    def walk(obj, prefix_ok):
        for v in list(vars(obj).values()):
            if isinstance(v, type) and v.__module__ == mod.__name__ and id(v) not in seen:
                seen.add(id(v))
                out.append(v)
                walk(v, True)
    walk(mod, True)
    return out


_SKIP_ATTRS = {"__dict__", "__weakref__", "__doc__", "__module__", "__qualname__", "__annotations__",
               "__firstlineno__", "__static_attributes__", "__orig_bases__", "__parameters__",
               "__annotate_func__", "__annotations_cache__"}


def diff_state(old_mods: dict, new_mods: dict, only=None):
    """Returns (changes: set of tuples, complete: bool)."""
    cmp = _Cmp()
    changes = set()
    for name, new in new_mods.items():
        if only is not None and name not in only:
            continue
        old = old_mods.get(name)
        if old is None:
            changes.add(("mod", name, "*", None))
            continue
        # module globals
        og, ng = vars(old), vars(new)
        for k in set(og) | set(ng):
            if k.startswith("__") and k.endswith("__"):
                continue
            if k not in og or k not in ng:
                if "__getattr__" in og and "__getattr__" in ng:
                    continue  # lazily resolved module attribute
                v = og.get(k, ng.get(k))
                # lazily populated package attributes (module __getattr__ caches)
                if isinstance(v, (type, types.ModuleType, types.FunctionType)) and \
                        getattr(v, "__module__", getattr(v, "__name__", "")).startswith("sqlglot"):
                    continue
                changes.add(("mod", name, k, None))
                continue
            a, b = og[k], ng[k]
            if isinstance(a, type) and isinstance(b, type):
                if a.__qualname__ != b.__qualname__ or a.__module__ != b.__module__:
                    changes.add(("mod", name, k, None))
                continue
            if isinstance(a, types.ModuleType):
                continue
            if isinstance(a, types.FunctionType) and isinstance(b, types.FunctionType) \
                    and a.__module__ == name and b.__module__ == name:
                # a module-level function: its body is line-covered like a method's
                if not cmp.same(a, b):
                    kind = "methcode" if not cmp.same_code(a.__code__, b.__code__) else "meth"
                    changes.add((kind, name, k, None))
                continue
            if isinstance(a, types.FunctionType) and isinstance(b, types.FunctionType) \
                    and a.__module__ == b.__module__ and a.__module__ != name \
                    and a.__qualname__ == b.__qualname__ and "." not in a.__qualname__ \
                    and vars(old_mods.get(a.__module__) or types).get(a.__qualname__) is a \
                    and vars(new_mods.get(b.__module__) or types).get(b.__qualname__) is b:
                # a re-export of a function another sqlglot module defines at top level:
                # that module's own entry reports the change
                continue
            if not cmp.same(a, b):
                _record(cmp, changes, ("mod", name, k), a, b)
        # classes defined in the module
        oc = {c.__qualname__: c for c in _classes_of(old)}
        nc = {c.__qualname__: c for c in _classes_of(new)}
        for q in set(oc) | set(nc):
            if q not in oc or q not in nc:
                changes.add(("cls", f"{name}:{q}", "*", None))
                continue
            a, b = oc[q], nc[q]
            if [x.__qualname__ for x in a.__mro__] != [x.__qualname__ for x in b.__mro__]:
                changes.add(("cls", f"{name}:{q}", "__mro__", None))
            da, db = vars(a), vars(b)
            for k in set(da) | set(db):
                if k in _SKIP_ATTRS:
                    continue
                if k not in da or k not in db:
                    changes.add(("cls", f"{name}:{q}", k, None))
                    continue
                x, y = da[k], db[k]
                if isinstance(x, type) and isinstance(y, type):
                    if x.__qualname__ != y.__qualname__:
                        changes.add(("cls", f"{name}:{q}", k, None))
                    continue
                if type(x) is type(y) and type(x) in (staticmethod, classmethod):
                    x, y = x.__func__, y.__func__
                if isinstance(x, types.FunctionType) and isinstance(y, types.FunctionType):
                    # body changes are line-covered; defaults/closures are import-time data
                    if not cmp.same(x, y):
                        kind = "methcode" if not cmp.same_code(x.__code__, y.__code__) else "meth"
                        changes.add((kind, f"{name}:{q}", k, None))
                    continue
                if not cmp.same(x, y):
                    _record(cmp, changes, ("cls", f"{name}:{q}", k), x, y)
    return changes


def reordered_only(old_mods, new_mods, change):
    """For a module global that differs after every re-import (a tuple made from a set
    comes out in another order): True when old and new hold the same members."""
    if change[0] != "mod" or change[3] is not None:
        return True
    try:
        a = vars(old_mods[change[1]])[change[2]]
        b = vars(new_mods[change[1]])[change[2]]
    except Exception:  # noqa
        return False
    if not isinstance(a, (tuple, list)) or type(a) is not type(b) or len(a) != len(b):
        return False
    cmp = _Cmp()
    try:
        ka = sorted(repr(cmp.key(x)) for x in a)
        kb = sorted(repr(cmp.key(x)) for x in b)
    except Exception:  # noqa
        return False
    return ka == kb


def _record(cmp, changes, base, a, b):
    """Record a change, at dict-key level when both sides are dicts."""
    if isinstance(a, dict) and isinstance(b, dict):
        ka = {cmp.key(k): v for k, v in a.items()}
        kb = {cmp.key(k): v for k, v in b.items()}
        n = 0
        for k in set(ka) | set(kb):
            if k not in ka or k not in kb:
                changes.add(base + (_keystr(k),))
                n += 1
            elif not cmp.same(ka[k], kb[k]):
                va, vb = ka[k], kb[k]
                if (type(va) is types.FunctionType and type(vb) is types.FunctionType
                        and va.__module__ == vb.__module__
                        and va.__qualname__ == vb.__qualname__
                        and cmp.same(va.__defaults__, vb.__defaults__)
                        and cmp.same(va.__kwdefaults__, vb.__kwdefaults__)
                        and cmp.same_closure(va, vb, 0)):
                    code = va.__code__
                    lines = [l for _, _, l in code.co_lines() if l]
                    if lines:
                        changes.add(("dictcode", base[1], base[2], (code.co_filename, min(lines), max(lines))))
                        n += 1
                        continue
                changes.add(base + (_keystr(k),))
                n += 1
        if n == 0:
            changes.add(base + (None,))
        return
    changes.add(base + (None,))


def _keystr(k):
    if isinstance(k, tuple) and len(k) == 3 and k[0] in ("E", "T", "O"):
        return f"{k[1]}.{k[2]}"
    return repr(k) if not isinstance(k, str) else k
