"""Prove that a patch cannot change any test outcome, without running tests.

Only rewrites whose old and new expressions agree for every value they can
actually take are accepted. Anything else is left to the rest of the selector.
"""
from __future__ import annotations

import ast
import enum
import os
import sys

import patchlib

_TREES = {}  # path -> parsed tree, filled as proofs ask for files


def _tree(repo, rel):
    t = _TREES.get(rel)
    if t is None:
        try:
            t = ast.parse(open(os.path.join(str(repo), rel), encoding="utf-8").read())
        except Exception:
            t = False
        _TREES[rel] = t
    return t or None


def _module_name(rel):
    name = rel[:-3].replace("/", ".")
    if name.endswith(".__init__"):
        name = name[: -len(".__init__")]
    return name


def _import_module(repo, rel):
    repo = str(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        return sys.modules.get(_module_name(rel)) or __import__(_module_name(rel), fromlist=["*"])
    except Exception:
        return None


def _dump(n):
    if isinstance(n, list):
        return "[" + ",".join(_dump(x) for x in n) + "]"
    if not isinstance(n, ast.AST):
        return repr(n)
    return ast.dump(n, include_attributes=False)


def _pairs(old_src, new_src, new_tree=None):
    """Expression pairs that are the whole difference, or None if we cannot see it all."""
    if old_src == new_src:
        return []
    # _same_ast compares statements by their source lines, which it reads from these
    patchlib._LA = old_src.splitlines()
    patchlib._LB = new_src.splitlines()
    old, new = patchlib._parse_old(old_src), new_tree if new_tree is not None else ast.parse(new_src)
    if patchlib._same_ast(old, new):
        return []
    d = patchlib._PairDiffer()
    d.diff(old, new, False, None, None)
    if not d.ok or not d.pairs:
        return None
    return d.pairs


def _const(node, module):
    """(ok, value) when the expression is a literal, a builtin constant, or an enum member."""
    if isinstance(node, ast.Constant):
        return True, node.value
    if isinstance(node, ast.Name) and node.id in ("True", "False", "None"):
        return True, {"True": True, "False": False, "None": None}[node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and module is not None:
        cls = getattr(module, node.value.id, None)
        if isinstance(cls, type) and issubclass(cls, enum.Enum) and isinstance(getattr(cls, node.attr, None), cls):
            return True, getattr(cls, node.attr)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.UAdd, ast.USub)):
        ok, v = _const(node.operand, module)
        if not ok:
            return False, None
        if isinstance(node.op, ast.Not):
            return True, not v
        if isinstance(node.op, ast.UAdd):
            return True, +v
        return True, -v
    return False, None


_CMP = {
    ast.Is: lambda a, b: a is b,
    ast.IsNot: lambda a, b: a is not b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def _enum_class(node, module):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and module is not None:
        cls = getattr(module, node.value.id, None)
        if isinstance(cls, type) and issubclass(cls, enum.Enum) and isinstance(getattr(cls, node.attr, None), cls):
            return cls
    if isinstance(node, ast.Name) and module is not None:
        cls = getattr(module, node.id, None)
        if isinstance(cls, type) and issubclass(cls, enum.Enum):
            return cls
    return None


def _enclosing_class(tree, lineno):
    found = None

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_ClassDef(self, n):
            self.stack.append(n)
            self.generic_visit(n)
            self.stack.pop()

        def visit_FunctionDef(self, n):
            if n.lineno <= lineno <= (n.end_lineno or n.lineno) and self.stack:
                nonlocal found
                found = self.stack[-1]
            self.generic_visit(n)

        visit_AsyncFunctionDef = visit_FunctionDef

    V().visit(tree)
    return found


def _self_assigns(class_node, attr):
    out = []
    for n in ast.walk(class_node):
        if isinstance(n, ast.AnnAssign) and n.value is not None:
            targets = [n.target]
            value = n.value
        elif isinstance(n, ast.Assign):
            targets, value = n.targets, n.value
        elif isinstance(n, ast.AugAssign):
            t = n.target
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self" and t.attr == attr:
                return None
            continue
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self" and t.attr == attr:
                out.append(value)
    return out


def _class_assigns(class_node, name):
    out = []
    for stmt in class_node.body:
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        elif isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name:
                out.append(value)
    return out


def _bases(node):
    names = []
    for b in node.bases:
        if isinstance(b, ast.Name):
            names.append(b.id)
        elif isinstance(b, ast.Attribute):
            names.append(b.attr)
    return names


def _hierarchy(repo, class_name):
    """The class and every class in the pinned checkout that names it as a base."""
    key = ("h", class_name)
    hit = _TREES.get(key)
    if hit is not None:
        return hit
    classes = []
    root = os.path.join(str(repo), "sqlglot")
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, str(repo))
            try:
                text = open(path, encoding="utf-8").read()
            except OSError:
                continue
            if class_name not in text:
                continue
            tree = _TREES.get(rel)
            if tree is None:
                try:
                    tree = ast.parse(text)
                except SyntaxError:
                    tree = False
                _TREES[rel] = tree
            if not tree:
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.ClassDef) and (n.name == class_name or class_name in _bases(n)):
                    classes.append(n)
    _TREES[key] = classes
    return classes


def _rhs_is_member(node, enum_cls, module):
    ok, v = _const(node, module)
    return ok and isinstance(v, enum_cls)


def _attr_stays_in_enum(repo, rel, lineno, attr, enum_cls, module):
    """True when every assignment of self.<attr> in this class hierarchy yields a member."""
    tree = _tree(repo, rel)
    if tree is None:
        try:
            tree = ast.parse(open(os.path.join(str(repo), rel), encoding="utf-8").read())
        except Exception:
            return False
    cls = _enclosing_class(tree, lineno)
    if cls is None:
        return False
    classes = _hierarchy(repo, cls.name)
    if not any(c.name == cls.name for c in classes):
        classes.append(cls)
    saw = False
    class_attrs = set()
    for c in classes:
        assigns = _self_assigns(c, attr)
        if assigns is None:
            return False
        for rhs in assigns:
            saw = True
            if isinstance(rhs, ast.Attribute) and isinstance(rhs.value, ast.Name) and rhs.value.id == "self":
                class_attrs.add(rhs.attr)
                continue
            if isinstance(rhs, ast.Call) and _enum_class(rhs.func, module) is enum_cls:
                continue
            if _rhs_is_member(rhs, enum_cls, module):
                continue
            return False
    if not saw:
        return False
    for name in class_attrs:
        defined = False
        for c in classes:
            for rhs in _class_assigns(c, name):
                defined = True
                if not _rhs_is_member(rhs, enum_cls, module):
                    return False
        if not defined:
            return False
    # the two operators agree on every member, which is the whole domain
    return True


def _ops_agree(enum_cls, member, old_op, new_op):
    fo, fn = _CMP.get(type(old_op)), _CMP.get(type(new_op))
    if fo is None or fn is None:
        return False
    try:
        for value in enum_cls:
            if fo(value, member) != fn(value, member):
                return False
            if fo(member, value) != fn(member, value):
                return False
    except Exception:
        return False
    return True


def _none_ops_agree(old_op, new_op, samples):
    fo, fn = _CMP.get(type(old_op)), _CMP.get(type(new_op))
    if fo is None or fn is None:
        return False
    try:
        for value in samples:
            if fo(value, None) != fn(value, None) or fo(None, value) != fn(None, value):
                return False
    except Exception:
        return False
    return True


def _no_custom_eq(cls):
    if not isinstance(cls, type):
        return False
    for c in cls.__mro__:
        if c is object:
            continue
        if "__eq__" in c.__dict__ or "__ne__" in c.__dict__:
            return False
    return True


def _resolve_class(node, module, repo=None, rel=None):
    if isinstance(node, ast.Name):
        cls = getattr(module, node.id, None) if module is not None else None
        if isinstance(cls, type):
            return cls
        # a name imported only for annotations (under TYPE_CHECKING) is still the class it names
        if repo is not None and rel is not None:
            tree = _tree(repo, rel)
            if tree is not None:
                for n in ast.walk(tree):
                    if isinstance(n, ast.ImportFrom) and n.module:
                        for alias in n.names:
                            if (alias.asname or alias.name) == node.id:
                                try:
                                    mod = __import__(n.module, fromlist=[alias.name])
                                    got = getattr(mod, alias.name, None)
                                except Exception:
                                    got = None
                                if isinstance(got, type):
                                    return got
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and module is not None:
        mod = getattr(module, node.value.id, None)
        cls = getattr(mod, node.attr, None) if mod is not None else None
        if isinstance(cls, type):
            return cls
    return None


def _annotated_classes(ann, module, repo=None, rel=None):
    """Classes named by an annotation like `Table | None` or `Optional[Table]`."""
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        return _annotated_classes(ann.left, module, repo, rel) + _annotated_classes(ann.right, module, repo, rel)
    if isinstance(ann, ast.Constant) and ann.value is None:
        return []
    if isinstance(ann, ast.Name) and ann.id in ("None", "Optional"):
        return []
    if isinstance(ann, ast.Subscript):
        return _annotated_classes(ann.slice, module, repo, rel)
    cls = _resolve_class(ann, module, repo, rel)
    return [cls] if cls is not None else [object()]  # unknown annotation: not a safe class


def _attr_annotation(class_node, attr):
    for n in ast.walk(class_node):
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Attribute) \
                and isinstance(n.target.value, ast.Name) and n.target.value.id == "self" and n.target.attr == attr:
            return n.annotation
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == attr:
            return n.annotation
    return None


def _none_domain(repo, rel, lineno, attr, module):
    """Values self.<attr> can hold, when every one of them compares with None by identity."""
    tree = _tree(repo, rel)
    if tree is None:
        return None
    cls = _enclosing_class(tree, lineno)
    if cls is None:
        return None
    ann = _attr_annotation(cls, attr)
    if ann is None:
        return None
    classes = [c for c in _annotated_classes(ann, module, repo, rel) if c is not None]
    if not classes or any(not _no_custom_eq(c) for c in classes):
        return None
    assigns = _self_assigns(cls, attr)
    if not assigns:
        return None
    for rhs in assigns:
        ok, v = _const(rhs, module)
        if ok and v is None:
            continue
        # constructed value: a call, a subscript, or an attribute of self. Its static type is the
        # annotation, and that type has no __eq__, so it compares with None by identity.
        if not isinstance(rhs, (ast.Call, ast.Subscript, ast.Attribute, ast.Name)):
            return None
    samples = [None]
    for c in classes:
        try:
            samples.append(object.__new__(c))
        except Exception:
            return None
    return samples


def _prove(old, new, repo, rel):
    module = _import_module(repo, rel)
    if _syntax_not(old, new):
        return True
    if not (isinstance(old, ast.Compare) and isinstance(new, ast.Compare)):
        ok_a, va = _const(old, module)
        ok_b, vb = _const(new, module)
        return bool(ok_a and ok_b and va == vb and type(va) is type(vb))
    if len(old.ops) != 1 or len(new.ops) != 1:
        return False
    if _dump(old.left) != _dump(new.left) or _dump(old.comparators) != _dump(new.comparators):
        return False
    # both sides constant: the comparison itself is constant, so evaluate it
    sides = [old.left, *old.comparators]
    consts = [_const(s, module) for s in sides]
    if all(ok for ok, _ in consts):
        fo, fn = _CMP.get(type(old.ops[0])), _CMP.get(type(new.ops[0]))
        if fo is None or fn is None:
            return False
        vals = [v for _, v in consts]
        try:
            return fo(vals[0], vals[1]) == fn(vals[0], vals[1])
        except Exception:
            return False
    if {type(old.ops[0]), type(new.ops[0])} not in ({ast.Is, ast.Eq}, {ast.IsNot, ast.NotEq}):
        return False
    enum_side, other = None, None
    for side in (old.left, old.comparators[0]):
        cls = _enum_class(side, module)
        if cls is not None and enum_side is None:
            enum_side = side
        else:
            other = side
    if enum_side is not None and isinstance(other, ast.Attribute) and isinstance(other.value, ast.Name) \
            and other.value.id == "self":
        enum_cls = _enum_class(enum_side, module)
        member = _const(enum_side, module)[1]
        return _attr_stays_in_enum(repo, rel, getattr(old, "lineno", 1), other.attr, enum_cls, module) \
            and _ops_agree(enum_cls, member, old.ops[0], new.ops[0])
    if any(isinstance(s, ast.Constant) and s.value is None for s in (old.left, old.comparators[0])):
        other = old.comparators[0] if (isinstance(old.left, ast.Constant) and old.left.value is None) else old.left
        if isinstance(other, ast.Attribute) and isinstance(other.value, ast.Name) and other.value.id == "self":
            samples = _none_domain(repo, rel, getattr(old, "lineno", 1), other.attr, module)
            return bool(samples) and _none_ops_agree(old.ops[0], new.ops[0], samples)
    return False


def _syntax_not(old, new):
    """`not (a is b)` and `a is not b` are the same expression, and the same for `is not`."""
    def unwrap(n):
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) and isinstance(n.operand, ast.Compare) \
                and len(n.operand.ops) == 1 and isinstance(n.operand.ops[0], (ast.Is, ast.IsNot)):
            return n.operand
        return None

    wrapped_old, wrapped_new = unwrap(old), unwrap(new)
    if (wrapped_old is None) == (wrapped_new is None):
        return False
    wrapped = wrapped_old or wrapped_new
    bare = new if wrapped_old else old
    if not isinstance(bare, ast.Compare) or len(bare.ops) != 1 or len(wrapped.ops) != 1:
        return False
    flip = {ast.Is: ast.IsNot, ast.IsNot: ast.Is}
    if type(bare.ops[0]) is not flip.get(type(wrapped.ops[0])):
        return False
    return _dump(bare.left) == _dump(wrapped.left) and _dump(bare.comparators) == _dump(wrapped.comparators)


_CONST_SHAPES = (ast.Constant, ast.Name, ast.Attribute, ast.UnaryOp)


def _provable_shape(old, new):
    """Whether _prove could accept this pair at all. Needs no import of the edited module."""
    if _syntax_not(old, new):
        return True
    if not (isinstance(old, ast.Compare) and isinstance(new, ast.Compare)):
        return isinstance(old, _CONST_SHAPES) and isinstance(new, _CONST_SHAPES)
    if len(old.ops) != 1 or len(new.ops) != 1:
        return False
    if _dump(old.left) != _dump(new.left) or _dump(old.comparators) != _dump(new.comparators):
        return False
    if {type(old.ops[0]), type(new.ops[0])} in ({ast.Is, ast.Eq}, {ast.IsNot, ast.NotEq}):
        return True
    return all(isinstance(s, _CONST_SHAPES) for s in (old.left, *old.comparators))


def maybe_equivalent(files: dict, repo, trees=None) -> bool:
    """Cheap screen for proven_equivalent: False means the proof would fail."""
    return proven_equivalent(files, repo, _provable_shape, trees)


def proven_equivalent(files: dict, repo, prove=None, trees=None) -> bool:
    """True when every edited file is behavior-preserving and at least one edit was proven so.

    `files` maps a repo-relative path to its new text (None when the file is deleted).
    """
    proved = False
    for rel, new in files.items():
        if new is None or not rel.endswith(".py"):
            return False
        old_path = os.path.join(str(repo), rel)
        try:
            old = open(old_path, encoding="utf-8").read()
        except OSError:
            return False
        if old == new:
            continue
        pairs = _pairs(old, new, trees.get(rel) if trees else None)
        if pairs is None:
            return False
        if not pairs:
            continue
        for a, b in pairs:
            if not (prove(a, b) if prove else _prove(a, b, repo, rel)):
                return False
        proved = True
    return proved
