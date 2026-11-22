"""Unified diff parsing/applying and Python AST change localisation."""
from __future__ import annotations

import ast
import difflib
import re

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class FilePatch:
    def __init__(self):
        self.old_path = None
        self.new_path = None
        self.hunks = []  # (old_start, old_len, new_start, new_len, [lines])
        self.binary = False
        self.is_new = False
        self.is_deleted = False
        self.mode_change = False

    @property
    def path(self):
        return self.new_path if self.new_path and not self.is_deleted else self.old_path

    def __repr__(self):
        return f"FilePatch({self.old_path}->{self.new_path}, {len(self.hunks)} hunks)"


def _strip_prefix(p):
    p = p.strip()
    if "\t" in p:
        p = p.split("\t", 1)[0]
    if p == "/dev/null":
        return None
    if p.startswith('"') and p.endswith('"'):
        p = p[1:-1]
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    return p


def parse_patch(text: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    cur = None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            cur = FilePatch()
            files.append(cur)
            m = re.match(r"diff --git a/(.*) b/(.*)$", line)
            if m:
                cur.old_path, cur.new_path = m.group(1), m.group(2)
            i += 1
            continue
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            if cur is None or cur.hunks:
                cur = FilePatch()
                files.append(cur)
            old = _strip_prefix(line[4:])
            new = _strip_prefix(lines[i + 1][4:])
            if old is None:
                cur.is_new = True
            else:
                cur.old_path = old
            if new is None:
                cur.is_deleted = True
            else:
                cur.new_path = new
            if old is None and cur.old_path is None:
                pass
            i += 2
            continue
        if cur is not None and (line.startswith("new file mode")):
            cur.is_new = True
        if cur is not None and (line.startswith("deleted file mode")):
            cur.is_deleted = True
        if cur is not None and (line.startswith("old mode") or line.startswith("new mode")):
            cur.mode_change = True
        if cur is not None and line.startswith("rename from "):
            cur.old_path = line[len("rename from "):]
        if cur is not None and line.startswith("rename to "):
            cur.new_path = line[len("rename to "):]
        if cur is not None and (line.startswith("Binary files") or line.startswith("GIT binary patch")):
            cur.binary = True
        m = HUNK_RE.match(line)
        if m and cur is not None:
            os_, ol, ns, nl = m.groups()
            os_, ns = int(os_), int(ns)
            ol = 1 if ol is None else int(ol)
            nl = 1 if nl is None else int(nl)
            body = []
            i += 1
            o_left, n_left = ol, nl
            while i < len(lines) and (o_left > 0 or n_left > 0):
                l = lines[i]
                if l.startswith("\\"):
                    body.append(l)
                    i += 1
                    continue
                tag = l[:1] if l else " "
                if tag == " ":
                    o_left -= 1
                    n_left -= 1
                elif tag == "-":
                    o_left -= 1
                elif tag == "+":
                    n_left -= 1
                else:
                    break
                body.append(l if l else " ")
                i += 1
            # trailing "\ No newline" markers
            while i < len(lines) and lines[i].startswith("\\"):
                body.append(lines[i])
                i += 1
            cur.hunks.append((os_, ol, ns, nl, body))
            continue
        i += 1
    return files


def apply_hunks(old_text: str | None, fp: FilePatch) -> str:
    """Apply the hunks of fp to old_text. Raises ValueError on mismatch."""
    if fp.is_deleted:
        return None
    old_lines = (old_text or "").splitlines(keepends=True)
    out = []
    pos = 0  # 0-based index into old_lines
    for (os_, ol, ns, nl, body) in fp.hunks:
        start = os_ - 1 if ol > 0 else os_
        if start < pos:
            raise ValueError("overlapping hunks")
        out.extend(old_lines[pos:start])
        pos = start
        new_seg = []
        prev_tag = None
        for l in body:
            if l.startswith("\\"):
                # no newline at end of the previous line
                if prev_tag in (" ", "+") and new_seg:
                    new_seg[-1] = new_seg[-1].rstrip("\n")
                continue
            tag, content = l[0], l[1:]
            if tag in (" ", "-"):
                if pos >= len(old_lines):
                    raise ValueError("hunk past end")
                if old_lines[pos].rstrip("\r\n") != content.rstrip("\r\n"):
                    raise ValueError(f"context mismatch at {pos + 1}")
                if tag == " ":
                    new_seg.append(old_lines[pos] if old_lines[pos].endswith("\n") else old_lines[pos] + "\n")
                pos += 1
            elif tag == "+":
                new_seg.append(content + "\n")
            prev_tag = tag
        out.extend(new_seg)
    out.extend(old_lines[pos:])
    res = "".join(out)
    if old_text is not None and not old_text.endswith("\n") and pos >= len(old_lines) and res.endswith("\n"):
        # keep "no newline" status only if last hunk said so (handled above)
        pass
    return res


def changed_old_lines(fp: FilePatch):
    """Old-file line numbers that were removed/replaced, plus insertion points (between lines)."""
    removed = set()
    inserts = set()  # old line number after which content was inserted (0 = before first)
    for (os_, ol, ns, nl, body) in fp.hunks:
        o = os_ if ol > 0 else os_ + 1
        for l in body:
            if l.startswith("\\"):
                continue
            tag = l[0]
            if tag == " ":
                o += 1
            elif tag == "-":
                removed.add(o)
                o += 1
            elif tag == "+":
                inserts.add(o - 1)
    return removed, inserts


# ---------------------------------------------------------------- AST diff

class Change:
    __slots__ = ("kind", "lines", "scope", "node")

    def __init__(self, kind, lines, scope, node=None):
        self.kind = kind      # "runtime" | "import" | "signature"
        self.lines = lines    # (start, end) old-file lines
        self.scope = scope    # tuple of enclosing names (class/def chain)
        self.node = node

    def __repr__(self):
        return f"Change({self.kind}, {self.lines}, {'.'.join(self.scope)})"


_COMPOUND = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try,
             ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Match)
if hasattr(ast, "TryStar"):
    _COMPOUND = _COMPOUND + (ast.TryStar,)


def _dump(n):
    if isinstance(n, ast.AST):
        return ast.dump(n, include_attributes=False)
    return repr(n)


class Differ:
    def __init__(self):
        self.changes: list[Change] = []

    def record(self, node, ctx, scope, stmt):
        """node: changed old node; stmt: the innermost enclosing old statement."""
        if node is None or not hasattr(node, "lineno"):
            node = stmt
        if node is None:
            self.changes.append(Change(ctx, (1, 10**9), scope))
            return
        s = stmt if stmt is not None else node
        start = min(getattr(s, "lineno", node.lineno), node.lineno)
        if isinstance(s, _COMPOUND) and s is not node:
            # header part only: from stmt start to end of the changed node
            end = max(node.end_lineno or node.lineno, start)
        else:
            end = max(getattr(s, "end_lineno", None) or s.lineno, node.end_lineno or node.lineno)
        self.changes.append(Change(ctx, (start, end), scope, node))

    def diff(self, a, b, ctx, scope, stmt):
        if type(a) is not type(b):
            self.record(a if isinstance(a, ast.AST) else None, ctx, scope, stmt)
            return
        if not isinstance(a, ast.AST):
            if a != b:
                self.record(None, ctx, scope, stmt)
            return
        if isinstance(a, ast.stmt):
            stmt = a
        for field in a._fields:
            va = getattr(a, field, None)
            vb = getattr(b, field, None)
            nctx, nscope = ctx, scope
            if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if field == "body":
                    nctx = "runtime"
                    nscope = scope + (a.name,)
                elif field in ("args",):
                    # defaults are import-time; parameter shapes affect callers
                    nctx = "signature" if ctx == "import" else ctx
                    nscope = scope + (a.name,)
                elif field == "name":
                    nctx = ctx
                elif field in ("returns", "type_comment", "type_params"):
                    nctx = "annotation"
            elif isinstance(a, ast.ClassDef) and field == "body":
                nscope = scope + (a.name,)
            elif isinstance(a, ast.Lambda) and field == "body":
                nctx = "runtime"
                nscope = scope + ("<lambda>",)
                if ctx != "runtime":
                    # a lambda defined at import time: only its own body lines run later
                    # (a changed node without a position, e.g. an operator, is placed
                    # on the body's lines)
                    self.diff(va, vb, nctx, nscope, va if hasattr(va, "lineno") else None)
                    continue
            elif isinstance(a, (ast.AnnAssign, ast.arg)) and field == "annotation":
                nctx = "annotation"
            elif isinstance(a, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                pass
            if isinstance(va, list) and isinstance(vb, list):
                self.diff_list(a, field, va, vb, nctx, nscope, stmt)
            elif isinstance(va, ast.AST) and isinstance(vb, ast.AST):
                self.diff(va, vb, nctx, nscope, stmt)
            elif va != vb and not (isinstance(va, ast.AST) or isinstance(vb, ast.AST)):
                self.record(a, nctx if field != "name" else ctx, scope, stmt)
            elif va is not vb and (va is None or vb is None):
                self.record(a, nctx, scope, stmt)

    def diff_list(self, parent, field, la, lb, ctx, scope, stmt):
        if len(la) == len(lb):
            for x, y in zip(la, lb):
                if not _same_text(x, y):
                    self.diff(x, y, ctx, scope, stmt)
            return
        da = [_dump(x) for x in la]
        db = [_dump(y) for y in lb]
        sm = difflib.SequenceMatcher(a=da, b=db, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace" and (i2 - i1) == (j2 - j1):
                for x, y in zip(la[i1:i2], lb[j1:j2]):
                    self.diff(x, y, ctx, scope, stmt)
                continue
            # deletions / insertions: record removed old items and the neighbours
            for x in la[i1:i2]:
                self.record(x if isinstance(x, ast.AST) else None, ctx, scope,
                            x if isinstance(x, ast.stmt) else stmt)
            neigh = []
            if i1 - 1 >= 0:
                neigh.append(la[i1 - 1])
            if i2 < len(la):
                neigh.append(la[i2])
            for x in neigh:
                if isinstance(x, ast.AST) and hasattr(x, "lineno"):
                    if isinstance(x, _COMPOUND):
                        # only its header
                        self.changes.append(Change(ctx, (x.lineno, x.lineno), scope, x))
                    else:
                        self.record(x, ctx, scope, x if isinstance(x, ast.stmt) else stmt)
            if not neigh or i1 == 0:
                # inserted at start of block: parent header
                if isinstance(parent, ast.AST) and hasattr(parent, "lineno"):
                    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and ctx == "runtime":
                        # the function's body start: any test that called it
                        body = parent.body
                        self.changes.append(Change(ctx, (body[0].lineno, body[0].lineno), scope, parent))
                    else:
                        self.changes.append(Change(ctx, (parent.lineno, parent.lineno), scope, parent))
                elif not la:
                    self.changes.append(Change(ctx, (1, 10**9), scope))


_LA: list = []   # source lines of the old / new file being compared
_LB: list = []
_PARSED: dict = {}


def _parse_old(src):
    """Parsed tree of an unpatched file; the same file is diffed for many commits."""
    t = _PARSED.get(src)
    if t is None:
        if len(_PARSED) > 256:
            _PARSED.clear()
        t = _PARSED[src] = ast.parse(src)
    return t


def _start(n):
    d = getattr(n, "decorator_list", None)
    return min([n.lineno] + [x.lineno for x in d]) if d else n.lineno


def _same_text(x, y):
    """Two statements whose source lines are identical have identical trees."""
    if not (isinstance(x, ast.stmt) and isinstance(y, ast.stmt)) or type(x) is not type(y):
        return False
    sa, sb = _start(x), _start(y)
    ea, eb = x.end_lineno, y.end_lineno
    if ea is None or eb is None or ea - sa != eb - sb or x.col_offset != y.col_offset:
        return False
    return _LA[sa - 1:ea] == _LB[sb - 1:eb]


def _same_ast(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if not (_same_text(x, y) or _same_ast(x, y)):
                return False
        return True
    if not isinstance(a, ast.AST):
        return a == b
    for f in a._fields:
        if not _same_ast(getattr(a, f, None), getattr(b, f, None)):
            return False
    return True


def ast_changes(old_src: str, new_src: str, new_tree=None):
    """Returns (identical: bool, changes: list[Change]) or raises SyntaxError for new."""
    global _LA, _LB
    a = _parse_old(old_src)
    b = new_tree if new_tree is not None else ast.parse(new_src)
    _LA, _LB = old_src.splitlines(), new_src.splitlines()
    if _same_ast(a, b):
        return True, []
    d = Differ()
    d.diff(a, b, "import", (), None)
    return False, d.changes


def literal_default_edit(path, old_src, new_tree):
    """Identify a single changed literal default with identical syntax everywhere else."""
    old_tree = _parse_old(old_src)

    def defaults(tree):
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef):
                continue
            for fn in cls.body:
                if not isinstance(fn, ast.FunctionDef):
                    continue
                args = fn.args
                pos = args.posonlyargs + args.args
                for params, values in ((pos[len(pos) - len(args.defaults):], args.defaults),
                                       (args.kwonlyargs, args.kw_defaults)):
                    for i, (param, value) in enumerate(zip(params, values)):
                        if isinstance(value, ast.Constant):
                            yield (cls.name, fn.name, param.arg), (values, i)

    old = dict(defaults(old_tree))
    for key, (values, i) in defaults(new_tree):
        if key not in old:
            continue
        previous, j = old[key]
        if ast.dump(values[i]) == ast.dump(previous[j]):
            continue
        # Restore the literal temporarily to check the entire module for other edits.
        saved = values[i]
        values[i] = previous[j]
        try:
            if ast.dump(old_tree) == ast.dump(new_tree):
                return (path, *key)
        finally:
            values[i] = saved
        break
    return None


# ---------------------------------------------------------------- differential instrumentation

_PURE_CALLS = {"isinstance", "len", "get", "getattr", "hasattr", "type", "bool", "str", "int", "float",
               "tuple", "list", "set", "frozenset", "issubclass", "startswith", "endswith", "lower", "upper",
               "seq_get", "is_type", "find", "name", "text", "items", "keys", "values"}


def _wrappable(e):
    for n in ast.walk(e):
        if isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await, ast.NamedExpr, ast.Starred, ast.Slice,
                          ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                          ast.JoinedStr, ast.FormattedValue)):
            return False
        if isinstance(n, (ast.Name, ast.Attribute, ast.Subscript)) and not isinstance(getattr(n, "ctx", ast.Load()), ast.Load):
            return False
        if isinstance(n, ast.Name) and n.id in ("super", "__class__", "locals", "vars"):
            return False
        if isinstance(n, ast.Call):
            f = n.func
            nm = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if nm not in _PURE_CALLS:
                return False
    return True


class _PairDiffer:
    """Find (old_expr, new_expr) pairs for changes confined to expressions in function bodies."""

    def __init__(self):
        self.pairs = []
        self.ok = True
        self.stack = []   # enclosing new-side function definitions
        self.roots = []   # outermost function of every pair

    def fail(self):
        self.ok = False

    def diff(self, a, b, in_func, expr_pair, parent):
        if not self.ok:
            return
        if type(a) is not type(b):
            if isinstance(a, ast.expr) and isinstance(b, ast.expr) and not isinstance(parent, (ast.JoinedStr, ast.FormattedValue)):
                self.add((a, b), in_func)
            else:
                self.add(expr_pair, in_func)
            return
        if not isinstance(a, ast.AST):
            if a != b:
                self.add(expr_pair, in_func)
            return
        if isinstance(a, ast.expr) and not isinstance(a, ast.Slice) and not isinstance(parent, (ast.JoinedStr, ast.FormattedValue)):
            expr_pair = (a, b)
        elif isinstance(a, ast.stmt):
            expr_pair = None
        is_fn = isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
        if is_fn:
            self.stack.append(b)
        try:
            self._fields(a, b, in_func, expr_pair)
        finally:
            if is_fn:
                self.stack.pop()

    def _fields(self, a, b, in_func, expr_pair):
        for field in a._fields:
            va, vb = getattr(a, field, None), getattr(b, field, None)
            f_in = in_func
            if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
                f_in = field == "body"
            elif isinstance(a, ast.ClassDef):
                f_in = False
            if isinstance(va, list) and isinstance(vb, list):
                if len(va) != len(vb):
                    # e.g. different number of comparators/args: whole enclosing expression
                    if expr_pair is not None and not isinstance(a, ast.stmt):
                        self.add(expr_pair, f_in)
                    else:
                        self.fail()
                    continue
                for x, y in zip(va, vb):
                    if not _same_text(x, y):
                        self.diff(x, y, f_in, expr_pair, a)
            elif isinstance(va, ast.AST) and isinstance(vb, ast.AST):
                self.diff(va, vb, f_in, expr_pair, a)
            elif va != vb:
                if isinstance(va, ast.AST) or isinstance(vb, ast.AST):
                    self.add(expr_pair, f_in)
                elif not isinstance(a, ast.stmt) or field in ("lineno",):
                    self.add(expr_pair, f_in)
                else:
                    self.fail()

    def add(self, pair, in_func):
        if pair is None or not in_func:
            self.fail()
            return
        a, b = pair
        if not (_wrappable(a) and _wrappable(b)):
            self.fail()
            return
        for p in self.pairs:
            if p[1] is b:
                return
        if not self.stack:
            self.fail()
            return
        self.pairs.append(pair)
        self.roots.append(self.stack[0])


def instrument(old_src, new_src, new_tree=None):
    """Return (instrumented new source, number of sites) or None if the change is not
    confined to side-effect-free expressions inside function bodies."""
    global _LA, _LB
    a = _parse_old(old_src)
    b = new_tree if new_tree is not None else ast.parse(new_src)
    _LA, _LB = old_src.splitlines(), new_src.splitlines()
    d = _PairDiffer()
    d.diff(a, b, False, None, None)
    if not d.ok or not d.pairs:
        return None
    roots = list({id(r): r for r in d.roots}.values())
    # collapse nested pairs: keep outermost
    news = [p[1] for p in d.pairs]
    inner = set()
    for p in d.pairs:
        for q in d.pairs:
            if p is not q and any(n is q[1] for n in ast.walk(p[1])):
                inner.add(id(q[1]))
    pairs = [p for p in d.pairs if id(p[1]) not in inner]
    targets = {id(nb): (k, na) for k, (na, nb) in enumerate(pairs)}
    bool_ctx = set()
    for n in (n for r in roots for n in ast.walk(r)):
        if isinstance(n, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            bool_ctx.add(id(n.test))
        elif isinstance(n, ast.comprehension):
            for c in n.ifs:
                bool_ctx.add(id(c))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            bool_ctx.add(id(n.operand))
        elif isinstance(n, ast.BoolOp) and id(n) in bool_ctx:
            for v in n.values:
                bool_ctx.add(id(v))

    class T(ast.NodeTransformer):
        def generic_visit(self, node):
            t = targets.get(id(node))
            if t is not None:
                k, old = t
                mk = lambda body: ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=body)
                call = ast.Call(func=ast.Name("_br_diff_", ast.Load()),
                                args=[ast.Constant(k), mk(old), mk(node), ast.Constant(id(node) in bool_ctx)], keywords=[])
                return ast.copy_location(call, node)
            return super().generic_visit(node)

    # bool context for BoolOp children must be computed before transforming
    tr = T()
    for r in roots:
        tr.visit(r)   # a function definition is never a target: transformed in place
        ast.fix_missing_locations(r)
    return b, len(pairs)
