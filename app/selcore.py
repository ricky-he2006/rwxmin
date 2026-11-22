"""Static candidate selection: which tests could a patch possibly break."""
from __future__ import annotations

import ast
import functools
import gzip
import json
import os
import re
from pathlib import Path

from patchlib import apply_hunks, ast_changes, changed_old_lines, parse_patch

DIALECT_DIRS = ("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/",
                "sqlglot/tokenizers/", "sqlglot/typing/")


class Map:
    def __init__(self, path):
        with gzip.open(path, "rt") as fh:
            d = json.load(fh)
        self.tests = d["tests"]
        self.n = len(self.tests)
        self.sets = [frozenset(s) for s in d["sets"]]
        self.lines = {f: {int(l): i for l, i in m.items()} for f, m in d["lines"].items()}
        self.importlines = {f: set(ls) for f, ls in d["importlines"].items()}
        self.opened = {p: set(s) for p, s in d["opened"].items()}
        self.file_of = [t.split("::", 1)[0] for t in self.tests]
        self._file_users = {}
        self._tests_by_file = {}
        for i, f in enumerate(self.file_of):
            self._tests_by_file.setdefault(f, set()).add(i)

    def covering(self, f, start, end):
        m = self.lines.get(f)
        out = set()
        if not m:
            return out
        if end - start > 5000:
            for l, i in m.items():
                if start <= l <= end:
                    out |= self.sets[i]
            return out
        for l in range(start, end + 1):
            i = m.get(l)
            if i is not None:
                out |= self.sets[i]
        return out

    def file_users(self, f):
        """Tests that executed any line of file f at test time."""
        if f not in self._file_users:
            s = set()
            for i in set(self.lines.get(f, {}).values()):
                s |= self.sets[i]
            self._file_users[f] = s
        return self._file_users[f]

    def tests_in_file(self, tf):
        return set(self._tests_by_file.get(tf, ()))


def dialect_name(path):
    for d in DIALECT_DIRS:
        if path.startswith(d):
            return Path(path).stem
    return None


class Decision:
    def __init__(self):
        self.cands: set[int] = set()
        self.precise: set[int] = set()   # candidates not coming from broad import-time guesses
        self.all = False
        self.default_edit = None
        self.ast_unchanged = set()
        self.parsed = {}  # patched trees reused when instrumenting this change
        self.must: set[int] = set()      # tests that fail for certain: selected without being run
        self.reasons: list[str] = []
        self.import_change = False   # something changed at import time
        self.files: dict[str, str] = {}  # path -> new content (None = deleted)
        self.rt_lines: dict[str, set] = {}  # runtime-changed old lines per sqlglot file
        self.ch_lines: dict[str, set] = {}  # every changed old line per non-test python file

    def add(self, s, why, broad=False):
        self.cands |= s
        if not broad:
            self.precise |= s
        self.reasons.append(f"{why}:{len(s)}")


def analyze(patch_text: str, repo: Path, M: Map) -> Decision:
    D = Decision()
    try:
        fps = parse_patch(patch_text)
    except Exception as e:  # noqa
        D.all = True
        D.reasons.append(f"parse-fail {e!r}")
        return D
    if not fps:
        D.reasons.append("empty patch")
        return D
    for fp in fps:
        _analyze_file(fp, repo, M, D)
    if len(fps) == 1 and not D.all:
        _narrow_tpc_fixture(fps[0], D, repo, M)
    return D


def separate_source(patch_text, repo, coverage):
    """Separate one library edit from unchanged Python syntax and SQL fixtures.

    Each test outside the union of their baseline reaches sees neither edit.
    Preserve that union when multiple files have observable changes.
    """
    pieces = []
    fixtures = set()
    data = {}
    files = parse_patch(patch_text)
    if len(files) < 2:
        return None
    for file in files:
        part = Decision()
        _analyze_file(file, repo, coverage, part)
        if part.ast_unchanged and set(part.files) == part.ast_unchanged:
            continue
        if (not part.all and part.files and not part.import_change
                and all(path.startswith("tests/fixtures/") and path.endswith(".sql") and content is not None
                        for path, content in part.files.items())):
            fixtures.update(part.cands | part.must)
            data.update(part.files)
        else:
            pieces.append(part)
    if len(pieces) == 1 and all(path.startswith("sqlglot/") for path in pieces[0].files):
        return pieces[0], fixtures, data
    return None


@functools.lru_cache(maxsize=256)
def _read(repo, rel):
    p = repo / rel
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


_TEST_IMPORT_RE = re.compile(r"^[ \t]*(?:from[ \t]+([.\w]+)[ \t]+import[ \t]+(\([^)]*\)|[^\n]*)|import[ \t]+([^\n]+))", re.M)


@functools.lru_cache(maxsize=4)
def _test_importers(repo):
    """tests/ file -> the tests/ files that import it, directly or through others (and
    itself). A package's __init__.py is imported by everything below it."""
    files = {}
    for path in sorted((repo / "tests").rglob("*.py")):
        rel = str(path.relative_to(repo))
        try:
            files[rel] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            files[rel] = ""
    by_mod = {}
    for rel in files:
        mod = rel[:-3].replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        by_mod[mod] = rel
    deps = {rel: set() for rel in files}

    def add(rel, name):
        while name:
            f = by_mod.get(name)
            if f and f != rel:
                deps[rel].add(f)
            name = name.rpartition(".")[0]

    for rel, text in files.items():
        mod = rel[:-3].replace("/", ".")
        is_pkg = mod.endswith(".__init__")
        if is_pkg:
            mod = mod[: -len(".__init__")]
        pkg = mod if is_pkg else mod.rpartition(".")[0]
        add(rel, pkg)
        for m in _TEST_IMPORT_RE.finditer(text):
            src, names, plain = m.groups()
            if plain is not None:
                for part in plain.split("#")[0].split(","):
                    add(rel, part.strip().split(" ")[0])
                continue
            if src.startswith("."):
                dots = len(src) - len(src.lstrip("."))
                b = pkg
                for _ in range(dots - 1):
                    b = b.rpartition(".")[0]
                rest = src[dots:]
                src = b + "." + rest if rest else b
            add(rel, src)
            for part in re.sub(r"#[^\n]*", "", names).strip("()").split(","):
                part = part.strip().split(" ")[0]
                if part:
                    add(rel, src + "." + part)
    rev = {rel: {rel} for rel in files}
    for rel, ds in deps.items():
        for d in ds:
            rev[d].add(rel)
    out = {}
    for rel in files:
        seen, todo = {rel}, [rel]
        while todo:
            for x in rev[todo.pop()]:
                if x not in seen:
                    seen.add(x)
                    todo.append(x)
        out[rel] = seen
    return out


def _test_reach(p, repo, M):
    """Tests whose module is p or imports p."""
    files = _test_importers(repo).get(p)
    if files is None:
        return set(M.tests_in_file(p))
    out = set()
    for f in files:
        out |= M.tests_in_file(f)
    return out


def _analyze_file(fp, repo, M, D):
    old_path = fp.old_path
    new_path = fp.new_path
    path = fp.path or old_path or new_path
    if fp.binary:
        D.all = True
        D.reasons.append(f"binary {path}")
        return
    old_text = _read(repo, old_path) if old_path and not fp.is_new else None
    try:
        new_text = apply_hunks(old_text, fp) if fp.hunks or fp.is_deleted else old_text
    except Exception as e:
        D.all = True
        D.reasons.append(f"apply-fail {path} {e!r}")
        return
    renamed = old_path and new_path and old_path != new_path
    if renamed:
        D.files[old_path] = None
        D.files[new_path] = new_text
    else:
        D.files[path] = new_text

    if renamed or fp.is_new or fp.is_deleted:
        # file appears/disappears: anything importing or globbing it
        _structural(path, old_path, new_path, fp, M, D, repo)
        if fp.is_new and not renamed:
            return

    p = old_path or path
    if p.endswith(".py"):
        _python(p, old_text, new_text, fp, M, D, repo)
    elif p.startswith("tests/fixtures/") or p.startswith("tests/"):
        _fixture(p, fp, M, D)
    elif p.startswith("sqlglot/"):
        # non-python file inside the package (e.g. data file)
        users = _opened_users(p, M)
        D.add(users, f"pkgdata {p}")
        if not users and Path(p).name == "py.typed" and new_text is not None:
            D.reasons.append(f"ignored {p}")    # a marker for type checkers; nothing reads it
        elif not users:
            D.all = True
            D.reasons.append(f"unknown pkg file {p}")
    else:
        # docs, setup, CI configs etc. Things that tests could read
        users = _opened_users(p, M)
        if users:
            D.add(users, f"opened {p}")
        elif p in ("setup.py", "setup.cfg", "pyproject.toml") or p.startswith("sqlglotc/"):
            D.add(M.tests_in_file("tests/test_sqlglotc.py") | M.tests_in_file("tests/test_build.py"), f"build {p}")
            if p in ("setup.cfg", "pyproject.toml"):
                if _pytest_sections(old_text) != _pytest_sections(new_text):
                    # pytest reads its options from here: a bad one stops every test
                    D.all = True
                    D.reasons.append(f"pytest-config {p}")
                else:
                    D.import_change = True
                    D.reasons.append(f"config {p}")
        else:
            D.reasons.append(f"ignored {p}")


def _pytest_sections(text):
    """The lines of the sections pytest reads in setup.cfg / pyproject.toml."""
    out, on = [], False
    for line in (text or "").splitlines():
        st = line.strip()
        if st.startswith("["):
            name = st.strip("[] \t").lower()
            on = "pytest" in name
            if on:
                out.append(st)
            continue
        if on and st and not st.startswith("#"):
            out.append(st)
    return out


def _opened_users(p, M):
    s = set()
    for q, users in M.opened.items():
        if q == p or (q.endswith("/") and p.startswith(q)):
            s |= users
    if "tests/fixtures/optimizer/tpc-ds" in p:
        s = {u for u in s if not (M.file_of[u] == "tests/test_executor.py" and "test_execute_tpcds" not in M.tests[u])}
    elif "tests/fixtures/optimizer/tpc-h" in p:
        s = {u for u in s if not (M.file_of[u] == "tests/test_executor.py" and "test_execute_tpch" not in M.tests[u] and "test_optimized_tpch" not in M.tests[u])}
    return s


def _structural(path, old_path, new_path, fp, M, D, repo=None):
    for p in (old_path, new_path):
        if not p:
            continue
        gone = p == old_path and (fp.is_deleted or old_path != new_path)
        name = Path(p).name
        if name == "conftest.py" or name in ("pytest.ini", "tox.ini"):
            # pytest loads these itself: anything can happen
            D.all = True
            D.reasons.append(f"pytest-config {p}")
        elif p.endswith(".py"):
            if p.startswith("tests/"):
                if gone:
                    # the module is gone: its tests and those of every module importing it
                    # cannot be imported any more
                    s = _test_reach(p, repo, M) if repo is not None else None
                    if s is None:
                        D.all = True
                    else:
                        D.must |= s
                    D.reasons.append(f"test-module-gone {p}")
                elif _shadows(p, repo):
                    D.all = True
                    D.reasons.append(f"new module shadows {p}")
                else:
                    D.reasons.append(f"new test-tree module {p}")
            elif p.startswith("sqlglot/") and not gone and not _shadows(p, repo):
                # a module nothing imports yet; tests that list its directory may see it
                users = set()
                parent = str(Path(p).parent) + "/"
                for q, u in M.opened.items():
                    if q == parent or q == p:
                        users |= u
                users |= M.tests_in_file("tests/test_typing.py") | M.tests_in_file("tests/test_dialect_entry_points.py") \
                    | M.tests_in_file("tests/test_dialect_imports.py")
                D.add(users, f"new-module {p}")
                D.import_change = True
            else:
                D.import_change = True
                D.add(M.file_users(p) if p in M.lines else set(), f"py-struct {p}")
                if p.startswith("sqlglot/") or p.startswith("tests/") or _shadows(p, repo):
                    D.all = True
                    D.reasons.append(f"module add/remove {p}")
        else:
            users = _opened_users(p, M)
            parent = str(Path(p).parent) + "/"
            for q, u in M.opened.items():
                if q == parent:
                    users |= u
            D.add(users, f"file-struct {p}")


def _shadows(p, repo):
    """Whether a new .py file takes the place of something importable today."""
    if repo is None:
        return True
    path = Path(p)
    stem = path.stem
    if stem == "__init__":
        # a new package: shadows a module of the same name beside it
        return (repo / path.parent.parent / (path.parent.name + ".py")).exists() or "/" not in str(path.parent)
    sib = repo / path.parent
    if (sib / stem).is_dir() or (sib / (stem + ".py")).exists():
        return True
    if str(path.parent) in (".", ""):
        # top of the checkout is on sys.path: an installed module of that name is hidden
        try:
            import importlib.util
            return importlib.util.find_spec(stem) is not None
        except Exception:  # noqa
            return True
    return False


def _tpc_pairs(text):
    # Match tests.helpers.load_sql_fixture_pairs at the pinned revision.
    sql = "\n".join(line for line in text.splitlines() if line and not line.startswith("--"))
    statements = sql.split(";")
    pairs = []
    for query, expected in zip(statements[::2], statements[1::2]):
        lines = query.strip().split("\n")
        meta = {}
        first = 0
        while lines[first].startswith("#"):
            key, value = lines[first].split(":", 1)
            meta[key.lstrip("#").strip()] = value.strip()
            first += 1
        pairs.append((meta, "\n".join(lines[first:]), expected.strip()))
    return pairs


def _narrow_tpc_fixture(fp, decision, repo, coverage):
    paths = {"tests/fixtures/optimizer/tpc-h/tpc-h.sql": "tpch",
             "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql": "tpcds"}
    path = fp.path
    if fp.old_path not in paths:
        return
    prefix = "tests/test_executor.py::TestExecutor::"
    if fp.is_deleted or fp.old_path != fp.new_path:
        decision.must.update(t for t, tid in enumerate(coverage.tests) if tid.startswith(prefix))
        return
    try:
        new = _tpc_pairs(decision.files[path])
    except (IndexError, ValueError):
        # setUpClass materializes both files before any TestExecutor method.
        decision.must.update(t for t, tid in enumerate(coverage.tests) if tid.startswith(prefix))
        return
    old = _tpc_pairs(_read(repo, path))
    if len(old) != len(new):
        return
    unchanged = set()
    if all((sql, bool(meta.get("execute"))) == (sql2, bool(meta2.get("execute")))
           for (meta, sql, _), (meta2, sql2, _) in zip(old, new)):
        unchanged.add("test_execute_" + paths[path])
    if paths[path] == "tpch" and all(a[1:] == b[1:] for a, b in zip(old, new)):
        unchanged.add("test_optimized_tpch")
    ids = {prefix + name for name in unchanged}
    drop = {t for t in decision.cands if t is not None and coverage.tests[t] in ids}
    decision.cands.difference_update(drop)
    decision.precise.difference_update(drop)


def _fixture(p, fp, M, D):
    users = _opened_users(p, M)
    if users:
        D.add(users, f"fixture {p}")
    else:
        D.reasons.append(f"fixture-unused {p}")


# Lines of the pinned commit that fill a process-level cache (the generator's dispatch
# table, built once per generator class and kept in _DISPATCH_CACHE).
# The same goes for the lazy loaders: the dialect registry (a dialect is imported by name
# the first time anything asks for it) and the packages that resolve attributes on demand.
CACHE_FILL = {"sqlglot/generator.py": [(71, 92), (1003, 1009)],
              "sqlglot/dialects/dialect.py": [(98, 185), (214, 305)],   # the Dialects enum and the registry built from it
              "sqlglot/dialects/__init__.py": [(1, 10 ** 6)],
              "sqlglot/optimizer/__init__.py": [(1, 10 ** 6)]}


def _python(p, old_text, new_text, fp, M, D, repo=None):
    if new_text is None:
        return
    if old_text is None:
        return
    # a byte order mark is not part of the source
    old_text, new_text = old_text.lstrip("\ufeff"), new_text.lstrip("\ufeff")
    try:
        tree = ast.parse(new_text)
        same, changes = ast_changes(old_text, new_text, tree)
        if not same:
            D.parsed[p] = tree
    except (SyntaxError, ValueError, RecursionError):
        # module cannot be imported any more
        D.reasons.append(f"syntax-error {p}")
        if p.startswith("tests/") and repo is not None and Path(p).name != "conftest.py":
            D.must |= _test_reach(p, repo, M)
            return
        D.import_change = True
        _import_break(p, M, D)
        return
    if same:
        # comments / formatting only. Line numbers may shift; only tests that
        # inspect source positions could notice.
        if not fp.is_new and not fp.is_deleted and fp.old_path == fp.new_path:
            D.ast_unchanged.add(p)
        D.reasons.append(f"ast-same {p}")
        return
    if p.startswith("sqlglotc/"):
        # run from its path by tests/test_sqlglotc.py, outside what a child can patch
        D.must |= M.tests_in_file("tests/test_sqlglotc.py")
        D.reasons.append(f"sqlglotc {p}")
        return
    if Path(p).name == "conftest.py":
        D.all = True
        D.reasons.append(f"pytest-config {p}")
        return
    imp_lines = M.importlines.get(p, set())
    is_test = p.startswith("tests/")
    n_lines = old_text.count("\n") + 1
    fill = CACHE_FILL.get(p)
    if fill and any(lo <= l <= hi for ch in changes for l in range(max(1, ch.lines[0]), min(ch.lines[1], n_lines) + 1)
                    for lo, hi in fill):
        # code that runs once per process and whose result every later test reuses: the
        # per-test coverage only names the test that happened to run it first
        D.import_change = True
        D.cands |= set(range(M.n))
        D.reasons.append(f"cache-fill {p}")
        return
    if p.startswith("sqlglot/") and all(ch.kind == "signature" for ch in changes):
        from patchlib import literal_default_edit
        D.default_edit = literal_default_edit(p, old_text, tree)
    for ch in changes:
        s, e = ch.lines
        e = min(e, n_lines)     # "to the end of the file" is recorded as a huge line number
        s = min(s, e)
        if ch.kind == "annotation":
            # with "from __future__ import annotations" these are never evaluated
            if "from __future__ import annotations" in old_text:
                D.reasons.append(f"annotation {p}:{s}")
                continue
            ch.kind = "import"
        if is_test:
            _test_change(p, ch, M, D, repo)
            continue
        D.ch_lines.setdefault(p, set()).update(range(s, e + 1))
        cov = M.covering(p, s, e)
        if ch.kind == "runtime":
            D.rt_lines.setdefault(p, set()).update(range(s, e + 1))
            D.add(cov, f"rt {p}:{s}-{e}")
            if any(l in imp_lines for l in range(s, e + 1)):
                D.import_change = True
                _import_change(p, ch, M, D)
        else:
            D.import_change = True
            D.add(cov, f"cov-imp {p}:{s}-{e}", broad=True)
            _import_change(p, ch, M, D)


def _test_change(p, ch, M, D, repo=None):
    s, e = ch.lines
    if p.endswith(".py") and Path(p).name.startswith("test_"):
        if ch.kind == "runtime" and len(ch.scope) >= 1:
            # a change inside a test method or helper: tests covering those lines
            cov = M.covering(p, s, e)
            D.add(cov, f"testline {p}:{s}")
            # the test method itself (in case coverage missed the line)
            name = ch.scope[-1] if ch.scope[-1] != "<lambda>" else None
            if len(ch.scope) >= 2 and name and name.startswith("test"):
                cls = ch.scope[-2]
                for i in M.tests_in_file(p):
                    parts = M.tests[i].split("::")
                    if parts[-1] == name and (len(parts) < 3 or parts[-2] == cls):
                        D.cands.add(i)
                        D.precise.add(i)
            elif len(ch.scope) == 1 and name and name.startswith("test"):
                for i in M.tests_in_file(p):
                    if M.tests[i].split("::")[-1] == name:
                        D.cands.add(i)
                        D.precise.add(i)
        else:
            # class body, module level, imports: the module has to be imported again, and
            # so do the test modules that import it
            D.add(_test_reach(p, repo, M) if repo is not None else M.tests_in_file(p), f"testfile {p}")
            D.add(M.covering(p, s, e), f"testfile-cov {p}")
            D.import_change = True
    else:
        # helpers.py, __init__.py, etc.
        cov = M.covering(p, s, e)
        D.add(cov, f"helper {p}:{s}")
        if ch.kind != "runtime":
            D.import_change = True
            if repo is None:
                _import_break(p, M, D)
            else:
                D.add(_test_reach(p, repo, M), f"helper-import {p}")


def _module_importers(p, M):
    """Tests that use module p in any way at runtime (rough)."""
    return M.file_users(p)


def _import_change(p, ch, M, D):
    """A change evaluated at import time: constant, class attribute, default, decorator."""
    users = set(_module_importers(p, M))
    dn = dialect_name(p)
    if dn:
        for d in DIALECT_DIRS:
            users |= M.file_users(f"{d}{dn}.py")
        # tests named after the dialect
        users |= M.tests_in_file(f"tests/dialects/test_{dn}.py")
        D.add(users, f"imp-dialect {dn}", broad=True)
    else:
        D.add(users, f"imp-module {p}", broad=True)
    D.reasons.append("IMPORT")


def _import_break(p, M, D):
    # a module that fails to import: everything that imports it (practically all)
    D.all = True
    D.reasons.append(f"import-break {p}")
