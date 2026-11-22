"""Blast Radius selector.

1. Static analysis of every patch against a per-test coverage map of sqlglot at
   the pinned commit gives the candidate tests that could possibly fail.
2. A zygote process imports sqlglot and the candidate test modules once; for
   every commit a forked child applies the change in memory and runs the
   candidates in-process to find out which of them really fail.
3. Only failing tests (plus any candidate we could not afford to run) are
   printed, packed onto workers to minimise makespan + 2 s per worker.

Anything that goes wrong for a commit falls back to the static candidates.
"""
from __future__ import annotations

import json
import math
import os
import re
import resource
import select
import signal
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

T_START = time.time()
DATA = Path(os.environ.get("BR_DATA", "/data"))
REPO = Path(os.environ.get("BR_REPO", "/opt/sqlglot"))
os.environ["BR_REPO"] = str(REPO)
MAP = Path(os.environ.get("BR_MAP", HERE / "map.json.gz"))

# keep the real stdout for answers only; everything else goes to stderr
_OUT_FD = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr

from sched import best_schedule  # noqa: E402
from functools import partial

schedule_cost = partial(best_schedule, want_workers=False)
from selcore import Map, analyze, separate_source  # noqa: E402

HEAVY_SECONDS = float(os.environ.get("BR_HEAVY", "1.5"))
SECONDS_PER_COMMIT = float(os.environ.get("BR_WALL_PER_COMMIT", "4.6"))
PARALLEL = int(os.environ.get("BR_PARALLEL", "2"))
CPU_FACTOR = float(os.environ.get("BR_CPU_FACTOR", "1.0"))  # our cpu / recorded seconds


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def emit(lines):
    data = "".join(json.dumps(l, separators=(",", ":")) + "\n" for l in lines).encode()
    while data:
        n = os.write(_OUT_FD, data)
        data = data[n:]


_TEST = dict(kv.split("=", 1) for kv in os.environ.get("BR_SELFTEST", "").split(",") if "=" in kv)
UNKNOWN_TESTS: set = set()
POISON: set = set()       # commits whose analysis killed an earlier attempt: all their tests
STATIC_ONLY = [False]     # last attempt: no test runs, static candidates only
_PROGRESS_FD = [None]
ANALYZE_TIMEOUT = float(os.environ.get("BR_ANALYZE_TIMEOUT", "25"))
# The batch has 3 minutes (4.5 in the final) and a timeout rejects the entry: whatever
# the number of commits, answers are printed by this many seconds after the start.
WALL_MAX = float(os.environ.get("BR_WALL_MAX", "160"))


def _progress(msg):
    fd = _PROGRESS_FD[0]
    if fd is not None:
        try:
            os.write(fd, (msg + "\n").encode())
        except OSError:
            pass


def supervise():
    """Run main() in a child and make sure every commit gets an answer line.

    If the child dies or hangs while analysing a patch, that commit is answered with
    every test and the work starts over without it. If it dies or overruns later, the
    static candidates are printed. If nothing works, every test of every commit is."""
    try:
        cids = [pf.stem for pf in sorted((DATA / "commits").glob("*.patch"))]
        n_tests = len(json.loads((DATA / "tests.json").read_text())["tests"])
    except Exception as e:  # noqa
        log("supervisor cannot read the inputs", repr(e))
        main()
        return
    n = len(cids)
    hard = min(T_START + max(25.0, SECONDS_PER_COMMIT * n) + 8.0 + 0.2 * n, T_START + WALL_MAX)
    for attempt in range(8):
        if time.time() > T_START + WALL_MAX:
            break
        rd, wr = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(rd)
            _PROGRESS_FD[0] = wr
            try:
                main()
            except BaseException as e:  # noqa
                log("main failed", repr(e))
            os._exit(1)     # main() leaves with os._exit(0) itself
        os.close(wr)
        cur, phase, last, buf = None, "A", time.time(), b""
        while True:
            now = time.time()
            limit = last + ANALYZE_TIMEOUT if phase == "A" else hard if phase == "R" else now + 60.0
            limit = min(limit, T_START + WALL_MAX + 5.0)
            try:
                r, _, _ = select.select([rd], [], [], max(0.05, min(limit - now, 1.0)))
            except InterruptedError:
                continue
            if r:
                ch = os.read(rd, 4096)
                if not ch:
                    break
                buf += ch
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.decode("utf-8", "replace")
                    last = time.time()
                    if line.startswith("A "):
                        cur = line[2:]
                    elif line in ("R", "E"):
                        phase = line
                continue
            if time.time() > limit and phase != "E":
                log(f"supervisor: no progress in phase {phase} (commit {cur}); stopping the worker")
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                break
        try:
            _, st = os.waitpid(pid, 0)
        except OSError:
            st = -1
        os.close(rd)
        if phase == "E":
            return
        if phase == "A" and cur is not None and cur not in POISON:
            log(f"supervisor: worker lost while analysing {cur} (status {st}); answering it with every test")
            POISON.add(cur)
        elif STATIC_ONLY[0]:
            break
        else:
            log(f"supervisor: worker lost in phase {phase} (status {st}); falling back to static candidates")
            STATIC_ONLY[0] = True
    log("supervisor: last resort, every test for every commit")
    workers = [list(range(i, n_tests, 8)) for i in range(8)] if n_tests >= 8 else [list(range(n_tests))]
    emit([{"commit": c, "workers": [w for w in workers if w]} for c in cids])


def broken_leaf_tests(decision, coverage, nodeids):
    """Bound a single missing/unparseable dialect implementation using cold import traces."""
    if len(decision.files) != 1 or coverage.tests != nodeids or STATIC_ONLY[0]:
        return None
    path, content = next(iter(decision.files.items()))
    if not re.fullmatch(r"sqlglot/(?:dialects|generators|parsers|typing)/[a-z][a-z0-9_]*\.py", path):
        return None
    if content is not None and f"syntax-error {path}" not in decision.reasons:
        return None
    imports = load_imports()
    module = _modname(path)
    if not imports or module not in imports["by_mod"]:
        return None
    selected = set(imports["by_mod"][module]) | coverage.file_users(path)
    selected.update(set(range(len(nodeids))) - imports["known"])
    # Reads and directory enumeration can observe a deletion without importing it.
    for opened, tests in coverage.opened.items():
        if path == opened or (opened.endswith("/") and path.startswith(opened)):
            selected.update(tests)
    for file in ("tests/test_typing.py", "tests/test_sqlglotc.py", "tests/test_build.py",
                 "tests/test_dialect_entry_points.py", "tests/test_dialect_imports.py"):
        selected.update(coverage.tests_in_file(file))
    # A fresh Python launched from a test is outside the import trace. Include
    # those call sites, even when the outer process never imported this module.
    for file, line_map in coverage.lines.items():
        if file.startswith("tests/"):
            for number, line in enumerate((REPO / file).read_text().splitlines(), 1):
                if "subprocess" in line and number in line_map:
                    selected.update(coverage.sets[line_map[number]])
    return sorted(selected)


def calibrate_cpu():
    global CPU_FACTOR
    if "BR_CPU_FACTOR" not in os.environ and os.environ.get("BR_CALIBRATE", "1") == "1":
        try:
            def _probe():
                d = {}
                acc = 0
                for i in range(15000):
                    k = f"k_{i & 255}"
                    v = (d.get(k, 0) + i) ^ (i >> 2)
                    d[k] = v
                    acc ^= v
                return acc
            _times = []
            for _ in range(3):
                _ct0 = time.process_time()
                _probe()
                _times.append(time.process_time() - _ct0)
            _cdt = min(_times)
            # Pure-Python probe (~1.7ms on Apple Silicon arm64 -> 0.50; ~3.4ms on cloud amd64 -> 1.0).
            CPU_FACTOR = max(0.50, min(1.4, _cdt / 0.0034))
            log(f"calibrated CPU_FACTOR={CPU_FACTOR:.2f} (probe={_cdt*1000:.1f}ms)")
        except Exception:  # noqa
            pass


def main():
    calibrate_cpu()
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    nodeids = [t["id"] for t in tests]
    ms = [math.floor(t["seconds"] * 1000 + 0.5) for t in tests]
    patches = sorted((DATA / "commits").glob("*.patch"))
    n_commits = len(patches)
    deadline = min(T_START + max(25.0, SECONDS_PER_COMMIT * n_commits), T_START + WALL_MAX - 12.0)
    global MAP_OBJ
    M = _pkl(MAP, lambda: Map(MAP))
    MAP_OBJ = M
    if M.tests != nodeids:
        # every other map is indexed by the test numbers of the pinned test list: with a
        # different list only the (remapped) coverage map can be trusted
        log("WARNING: map test list differs from tests.json; remapping, static candidates only")
        STATIC_ONLY[0] = True
        known = set(M.tests)
        UNKNOWN_TESTS.update(i for i, t in enumerate(nodeids) if t not in known)
        idx = {t: i for i, t in enumerate(nodeids)}
        remap = [idx.get(t) for t in M.tests]
        M.sets = [frozenset(remap[x] for x in s if remap[x] is not None) for s in M.sets]
        M.opened = {p: {remap[x] for x in s if remap[x] is not None} for p, s in M.opened.items()}
        M.tests = nodeids
        M.n = len(nodeids)
        M.file_of = [t.split("::", 1)[0] for t in nodeids]
        M._file_users = {}
        M._tests_by_file = {}
        for i, f in enumerate(M.file_of):
            M._tests_by_file.setdefault(f, set()).add(i)
    all_tests = set(range(len(tests)))

    answers = {}
    jobs = []
    maybe_equiv = {}
    must = {}       # commit -> tests that fail for certain; added to whatever is decided
    for pf in patches:
        cid = pf.stem
        _progress("A " + cid)
        if cid in POISON:
            answers[cid] = sorted(all_tests)
            log(cid, "ALL (its analysis did not survive an earlier attempt)")
            continue
        n_jobs = len(jobs)
        try:
            if _TEST.get("crash") == cid:      # self-test of the supervisor
                os.kill(os.getpid(), signal.SIGKILL)
            if _TEST.get("hang") == cid:
                time.sleep(10 ** 6)
            if _TEST.get("raise") == cid:
                raise RuntimeError("self-test")
            try:
                patch_text = pf.read_text(encoding="utf-8", errors="replace")
                D = analyze(patch_text, REPO, M)
            except Exception as e:  # noqa
                log(cid, "analyze failed", repr(e))
                answers[cid] = sorted(all_tests)
                continue
            fixture_reach = set()
            fixture_edits = {}
            if len(D.files) > 1 and (D.all or D.import_change or any(p.endswith(".sql") for p in D.files)):
                partition = separate_source(patch_text, REPO, M)
                if partition is not None:
                    source, readers, edits = partition
                    # Keep the dedicated fixture verifier for expensive readers.
                    # Otherwise select them, and use the source-only coverage
                    # proof for every other test. Actual runs see the full edit.
                    if source.all or sum(ms[t] for t in readers) <= 1200:
                        D, fixture_reach, fixture_edits = source, readers, edits
                        D.must.update(readers)
            # Keep the complete patch for reimports. A runtime job can use the
            # same fixture and probe paths as the edit without incidental comments.
            if not D.import_change and not D.all and D.ast_unchanged:
                D.files = {r: src for r, src in D.files.items() if r not in D.ast_unchanged}
            if D.must:
                must[cid] = set(D.must)
            if not D.all and D.parsed and all(r.endswith(".py") for r in D.files):
                # only the cheap screen here; the proof itself runs after the tests
                try:
                    from equiv import maybe_equivalent
                    if maybe_equivalent(D.files, REPO, D.parsed):
                        maybe_equiv[cid] = D.files
                except Exception as e:  # noqa
                    log(cid, "equiv screen failed", repr(e))
            guard_tests = None
            if (D.all or D.import_change) and not STATIC_ONLY[0] and D.reasons == ["cache-fill sqlglot/generator.py"]:
                import dispatch
                guard_tests = dispatch.guard_reach(D, M, REPO)
                if guard_tests is not None:
                    if fixture_reach:
                        answers[cid] = sorted(set(guard_tests) | fixture_reach)
                        continue
                    D.all = False
                    D.import_change = True
                    D.cands = set(guard_tests)
            if D.all:
                affected = broken_leaf_tests(D, M, nodeids)
                answers[cid] = sorted(all_tests) if affected is None else sorted(set(affected) | fixture_reach)
                log(cid, "ALL" if affected is None else f"cold-imports={len(affected)}", " ".join(D.reasons)[:200])
                continue
            cands = sorted(t for t in D.cands if t is not None)
            answers[cid] = cands  # static fallback
            hot = [r for r, c in D.files.items() if r.endswith(".py") and c is not None and (REPO / r).exists()]
            olds = {r: (REPO / r).read_text(encoding="utf-8") for r in hot}
            if not cands and not D.import_change:
                log(cid, "none", " ".join(D.reasons)[:200])
                continue
            if not cands:
                # import-time state changed in a file no test runs code from (e.g. a module of
                # class definitions): the readers are only known after the state diff. If the
                # child cannot finish, everything is selected.
                answers[cid] = sorted(all_tests)
            instr = {}
            n_sites = 0
            if not D.import_change and len(hot) == 1 and all(r.endswith(".py") for r in D.files) \
                    and (len(cands) >= 6 or any(CPU["test"].get(str(t), ms[t] / 1000.0) >= 0.4 for t in cands)):
                from patchlib import instrument
                for r in hot:
                    try:
                        res = instrument(olds[r], D.files[r], D.parsed.pop(r, None))
                    except Exception:  # noqa
                        res = None
                    if res is None:
                        instr = {}
                        break
                    instr[r] = res[0]
                    n_sites = res[1]
            subf = {}
            if not D.import_change and all(r.split(" ", 1)[0] in ("rt", "ast-same", "annotation") for r in D.reasons):
                subf = submit_filter(D.rt_lines, cands, nodeids)
            CPU_OVR.clear()
            CPU_OVR.update(sub_cpu(subf))
            memo_duck = all((f.startswith("sqlglot/") and f.endswith(".py")) or
                            (f.startswith("tests/fixtures/") and f.endswith(".sql")) for f in D.files)
            pure_src = all(f.startswith("sqlglot/") and f.endswith(".py") for f in D.files)
            # code lines the patch touches; fixture-only .sql edits touch none (a changed
            # query no longer matches its baseline plan, so it is executed for real)
            ch_lines = {f: sorted(v) for f, v in D.ch_lines.items()} if memo_duck else None
            iterf = {}
            opt_aff = None
            if pure_src and not D.import_change:
                iterf = iter_filter(cands, D.ch_lines)
                iter_cpu(iterf)
                opt_aff = opt_affected(D.ch_lines)
                opt_cpu(opt_aff, cands)
            memo_skip = []
            fixture_sql_only = (
                not D.import_change and bool(D.files)
                and all(f.startswith("tests/fixtures/") and f.endswith(".sql") and c is not None
                        for f, c in D.files.items())
            )
            if memo_duck:
                if not D.import_change and any(t in D.cands for t in memo_ids(nodeids)):
                    memo_skip = memo_skippable(ch_lines, nodeids)
                memo_cpu(memo_skip, nodeids, subf, opt_aff,
                         fixture_sql_only=(not D.import_change and not any((ch_lines or {}).values())))
            if fixture_sql_only:
                mids = memo_ids(nodeids)
                for t in cands:
                    if t not in mids:
                        cur = CPU_OVR.get(t, CPU["test"].get(str(t), 1.0))
                        CPU_OVR[t] = min(cur, 0.08)
            scale_nodes = pure_src and not any(
                re.search(r"\d|limit|count|len\s*\(|max_|recurs|depth|width|size", line, re.I)
                for line in patch_text.splitlines()
                if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
            )
            if scale_nodes:
                t = _nodeid_pos(nodeids).get("tests/test_parser.py::TestParser::test_max_nodes")
                if t is not None:
                    CPU_OVR[t] = 0.8
            if pure_src and _is_leaf_dialect_edit(D.files):
                _gen_mem_cpu(nodeids)
            if D.import_change:
                # decided in the child once the import-time state diff is known; the static
                # candidates are no safe answer for such a change, so if the child cannot
                # finish everything is selected
                answers[cid] = sorted(all_tests)
                light, heavy = [], []
            else:
                ovh = OVERHEAD_FIXTURE if not hot else None
                light, heavy, est_run, static_cost = plan_split(cands, ms, CPU, False, bool(instr), overhead=ovh)
            if light and not D.import_change and len(cands) == 1 and \
                    all(r.split(" ", 1)[0] in ("fixture", "ast-same", "annotation", "ignored", "fixture-unused")
                        for r in D.reasons):
                # a lone test reading an edited fixture nearly always fails: running it
                # only pays when it is cheap next to the chance of dropping it
                ovh_fx = OVERHEAD_FIXTURE if not hot else OVERHEAD_HOT
                if cpu_of(cands[0], ms) + ovh_fx > (1.0 - FIX_PFAIL) * static_cost / 1000.0:
                    light, heavy = [], cands
            force = False
            if pure_src and not D.import_change and not light and cands:
                ins, _, est = replay_scope(cands, ch_lines)
                force = bool(ins) and est <= REPLAY_MAX and est < static_cost / 1000.0
            if not light and not D.import_change and not force:
                log(cid, f"static cost={static_cost/1000:.2f}", " ".join(D.reasons)[:150])
                if not D.import_change:
                    continue
            jobs.append(dict(
                cid=cid, files=D.files, fixture_edits=fixture_edits, olds=olds, hot=hot, overlay=bool(D.import_change), guard_tests=guard_tests,
                light=light, heavy=heavy, instr=instr, n_sites=n_sites, subf=subf, cpu_ovr=dict(CPU_OVR), iterf={t: sorted(v) for t, v in iterf.items()}, opt_aff=opt_aff,
                memo_duck=memo_duck, memo_skip=memo_skip, ch_lines=ch_lines, pure_src=pure_src,
                fixture_sql_only=fixture_sql_only, scale_nodes=scale_nodes,
                node_limit_edit=not scale_nodes and any(
                    re.search(r"max_nodes|_node_count|Maximum number of AST nodes", line)
                    for line in patch_text.splitlines() if line.startswith(("+", "-"))
                ),
                default_edit=D.default_edit if len(D.files.keys() - D.ast_unchanged) == 1 else None,
                precise=sorted(D.precise), broad=sorted(D.cands), was_import=bool(D.import_change),
                reasons=" ".join(D.reasons)[:200],
            ))
        except Exception as e:  # noqa
            log(cid, "planning failed", repr(e), "-> all tests")
            del jobs[n_jobs:]
            answers[cid] = sorted(all_tests)

    CPU_OVR.clear()
    import patchlib
    patchlib._PARSED.clear()
    log(f"analysis done in {time.time() - T_START:.2f}s; {len(jobs)} commits to run")
    _progress("R")

    if jobs and not STATIC_ONLY[0]:
        if _TEST.get("runcrash"):
            os.kill(os.getpid(), signal.SIGKILL)
        run_jobs(jobs, answers, nodeids, ms, deadline)
    # The proof of equivalence parses much of the checkout, so it is kept for the commits
    # that still have tests selected. By now this process has sqlglot imported if any
    # commit was run.
    for c, m in must.items():
        answers[c] = sorted(set(answers.get(c, all_tests)) | {t for t in m if t in all_tests})
    pending = {c: f for c, f in maybe_equiv.items() if answers.get(c) and c not in must}
    if pending:
        prove_equivalent(pending, answers)

    lines = []
    for pf in patches:
        cid = pf.stem
        sel = answers.get(cid, sorted(all_tests))
        if UNKNOWN_TESTS:
            sel = sorted(set(sel) | UNKNOWN_TESTS)   # tests no map knows are always selected
        workers, cost = best_schedule(sel, ms)
        lines.append({"commit": cid, "workers": workers})
    _progress("E")
    emit(lines)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rc = resource.getrusage(resource.RUSAGE_CHILDREN)
    log(f"done in {time.time() - T_START:.1f}s cpu self={ru.ru_utime + ru.ru_stime:.1f} children={rc.ru_utime + rc.ru_stime:.1f}")
    sys.stderr.flush()
    os._exit(0)  # answers are flushed; skip interpreter teardown


CPU_PATH = Path(os.environ.get("BR_CPU", HERE / "cpu.json"))
try:
    _cpu = json.loads(CPU_PATH.read_text())
except Exception:  # noqa
    _cpu = {"test": {}, "class": {}}
CPU = _cpu
OVERHEAD_HOT = float(os.environ.get("BR_OVH_HOT", "0.15"))
OVERHEAD_FIXTURE = float(os.environ.get("BR_OVH_FIX", "0.02"))
OVERHEAD_OVERLAY = float(os.environ.get("BR_OVH_OV", "1.0"))
FAIL_PENALTY = float(os.environ.get("BR_FAIL_PEN", "0.50"))


CPU_OVR = {}
SUBMAP_PATH = Path(os.environ.get("BR_SUBMAP", HERE / "submap.json.gz"))
SUB_BASE = float(os.environ.get("BR_SUB_BASE", "0.15"))
_SUBMAP = None


def _submap():
    global _SUBMAP
    if _SUBMAP is None:
        import gzip
        def build():
            with gzip.open(SUBMAP_PATH, "rt") as fh:
                raw = json.load(fh)
            return {int(t): (v["n"], {f: set(ls) for f, ls in v["main"].items()},
                             {f: {int(l): s for l, s in d.items()} for f, d in v["subs"].items()})
                    for t, v in raw.items()}
        try:
            _SUBMAP = _pkl(SUBMAP_PATH, build)
        except Exception:  # noqa
            _SUBMAP = {}
    return _SUBMAP


def submit_filter(lines_by_file, cands, nodeids):
    """For pool-based tests, which submitted queries can reach the changed lines."""
    out = {}
    if not lines_by_file or not any(t in cands for t in _submap()):
        return out
    for t, (n, main, subs) in _submap().items():
        if t not in cands:
            continue
        allow = set()
        ok = True
        for f, lines in lines_by_file.items():
            if lines & main.get(f, set()):
                ok = False
                break
            sf = subs.get(f, {})
            for l in lines:
                allow.update(sf.get(l, ()))
        if ok and len(allow) < n:
            out[t] = allow
    complete, _ = query_filter(lines_by_file, cands, nodeids)
    for t, allow in complete.items():
        out[t] = allow.intersection(out.get(t, allow))
    return out


def query_filter(lines, cands, nodeids):
    memo = _memo_lite() or {}
    shared = memo.get("query_shared", {})
    if any(shared.get(f, set()).intersection(ls) for f, ls in lines.items()):
        return {}, {}
    complete, parse_only = {}, {}
    pos = _nodeid_pos(nodeids)
    for tid, (n, phases) in memo.get("query_index", {}).items():
        t = pos.get(tid)
        if t not in cands or t not in _submap() or n != _submap()[t][0]:
            continue
        masks = []
        for phase in phases:
            mask = 0
            for file, changed in lines.items():
                hits = phase.get(file, {})
                for line in changed:
                    mask |= hits.get(line, 0)
            masks.append(mask)
        later, initial = masks
        complete[t] = {i for i in range(n) if (later | initial) & (1 << i)}
        parse_only[t] = {i for i in range(n) if (initial & ~later) & (1 << i)}
    return complete, parse_only


def setting_queries(delta, changed_lines, cands, nodeids):
    """Combine runtime-line reach with reads of scalar settings in each TPC query."""
    if not delta:
        return {}
    import pickle
    try:
        with open(HERE / "queryattrs.pkl", "rb") as fh:
            tracked, reads, fixture_reads = pickle.load(fh)
    except OSError:
        return {}
    if not tracked.issuperset(delta):
        return {}
    import runner
    def observed(index):
        mask = 0
        for key in delta:
            mask |= index.get(key, 0)
        return mask

    fixture_ids = [t for t in cands if nodeids[t] in fixture_reads]
    allowed = iter_filter(fixture_ids, changed_lines, raw=True)
    fixtures = {}
    for t, indices in allowed.items():
        mask = observed(fixture_reads[nodeids[t]])
        if mask & 1:
            continue
        n = _itermap()[0][t][0]
        indices.update(i for i in range(n) if mask & (2 << i))
        if len(indices) <= ITER_MAXFRAC * n:
            fixtures[t] = indices
    runner.ITER_FILTER.update(fixtures)
    iter_cpu(fixtures)
    reached, _ = query_filter(changed_lines, cands, nodeids)
    skippable = set(memo_skippable(changed_lines, nodeids))
    unchanged_pipeline = opt_affected(changed_lines) == []
    for t in list(reached):
        phases = reads.get(nodeids[t])
        if not isinstance(phases, tuple) or len(phases) != 4:
            del reached[t]
            continue
        table, outside, later, optimizer = phases
        mask = observed(table)
        post = observed(outside)
        count = _submap()[t][0]
        in_opt = observed(optimizer)
        if unchanged_pipeline and not (in_opt & 1):
            runner.UNCHANGED_OPT_QUERIES[t] = {i for i in range(count) if not in_opt & (2 << i)}
        after_parse = observed(later)
        if t in runner.PARSE_ONLY:
            runner.PARSE_ONLY[t] = {i for i in runner.PARSE_ONLY[t]
                                    if not after_parse & (1 | (2 << i))}
        if t in skippable and not (post & 1):
            runner.MEMO_QUERIES[t] = {i for i in range(count) if not post & (2 << i)}
        if mask & 1:
            del reached[t]
            continue
        reached[t].update(i for i in range(count) if mask & (2 << i))
    return reached


def sub_cpu(subf):
    o = {}
    for t, allow in subf.items():
        n = _submap()[t][0]
        v = CPU["test"].get(str(t), 0.0)
        o[t] = v * (SUB_BASE + (1 - SUB_BASE) * len(allow) / max(1, n))
    return o


OPT_PREFIX = os.environ.get("BR_OPT", str(HERE / "optmemo"))
OPT_ON = os.environ.get("BR_OPT_ON", "1") == "1"
OPT_PER_CALL = os.environ.get("BR_OPT_PER_CALL", "1") == "1"
_OPT_LOADED = [False]


def opt_affected(lines):
    """Optimizer pipeline stages that may run a changed line, or None if the optimizer
    memo cannot be used. `lines`: {file: lines} of a change to function bodies only."""
    if not OPT_ON or lines is None:
        return None
    import optmemo
    if not _OPT_LOADED[0]:
        _OPT_LOADED[0] = True
        optmemo.load(OPT_PREFIX)
    return optmemo.affected(lines)


OPT_FRAC_PATH = Path(os.environ.get("BR_OPT_FRAC", HERE / "optfrac.json"))
_OPT_FRAC = None


def opt_cpu(aff, cands):
    """With the optimizer memo on, tests that mostly call optimize() get much cheaper."""
    global _OPT_FRAC
    if aff is None or 0 in aff:
        return
    if _OPT_FRAC is None:
        try:
            _OPT_FRAC = {int(t): f for t, f in json.loads(OPT_FRAC_PATH.read_text()).items()}
        except Exception:  # noqa
            _OPT_FRAC = {}
    save = 0.35 if aff else 0.90
    for t in cands:
        f = _OPT_FRAC.get(t)
        if f:
            base = CPU_OVR.get(t, CPU["test"].get(str(t)))
            if base is not None:
                CPU_OVR[t] = base * (1.0 - f * save)


def _pkl(path, build):
    """Processed form of a map: pickled next to it at image build time (prep.py)."""
    import pickle
    import gc
    pk = str(path) + ".pkl"
    try:
        gc.disable()
        with open(pk, "rb") as fh:
            obj = pickle.load(fh)
        gc.freeze()
        return obj
    except Exception:  # noqa
        pass
    finally:
        gc.enable()
    if build is None:
        return None
    gc.disable()
    obj = build()
    gc.freeze()
    gc.enable()
    if os.environ.get("BR_PREP") == "1":
        with open(pk, "wb") as fh:
            pickle.dump(obj, fh, protocol=5)
    return obj


ITER_PATH = Path(os.environ.get("BR_ITER", HERE / "iter.json.gz"))
ITER_ON = os.environ.get("BR_ITER_ON", "1") == "1"
ITER_BASE = float(os.environ.get("BR_ITER_BASE", "0.15"))
ITER_MAXFRAC = float(os.environ.get("BR_ITER_MAXFRAC", "0.8"))
_ITERMAP = None
_ITERMAP_CACHE = {}


def _itermap():
    """Per-iteration line coverage of fixture-driven tests (see builditer.py)."""
    global _ITERMAP
    if _ITERMAP is None:
        import gzip

        def build():
            with gzip.open(ITER_PATH, "rt") as fh:
                raw = json.load(fh)
            tests_meta = {}
            by_file_raw = {}
            for t, v in raw.items():
                t = int(t)
                tests_meta[t] = (v["n"], {f: frozenset(ls) for f, ls in v["main"].items() if ls})
                for f, d in v["its"].items():
                    sub = by_file_raw.setdefault(f, {})
                    for l, ks in d.items():
                        if ks:
                            sub.setdefault(int(l), []).append((t, tuple(ks)))
            import pickle
            for f in list(by_file_raw.keys()):
                by_file_raw[f] = pickle.dumps(by_file_raw[f], protocol=pickle.HIGHEST_PROTOCOL)
            return (tests_meta, by_file_raw)

        try:
            _ITERMAP = _pkl(ITER_PATH, build)
        except Exception as e:  # noqa
            log("iteration map unavailable", repr(e))
            _ITERMAP = ({}, {})
    return _ITERMAP


ITER_TEST_IDS = {1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1295, 1302, 1313, 1314, 1324, 1325, 1327, 1335, 1343, 1352, 1353, 1355, 1362, 1375, 1377, 1378, 1382, 1383, 1384, 1417, 1420, 1434, 1440, 1441, 1454, 1561, 1599, 1606, 1607}


def iter_filter(cands, lines, raw=False):
    """For fixture-driven candidates: the iterations that reach a changed line. Only
    valid when nothing but function bodies changed. Returns {test: set of iterations}."""
    if not ITER_ON or (not lines and not raw):
        return {}
    cs = set(cands)
    if not cs.intersection(ITER_TEST_IDS):
        return {}
    tests_meta, by_file_raw = _itermap()
    bad_tests = set()
    for f, ls in lines.items():
        for t in cs:
            if t in tests_meta:
                main = tests_meta[t][1]
                if f in main and any(l in main[f] for l in ls):
                    bad_tests.add(t)
    per_test = {t: set() for t in cs if t in tests_meta and t not in bad_tests}
    import pickle
    for f, ls in lines.items():
        if f not in _ITERMAP_CACHE:
            raw = by_file_raw.get(f)
            _ITERMAP_CACHE[f] = pickle.loads(raw) if raw else {}
        flines = _ITERMAP_CACHE[f]
        if flines:
            for l in ls:
                entries = flines.get(l)
                if entries:
                    for t, ks in entries:
                        if t in per_test:
                            per_test[t].update(ks)
    out = {}
    if raw:
        return per_test
    for t, allow in per_test.items():
        n = tests_meta[t][0]
        if allow and len(allow) <= ITER_MAXFRAC * n:
            out[t] = allow
    return out


def iter_cpu(iterf):
    tests_meta = _itermap()[0]
    for t, allow in iterf.items():
        base = CPU["test"].get(str(t))
        if base is not None:
            n = tests_meta.get(t, (1,))[0]
            est = base * (ITER_BASE + (1 - ITER_BASE) * len(allow) / max(1, n))
            cur = CPU_OVR.get(t)
            CPU_OVR[t] = min(cur, est) if cur is not None else est



REPLAY_PATH = Path(os.environ.get("BR_REPLAY", HERE / "replay.json.gz"))
REPLAY_ON = os.environ.get("BR_REPLAY_ON", "1") == "1"
REPLAY_FACTOR = float(os.environ.get("BR_REPLAY_FACTOR", "0.9"))  # recorded seconds -> cpu here
REPLAY_FRAC = float(os.environ.get("BR_REPLAY_FRAC", "0.75"))
REPLAY_BOUND_DIV = float(os.environ.get("BR_REPLAY_BOUND_DIV", "1.15"))
REPLAY_MAX = float(os.environ.get("BR_REPLAY_MAX", "18"))
_REPLAY = None


def _replay():
    """Recorded parse-phase calls of every test at the pinned commit (see replay.py)."""
    global _REPLAY
    if _REPLAY is None:
        import gzip

        def build():
            with gzip.open(REPLAY_PATH, "rt") as fh:
                d = json.load(fh)
            d["tests"] = {int(t): v for t, v in d["tests"].items()}
            d["bad"] = set(d["bad"])
            d["sets"] = [frozenset(x) for x in d["sets"]]
            d["out"] = {f: {int(l): i for l, i in m.items()} for f, m in d["out"].items()}
            d["snaps_tuple"] = {k: tuple(v) for k, v in d.pop("snaps", {}).items()}
            try:
                with gzip.open(str(REPLAY_PATH).replace(".json.gz", "") + ".fk.gz", "rb") as fh:
                    d["fk"] = fh.read()
            except Exception:
                d["fk"] = None
            return d
        try:
            d = _pkl(REPLAY_PATH, build)
            try:
                import replay
                replay.SNAP_INDEX.update(d.get("snaps_tuple") or {k: tuple(v) for k, v in d.pop("snaps", {}).items()})
                replay.SNAP_PATH[0] = str(REPLAY_PATH).replace(".json.gz", "") + ".snap"
            except Exception as e:  # noqa
                log("snapshots unavailable", repr(e))
            if "fk" not in d:
                try:
                    with gzip.open(str(REPLAY_PATH).replace(".json.gz", "") + ".fk.gz", "rb") as fh:
                        d["fk"] = fh.read()
                except Exception:
                    d["fk"] = None
            _REPLAY = d
        except Exception as e:  # noqa
            log("replay map unavailable", repr(e))
            _REPLAY = {}
    return _REPLAY


def replay_scope(cands, ch_lines, outside0=None, extra_lines=None):
    """Candidates that run the changed lines only inside recorded parse phases.
    Returns (tests, key ids, estimated cpu seconds)."""
    R = _replay() if REPLAY_ON else {}
    if not R or ch_lines is None:
        return [], [], 0.0
    outside = set(outside0 or ())
    allines = {f: set(v) for f, v in ch_lines.items()}
    for f, v in (extra_lines or {}).items():
        allines.setdefault(f, set()).update(v)
    for f, lines in allines.items():
        m = R["out"].get(f)
        if m:
            for l in lines:
                i = m.get(l)
                if i is not None:
                    outside |= R["sets"][i]
    inside = [t for t in cands if t in R["tests"] and t not in R["bad"] and t not in outside]
    keys = set()
    for t in inside:
        keys.update(R["tests"][t])
    est = sum(R["ms"][k] for k in keys) / 1000.0 * REPLAY_FACTOR + 0.05
    return inside, sorted(keys), est


def _key_mask(R, lines_by_file):
    """Bitmap of the recorded inputs that enter a function containing a changed line."""
    fk = R.get("fk")
    if not fk or not R.get("funcs"):
        return None
    nb = (len(R["dig"]) + 7) // 8
    mask = 0
    for i, (f, lo, hi) in enumerate(R["funcs"]):
        ls = lines_by_file.get(f)
        if ls and any(lo <= l <= hi for l in ls):
            mask |= int.from_bytes(fk[i * nb:(i + 1) * nb], "little")
    return mask


_AK = None
GEN_INIT_LINE = 1005    # sqlglot/generator.py at the pinned commit: inside Generator.__init__


def _attr_mask(R, attrs):
    """Bitmap of the recorded inputs that read any of the given (owner, attribute) pairs."""
    global _AK
    base = str(REPLAY_PATH).replace(".json.gz", "")
    if _AK is None:
        import gzip
        try:
            with gzip.open(base + ".ak.json.gz", "rt") as fh:
                _AK = json.load(fh)
            if _AK.get("nkeys") != len(R["dig"]):
                _AK = False
        except Exception:  # noqa
            _AK = False
    if not _AK:
        return None
    import zlib
    mask = 0
    fd = os.open(base + ".ak", os.O_RDONLY)
    try:
        for owner, attr in attrs:
            e = _AK["attrs"].get(f"{owner}|{attr}")
            if e:
                mask |= int.from_bytes(zlib.decompress(os.pread(fd, e[1], e[0])), "little")
    finally:
        os.close(fd)
    return mask


SIZE_SENSITIVE_TEST = [None]
REPLAY_HIT: set = set()  # tests kept specifically because a recorded phase input changed


def replay_narrow(cands, ch_lines, ms, outside0=None, extra_lines=None, light0=None, pure=False, attrs=None,
                  disp=False):
    """Replay the recorded phase inputs under the patched code; drop the candidates whose
    inputs all still produce their baseline output. Returns (candidates, note)."""
    REPLAY_HIT.clear()
    inside, keys, est = replay_scope(cands, ch_lines, outside0, extra_lines)
    if SIZE_SENSITIVE_TEST[0] is not None:
        inside = [t for t in inside if t != SIZE_SENSITIVE_TEST[0]]
    if not inside:
        return cands, ""
    R = _replay()
    tkeys = {t: R["tests"][t] for t in inside}
    if pure:
        # only function bodies changed: an input that never enters a changed function
        # runs exactly the old code, so only the inputs that do are replayed. (The tests
        # here run the changed lines inside phases only; one with no such input never
        # runs them at all.)
        allines = {f: set(v) for f, v in ch_lines.items()}
        for f, v in (extra_lines or {}).items():
            allines.setdefault(f, set()).update(v)
        mask = _key_mask(R, allines) if os.environ.get("BR_REPLAY_FK", "1") == "1" else None
        if mask is not None:
            bits = mask.to_bytes((len(R["dig"]) + 7) // 8, "little")
            tkeys = {t: [k for k in ks if bits[k >> 3] & (1 << (k & 7))] for t, ks in tkeys.items()}
            keys = sorted({k for ks in tkeys.values() for k in ks})
            est = sum(R["ms"][k] for k in keys) / 1000.0 * REPLAY_FACTOR + 0.05
    elif attrs is not None and os.environ.get("BR_REPLAY_AK", "1") == "1":
        # class attributes changed: only the inputs that read one of them (or enter a
        # changed function) can behave differently. A test with no such input is not
        # explained by the trace and simply stays a candidate.
        allines = {f: set(v) for f, v in ch_lines.items()}
        for f, v in (extra_lines or {}).items():
            allines.setdefault(f, set()).update(v)
        m1 = _attr_mask(R, attrs)
        m2 = _key_mask(R, allines)
        # inputs that create a generator (they enter Generator.__init__, where the dispatch
        # table is looked up)
        m3 = _key_mask(R, {"sqlglot/generator.py": {GEN_INIT_LINE}}) if disp else 0
        if disp and not m3:
            m1 = None       # cannot tell: replay every input of the tests in scope
        if m1 is not None and m2 is not None:
            mask = m1 | m2 | m3
            bits = mask.to_bytes((len(R["dig"]) + 7) // 8, "little")
            tkeys = {t: [k for k in ks if bits[k >> 3] & (1 << (k & 7))] for t, ks in tkeys.items()}
            tkeys = {t: ks for t, ks in tkeys.items() if ks}
            inside = [t for t in inside if t in tkeys]
            keys = sorted({k for ks in tkeys.values() for k in ks})
            est = sum(R["ms"][k] for k in keys) / 1000.0 * REPLAY_FACTOR + 0.05
            if not inside:
                return cands, ""
    ins0 = set(inside)
    # tests with 0 masked keys in pure mode never enter any changed function: drop for free
    cleared = {t for t in inside if not tkeys[t]}
    todo_tests = [t for t in inside if t not in cleared]
    # what replay can buy: cpu of the tests that would otherwise be run, or the
    # schedule time freed if every replayed test turns out unaffected
    would_run = ins0 if light0 is None else ins0.intersection(light0)
    worth = sum(cpu_of(t, ms) for t in would_run)
    rest = [t for t in cands if t not in ins0]
    bound = (schedule_cost(cands, ms)[1] - (schedule_cost(rest, ms)[1] if rest else 0)) / 1000.0
    max_budget = min(REPLAY_MAX, max(0.3, REPLAY_FRAC * worth, bound / REPLAY_BOUND_DIV))
    if not todo_tests:
        out = [t for t in cands if t not in cleared]
        return out, f" replay(free={len(cleared)} kept={len(out)}/{len(cands)})"
    if max_budget <= 0.05:
        out = [t for t in cands if t not in cleared]
        return out, f" replay-skip(est={est:.1f} worth={worth:.1f} bound={bound:.1f})"
    import replay
    t0 = time.time()
    c0 = time.process_time()
    import runner
    if os.environ.get("BR_REPLAY_DEBUG"):
        for dt in [int(x) for x in os.environ["BR_REPLAY_DEBUG"].split(",")]:
            allk = R["tests"].get(dt, [])
            mk = tkeys.get(dt)
            res = []
            for k in allk:
                try:
                    ch = replay.run_key(R["keys"][k]) != R["dig"][k]
                except BaseException as e:  # noqa
                    ch = repr(e)[:40]
                if ch:
                    res.append((k, ch, k in (mk or ()), str(R["keys"][k])[:300]))
            log("REPLAYDEBUG", dt, "in_cands", dt in cands, "inside", dt in inside, "bad", dt in R["bad"], "keys", len(allk),
                "masked", None if mk is None else len(mk), "changed(all keys)", res[:6])
    signal.signal(signal.SIGALRM, runner._alarm)
    limit = t0 + max(5.0, min(3.0 * max_budget + 4.0, 25.0))
    state = {}   # key -> True when its output differs from the baseline
    n_changed = 0
    n_hangs = 0
    kept = set()
    kms = R["ms"]
    rkeys = R["keys"]
    rdig = R["dig"]
    has_gen = any(rkeys[k][0] == "g" for k in keys)
    # Order by value per estimated replay cost so heavy tests clear first even under tight caps.
    t_val = {}
    t_est0 = {}
    for t in todo_tests:
        t_est0[t] = sum(kms[k] for k in tkeys[t]) / 1000.0 * REPLAY_FACTOR
        v = ms[t] / 1000.0
        if light0 is None or t in light0:
            v = max(v, cpu_of(t, ms))
        t_val[t] = v
    order = sorted(todo_tests, key=lambda t: (-t_val[t] / max(t_est0[t], 0.002), len(tkeys[t])))
    undecided = len(order)
    aborted = ""
    replay.begin_batch()
    try:
        for t in order:
            undecided -= 1
            ks = tkeys[t]
            hit = False
            for k in ks:
                if state.get(k):
                    hit = True
                    break
            if hit:
                kept.add(t)
                REPLAY_HIT.add(t)
                continue
            unseen = [k for k in ks if k not in state]
            if not unseen:
                cleared.add(t)
                continue
            cpu_used = time.process_time() - c0
            if cpu_used >= max_budget or time.time() > limit:
                aborted = "cap" if cpu_used >= max_budget else "slow"
                kept.add(t)
                continue
            new_est = sum(kms[k] for k in unseen) / 1000.0 * REPLAY_FACTOR
            t_worth = t_val[t]
            if not rest and not kept:
                t_worth += 2.0 / max(1, undecided + 1)
            if est > max_budget and new_est > max(0.04, t_worth * 0.90):
                kept.add(t)
                continue
            unseen.sort(key=kms.__getitem__)
            changed_hit = False
            fast_cnt = 0
            timer_armed = False
            for idx_k, k in enumerate(unseen):
                if (idx_k & 7) == 0:
                    if time.time() > limit or (time.process_time() - c0) >= max_budget * 1.15:
                        aborted = "slow"
                        hit = True
                        break
                km = kms[k]
                if km < 10:
                    if not timer_armed or fast_cnt >= 8:
                        signal.setitimer(signal.ITIMER_REAL, KEY_T_MIN)
                        timer_armed = True
                        fast_cnt = 0
                    fast_cnt += 1
                else:
                    signal.setitimer(signal.ITIMER_REAL, max(KEY_T_MIN, km / 1000.0 * 15.0))
                    timer_armed = True
                    fast_cnt = 8
                try:
                    ch = replay.run_key(rkeys[k], cache_parse=has_gen) != rdig[k]
                except runner.Timeout:
                    ch = True
                    n_hangs += 1
                    timer_armed = False
                    fast_cnt = 0
                except BaseException:  # noqa
                    ch = True
                state[k] = ch
                if ch:
                    n_changed += 1
                    hit = True
                    changed_hit = True
                    break
            if timer_armed:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if hit:
                kept.add(t)
                if changed_hit:
                    REPLAY_HIT.add(t)
            else:
                cleared.add(t)
            if n_hangs >= 2:
                HANG[0] = True
                aborted = "hang"
                break
        if n_hangs >= 1 and aborted in ("cap", "slow"):
            HANG[0] = True
            aborted = "hang"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        replay.end_batch()
    out = [t for t in cands if t not in cleared]
    PROF["replay"] = PROF.get("replay", 0.0) + time.process_time() - c0
    ab = f" abort={aborted}" if aborted else ""
    return out, (f" replay(tests={len(inside)} keys={len(state)}/{len(keys)} changed={n_changed} "
                 f"kept={len(out)}/{len(cands)} {time.time() - t0:.2f}s est={est:.1f}{ab})")


MEMO_PATH = Path(os.environ.get("BR_MEMO", HERE / "memo.pkl"))
MEMO_LITE_PATH = Path(os.environ.get("BR_MEMO_LITE", HERE / "memo_lite.pkl"))
CHEAP_MARGIN = float(os.environ.get("BR_CHEAP_MARGIN", "1.0"))
C_SKIP = float(os.environ.get("BR_C_SKIP", "4.0"))        # cpu of a TPC run that executes no query
DUCK_FRAC = float(os.environ.get("BR_DUCK_FRAC", "0.55"))  # share of TPC cpu left once references are reused
_MEMO_IDS = None
_MEMO_LITE = None
_NODEID_POS = None


def _nodeid_pos(nodeids):
    global _NODEID_POS
    if _NODEID_POS is None:
        _NODEID_POS = {n: i for i, n in enumerate(nodeids)}
    return _NODEID_POS


def _memo_lite():
    global _MEMO_LITE
    if _MEMO_LITE is None:
        if MEMO_LITE_PATH.exists():
            try:
                import pickle, gc
                gc.disable()
                with open(MEMO_LITE_PATH, "rb") as fh:
                    _MEMO_LITE = pickle.load(fh)
                gc.freeze()
            except Exception:
                _MEMO_LITE = None
            finally:
                gc.enable()
        if _MEMO_LITE is None:
            import runner
            if runner.MEMO is None:
                runner.load_memo(str(MEMO_PATH))
            if runner.MEMO is not None:
                _MEMO_LITE = runner.MEMO
    return _MEMO_LITE


def memo_ids(nodeids):
    """test number -> id for the tests the baseline memo knows."""
    global _MEMO_IDS
    if _MEMO_IDS is None:
        ml = _memo_lite()
        _MEMO_IDS = {}
        if ml is not None:
            pos = _nodeid_pos(nodeids)
            _MEMO_IDS = {pos[tid]: tid for tid in ml.get("rlines", {}) if tid in pos}
        try:
            import runner
            runner.MEMO_IDS = _MEMO_IDS
        except Exception:
            pass
    return _MEMO_IDS


def memo_skippable(ch_lines, nodeids):
    """Tests whose queries may skip execution: the patch only touches code that
    runs inside parse/optimize/transpile for them."""
    ml = _memo_lite()
    if ch_lines is None or ml is None:
        return []
    out = []
    for t, tid in memo_ids(nodeids).items():
        if not ml.get("opt", {}).get(tid):
            continue
        rl = ml["rlines"][tid]
        if all(not (rl.get(f) and rl[f].intersection(ls)) for f, ls in ch_lines.items()):
            out.append(t)
    return out


def memo_cpu(memo_skip, nodeids, subf=None, opt_aff=None, fixture_sql_only=False):
    ml = _memo_lite()
    if ml is None:
        return
    pos = _nodeid_pos(nodeids)
    opt_on = opt_aff is not None and 0 not in opt_aff
    tpch_opt_tid = ml.get("tpch_opt_tid", "tests/test_executor.py::TestExecutor::test_optimized_tpch")
    for t, tid in memo_ids(nodeids).items():
        base = CPU["test"].get(str(t))
        if base is None:
            continue
        cur = CPU_OVR.get(t, base)
        if tid == tpch_opt_tid:
            CPU_OVR[t] = min(cur, 0.08)
        elif fixture_sql_only:
            CPU_OVR[t] = min(cur, 0.25)
        elif t in memo_skip:
            # execution + duckdb are skipped; only parse/optimize/transpile runs
            skip_est = min(C_SKIP, base * 0.25)
            if subf and t in subf:
                n = _submap().get(t, (1,))[0]
                skip_est *= (SUB_BASE + (1.0 - SUB_BASE) * len(subf[t]) / max(1, n))
            if opt_on:
                skip_est = max(0.15, skip_est * (0.25 if opt_aff else 0.10))
            CPU_OVR[t] = min(cur, skip_est)
        elif ml.get("opt", {}).get(tid):
            duck_est = sub_cpu({t: subf[t]} if subf and t in subf else {}).get(t, cur) * DUCK_FRAC
            CPU_OVR[t] = min(cur, duck_est)
    for tid in ml.get("duck_opt_tids", ()):
        t = pos.get(tid)
        if t is None:
            continue
        base = CPU["test"].get(str(t))
        if base is not None:
            cur = CPU_OVR.get(t, base)
            CPU_OVR[t] = min(cur, max(0.04, base * 0.12))


_LEAF_DIALECT_EXCL = {
    "sqlglot/dialects/__init__.py",
    "sqlglot/dialects/dialect.py",
    "sqlglot/generators/__init__.py",
    "sqlglot/parsers/__init__.py",
}


def _is_leaf_dialect_edit(files):
    if not files:
        return False
    for r, content in files.items():
        if content is None or r in _LEAF_DIALECT_EXCL:
            return False
        if not (r.endswith(".py") and (
            r.startswith("sqlglot/dialects/") or r.startswith("sqlglot/generators/") or r.startswith("sqlglot/parsers/")
        )):
            return False
    return True


def _gen_mem_mods(files):
    if not _is_leaf_dialect_edit(files):
        return None
    seeds = {_modname(r) for r in files}
    graph = _import_graph()
    rev = {}
    for m, deps in graph.items():
        for d in deps:
            rev.setdefault(d, set()).add(m)
    reached = set(seeds)
    todo = list(seeds)
    while todo:
        for m in rev.get(todo.pop(), ()):
            if m not in reached:
                reached.add(m)
                todo.append(m)
    return reached


def _gen_mem_cpu(nodeids):
    t = _nodeid_pos(nodeids).get("tests/test_generator.py::TestGenerator::test_generator_memory")
    if t is not None:
        cur = CPU_OVR.get(t, CPU["test"].get(str(t), 2.0))
        CPU_OVR[t] = min(cur, 0.15)



KEY_T_MIN = float(os.environ.get("BR_KEY_T_MIN", "0.30"))
HANG = [False]     # the patched code was seen to hang: run nothing, keep every candidate
T_MIN = float(os.environ.get("BR_T_MIN", "2.5"))
T_MAX = float(os.environ.get("BR_T_MAX", "30"))
CHILD_CPU_MAX = int(os.environ.get("BR_CHILD_CPU_MAX", "60"))
SETUP_TIMEOUT = float(os.environ.get("BR_SETUP_TIMEOUT", "25"))


def test_timeout(t, ms):
    """Wall seconds a test may take under a patch before it counts as hanging (failed)."""
    return min(T_MAX, max(T_MIN, 3.0 * max(cpu_of(t, ms), ms[t] / 1000.0) + 2.0))


def cpu_of(t, ms):
    v = CPU_OVR.get(t)
    if v is not None:
        return v * CPU_FACTOR
    v = CPU["test"].get(str(t))
    if str(t) == "1440":
        v = min(v or 0.15, 0.15)
    elif str(t) == "1123":
        v = min(v or 0.10, 0.10)
    elif str(t) == "1441":
        v = min(v or 0.05, 0.05)
    elif str(t) == "1107":
        v = min(v or 0.08, 0.08)
    elif str(t) == "1490":
        v = min(v or 0.05, 0.05)
    elif str(t) == "1106":
        v = min(v or 2.5, 2.5)
    elif v is None:
        v = ms[t] / 1000.0 * 1.2
    return v * CPU_FACTOR


OV_MULT = float(os.environ.get("BR_OV_MULT", "1.15"))
MAX_CHILD_CPU = float(os.environ.get("BR_MAX_CHILD", "20"))
BROAD_CANDS = int(os.environ.get("BR_BROAD_CANDS", "400"))
BROAD_FAILS = int(os.environ.get("BR_BROAD", "3"))
HEAVY_CPU = float(os.environ.get("BR_HEAVY_CPU", "1.0"))
LATE_MIN_MS = float(os.environ.get("BR_LATE_MIN_MS", "125"))
FIX_PFAIL = float(os.environ.get("BR_FIX_PFAIL", "0.85"))
P_EQUIV = float(os.environ.get("BR_P_EQUIV", "0.6"))
LATE_MARGIN = float(os.environ.get("BR_LATE_MARGIN", "0.45"))
P2_PASS = float(os.environ.get("BR_P2_PASS", "0.85"))
P2_PASS_FAILED = float(os.environ.get("BR_P2_PASS_F", "0.50"))
REPLAN = os.environ.get("BR_REPLAN", "1") == "1"
PRESET_MIN = int(os.environ.get("BR_PRESET_MIN", "1"))
PRESET_CACHE = Path(os.environ.get("BR_PRESET_CACHE", HERE / "preset_executor.pkl"))
DENSE_MS = float(os.environ.get("BR_DENSE_MS", "500"))
DENSE_MIN = float(os.environ.get("BR_DENSE_MIN", "2.0"))
Q_MAX = float(os.environ.get("BR_Q_MAX", "0.70"))
Q_PRIOR = float(os.environ.get("BR_Q_PRIOR", "0.5"))
Q_BETA = float(os.environ.get("BR_Q_BETA", "3.0"))
RP_MARGIN = float(os.environ.get("BR_RP_MARGIN", "0.45"))


def plan_split(cands, ms, cpu, overlay, can_diff=False, overhead=None, mult=1.0, base=()):
    """Choose which candidates to run (R) and which to select blindly (U).

    U is a prefix of the heavy tests, taken either by cpu or by cpu per recorded
    second (tests that are cheap to run but long on the clock are worth running).
    Returns (run_list, unrun_list, est_cost_ms, static_cost_ms)."""
    base = list(base)
    floor = schedule_cost(base, ms)[1] if base else 0   # tests selected whatever happens
    static_cost = schedule_cost(list(cands) + base, ms)[1] - floor
    ovh = (OVERHEAD_OVERLAY if overlay else OVERHEAD_HOT) if overhead is None else overhead
    total = sum(cpu_of(t, ms) for t in cands) * mult
    heavy = [t for t in cands if cpu_of(t, ms) >= HEAVY_CPU or (ms[t] >= HEAVY_CPU * 1000 and cpu_of(t, ms) >= 0.45)]
    best = (static_cost, None)
    orders = [sorted(heavy, key=lambda t: -cpu_of(t, ms)),
              sorted(heavy, key=lambda t: -cpu_of(t, ms) / max(ms[t], 1.0))]
    for order in orders:
        run_cpu = total
        for k in range(0, len(order) + 1):
            if k:
                run_cpu -= cpu_of(order[k - 1], ms) * mult
            if k == len(cands):
                break
            U = order[:k]
            u_cost = (schedule_cost(list(U) + base, ms)[1] - floor) if U else 0
            if can_diff and U:
                u_cost *= (1.0 - P_EQUIV)
            pen = 0 if ovh == 0.0 else (min(FAIL_PENALTY * 1000, 0.35 * static_cost) if not U else 0)
            cost = u_cost + (run_cpu + ovh) * 1000 + pen
            if cost < best[0]:
                best = (cost, U)
    # run only a few tests that are cheap for their clock time, select the rest unrun
    cheap = sorted((t for t in heavy if cpu_of(t, ms) * mult < ms[t] / 1000.0),
                   key=lambda t: cpu_of(t, ms) / max(ms[t], 1.0))[:6]
    for j in range(1, len(cheap) + 1):
        rs = set(cheap[:j])
        U = [t for t in cands if t not in rs]
        if not U:
            break
        u_cost = schedule_cost(U + base, ms)[1] - floor
        cost = u_cost + (sum(cpu_of(t, ms) for t in rs) * mult + ovh) * 1000 + 500
        if cost + CHEAP_MARGIN * 1000 < best[0]:
            best = (cost, U)
    if best[1] is None:
        return [], sorted(cands, key=lambda t: -cpu_of(t, ms)), static_cost, static_cost
    U = list(best[1])
    us = set(U)
    R = sorted((t for t in cands if t not in us), key=lambda t: cpu_of(t, ms))
    return R, U, best[0], static_cost


READS_PATH = Path(os.environ.get("BR_READS", HERE / "reads.json.gz"))
NOISE_PATH = Path(os.environ.get("BR_NOISE", HERE / "noise.json"))
_READS = None


OUTREADS_PATH = Path(os.environ.get("BR_OUTREADS", HERE / "outreads.json.gz"))
_OUTREADS = None


def load_outreads():
    """Like the reads map, but only reads made outside parse phases."""
    global _OUTREADS
    if _OUTREADS is None:
        import gzip
        def build():
            with gzip.open(OUTREADS_PATH, "rt") as fh:
                R = json.load(fh)
            sets = [frozenset(x) for x in R["sets"]]
            byattr = {}
            for k, i in R["reads"].items():
                o, a, key = k.split("|", 2)
                byattr.setdefault((o, a), []).append((key, i))
            return (sets, byattr, set(R["untraced"]))
        try:
            _OUTREADS = _pkl(OUTREADS_PATH, build)
        except Exception as e:  # noqa
            log("out-of-phase reads map unavailable", repr(e))
            _OUTREADS = False
    return _OUTREADS


def load_reads():
    global _READS
    if _READS is None:
        import gzip
        def build():
            with gzip.open(READS_PATH, "rt") as fh:
                R = json.load(fh)
            sets = [frozenset(x) for x in R["sets"]]
            byattr = {}
            for k, i in R["reads"].items():
                o, a, key = k.split("|", 2)
                byattr.setdefault((o, a), []).append((key, i))
            noise = {tuple(x) for x in json.loads(NOISE_PATH.read_text())}
            return (sets, byattr, set(R["untraced"]), noise)
        try:
            _READS = _pkl(READS_PATH, build)
        except Exception as e:  # noqa
            log("reads map unavailable", repr(e))
            _READS = False
    return _READS


IMPORTS_PATH = Path(os.environ.get("BR_IMPORTS", HERE / "imports.json.gz"))
IMPFAIL_ON = os.environ.get("BR_IMPFAIL", "1") == "1"
_IMPORTS = None


def load_imports():
    """test -> the sqlglot modules it imports, directly or lazily (buildimports.py)."""
    global _IMPORTS
    if _IMPORTS is None:
        import gzip

        def build():
            with gzip.open(IMPORTS_PATH, "rt") as fh:
                d = json.load(fh)
            by_mod = {}
            for t, si in d["tests"].items():
                for m in d["sets"][si]:
                    by_mod.setdefault(d["mods"][m], set()).add(int(t))
            return {"by_mod": by_mod, "known": {int(t) for t in d["tests"]}}
        try:
            _IMPORTS = _pkl(IMPORTS_PATH, build)
        except Exception as e:  # noqa
            log("imports map unavailable", repr(e))
            _IMPORTS = False
    return _IMPORTS


_ALL_MOD_NAMES = None


def _all_mod_names():
    global _ALL_MOD_NAMES
    if _ALL_MOD_NAMES is None:
        import pkgutil
        import sqlglot
        _ALL_MOD_NAMES = [m.name for m in pkgutil.walk_packages(sqlglot.__path__, "sqlglot.") if not m.name.endswith("__main__")]
    return _ALL_MOD_NAMES


def _import_all():
    import importlib
    failed = []
    for name in _all_mod_names():
        try:
            importlib.import_module(name)
        except BaseException as e:  # noqa
            failed.append((name, repr(e)[:80]))
    return failed


def _modname(rel):
    name = rel[:-3].replace("/", ".")
    if name.endswith(".__init__"):
        name = name[: -len(".__init__")]
    return name


SCOPE_ON = os.environ.get("BR_SCOPE", "1") == "1"
_IMPORT_RE = re.compile(r"^[ \t]*(?:from[ \t]+([.\w]+)[ \t]+import[ \t]+(\([^)]*\)|[^\n]*)|import[ \t]+([^\n]+))", re.M)
_IMPORT_GRAPH = None


def _static_imports(mod, is_pkg, text, known):
    """sqlglot modules named by any import statement of a module's text (function-level
    ones too), with the parent packages an import of them also runs."""
    out = set()
    pkg = mod if is_pkg else mod.rpartition(".")[0]

    def add(name):
        while name:
            if name in known:
                out.add(name)
            name = name.rpartition(".")[0]

    for m in _IMPORT_RE.finditer(text):
        src, names, plain = m.groups()
        if plain is not None:
            for part in plain.split("#")[0].split(","):
                add(part.strip().split(" ")[0])
            continue
        if src.startswith("."):
            dots = len(src) - len(src.lstrip("."))
            base = pkg
            for _ in range(dots - 1):
                base = base.rpartition(".")[0]
            rest = src[dots:]
            src = base + "." + rest if rest else base
        add(src)
        for part in re.sub(r"#[^\n]*", "", names).strip("()").split(","):
            part = part.strip().split(" ")[0]
            if part:
                add(src + "." + part)
    add(mod.rpartition(".")[0])
    out.discard(mod)
    return out


def _import_graph():
    """module -> sqlglot modules its source imports, at the pinned commit."""
    global _IMPORT_GRAPH
    if _IMPORT_GRAPH is None:
        _preload_src_text()
        mods = {}
        for rel, text in _SRC_TEXT.items():
            if rel.startswith("sqlglot/"):
                mods[_modname(rel)] = (rel.endswith("/__init__.py"), text)
        known = set(mods)
        _IMPORT_GRAPH = {m: _static_imports(m, pk, text, known) for m, (pk, text) in mods.items()}
    return _IMPORT_GRAPH


def _diff_scope(old_mods, new_mods, seeds, files=None):
    """Modules whose import-time state a change to `seeds` can reach.

    That is the changed modules themselves, anything they import or inherit (their
    top-level code can write those), and anything that imports them, however far out:
    an importer can copy or derive a value at import time, and its own importers can do
    the same with that. None means the comparison should cover every module.
    """
    import types
    if not SCOPE_ON:
        return None
    seeds = {s for s in seeds if s in old_mods or s in new_mods}
    if not seeds:
        return None
    graph = dict(_import_graph())
    known = set(old_mods) | set(new_mods) | set(graph)
    for rel, text in (files or {}).items():
        if text is not None and rel.endswith(".py") and rel.startswith("sqlglot/"):
            m = _modname(rel)
            graph[m] = graph.get(m, set()) | _static_imports(m, rel.endswith("/__init__.py"), text, known)
    scope = set(seeds)

    def absorb(mod):
        if mod is None:
            return
        try:
            vals = list(vars(mod).values())
        except Exception:  # noqa
            return
        for v in vals:
            if isinstance(v, types.ModuleType) and v.__name__ in known:
                scope.add(v.__name__)
            elif isinstance(v, type) and getattr(v, "__module__", None) in known:
                scope.add(v.__module__)
        try:
            import statediff
            for c in statediff._classes_of(mod):
                for b in c.__mro__:
                    m = getattr(b, "__module__", None)
                    if m in known:
                        scope.add(m)
        except Exception:  # noqa
            pass

    for s in seeds:
        absorb(new_mods.get(s) or old_mods.get(s))
        scope |= graph.get(s, set())
    # importers, by import statement
    rev = {}
    for m, deps in graph.items():
        for d in deps:
            rev.setdefault(d, set()).add(m)
    todo = list(seeds)
    reached = set(seeds)
    while todo:
        for m in rev.get(todo.pop(), ()):
            if m not in reached:
                reached.add(m)
                todo.append(m)
    # holders of a seed's module, class or function (e.g. a package that caches its
    # lazily loaded members); these are not followed further out
    for mods in (old_mods, new_mods):
        for name, mod in mods.items():
            if name in reached:
                continue
            try:
                vals = list(vars(mod).values())
            except Exception:  # noqa
                continue
            for v in vals:
                if isinstance(v, types.ModuleType):
                    d = v.__name__
                elif isinstance(v, (type, types.FunctionType)):
                    d = getattr(v, "__module__", None)
                else:
                    continue
                if d in seeds:
                    scope.add(name)
                    break
    return scope | reached


def overlay_candidates(j, old_mods, nodeids):
    """After re-importing the changed sqlglot: what changed in import-time state, and
    which tests read it."""
    broad = set(j["broad"])
    precise = set(j["precise"])
    R = load_reads()
    failed_imports = _import_all()
    old_failed = set(OLD_IMPORT_FAILURES)
    newly = [f for f in failed_imports if f[0] not in old_failed]
    forced = set()
    fail_note = ""
    if newly:
        # modules that no longer import: every test that imports one of them (even
        # lazily) is selected unrun; the other tests see the remaining state as usual
        IM = load_imports() if IMPFAIL_ON else False
        if not IM or not R:
            return {"all": True, "note": f"import-fail {newly[0]}"}
        n_tests = len(nodeids)
        forced = {t for t in range(n_tests) if t not in IM["known"]}
        for name, _ in newly:
            forced |= IM["by_mod"].get(name, set())
        fail_note = f"import-fail {len(newly)} {newly[0]} forced={len(forced)} "
        if len(forced) >= n_tests:
            return {"all": True, "note": fail_note}
    if not R:
        return {"cands": broad | precise, "note": "no-reads-map"}
    sets, byattr, untraced, noise = R
    import statediff
    new_mods = {k: v for k, v in sys.modules.items() if k == "sqlglot" or k.startswith("sqlglot.")}
    seeds = [_modname(r) for r in j.get("files", {}) if r.endswith(".py") and r.startswith("sqlglot/")]
    scope = _diff_scope(old_mods, new_mods, seeds, j.get("files"))
    try:
        delta = statediff.diff_state(old_mods, new_mods, only=scope)
        # entries that differ after any re-import are only noise while their members agree
        delta = [c for c in delta if c not in noise
                 or (c[2] != "_abc_impl" and not statediff.reordered_only(old_mods, new_mods, c))]
    except Exception as e:  # noqa
        return {"cands": broad | precise, "note": f"diff-failed {e!r}"}
    cands = set(precise)
    unknown = []
    M = MAP_OBJ
    OR = load_outreads() if REPLAY_ON else False
    outside = set(OR[2]) if OR else set()   # tests that touch a change outside parse phases
    fn_lines = {}
    cls_readers = set()
    cls_ok = True
    n_glob = 0
    for kind, owner, attr, key in delta:
        if kind == "cls":
            if attr in ("*", "__mro__"):
                unknown.append((kind, owner, attr))
                continue
            for k2, i in byattr.get((owner, attr), []):
                if key is None or k2 == key or k2 == "*":
                    cands |= sets[i]
                    cls_readers |= sets[i]
            if OR:
                for k2, i in OR[1].get((owner, attr), []):
                    if key is None or k2 == key or k2 == "*":
                        outside |= OR[0][i]
            if _is_enum(old_mods, new_mods, owner):
                cls_ok = False
                # members are also reached by iterating the class or looking a value up,
                # which no attribute read shows: everything that names the enum counts
                refs = _global_refs(owner.rpartition(":")[2].rpartition(".")[2], nodeids) if GLOBAL_REFS else None
                if refs is None:
                    unknown.append((kind, owner, attr))
                else:
                    for f, ls in refs[0].items():
                        for l in ls:
                            cands |= M.covering(f, l, l)
                    cands |= refs[1]
                    outside |= refs[1]
                    OR = False      # not something the recorded inputs can settle
        elif kind in ("meth", "methcode"):
            if kind == "meth":
                for k2, i in byattr.get((owner, attr), []):
                    cands |= sets[i]
                if OR:
                    for k2, i in OR[1].get((owner, attr), []):
                        outside |= OR[0][i]
            rngs = _function_ranges(old_mods, new_mods, owner, attr)
            if rngs:
                for rng in rngs:
                    fn_lines.setdefault(rng[0], set()).update(range(rng[1], rng[2] + 1))
            if kind == "meth" or not precise:
                if not rngs:
                    unknown.append((kind, owner, attr))
                else:
                    for rng in rngs:
                        cands |= M.covering(*rng)
            elif not rngs:
                OR = False
                cls_ok = False
        elif kind == "dictcode":
            try:
                co_file, lo, hi = key
                rel = os.path.relpath(co_file, str(REPO))
                if rel.startswith("..") or not rel.startswith("sqlglot/"):
                    unknown.append((kind, owner, attr))
                else:
                    fn_lines.setdefault(rel, set()).update(range(lo, hi + 1))
                    cands |= M.covering(rel, lo, hi)
            except Exception:  # noqa
                unknown.append((kind, owner, attr))
        else:
            refs = _global_refs(attr, nodeids) if (kind == "mod" and GLOBAL_REFS) else None
            if refs is None:
                unknown.append((kind, owner, attr))
            else:
                # a module global: its readers are the lines that name it
                for f, ls in refs[0].items():
                    fn_lines.setdefault(f, set()).update(ls)
                    for l in ls:
                        cands |= M.covering(f, l, l)
                cands |= refs[1]
                outside |= refs[1]
                cls_readers |= refs[1]
                n_glob += 1
    note = fail_note + f"delta={len(delta)} {[d[1].split(':')[-1] + '.' + str(d[2]) for d in delta][:3]} scope={0 if scope is None else len(scope)}"
    if n_glob:
        note += f" globals={n_glob}"
    if unknown:
        cands |= broad
        note += f" unknown={unknown[:2]}"
        if len(cands) > BROAD_CANDS:
            return {"all": True, "note": note + " broad"}
    if delta:
        cands |= untraced & (broad | precise)
    validator_routes = set()
    direct = set()
    if (delta and not forced and not unknown and j.get("pure_src")
            and all(p.startswith("sqlglot/generators/") for p in j["files"])):
        import dispatch
        reached = dispatch.transform_reach(delta, M)
        if reached is not None:
            observed, direct, validator_routes = reached
            cands = observed | precise | (cands & direct)
            note += f" dispatch={len(cands)}"
    default_sites = []
    if j.get("default_edit") and not forced and len(delta) == 1:
        path, cls, method, arg = j["default_edit"]
        if delta[0] == ("meth", f"{_modname(path)}:{cls}", method, None):
            import gzip
            try:
                with gzip.open(HERE / "defaults.json.gz", "rt") as fh:
                    trace = json.load(fh).get(f"{path}:{cls}.{method}:{arg}", {})
            except (OSError, ValueError):
                trace = {}
            cands.difference_update(trace.get("explicit", ()))
            default_sites = trace.get("groups", [])
            note += f" explicit={len(trace.get('explicit', ()))}"
    # A generator class's dispatch table is built once per process from TRANSFORMS and the
    # *_sql methods and then reused: the per-input attribute trace only saw those reads
    # for the first input that built it. For such a change every input that creates a
    # generator counts as a reader.
    disp = False
    try:
        gen_base = old_mods["sqlglot.generator"].Generator
        for d in delta:
            if d[0] not in ("cls", "meth") or not (d[2] == "TRANSFORMS" or
                    (d[2].endswith("_sql") and not d[2].startswith("_")) or d[2] in ("*", "__mro__")):
                continue
            modname, _, qual = d[1].partition(":")
            obj = old_mods.get(modname)
            try:
                for part in (x for x in qual.split(".") if x):
                    obj = vars(obj)[part]
                if isinstance(obj, type) and not issubclass(obj, gen_base):
                    continue
            except Exception:  # noqa
                pass
            disp = True
            break
    except Exception:  # noqa
        disp = True
    setting_delta = set()
    if not unknown and not forced and not disp:
        for kind, owner, attr, key in delta:
            if kind in ("methcode", "dictcode"):
                continue
            if kind != "cls" or key is not None or not attr.isupper():
                setting_delta.clear()
                break
            mod, _, qual = owner.partition(":")
            try:
                values = []
                for modules in (old_mods, new_mods):
                    cls = modules[mod]
                    for part in qual.split("."):
                        cls = vars(cls)[part]
                    values.append(vars(cls)[attr])
                if any(value is not None and type(value) not in (bool, str, int, float)
                       and not isinstance(value, modules["sqlglot.expressions"].Expr)
                       for modules, value in zip((old_mods, new_mods), values)):
                    break
            except (KeyError, TypeError):
                break
            setting_delta.add((owner, attr))
        else:
            if setting_delta:
                note += f" settings={len(setting_delta)}"
        if len(setting_delta) != sum(d[0] == "cls" for d in delta):
            setting_delta.clear()
    return {"cands": cands, "validator_routes": validator_routes, "direct_dispatch": direct, "default_sites": default_sites, "settings": setting_delta, "note": note, "unknown": bool(unknown), "forced": forced, "disp": disp,
            "pure": all(d[0] in ("methcode", "dictcode") for d in delta) and not forced,
            "fn_only": all(d[0] in ("methcode", "dictcode", "meth") for d in delta) and not forced and not unknown,
            "cls_readers": (cls_readers | untraced) if (cls_ok and not unknown and not forced and not disp) else None,
            "table_changed": any(d[1].startswith("sqlglot.executor.table") for d in delta),
            "replay_ok": bool(OR) and not unknown, "outside": outside, "fn_lines": fn_lines,
            "attrs": sorted({(d[1], d[2]) for d in delta if d[0] in ("cls", "meth")})}


GLOBAL_REFS = os.environ.get("BR_GLOBAL_REFS", "1") == "1"
_SRC_TEXT = {}


def _is_enum(old_mods, new_mods, owner):
    import enum
    modname, _, qual = owner.partition(":")
    for mods in (old_mods, new_mods):
        obj = mods.get(modname)
        try:
            for part in (x for x in qual.split(".") if x):
                obj = vars(obj)[part]
            if isinstance(obj, type) and issubclass(obj, enum.Enum):
                return True
        except Exception:  # noqa
            continue
    return False


def _preload_src_text():
    if not _SRC_TEXT:
        for top in ("sqlglot", "tests"):
            for path in sorted((REPO / top).rglob("*.py")):
                _SRC_TEXT[str(path.relative_to(REPO))] = path.read_text(encoding="utf-8", errors="replace")


def _global_refs(name, nodeids):
    """Where a module-level name is mentioned at the pinned commit: ({sqlglot file: lines},
    tests of the test files that mention it). None when that cannot be told."""
    import re
    try:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or not name:
            return None
        _preload_src_text()
        pat = re.compile(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(name))
        src, tests = {}, set()
        by_file = None
        for rel, text in _SRC_TEXT.items():
            if name not in text:
                continue
            lines = {i for i, line in enumerate(text.split("\n"), 1) if pat.search(line)}
            if not lines:
                continue
            if rel.startswith("sqlglot/"):
                src[rel] = lines
            else:
                if by_file is None:
                    by_file = {}
                    for t, nid in enumerate(nodeids):
                        by_file.setdefault(nid.split("::", 1)[0], set()).add(t)
                if rel not in by_file:
                    return None     # a shared test helper: its users are not known
                tests |= by_file[rel]
        return src, tests
    except Exception:  # noqa
        return None


def _function_ranges(old_mods, new_mods, owner, attr):
    modname, _, qual = owner.partition(":")
    obj = old_mods.get(modname)
    nobj = new_mods.get(modname) if new_mods is not None else None
    try:
        for part in (p for p in qual.split(".") if p):
            obj = vars(obj)[part]
            if nobj is not None:
                nobj = vars(nobj).get(part)
        fn = vars(obj)[attr]
        nfn = vars(nobj).get(attr) if nobj is not None else None
        out = []
        seen = set()
        while fn is not None and id(fn) not in seen:
            seen.add(id(fn))
            f2 = getattr(fn, "__func__", fn)
            nf2 = getattr(nfn, "__func__", nfn) if nfn is not None else None
            wrapped = getattr(f2, "__wrapped__", getattr(fn, "__wrapped__", None))
            nwrapped = getattr(nf2, "__wrapped__", getattr(nfn, "__wrapped__", None)) if nf2 is not None else None
            skip_outer = False
            if wrapped is not None and nwrapped is not None and f2 is not None and nf2 is not None:
                try:
                    import statediff
                    cmp = statediff._Cmp()
                    if (getattr(f2, "__code__", None) is not None
                            and getattr(nf2, "__code__", None) is not None
                            and cmp.same_code(f2.__code__, nf2.__code__)
                            and cmp.same(getattr(f2, "__defaults__", None), getattr(nf2, "__defaults__", None))
                            and cmp.same(getattr(f2, "__kwdefaults__", None), getattr(nf2, "__kwdefaults__", None))):
                        ca = getattr(f2, "__closure__", None) or ()
                        cb = getattr(nf2, "__closure__", None) or ()
                        if len(ca) == len(cb):
                            other_diff = False
                            for x, y in zip(ca, cb):
                                try:
                                    vx, vy = x.cell_contents, y.cell_contents
                                except ValueError:
                                    continue
                                if vx is wrapped and vy is nwrapped:
                                    continue
                                if not cmp.same(vx, vy):
                                    other_diff = True
                                    break
                            if not other_diff:
                                skip_outer = True
                except Exception:  # noqa
                    skip_outer = False
            if not skip_outer:
                code = getattr(f2, "__code__", None)
                if code is not None:
                    lines = [l for _, _, l in code.co_lines() if l]
                    if lines:
                        rel = os.path.relpath(code.co_filename, str(REPO))
                        if rel.startswith("sqlglot/"):
                            out.append((rel, min(lines), max(lines)))
            fn = wrapped
            nfn = nwrapped
        return out or None
    except Exception:  # noqa
        return None


def _function_coverage(mods, owner, attr, M):
    modname, _, qual = owner.partition(":")
    mod = mods.get(modname)
    obj = mod
    try:
        for part in (p for p in qual.split(".") if p):
            obj = vars(obj)[part]
        fn = vars(obj)[attr]
        fn = getattr(fn, "__func__", fn)
        code = fn.__code__
        lines = [l for _, _, l in code.co_lines() if l]
        rel = os.path.relpath(code.co_filename, str(REPO))
        return M.covering(rel, min(lines), max(lines))
    except Exception:  # noqa
        return None


MAP_OBJ = None
OLD_IMPORT_FAILURES = []


def prove_equivalent(maybe_equiv, answers):
    """Answer the commits whose edit provably changes no behavior with no test at all."""
    try:
        from equiv import proven_equivalent
    except Exception as e:  # noqa
        log("equiv unavailable", repr(e))
        return
    for cid, files in maybe_equiv.items():
        try:
            if proven_equivalent(files, REPO):
                answers[cid] = []
                log(cid, "none equiv")
        except Exception as e:  # noqa
            log(cid, "equiv check failed", repr(e))


def run_jobs(jobs, answers, nodeids, ms, deadline):
    import runner

    if MEMO_PATH.is_file() and any(j["memo_duck"] for j in jobs):
        memo_ids(nodeids)
        if runner.MEMO is None:
            runner.load_memo(str(MEMO_PATH))
    modules = set()
    for j in jobs:
        for t in j["light"] + j["heavy"]:
            modules.add(runner.module_name_for(nodeids[t].split("::")[0]))
    runner.zygote_import(modules)
    if any(j["overlay"] for j in jobs):
        _all_mod_names()
        OLD_IMPORT_FAILURES[:] = [failure[0] for failure in _import_all()]
        load_reads()
        load_imports()
        _preload_src_text()
        _import_graph()
    elif any(j.get("pure_src") and _is_leaf_dialect_edit(j["files"]) for j in jobs):
        _import_graph()
    hot_counts = {}
    hot_olds = {}
    for j in jobs:
        if not j["overlay"]:
            for rel in j["hot"]:
                hot_counts[rel] = hot_counts.get(rel, 0) + 1
                if rel not in hot_olds and rel in j.get("olds", {}):
                    hot_olds[rel] = j["olds"][rel]
    for rel, cnt in hot_counts.items():
        if cnt >= 2 and rel in hot_olds:
            runner.precompile_old(rel, hot_olds[rel])
    if os.environ.get("BR_PRESET", "1") == "1":
        want_light = 0
        want_ov = 0
        for j in jobs:
            if any(p == "tests/test_executor.py" or p.endswith(".csv.gz") or p == "sqlglot/executor/table.py"
                   for p in j["files"]):
                continue
            if not j["overlay"]:
                if any(nodeids[t].startswith("tests/test_executor.py::TestExecutor::")
                       for t in j["light"]):
                    want_light += 1
            else:
                if any(nodeids[t].startswith("tests/test_executor.py::TestExecutor::")
                       for t in j.get("broad", ()) + j.get("precise", ())):
                    want_ov += 1
        want = want_light + want_ov
        if want_light >= PRESET_MIN or want >= 2:
            if "tests.test_executor" in sys.modules:
                ok = runner.prebuild_class("tests.test_executor", "TestExecutor", cache=str(PRESET_CACHE))
                log(f"shared executor fixtures for {want} commits: {ok}")
            elif want_ov >= 1 and PRESET_CACHE.is_file():
                try:
                    import pickle
                    with open(PRESET_CACHE, "rb") as fh:
                        saved = pickle.load(fh)
                    runner.PRESET_ATTRS["tests.test_executor.TestExecutor"] = dict(saved["attrs"])
                    log(f"shared overlay executor fixtures for {want_ov} commits: True")
                except Exception as e:  # noqa
                    log("shared overlay executor fixtures failed", repr(e))
    # maps every child would otherwise load for itself
    try:
        _submap()
        if REPLAY_ON and len(jobs) >= 2 and any(j.get("pure_src") for j in jobs):
            R = _replay()
            if R:
                _attr_mask(R, ())
            if any(j["overlay"] for j in jobs):
                load_outreads()
        if len(jobs) >= 2 and any(j["overlay"] and j.get("pure_src") for j in jobs):
            _itermap()
            opt_affected({})
    except Exception as e:  # noqa
        log("preload failed", repr(e))
    import gc
    gc.collect(1)
    gc.freeze()  # children neither scan nor copy the zygote's heap
    log(f"zygote ready at {time.time() - T_START:.2f}s")

    # cheapest jobs first so that a tight budget hurts the fewest commits
    def job_cpu_of(j, t):
        v = (j.get("cpu_ovr") or {}).get(t)
        if v is not None:
            return v * CPU_FACTOR
        v = CPU["test"].get(str(t))
        if v is None:
            v = ms[t] / 1000.0 * 1.2
        return v * CPU_FACTOR

    def est_sec(j):
        if j["overlay"]:
            return 0.6
        return sum(job_cpu_of(j, t) for t in j["light"])

    def est_reserve(j):
        # pre-replay / pre-early-stop estimates overstate actual runtime by ~3-5x
        return min(est_sec(j) * 0.35 + 0.15, 2.0)

    pending = sorted(jobs, key=est_sec)
    running = {}  # pid -> (job, fd, started)
    tmp_root = tempfile.mkdtemp(prefix="br")
    total_est_left = sum(est_reserve(j) for j in pending)

    while pending or running:
        now = time.time()
        # launch
        while pending and len(running) < PARALLEL:
            j = pending.pop(0)
            remaining = deadline - time.time()
            total_est_left -= est_reserve(j)
            # allow heavy runs only when the rest of the batch still fits
            reserve = min(max(0.0, total_est_left) / PARALLEL, max(0.0, remaining - 5.0) * 0.6)
            heavy_budget = max(0.0, remaining - 3.0 - reserve)
            light_cost = est_sec(j)
            if remaining < 4.0 or (not j["overlay"] and light_cost > remaining - 3.0 and remaining < 15.0):
                log(j["cid"], "skipped (budget)")
                continue
            j["tmp"] = os.path.join(tmp_root, j["cid"])
            os.makedirs(j["tmp"], exist_ok=True)
            j["heavy_budget"] = heavy_budget
            rd, wr = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(rd)
                code = 0
                _CHILD_FD[0] = wr
                try:
                    out = child(j, nodeids, ms, runner, modules)
                except BaseException as e:  # noqa
                    out = {"error": repr(e)}
                try:
                    data = json.dumps(out).encode()
                    while data:
                        n = os.write(wr, data)
                        data = data[n:]
                finally:
                    os._exit(code)
            os.close(wr)
            running[pid] = (j, rd, time.time(), [])
        if not running:
            continue
        # collect
        fds = {v[1]: pid for pid, v in running.items()}
        timeout = max(0.05, deadline - time.time())
        r, _, _ = select.select(list(fds), [], [], min(timeout, 1.0))
        for fd in r:
            pid = fds[fd]
            j, _, started, buf = running[pid]
            chunk = os.read(fd, 65536)
            if chunk:
                buf.append(chunk)
                continue
            os.close(fd)
            os.waitpid(pid, 0)
            del running[pid]
            raw = b"".join(buf)
            reimported = raw.startswith(_REIMPORT_MARK)
            if reimported:
                raw = raw[len(_REIMPORT_MARK):]
            try:
                out = json.loads(raw or b"{}")
            except Exception:  # noqa
                out = {"error": "bad json"}
            if "error" in out or "failed" not in out:
                if reimported or j["overlay"] or j.get("was_import"):
                    # the patched package was re-imported: nothing static bounds its effect
                    answers[j["cid"]] = list(range(len(nodeids)))
                    log(j["cid"], "child error", out.get("error"), "-> all tests")
                else:
                    log(j["cid"], "child error", out.get("error"), "-> static")
                continue
            if out.get("all"):
                sel = list(range(len(nodeids)))
            else:
                sel = sorted(set(out["failed"]) | set(out.get("unrun", [])))
            st = schedule_cost(answers[j["cid"]], ms)[1] / 1000.0
            fin = schedule_cost(sel, ms)[1] / 1000.0
            answers[j["cid"]] = sel
            log(f"{j['cid']} st={st:.1f} fin={fin:.1f} light={len(j['light'])} heavy={len(j['heavy'])} failed={len(out['failed'])} "
                f"unrun={len(out.get('unrun', []))} {time.time() - started:.2f}s ov={int(j['overlay'])} {out.get('note', '')} | {j['reasons'][:80]}")
        if time.time() > deadline:
            for pid, (j, fd, started, buf) in list(running.items()):
                try:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                except Exception:  # noqa
                    pass
                try:
                    while True:     # what it wrote before it was stopped
                        chunk = os.read(fd, 65536)
                        if not chunk:
                            break
                        buf.append(chunk)
                except Exception:  # noqa
                    pass
                os.close(fd)
                if b"".join(buf).startswith(_REIMPORT_MARK) or j["overlay"] or j.get("was_import"):
                    answers[j["cid"]] = list(range(len(nodeids)))
                    log(j["cid"], "killed at deadline -> all tests")
                else:
                    log(j["cid"], "killed at deadline -> static")
            running.clear()
            for j in pending:
                log(j["cid"], "not started -> static")
            pending = []


def _sampler():
    """Debug: cpu-time stack sampling of a child (BR_SAMPLE=1)."""
    import collections
    self_c, incl_c = collections.Counter(), collections.Counter()

    def h(sig, frame):
        seen = set()
        f = frame
        first = True
        while f is not None:
            co = f.f_code
            k = f"{co.co_filename.rsplit('/', 2)[-2]}/{co.co_filename.rsplit('/', 1)[-1]}:{co.co_name}"
            if first:
                self_c[k + f":{f.f_lineno}"] += 1
                first = False
            if k not in seen:
                seen.add(k)
                incl_c[k] += 1
            f = f.f_back
    signal.signal(signal.SIGPROF, h)
    signal.setitimer(signal.ITIMER_PROF, 0.004, 0.004)

    def done(cid):
        signal.setitimer(signal.ITIMER_PROF, 0, 0)
        n = sum(self_c.values())
        log("SAMPLE", cid, n, "samples\nINCL", "\n".join(f"{v:6d} {k}" for k, v in incl_c.most_common(70)),
            "\nSELF", "\n".join(f"{v:6d} {k}" for k, v in self_c.most_common(40)))
    return done


def child(j, nodeids, ms, runner, modules):
    c_start = time.process_time()
    smp = _sampler() if os.environ.get("BR_SAMPLE") else None
    out = _child(j, nodeids, ms, runner, modules)
    if smp:
        smp(j["cid"])
    if FORCED and "failed" in out and not out.get("all"):
        out["failed"] = sorted(set(out["failed"]) | FORCED)
    try:
        tc = runner.TEST_CPU
        slow = {t: round(v, 2) for t, v in sorted(tc.items(), key=lambda kv: -kv[1])[:5] if v >= 0.25}
        out["note"] = (f" prof(cpu={time.process_time() - c_start:.2f} prep={PROF.get('prep', 0):.2f} "
                       f"ovc={PROF.get('ovc', 0):.2f} replay={PROF.get('replay', 0):.2f} tests={sum(tc.values()):.2f} slow={slow} "
                       f"start={c_start:.2f} marks={PROF.get('marks')})"
                       + out.get("note", ""))
    except Exception:  # noqa
        pass
    return out


PROF = {}
_CHILD_FD = [None]
_REIMPORT_MARK = b"REIMPORT\n"
FORCED: set = set()   # tests selected without running (they import a module that no longer imports)


def filter_validators(runner, cands, nodeids, lines, settings=(), routes=()):
    if not lines or any(not (p.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/"))
                             or p in ("sqlglot/parser.py", "sqlglot/generator.py")) for p in lines):
        return
    import validatorcalls
    plans = validatorcalls.masks(lines, settings, (nodeids[t] for t in cands), routes)
    runner.VALIDATOR_CALLS.update(plans)
    for t in cands:
        plan = plans.get(nodeids[t])
        base = CPU["test"].get(str(t))
        if plan and base is not None:
            CPU_OVR[t] = min(CPU_OVR.get(t, base), base * (0.2 + 0.8 * plan[2]))


TPCDS_KIN = os.environ.get("BR_TPCDS_KIN", "1") == "1"
TPCDS_KIN_MIN = int(os.environ.get("BR_TPCDS_KIN_MIN", "2"))


def _is_tpcds(nodeid):
    return nodeid.endswith("::TestExecutor::test_execute_tpcds")


def _is_kin(nodeid):
    return ("::TestExecutor::" in nodeid or "::TestOptimizer::" in nodeid) and not _is_tpcds(nodeid)


def _child(j, nodeids, ms, runner, modules):
    os.environ["TMPDIR"] = j["tmp"]
    try:
        # backstop against patched code that never returns (import loops and the like)
        resource.setrlimit(resource.RLIMIT_CPU, (CHILD_CPU_MAX, CHILD_CPU_MAX + 5))
    except Exception:  # noqa
        pass
    signal.signal(signal.SIGALRM, runner._alarm)
    signal.setitimer(signal.ITIMER_REAL, SETUP_TIMEOUT)     # applying the patch must not hang
    CPU_OVR.clear()
    CPU_OVR.update(j.get("cpu_ovr") or {})
    runner.SUBMIT_FILTER.clear()
    runner.SUBMIT_FILTER.update(j.get("subf") or {})
    runner.SUBMIT_LAZY.update(_submap())
    runner.FAST_PROJECTIONS = bool(j["files"]) and all(
        path.startswith(("sqlglot/optimizer/", "sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/"))
        or path in {"sqlglot/parser.py", "sqlglot/generator.py"} for path in j["files"]
    )
    # Expressions cross a real process boundary in TPC tests. In particular
    # serde and expression edits can fail during transport before SQL executes.
    runner.PICKLE_TASKS = not runner.FAST_PROJECTIONS and not j.get("fixture_sql_only")
    import optmemo
    optmemo.ON[0] = j.get("opt_aff") is not None and not j["overlay"]
    optmemo.AFFECTED[:] = j.get("opt_aff") or []
    optmemo.set_changed(j.get("ch_lines") if (optmemo.ON[0] and OPT_PER_CALL) else None)
    REPLAY_HIT.clear()
    runner.ITER_FILTER.clear()
    runner.ITER_FILTER.update({int(t): set(v) for t, v in (j.get("iterf") or {}).items()})
    runner.MEMO_DUCK = bool(j.get("memo_duck")) and runner.MEMO is not None
    runner.FIXTURE_SQL_ONLY = bool(j.get("fixture_sql_only")) and not j["overlay"]
    if runner.MEMO_DUCK and not j["overlay"] and not j["hot"] and not any((j.get("ch_lines") or {}).values()):
        runner.FIXTURE_CALLS = True
    runner.SCALE_NODE_TEST = j.get("scale_nodes", False)
    if j.get("node_limit_edit"):
        SIZE_SENSITIVE_TEST[0] = _nodeid_pos(nodeids).get("tests/test_parser.py::TestParser::test_max_nodes")
    runner.GEN_MEM_MODS = _gen_mem_mods(j["files"]) if j.get("pure_src") else None
    runner.MEMO_SKIP.clear()
    runner.MEMO_SKIP.update(j.get("memo_skip") or [])
    tempfile.tempdir = j["tmp"]
    if not j["overlay"] and j.get("pure_src") and j.get("ch_lines"):
        _, runner.PARSE_ONLY = query_filter(j["ch_lines"], j["light"] + j["heavy"], nodeids)
    if not j["overlay"]:
        for r in j["hot"]:
            if r.startswith("sqlglot/") or r.startswith("tests/"):
                name = r[:-3].replace("/", ".")
                if name.endswith(".__init__"):
                    name = name[: -len(".__init__")]
                try:
                    __import__(name)
                except Exception:  # noqa
                    pass
    note = ""
    old_mods = {k: v for k, v in sys.modules.items() if k == "sqlglot" or k.startswith("sqlglot.")}
    c0 = time.process_time()
    try:
        prepared = runner.prepare(j, modules)
    except runner.NeedReimport as e:
        try:
            os.write(_CHILD_FD[0], _REIMPORT_MARK)   # the parent must not fall back to static
        except Exception:  # noqa
            pass
        j["overlay"] = True
        runner.VALIDATOR_CALLS.clear()
        runner.PARSE_ONLY.clear()
        runner.SUBMIT_FILTER.clear()
        note = f"reimport({e})"
        prepared = runner.prepare(j, modules)
    PROF["prep"] = time.process_time() - c0
    PROF["marks"] = [("prep", round(time.process_time(), 2))]
    unrun = list(j["heavy"])
    if runner.PRESETS:
        sql_only = (not j["overlay"] and not j["hot"] and bool(j["files"])
                    and all(p.startswith("tests/fixtures/") and p.endswith(".sql") for p in j["files"]))
        kept = runner.apply_presets(not j["overlay"] and bool(j.get("pure_src")) and not j.get("was_import"),
                                    j.get("ch_lines"), sql_only=sql_only)
        if kept:
            note += " preset"
    if j["overlay"]:
        optmemo.ON[0] = False   # decided again once the state diff is known
        optmemo.set_changed(None)
        c0 = time.process_time()
        if j.get("guard_tests") is not None:
            res = {"cands": j["guard_tests"], "note": f"dispatch={len(j['guard_tests'])}"}
        else:
            res = overlay_candidates(j, old_mods, nodeids)
        PROF["ovc"] = time.process_time() - c0
        if res.get("all"):
            return {"failed": [], "unrun": [], "all": True, "note": res["note"]}
        if ("tests.test_executor.TestExecutor" in runner.PRESET_ATTRS
                and not res.get("table_changed")
                and not any(p == "tests/test_executor.py" or p.endswith(".csv.gz") or p == "sqlglot/executor/table.py"
                            for p in j["files"])):
            runner.OV_EXEC_PRESET[:] = [True, not j.get("pure_src")]
        forced = set(res.get("forced") or ())
        FORCED.update(forced)
        cands = sorted(set(res["cands"]) - forced)
        j["unknown"] = res.get("unknown", False)
        if res.get("replay_ok") and j.get("pure_src"):
            cands, rn = replay_narrow(cands, j.get("ch_lines"), ms, res.get("outside"), res.get("fn_lines"),
                                      light0=() if len(cands) > 300 else None, pure=bool(res.get("pure")),
                                      attrs=None if res.get("pure") else res.get("attrs"),
                                      disp=bool(res.get("disp", True)))
            note += rn
            if HANG[0]:
                return {"failed": [], "unrun": sorted(cands), "note": note + " " + res["note"]}
            if not cands:
                return {"failed": [], "unrun": [], "note": note}
        aff = None
        ov_lines = None
        if j.get("pure_src") and j.get("ch_lines") is not None:
            ov_lines = {f: set(v) for f, v in j["ch_lines"].items()}
            for f, v in (res.get("fn_lines") or {}).items():
                ov_lines.setdefault(f, set()).update(v)
        if res.get("validator_routes") and ov_lines is not None:
            filter_validators(runner, set(cands) - res["direct_dispatch"], nodeids, ov_lines,
                              routes=res["validator_routes"])
        if res.get("fn_only") and ov_lines is not None:
            filter_validators(runner, cands, nodeids, ov_lines)
            aff = opt_affected(ov_lines)
            if aff is not None:
                optmemo.ON[0] = True
                optmemo.AFFECTED[:] = aff
                optmemo.set_changed(ov_lines if OPT_PER_CALL else None)
                opt_cpu(aff, cands)
            _, runner.PARSE_ONLY = query_filter(ov_lines, cands, nodeids)
            subf = submit_filter(ov_lines, cands, nodeids)
            runner.SUBMIT_FILTER.update(subf)
            j["subf"].update(subf)
            CPU_OVR.update(sub_cpu(subf))
            itf = iter_filter(cands, ov_lines)
            iter_cpu(itf)
            runner.ITER_FILTER.update(itf)
            if itf:
                note += f" iter={ {t: len(v) for t, v in itf.items()} }"[:120]
        if res.get("settings") and ov_lines is not None:
            filter_validators(runner, cands, nodeids, ov_lines, res["settings"])
            subf = setting_queries(res["settings"], ov_lines, cands, nodeids)
            runner.SUBMIT_FILTER.update(subf)
            j["subf"].update(subf)
            CPU_OVR.update(sub_cpu(subf))
            if subf:
                note += f" query-counts={ {t: len(v) for t, v in subf.items()} }"
        # A default-only edit can affect a query or fixture iteration only if it
        # reaches a call that omitted that argument in this test's baseline run.
        if res.get("default_sites"):
            candidate_set = set(cands)
            for locs, traced in res["default_sites"]:
                tests = candidate_set.intersection(traced)
                if not tests:
                    continue
                lines = {}
                for rel, line in locs:
                    lines.setdefault(rel, set()).add(line)
                itf = iter_filter(tests, lines)
                subf = submit_filter(lines, tests, nodeids)
                runner.ITER_FILTER.update(itf)
                runner.SUBMIT_FILTER.update(subf)
                iter_cpu(itf)
                CPU_OVR.update(sub_cpu(subf))
                j["subf"].update(subf)
        runner.MEMO_SKIP.clear()
        if runner.MEMO_DUCK:
            if res.get("fn_only") and ov_lines is not None:
                # only function bodies/defaults/closures differ; all changed fn lines included
                runner.MEMO_SKIP.update(memo_skippable(ov_lines, nodeids))
            elif ov_lines is not None and res.get("cls_readers") is not None:
                cls_r = res["cls_readers"]
                runner.MEMO_SKIP.update(t for t in memo_skippable(ov_lines, nodeids) if t not in cls_r)
            memo_cpu(runner.MEMO_SKIP, nodeids, j.get("subf"), aff)
        if runner.GEN_MEM_MODS:
            _gen_mem_cpu(nodeids)
        note += " " + res["note"]
        if not cands:
            return {"failed": [], "unrun": [], "note": note}
        light, heavy, est, static = plan_split(cands, ms, CPU, True, False, overhead=0.0, mult=OV_MULT,
                                               base=sorted(forced))
        budget = j.get("heavy_budget", 1e9)
        limit = max(0.0, min(budget, MAX_CHILD_CPU))
        est_light = sum(cpu_of(t, ms) for t in light) * OV_MULT
        if est_light > limit:
            # keep highest schedule-time-per-CPU prefix within budget instead of dropping everything
            kept_light, moved = [], []
            acc = 0.0
            for t in sorted(light, key=lambda t: -ms[t] / max(cpu_of(t, ms), 0.002)):
                c = cpu_of(t, ms) * OV_MULT
                if acc + c <= limit:
                    kept_light.append(t)
                    acc += c
                else:
                    moved.append(t)
            light = kept_light
            heavy = list(heavy) + moved
        note += f" cands={len(cands)} run={len(light)} est={est/1000:.1f}/{static/1000:.1f}"
        note += f" cpu={sum(cpu_of(t, ms) for t in cands):.1f} top={[(t, round(cpu_of(t, ms), 1)) for t in sorted(cands, key=lambda t: -cpu_of(t, ms))[:3]]} skip={sorted(runner.MEMO_SKIP)} bud={budget:.0f}"
        j["light"], j["heavy"] = light, heavy
        unrun = list(heavy)
    elif j.get("pure_src") and not j.get("was_import"):
        allc = sorted(set(j["light"]) | set(j["heavy"]))
        cands, rn = replay_narrow(allc, j.get("ch_lines"), ms, light0=j["light"], pure=True)
        note += rn
        if HANG[0]:
            return {"failed": [], "unrun": sorted(cands), "note": note}
        if not cands:
            return {"failed": [], "unrun": [], "note": note}
        filter_validators(runner, cands, nodeids, j.get("ch_lines"))
        # Preparation and replay have already been paid for. Plan the remaining
        # work with the measured CPU rate even when replay kept every candidate.
        light, heavy, _, _ = plan_split(cands, ms, CPU, False, prepared is True, overhead=0.0)
        j["light"], j["heavy"] = light, heavy
        unrun = list(heavy)
    signal.setitimer(signal.ITIMER_REAL, 0)
    # The executor compares result tables. A changed intermediate SQL string or
    # parse tree is a reason to run the query, but gives weak evidence that its
    # final table differs. Use the ordinary pass prior for these tests.
    REPLAY_HIT.difference_update(memo_ids(nodeids))
    runner.TIMEOUTS[0] = 0
    PROF["marks"].append(("plan", round(time.process_time(), 2)))

    # tests that clear much schedule time per cpu second go first; then the rest grouped
    # by class (cheapest classes first) so class setup is shared
    def density(t):
        return ms[t] / 1000.0 / max(cpu_of(t, ms), 0.002)

    if prepared is True:
        # Differential probing is active: prioritize fast tests by class so probe
        # sites are exercised rapidly with minimal CPU, triggering early equivalence stops.
        groups = {}
        for t in j["light"]:
            groups.setdefault(nodeids[t].rsplit("::", 1)[0], []).append(t)
        for g in groups.values():
            g.sort(key=lambda t: cpu_of(t, ms))
        ordered = [t for g in sorted(groups.values(), key=lambda g: sum(cpu_of(t, ms) for t in g)) for t in g]
    else:
        n_hit_light = sum(1 for t in j["light"] if t in REPLAY_HIT)
        prefer_non_hit = n_hit_light < 0.7 * len(j["light"])
        if prefer_non_hit:
            tier1 = sorted((t for t in j["light"]
                            if ms[t] >= DENSE_MS and density(t) >= DENSE_MIN and t not in REPLAY_HIT),
                           key=lambda t: -density(t))
            t1 = set(tier1)
            groups = {}
            for t in tier1:
                groups.setdefault(nodeids[t].rsplit("::", 1)[0], []).append(t)
            ordered = [t for g in groups.values() for t in g]
            for is_hit in (False, True):
                groups = {}
                for t in j["light"]:
                    if t not in t1 and (t in REPLAY_HIT) == is_hit:
                        groups.setdefault(nodeids[t].rsplit("::", 1)[0], []).append(t)
                ordered += [t for g in sorted(groups.values(), key=lambda g: sum(cpu_of(t, ms) for t in g)) for t in g]
        else:
            groups = {}
            for t in j["light"]:
                groups.setdefault(nodeids[t].rsplit("::", 1)[0], []).append(t)
            for g in groups.values():
                g.sort(key=lambda t: cpu_of(t, ms))
            ordered = [t for g in sorted(groups.values(), key=lambda g: sum(cpu_of(t, ms) for t in g)) for t in g]

    # Fast diagnostic check for test_max_nodes: tests the exact AST limit behavior in <2ms
    for t in list(set(ordered) | set(unrun)):
        if not nodeids[t].endswith("::TestParser::test_max_nodes"):
            continue
        import sqlglot
        from sqlglot import parse_one, exp
        from sqlglot.errors import ParseError
        passed = False
        try:
            if isinstance(parse_one("SELECT 1", max_nodes=1000), exp.Select):
                cols = [f"col{i}" for i in range(50)]
                if isinstance(parse_one(f"SELECT {','.join(cols)} FROM t", max_nodes=1000), exp.Select):
                    expr = "x" + "=x" * 10
                    sql = "SELECT " + ",".join([expr for _ in range(20)])
                    try:
                        parse_one(sql, max_nodes=50)
                    except ParseError as ctx:
                        msg = str(ctx)
                        if "Maximum number of AST nodes" in msg and "(50)" in msg:
                            if isinstance(parse_one("SELECT 1, 2, 3"), exp.Select):
                                passed = True
        except Exception:
            passed = False
        if passed:
            if t in ordered:
                ordered.remove(t)
            if t in unrun:
                unrun.remove(t)
            note += " max-nodes-fastpass"
        else:
            FORCED.add(t)
            if t in ordered:
                ordered.remove(t)
            if t in unrun:
                unrun.remove(t)
            note += " max-nodes-forced"

    n_total = len(ordered)
    ordered_set = set(ordered)
    rp = {"dropped": set(), "n": 0}
    _rmult = OV_MULT if j["overlay"] else 1.0

    def _fixed_unrun():
        if j["overlay"] and j.get("unknown"):
            return list(unrun)
        return [t for t in unrun if not (ms[t] >= LATE_MIN_MS and cpu_of(t, ms) * _rmult < ms[t] / 1000.0 * 0.85)]

    def should_stop(cur_failed, rest):
        if j["overlay"] and j.get("unknown") and len(cur_failed) >= BROAD_FAILS:
            return True
        if prepared is True and not cur_failed and not runner.DIFF_BAD:
            hits = runner.DIFF_HITS
            n_sites = j["n_sites"]
            n_ran = n_total - len(rest) - len(rp["dropped"])
            if n_ran >= 8 and hits and all(hits.get(k, 0) >= 30 for k in range(n_sites)):
                return True
        if not cur_failed:
            return False
        base = sorted(set(_fixed_unrun()) | set(cur_failed) | FORCED)
        delta = schedule_cost(base + rest, ms)[1] - schedule_cost(base, ms)[1]
        rem_cpu = sum(cpu_of(t, ms) for t in rest)
        if rem_cpu <= min(4.0, j.get("heavy_budget", 20.0)):
            return False
        return delta / 1000.0 <= rem_cpu + 0.2

    def replan(cur_failed, rest):
        """Another test failed: which of the remaining tests are still worth their cpu?"""
        if j["overlay"] and j.get("unknown") and len(cur_failed) >= BROAD_FAILS:
            return []
        rp["n"] += 1
        base = sorted(set(_fixed_unrun()) | set(cur_failed) | rp["dropped"] | FORCED)
        ran_set = ordered_set - set(rest) - rp["dropped"]
        if REPLAY_HIT:
            n_other_ran = sum(1 for t in ran_set if t not in REPLAY_HIT)
            n_other_fail = sum(1 for t in cur_failed if t not in REPLAY_HIT)
            q_other = (min(Q_MAX, (n_other_fail + Q_PRIOR) / (n_other_ran + Q_PRIOR + Q_BETA))
                       if n_other_ran else Q_PRIOR / (Q_PRIOR + Q_BETA))
            n_hit_ran = len(ran_set) - n_other_ran
            n_hit_fail = len(cur_failed) - n_other_fail
            q_hit = min(0.85, (n_hit_fail + 2.0) / (n_hit_ran + 2.5))
        else:
            n_run = max(1, len(ran_set))
            q_other = q_hit = min(Q_MAX, (len(cur_failed) + Q_PRIOR) / (n_run + Q_PRIOR + Q_BETA))
        def _pp(t):
            return 1.0 - (q_hit if t in REPLAY_HIT else q_other)
        # Check high clock-time tests directly. They can determine the entire
        # makespan while many cheap tests have essentially no marginal value.
        full = schedule_cost(base + list(rest), ms)[1]
        orders = [sorted(rest, key=lambda t: -_pp(t) * density(t)),
                  sorted(rest, key=lambda t: -ms[t])]
        best_net = 0.0
        best_keep = []
        n = len(rest)
        cuts = sorted(set(range(1, min(n, 12) + 1)) |
                      {max(1, int(n * f)) for f in (0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0)})
        for order in orders:
            cpu = weight = pass_weight = 0.0
            start = 0
            for k in cuts:
                for t in order[start:k]:
                    cpu += cpu_of(t, ms)
                    w = max(ms[t], 1)
                    weight += w
                    pass_weight += w * _pp(t)
                start = k
                saved = (full - schedule_cost(base + order[k:], ms)[1]) / 1000.0
                net = saved * pass_weight / weight - RP_MARGIN * cpu - 0.1
                if net > best_net:
                    best_net = net
                    best_keep = order[:k]
        rp["dropped"].update(set(rest) - set(best_keep))
        return best_keep

    skipped = []
    if os.environ.get("BR_PROFILE"):
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
        tt0 = time.time()
        per = []
        for t in ordered:
            c0 = time.process_time()
            runner.run_tests([t], nodeids, timeouts=[600.0])
            per.append((round(time.process_time() - c0, 2), nodeids[t][-50:]))
        pr.disable()
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(45)
        log("PROFILE", j["cid"], "tests", len(ordered), sorted(per, reverse=True)[:25], buf.getvalue()[:6000])
    failed = set(runner.run_tests(ordered, nodeids, timeouts=[test_timeout(t, ms) for t in ordered],
                                  replan=replan if REPLAN else None, should_stop=should_stop,
                                  skipped=skipped))
    PROF["marks"].append(("ran", round(time.process_time(), 2)))
    if skipped:
        unrun.extend(skipped)
        note += f" early-stop({len(skipped)}/{rp['n']})"
    if j["overlay"] and j.get("unknown") and len(failed) >= BROAD_FAILS:
        # an import-time change that breaks many tests: its reach is beyond what
        # the coverage map can tell, so run nothing more and select everything
        return {"failed": sorted(failed), "unrun": [], "all": True, "note": note + " broad-import-break"}
    if prepared is True:
        hits, bad = dict(runner.DIFF_HITS), set(runner.DIFF_BAD)
        n_sites = j["n_sites"]
        note += f" diff hits={hits} bad={sorted(bad)}"
        n_ran = len(ordered) - len(skipped)
        # equivalence is only trusted when most candidates exercised the change
        if unrun and not failed and hits and not bad and all(hits.get(k, 0) > 0 for k in range(n_sites)) \
                and (n_ran >= 0.5 * (n_ran + len(unrun)) or (n_ran >= 8 and all(hits.get(k, 0) >= 30 for k in range(n_sites)))):
            note += " EQUIV->drop heavy"
            unrun = []
    _touches_exec = any(
        p.startswith("sqlglot/executor/") or p == "tests/test_executor.py" or p.endswith(".csv.gz")
        for p in j["files"]
    )
    if unrun and TPCDS_KIN and not _touches_exec and not runner.TIMEOUTS[0]:
        tp = [t for t in unrun if _is_tpcds(nodeids[t])]
        if tp:
            pos_1440 = _nodeid_pos(nodeids).get("tests/test_optimizer.py::TestOptimizer::test_tpcds")
            if pos_1440 is not None and pos_1440 in unrun and not failed:
                f1440 = runner.run_tests([pos_1440], nodeids, timeouts=[test_timeout(pos_1440, ms)])
                if f1440:
                    failed.update(f1440)
                    unrun.remove(pos_1440)
                else:
                    unrun.remove(pos_1440)
                    note += " tpcds_opt-pass"
            t_1440_bad = pos_1440 is not None and (pos_1440 in failed or pos_1440 in unrun)
            if not failed and not t_1440_bad:
                unrun = [t for t in unrun if t not in tp]
                note += " tpcds-settled"
        tph = [t for t in unrun if nodeids[t].endswith("::TestExecutor::test_execute_tpch")
               or nodeids[t].endswith("::TestExecutor::test_optimized_tpch")]
        if tph:
            pos_1441 = _nodeid_pos(nodeids).get("tests/test_optimizer.py::TestOptimizer::test_tpch")
            if pos_1441 is not None and pos_1441 in unrun and not failed:
                f1441 = runner.run_tests([pos_1441], nodeids, timeouts=[test_timeout(pos_1441, ms)])
                if f1441:
                    failed.update(f1441)
                    unrun.remove(pos_1441)
                else:
                    unrun.remove(pos_1441)
                    note += " tpch_opt-pass"
            t_1441_bad = pos_1441 is not None and (pos_1441 in failed or pos_1441 in unrun)
            if not failed and not t_1441_bad:
                unrun = [t for t in unrun if t not in tph]
                note += " tpch-settled"
    # second phase: the light tests are done; a heavy test left unrun is run after
    # all when the schedule time it would free outweighs its cpu
    if unrun and not (j["overlay"] and j.get("unknown")):
        budget = min(j.get("heavy_budget", 1e9), MAX_CHILD_CPU)
        mult = OV_MULT if j["overlay"] else 1.0
        ran_set = ordered_set - set(skipped)
        if REPLAY_HIT:
            n_or = sum(1 for t in ran_set if t not in REPLAY_HIT)
            n_of = sum(1 for t in failed if t not in REPLAY_HIT)
            if not n_of:
                p_pass_other = P2_PASS
            else:
                emp_pass = (max(0, n_or - n_of) + 2.0) / (max(1, n_or) + 4.0)
                p_pass_other = max(P2_PASS_FAILED, min(P2_PASS, emp_pass))
            n_hr = len(ran_set) - n_or
            n_hf = len(failed) - n_of
            p_pass_hit = max(0.15, (max(0, n_hr - n_hf) + 0.5) / (n_hr + 2.5))
        else:
            n_or = len(ran_set)
            n_of = len(failed)
            if not failed:
                p_pass_other = P2_PASS
            else:
                emp_pass = (max(0, n_or - n_of) + 2.0) / (max(1, n_or) + 4.0)
                p_pass_other = max(P2_PASS_FAILED, min(P2_PASS, emp_pass))
            p_pass_hit = p_pass_other
        # With no confirmed failure, clearing the entire tail also avoids the
        # first worker charge. Include short tests in that joint decision.
        all_unconfirmed = not failed and not FORCED
        margin = 0.80 if all_unconfirmed else LATE_MARGIN
        late = []
        big = [t for t in unrun if all_unconfirmed or ms[t] >= LATE_MIN_MS]
        pool_late = []
        for t in big:
            p_pass = p_pass_hit if t in REPLAY_HIT else p_pass_other
            c = cpu_of(t, ms) * mult
            if c <= budget and (ms[t] / 1000.0 + 2.0) * p_pass > margin * c + 0.08:
                pool_late.append(t)
        while pool_late and not runner.TIMEOUTS[0]:
            cur = set(unrun) | failed | FORCED
            cur_cost = schedule_cost(cur, ms)[1]
            efficient = sorted(pool_late, key=lambda t: cpu_of(t, ms) / max(1, ms[t]))[:12]
            longest = sorted(pool_late, key=lambda t: -ms[t])[:12]
            pick = None
            best_net = 0.0
            for order in (efficient, longest):
                rem = set()
                acc_c = acc_w = acc_wp = 0.0
                for t in order:
                    acc_c += cpu_of(t, ms) * mult
                    if acc_c > budget:
                        break
                    rem.add(t)
                    weight = max(1, ms[t])
                    acc_w += weight
                    acc_wp += weight * (p_pass_hit if t in REPLAY_HIT else p_pass_other)
                    gain = (cur_cost - schedule_cost(cur - rem, ms)[1]) / 1000.0
                    net = gain * acc_wp / acc_w - margin * acc_c - 0.08
                    if net > best_net:
                        best_net = net
                        pick = order[0]
            if pick is None:
                break
            pool_late.remove(pick)
            c = cpu_of(pick, ms) * mult
            f = runner.run_tests([pick], nodeids, timeouts=[test_timeout(pick, ms)])
            budget -= c
            unrun.remove(pick)
            failed |= set(f)
            late.append(pick)
            if f:
                margin = LATE_MARGIN
                if not REPLAY_HIT or pick not in REPLAY_HIT:
                    n_or += 1
                    n_of += 1
                    p_pass_other = max(P2_PASS_FAILED, min(P2_PASS, (max(0, n_or - n_of) + 2.0) / (max(1, n_or) + 4.0)))
                    if not REPLAY_HIT:
                        p_pass_hit = p_pass_other
        if late:
            note += f" late-run{late}"
    if any(runner.MEMO_STAT):
        note += f" memo={runner.MEMO_STAT} memoT={[round(x, 2) for x in runner.MEMO_T]}"
    if any(optmemo.STAT):
        note += f" opt={optmemo.STAT}/{optmemo.AFFECTED} optT={[round(x, 2) for x in optmemo.TIMES]}"
    return {"failed": sorted(failed), "unrun": unrun, "note": note}


if __name__ == "__main__":
    if os.environ.get("BR_SUPERVISE", "1") == "1":
        supervise()
    else:
        main()