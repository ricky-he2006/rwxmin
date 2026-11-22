"""Run the public sample under the Makefile's Docker limits."""
import statistics
import subprocess
import sys
from pathlib import Path

sample, output, *docker_args = sys.argv[1:]
# Prefer the container CPU controller, which also counts orphaned workers.
measure = """
import os, sys
from pathlib import Path

def container_usage():
    root = Path("/sys/fs/cgroup")
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            directory = root / line[3:].lstrip("/")
            try:
                if os.getpid() not in map(int, (directory / "cgroup.procs").read_text().split()):
                    return None
                stats = dict(row.split() for row in (directory / "cpu.stat").read_text().splitlines())
                return int(stats["usage_usec"])
            except (OSError, KeyError, ValueError):
                return None
    return None

before = container_usage()
pid = os.posix_spawnp("timeout", ["timeout", "-s", "KILL", "180", "python", "/app/selector.py"], os.environ)
_, status, usage = os.wait4(pid, 0)
after = container_usage()
cpu = usage.ru_utime + usage.ru_stime
if before is not None and after is not None:
    cpu = (after - before) / 1_000_000
print("selector_cpu", cpu, file=sys.stderr)
sys.exit(os.waitstatus_to_exitcode(status))
"""
totals = []
for run in range(1, 4):
    with Path(output).open("wb") as predictions:
        result = subprocess.run(
            ["docker", "run", *docker_args, "python", "-c", measure],
            stdout=predictions, stderr=subprocess.PIPE, text=True,
        )
    if result.returncode:
        sys.exit(result.stderr or f"Docker exited with {result.returncode}")
    usage = next(line.split()[1] for line in reversed(result.stderr.splitlines())
                 if line.startswith("selector_cpu "))
    report = subprocess.check_output(
        [sys.executable, "score.py", sample, str(Path(sample) / "answers.json"),
         output, "--cpu-seconds", usage], text=True,
    )
    print(f"Run {run}\n{report}", flush=True)
    totals.append(float(next(line.split()[1] for line in report.splitlines()
                             if line.startswith("Total:"))))
print(f"Median total {statistics.median(totals):.1f} s")
