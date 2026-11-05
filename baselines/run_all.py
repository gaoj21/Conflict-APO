"""Run every (dataset, method, seed) job that has no scores yet, N at a time; logs in results/<WGC_RUN>/logs.
Re-running it resumes: finished jobs (all three score files present) and jobs still running are skipped.
A file results/<WGC_RUN>/STOP makes it start no new jobs.
Prints progress every --every seconds: jobs done / running / queued / failed, and each running job's last log line.
  python baselines/run_all.py [--seeds 0,1,2] [--ds hs2,hanna,ell,merlin] [--methods v0,fewshot,...] [--par 10]"""
import argparse, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--ds", default="hs2,hanna,ell,merlin")
ap.add_argument("--methods", default="")  # comma-separated subset of METHODS (default: all)
ap.add_argument("--par", type=int, default=10); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--every", type=int, default=60)  # seconds between progress reports
ap.add_argument("--skip", default="")  # job names already running elsewhere
a = ap.parse_args()
P = sys.executable
B = f"{common.ROOT}/baselines"
METHODS = {  # method dir -> command (seed-independent methods run for seed 0 only)
    "v0": ("run_simple.py", ["v0"], False), "fewshot": ("run_simple.py", ["fewshot"], True),
    "dspy_zeroshot": ("run_dspy.py", ["zeroshot"], False),
    "opro": ("run_opro.py", [], True),  # ProTeGi dropped (too slow with its sequential official loop)
    "textgrad_ccc": ("run_textgrad.py", ["ccc"], True), "textgrad_single": ("run_textgrad.py", ["single"], True),
    "dspy_mipro": ("run_dspy.py", ["mipro"], True), "dspy_gepa": ("run_dspy.py", ["gepa"], True)}
def running_cmds():
    """Command lines of all running processes (ps works on macOS and Linux; -ww: never truncate)."""
    try:
        return subprocess.run(["ps", "-axww", "-o", "command"], capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []


RUNNING = running_cmds()
jobs = []
for d in a.ds.split(","):
    ds = common.DS(d)
    for m, (script, args, seeded) in METHODS.items():
        if a.methods and m not in a.methods.split(","):
            continue
        for s in (map(int, a.seeds.split(",")) if seeded else [0]):
            cmd = [P, f"{B}/{script}", d] + args + ["--seed", str(s), "--workers", str(a.workers)]
            busy = any(f"{script} {' '.join([d] + args)} --seed {s} " in c for c in RUNNING)  # started by an earlier scheduler
            if not common.have_scores(ds, m, s) and not busy and f"{d}_{m}_{s}" not in a.skip.split(","):
                jobs.append((f"{d}_{m}_{s}", cmd))
# longest first: textgrad_single is sequential in the optimizer calls
jobs.sort(key=lambda j: 0 if "single" in j[0] else 1)
print(f"{len(jobs)} jobs: " + ", ".join(n for n, _ in jobs), flush=True)


def last_line(n):
    try:
        with open(f"{common.OUT}/logs/{n}.log", "rb") as f:
            f.seek(0, 2); f.seek(max(f.tell() - 4000, 0))
            lines = [l for l in f.read().decode(errors="replace").splitlines() if l.strip()]
        return lines[-1].strip()[:110] if lines else ""
    except OSError:
        return ""


def fmt(sec):
    return f"{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m"


total, done, failed, running, t0, last = len(jobs), [], [], [], time.time(), 0.0
while jobs or running:
    still = []
    for n, p, ts in running:
        if p.poll() is None:
            still.append((n, p, ts))
        elif p.returncode == 0:
            done.append(n); print(time.strftime("%H:%M"), f"done   {n} ({fmt(time.time() - ts)})", flush=True)
        else:
            failed.append(n); print(time.strftime("%H:%M"), f"FAILED {n} (exit {p.returncode}): {last_line(n)}", flush=True)
    running = still
    if os.path.exists(f"{common.OUT}/STOP") and jobs:
        print("STOP file found; not starting new jobs", flush=True); jobs = []
    while jobs and len(running) < a.par:
        n, cmd = jobs.pop(0)
        running.append((n, subprocess.Popen(cmd, stdout=open(f"{common.OUT}/logs/{n}.log", "w"), stderr=subprocess.STDOUT),
                        time.time()))
        print(time.strftime("%H:%M"), "start ", n, flush=True)
    if time.time() - last >= a.every and running:
        last = time.time()
        print(f"--- {time.strftime('%H:%M')}  {len(done)}/{total} done, {len(running)} running, {len(jobs)} queued, "
              f"{len(failed)} failed  (elapsed {fmt(time.time() - t0)})", flush=True)
        for n, p, ts in running:
            print(f"    {n:<28} {fmt(time.time() - ts)}  {last_line(n)}", flush=True)
    time.sleep(5)
print(f"ALL_JOBS_DONE: {len(done)} done, {len(failed)} failed" + (f" ({', '.join(failed)}; rerun to retry)" if failed else ""),
      flush=True)
