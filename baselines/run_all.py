"""Run every (dataset, method, seed) job that has no scores yet, N at a time; logs in results/<WGC_RUN>/logs.
Re-running it resumes: finished jobs (all three score files present) and jobs still running are skipped.
A file results/<WGC_RUN>/STOP makes it start no new jobs.
  python baselines/run_all.py [--seeds 0,1,2] [--ds hs2,hanna,ell,merlin] [--methods v0,fewshot,...] [--par 10]"""
import argparse, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--ds", default="hs2,hanna,ell,merlin")
ap.add_argument("--methods", default="")  # comma-separated subset of METHODS (default: all)
ap.add_argument("--par", type=int, default=10); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--skip", default="")  # job names already running elsewhere
a = ap.parse_args()
P = sys.executable
B = f"{gem.ROOT}/baselines"
METHODS = {  # method dir -> command (seed-independent methods run for seed 0 only)
    "v0": ("run_simple.py", ["v0"], False), "fewshot": ("run_simple.py", ["fewshot"], True),
    "dspy_zeroshot": ("run_dspy.py", ["zeroshot"], False),
    "opro": ("run_opro.py", [], True),  # ProTeGi dropped (too slow with its sequential official loop)
    "textgrad_ccc": ("run_textgrad.py", ["ccc"], True), "textgrad_single": ("run_textgrad.py", ["single"], True),
    "dspy_mipro": ("run_dspy.py", ["mipro"], True), "dspy_gepa": ("run_dspy.py", ["gepa"], True)}
def running_cmds():
    out = []
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                out.append(open(f"/proc/{pid}/cmdline").read().replace("\0", " "))
            except OSError:
                pass
    return out


RUNNING = running_cmds()
jobs = []
for d in a.ds.split(","):
    ds = gem.DS(d)
    for m, (script, args, seeded) in METHODS.items():
        if a.methods and m not in a.methods.split(","):
            continue
        for s in (map(int, a.seeds.split(",")) if seeded else [0]):
            cmd = [P, f"{B}/{script}", d] + args + ["--seed", str(s), "--workers", str(a.workers)]
            busy = any(f"{script} {' '.join([d] + args)} --seed {s} " in c for c in RUNNING)  # started by an earlier scheduler
            if not gem.have_scores(ds, m, s) and not busy and f"{d}_{m}_{s}" not in a.skip.split(","):
                jobs.append((f"{d}_{m}_{s}", cmd))
# longest first: textgrad_single is sequential in the optimizer calls
jobs.sort(key=lambda j: 0 if "single" in j[0] else 1)
print(f"{len(jobs)} jobs", flush=True)
running = []
while jobs or running:
    running = [(n, p) for n, p in running if p.poll() is None]
    while jobs and len(running) < a.par:
        n, cmd = jobs.pop(0)
        running.append((n, subprocess.Popen(cmd, stdout=open(f"{gem.OUT}/logs/{n}.log", "w"), stderr=subprocess.STDOUT)))
        print(time.strftime("%H:%M"), "start", n, flush=True)
    if os.path.exists(f"{gem.OUT}/STOP"):
        print("STOP file found; not starting new jobs", flush=True); jobs = []
    time.sleep(20)
print("ALL_JOBS_DONE", flush=True)
