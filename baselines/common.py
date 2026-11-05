"""Shared harness. Judge and optimizer = the llm package's default provider, through its public API only:
  batch_result(None, [messages, ...], temperature=0)   judging many items (order preserved; the llm package
                                                        parallelizes or uses the provider's batch endpoint)
  chat_result(None, messages, temperature=t)          single calls of the optimizers (OPRO, TextGrad, DSPy)
Every LLMResult goes into one UsageTracker per process; save_usage() writes its totals and cost next to the run's
scores. Model, parameters, retries and prices are all the llm package's.

Every method (v0, few-shot, OPRO, TextGrad CCC / Single-Task, DSPy zero-shot / MIPROv2 / GEPA) writes, per dataset and
seed, the judge's scores on fit / sel / test to results/<WGC_RUN>/<ds>/<method>/seed<s>/scores_<split>.csv (WGC_RUN
defaults to "gpt"); evaluate.py then reports raw and readout test Spearman for all of them in the same way.
"""
import json, os, re, sys, threading, time

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # the llm/ package lives at the repository root
OUT = f"{ROOT}/results/{os.environ.get('WGC_RUN', 'gpt')}"
DATASETS = ("hs2", "hanna", "ell", "merlin")
os.makedirs(f"{OUT}/logs", exist_ok=True)


class Abort(Exception):
    pass


from llm import ProviderError, UsageTracker, batch_result, chat_result  # noqa: E402

TRACKER = UsageTracker()  # every call of this process (one run per process)
_TOK = dict(calls=0, uncounted=0, input_tokens=0, output_tokens=0, cache_read_tokens=0, reasoning_tokens=0)
_LOCK = threading.Lock()


def _track(res):
    counted = TRACKER.add(res)
    with _LOCK:
        _TOK["calls"] += 1
        if not counted or res.usage is None:
            _TOK["uncounted"] += 1  # the provider reported no usage: visible, not counted as free
        else:
            for k in ("input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens"):
                _TOK[k] += getattr(res.usage, k, 0) or 0
    return res


def chat(messages, temperature=0.0):
    """One completion -> text (chat_result; its usage goes into TRACKER)."""
    return _track(chat_result(None, messages, temperature=temperature)).content or ""


def chat_batch(messages_list, temperature=0.0):
    """Many completions -> texts in the order given (batch_result; usage goes into TRACKER)."""
    return [_track(r).content or "" for r in batch_result(None, list(messages_list), temperature=temperature)]


def save_usage(outdir):
    """Write this process's token totals and the UsageTracker's own totals / cost to <outdir>/usage.json."""
    t = TRACKER.summary() if callable(getattr(TRACKER, "summary", None)) else {
        k: getattr(TRACKER, k) for k in dir(TRACKER) if not k.startswith("_") and
        isinstance(getattr(TRACKER, k, None), (int, float))}
    with _LOCK:
        info = dict(tokens=dict(_TOK), usage_tracker=t)
    json.dump(info, open(f"{outdir}/usage.json", "w"), indent=1, default=str)
    print("usage:", json.dumps(info, default=str), flush=True)


def parse_scores(txt):
    for js in reversed(re.findall(r"\{[^{}]*\}", txt)):
        try:
            return {k: float(v) for k, v in json.loads(js).items() if isinstance(v, (int, float))}
        except Exception:  # noqa: BLE001
            continue
    return {}


class DS:
    """One dataset (data/<name>/): attributes, output format, initial prompt v0, gold scores and the fit / sel / test
    items (one JSON object per line: {"id", "user"}, "user" being the text the judge rates)."""

    def __init__(self, name):
        self.name, self.R = name, f"{ROOT}/data/{name}"
        self.A = open(f"{self.R}/attrs.txt").read().split()
        self.FORMAT = open(f"{self.R}/format.txt").read().strip()
        self.LO, self.HI = map(int, re.search(r"integer from (\d+) to (\d+)", self.FORMAT).groups())
        self.v0 = open(f"{self.R}/v0.txt").read().strip()
        self.gold = pd.read_csv(f"{self.R}/gold.csv").set_index("id")
        self.rows = {s: [json.loads(l) for l in open(f"{self.R}/{s}.jsonl") if l.strip()] for s in ("fit", "sel", "test")}
        self.text = {r["id"]: r["user"] for s in self.rows for r in self.rows[s]}

    def out(self, method, seed=0):
        d = f"{OUT}/{self.name}/{method}/seed{seed}"
        os.makedirs(d, exist_ok=True)
        return d

    def subset(self, split, n, seed=0):
        rows = self.rows[split]
        return [rows[i] for i in sorted(np.random.default_rng(seed).permutation(len(rows))[:n])] if n < len(rows) else rows

    def human(self, ids):
        return self.gold.loc[list(ids), self.A].to_numpy(float)

    # ---------------------------------------------------------------- judging
    def judge(self, instruction, rows, workers=16, strict=True):
        """Judge rows with system = instruction + output format (the setting of all prompt-text methods).
        Returns a DataFrame id + one column per attribute (NaN where unparsed)."""
        system = instruction.strip() + "\n\n" + self.FORMAT
        return self.judge_msgs(lambda r: [dict(role="system", content=system), dict(role="user", content=r["user"])], rows,
                               workers, strict)

    def judge_msgs(self, make, rows, workers=None, strict=True):
        """Judge rows (make(row) -> messages) with one batch_result call. strict: stop if >20% of outputs are unparsed
        (outage / refusal). Optimizers scoring candidate prompts pass strict=False: a candidate that breaks the output
        format just scores badly (see worst_case). `workers` is unused (the llm package bounds the parallelism)."""
        t0 = time.time()
        try:
            texts, err = chat_batch([make(r) for r in rows]), None
        except ProviderError as e:  # the whole batch failed
            if strict:
                raise
            texts, err = [""] * len(rows), str(e)[:200]
        recs = []
        for r, txt in zip(rows, texts):
            sc = parse_scores(txt)
            recs.append(dict(id=r["id"], **{a: sc.get(a, np.nan) for a in self.A}, reason=(err or txt[-300:])))
        df = pd.DataFrame(recs)
        for a in self.A:
            if a not in df:
                df[a] = np.nan
        bad = df[self.A].isna().any(axis=1).mean()
        print(f"  [{time.strftime('%H:%M:%S')}] judged {len(rows)} items in {time.time() - t0:.0f}s, "
              f"{1 - bad:.0%} parsed" + (f" (batch failed: {err})" if err else ""), flush=True)
        if strict and bad > 0.2:  # outage / refusal / moderation: stop rather than optimize on garbage
            raise Abort(f"{self.name}: {bad:.0%} of {len(df)} judge calls unparsed, e.g. "
                        f"{df['reason'].astype(str).iloc[0][:200] if len(df) else ''}")
        return df

    def worst_case(self, df):
        """Unparsed ratings of a candidate -> the scale midpoint (no rank information, maximal-ish error)."""
        df = df.copy()
        df[self.A] = df[self.A].fillna((self.LO + self.HI) / 2)
        return df

    # ---------------------------------------------------------------- metrics
    def spearman(self, df, attrs=None):
        """Per-attribute Spearman of judge scores (df: id + attribute columns) with the human scores."""
        attrs = attrs or self.A
        H = self.gold.loc[df.id, attrs]
        return np.array([pd.Series(df[a].to_numpy(float)).corr(pd.Series(H[a].to_numpy(float)), method="spearman")
                         for a in attrs])

    def closeness(self, pred, ids, attrs=None):
        """Per-item agreement 1 - mean |pred - human| / range, for optimizers that need a per-example metric."""
        attrs = attrs or self.A
        P = np.asarray(pred, float).reshape(len(ids), len(attrs))
        H = self.gold.loc[list(ids), attrs].to_numpy(float)
        return 1 - np.nanmean(np.abs(P - H), axis=1) / (self.HI - self.LO)


def save_scores(ds, method, seed, split, df):
    df.to_csv(f"{ds.out(method, seed)}/scores_{split}.csv", index=False)


def have_scores(ds, method, seed):
    d = f"{OUT}/{ds.name}/{method}/seed{seed}"
    return all(os.path.exists(f"{d}/scores_{s}.csv") for s in ("fit", "sel", "test"))


def judge_all_splits(ds, method, seed, instruction, workers=16):
    """Final evaluation of a single-instruction judge: score fit, sel and test, save the scores and the prompt."""
    for s in ("fit", "sel", "test"):
        print(f"final evaluation: {s} ({len(ds.rows[s])} items)", flush=True)
        save_scores(ds, method, seed, s, ds.judge(instruction, ds.rows[s], workers))
    open(f"{ds.out(method, seed)}/final_prompt.txt", "w").write(instruction)
    save_usage(ds.out(method, seed))
