"""Shared harness: judge and optimizer = the model behind the machine's `llm` package (llm-v12 transport contract).

Every model call goes through `llm.chat_result` (or `llm.batch_result` for judge evaluation with WGC_BATCH=1); this
file never names a provider, an endpoint or a key. Which model answers is decided by `llm/providers.json` (its
`default`) or by LLM_PROVIDER / EAX_PROVIDER in the environment, as the contract specifies.

Every method (v0, few-shot, OPRO, TextGrad CCC / Single-Task, DSPy zero-shot / MIPROv2 / GEPA) writes, per dataset and
seed, the judge's scores on fit / sel / test to results/<run>/<ds>/<method>/seed<s>/scores_<split>.csv; evaluate.py
then reports raw and readout test Spearman for all of them in the same way.

Judge calls (temperature 0) are cached in one sqlite file per run; every call that reached the model is logged with
its tokens (NULL when the provider reported no usage) and its cost from llm.UsageTracker (prices as configured in llm).

Environment:
  WGC_RUN       name of the run = results/<WGC_RUN>/ (default: the llm default provider's name)
  WGC_BUDGET    stop every job once the run's cost (llm.UsageTracker) exceeds this many $ (default 300)
  WGC_BATCH     1 = final judge evaluations use llm.batch_result (the provider's batch endpoint when it has one)
  WGC_MAX_FAIL  consecutive failed calls (all threads together) after which everything stops (default 30)

Only the temperature is passed (judge 0; each optimizer at its official temperature); every other parameter
(max_tokens, reasoning, ...) is set in the llm package (providers.json `params`). Temperature-0 calls are cached.
"""
import hashlib, json, os, re, sqlite3, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # the llm/ package lives at the repository root
try:
    from llm import chat_result, batch_result, UsageTracker, default_provider  # noqa: E402
except ImportError as _e:  # evaluate.py needs no model; any call without the package fails loudly
    _MISSING = f"cannot import the llm/ package at {ROOT} ({_e}); see README"

    def chat_result(*a, **k):
        raise Abort(_MISSING)

    batch_result = chat_result

    class UsageTracker:  # noqa: D101
        pass

    def default_provider():
        return "run"

RUN = os.environ.get("WGC_RUN") or default_provider() or "run"  # results/<default provider> unless named
OUT = f"{ROOT}/results/{RUN}"
JUDGE = RUN  # model labels only (cache keys and cost log): the llm package decides where a call goes
OPTIM = JUDGE  # the optimizer is the judge model itself (as in ProTeGi and GEPA)
DATASETS = ("hs2", "hanna", "ell", "merlin")
BUDGET = float(os.environ.get("WGC_BUDGET", "300"))  # a safety stop, not part of any method
MAX_FAIL = int(os.environ.get("WGC_MAX_FAIL", "30"))
USE_BATCH = os.environ.get("WGC_BATCH", "0") == "1"
# errors a retry cannot fix (bad / disabled key, no credit, moderation, misconfiguration): stop everything at once
FATAL = re.compile(r"\b(401|402|403)\b|unauthori[sz]ed|forbidden|token_disabled|disabled|banned|insufficient|quota|"
                   r"missing|unknown provider|not configured|invalid api key|api key|绕过|封", re.I)
os.makedirs(f"{OUT}/logs", exist_ok=True)

_LOCK = threading.Lock()
_DB = None
_FAILS = [0]  # consecutive failed calls across all threads
_STOP = []  # set once a job must stop: every later call raises at once


class _Tag:
    v = ""


TAG = _Tag()  # which method/run the calls belong to (for the cost log); one run per process


class Abort(Exception):
    pass


def _db():
    global _DB
    if _DB is None:
        _DB = sqlite3.connect(f"{OUT}/cache.sqlite", check_same_thread=False, timeout=60)
        _DB.execute("create table if not exists cache (k text primary key, txt text, in_tok int, out_tok int, cost real)")
        _DB.execute("create table if not exists spend (ts real, tag text, model text, in_tok int, out_tok int, cost real)")
        _DB.commit()
    return _DB


def spent(tag=None):
    """(cost $, calls, input tokens, output tokens) logged so far, for the whole run or one tag prefix."""
    with _LOCK:
        q = "select coalesce(sum(cost),0), count(*), coalesce(sum(in_tok),0), coalesce(sum(out_tok),0) from spend"
        return _db().execute(q + (" where tag like ?" if tag else ""), ((tag + "%",) if tag else ())).fetchone()


def _key(model, temperature, max_tokens, messages):
    return hashlib.sha256(json.dumps([model, temperature, max_tokens, messages], ensure_ascii=False).encode()).hexdigest()


def _usage(res):
    """LLMResult -> (in_tok, out_tok, cost) priced by llm.UsageTracker (prices as configured in llm);
    (None, None, 0.0) when the provider reported no usage."""
    t = UsageTracker()
    if not t.add(res):
        return None, None, 0.0
    return res.usage.input_tokens, res.usage.output_tokens, t.cost


def _record(model, k, cache, txt, it, ot, c):
    with _LOCK:
        db = _db()
        db.execute("insert into spend values (?,?,?,?,?,?)", (time.time(), TAG.v, model, it, ot, c))
        if cache:
            db.execute("insert or replace into cache values (?,?,?,?,?)", (k, txt, it, ot, c))
        db.commit()
        _FAILS[0] = 0


def _failed(err):
    with _LOCK:
        _FAILS[0] += 1
        n = _FAILS[0]
    if FATAL.search(err):
        _STOP.append(f"non-retryable provider error: {err}")
    elif n >= MAX_FAIL:
        _STOP.append(f"{n} consecutive failed calls, last: {err}")
    if _STOP:
        raise Abort(_STOP[0])


def _cached(k):
    with _LOCK:
        return _db().execute("select txt, in_tok, out_tok, cost from cache where k=?", (k,)).fetchone()


def chat(messages, model=JUDGE, temperature=0.0, max_tokens=2000, cache=None):
    """One completion -> (text, in_tok, out_tok, cost). Cached when temperature is 0 (or cache=True).
    `model` is only a label: the llm package routes the call."""
    cache = (temperature == 0) if cache is None else cache
    k = _key(model, temperature, max_tokens, messages)
    if cache and (row := _cached(k)):
        return row[0], row[1], row[2], 0.0
    if _STOP:
        raise Abort(_STOP[0])
    if spent()[0] > BUDGET:
        raise Abort(f"budget ${BUDGET} reached")
    err = ""
    for t in range(4):  # the llm package already retries rate limits and 5xx twice; this covers longer outages
        try:
            res = chat_result(None, messages, temperature=temperature)  # other parameters: the llm package
            txt = res.content or ""
            it, ot, c = _usage(res)
            _record(model, k, cache, txt, it, ot, c)
            return txt, it, ot, c
        except Abort:
            raise
        except Exception as e:  # noqa: BLE001  ProviderError and anything the transport raised
            err = f"{type(e).__name__}: {str(e)[:300]}"
            _failed(err)
        time.sleep(15 * (t + 1))
    raise RuntimeError(err)


def chat_many(messages_list, model=JUDGE, max_tokens=2000, chunk=500):
    """Temperature-0 completions of many message lists through llm.batch_result (native batch endpoint when the
    provider has one), skipping cached ones. Returns texts in order; a failed chunk yields "" for its items."""
    keys = [_key(model, 0.0, max_tokens, m) for m in messages_list]
    out = [None] * len(keys)
    todo = []
    for i, k in enumerate(keys):
        row = _cached(k)
        if row:
            out[i] = row[0]
        else:
            todo.append(i)
    for j in range(0, len(todo), chunk):
        if spent()[0] > BUDGET:
            raise Abort(f"budget ${BUDGET} reached")
        idx = todo[j:j + chunk]
        try:
            results = batch_result(None, [messages_list[i] for i in idx], temperature=0.0)
        except Exception as e:  # noqa: BLE001
            _failed(f"{type(e).__name__}: {str(e)[:300]}")
            for i in idx:
                out[i] = ""
            continue
        for i, res in zip(idx, results):
            it, ot, c = _usage(res)
            out[i] = res.content or ""
            _record(model, keys[i], True, out[i], it, ot, c)
        print(f"  batch {j + len(idx)}/{len(todo)} done", flush=True)
    return out


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

    def judge_msgs(self, make, rows, workers=16, strict=True, batch=False):
        """strict: stop if >20% of outputs are unparsed (outage / refusal). Optimizers scoring candidate prompts pass
        strict=False: a candidate that breaks the output format just scores badly (see worst_case).
        batch: send the uncached calls through llm.batch_result (final evaluations with WGC_BATCH=1)."""
        def row_of(r, txt, it=None, ot=None):
            sc = parse_scores(txt)
            return dict(id=r["id"], **{a: sc.get(a, np.nan) for a in self.A}, in_tok=it, out_tok=ot, reason=txt[-300:])

        def one(r):
            try:
                txt, it, ot, c = chat(make(r))
            except Abort:
                raise
            except Exception as e:  # noqa: BLE001
                return dict(id=r["id"], error=str(e)[:200])
            return row_of(r, txt, it, ot)

        if batch:
            df = pd.DataFrame([row_of(r, t) for r, t in zip(rows, chat_many([make(r) for r in rows]))])
        else:
            with ThreadPoolExecutor(workers) as ex:
                df = pd.DataFrame(list(ex.map(one, rows)))
        for a in self.A:
            if a not in df:
                df[a] = np.nan
        bad = df[self.A].isna().any(axis=1).mean()
        if strict and bad > 0.2:  # outage / refusal / moderation: stop rather than optimize on garbage
            raise Abort(f"{self.name}: {bad:.0%} of {len(df)} judge calls unparsed, e.g. "
                        f"{df.get('error', df.get('reason')).dropna().astype(str).iloc[0][:200] if len(df) else ''}")
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
    system = instruction.strip() + "\n\n" + ds.FORMAT
    for s in ("fit", "sel", "test"):
        save_scores(ds, method, seed, s, ds.judge_msgs(
            lambda r: [dict(role="system", content=system), dict(role="user", content=r["user"])], ds.rows[s], workers,
            batch=USE_BATCH))
    open(f"{ds.out(method, seed)}/final_prompt.txt", "w").write(instruction)
