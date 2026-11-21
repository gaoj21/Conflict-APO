"""Shared harness for the Gemma experiments (judge and optimizer = Gemma 4 26B A4B via OpenRouter).

Every method (v0, few-shot, OPRO, ProTeGi, TextGrad CCC / Single-Task, DSPy MIPROv2 / GEPA) writes, per dataset and
seed, the judge's scores on fit / sel / test to results/gemma/<ds>/<method>/seed<s>/scores_<split>.csv; evaluate.py
then reports raw and readout test Spearman for all of them in the same way.

Judge calls (temperature 0) are cached in one sqlite file shared by all methods; every paid call (judge or optimizer)
is logged with its tokens and cost. The key is read from ~/.openrouter_key and never printed.
"""
import hashlib, json, os, re, sqlite3, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# GEM_PROVIDER=juapi GEM_MODEL=<gpt model> runs everything on a GPT model through juapi (streaming; key ~/.juapi_key)
PROVIDER = os.environ.get("GEM_PROVIDER", "openrouter")
# GEM_PROVIDER=lanyi GEM_MODEL=<model> runs everything through lanyiapi (OpenAI protocol; key ~/.lanyiapi_key)
if PROVIDER == "juapi":
    JUDGE = os.environ.get("GEM_MODEL", "gpt-6-luna")
    OUT = f"{ROOT}/results/{JUDGE}"
elif PROVIDER == "lanyi":
    JUDGE = os.environ.get("GEM_MODEL", "gpt-6-luna")
    OUT = f"{ROOT}/results/lanyi_{JUDGE}"
elif PROVIDER == "llm":
    # GEM_PROVIDER=llm: every call goes through the project's `llm` package (llm-v12; `from llm import chat_result`),
    # found at the repository root or under GEM_LLM_PATH. Which model and endpoint are used is decided by
    # llm/providers.json; GEM_LLM_PROVIDER picks a provider by name (default: the configured default provider).
    import sys as _sys
    _cands = [os.environ.get("GEM_LLM_PATH"), ROOT, os.path.dirname(ROOT)]  # this package, then the repo above it
    _where = next((c for c in _cands if c and os.path.isdir(os.path.join(c, "llm"))), None)
    if _where is None:
        raise ImportError("GEM_PROVIDER=llm but no `llm` package found in " + ", ".join(c for c in _cands if c)
                          + ". Copy your llm package there (cp -r /path/to/llm .) or set GEM_LLM_PATH to the folder that contains it.")
    _sys.path.insert(0, _where)
    from llm import chat_result as _llm_chat_result, ProviderError as _LLMError  # noqa: E402
    LLM_PROVIDER = os.environ.get("GEM_LLM_PROVIDER") or None
    JUDGE = os.environ.get("GEM_MODEL", LLM_PROVIDER or "llm-default")  # a label only (cache key, logs)
    # the package call has no timeout of its own: a stuck connection would block a worker (and the run) forever, so each
    # call runs in its own thread and is abandoned after GEM_TIMEOUT seconds (then retried like any transient error)
    LLM_TIMEOUT = float(os.environ.get("GEM_TIMEOUT", "180"))
    from concurrent.futures import TimeoutError as _FutTimeout  # noqa: E402
    _LLM_POOL = ThreadPoolExecutor(int(os.environ.get("GEM_LLM_THREADS", "128")), thread_name_prefix="llm")
    _N_CALLS, _N_TIMEOUTS, _T0 = [0], [0], time.time()

    def _heartbeat():  # one progress line a minute on stderr, so a long step is visibly alive (or visibly stuck)
        while True:
            time.sleep(60)
            print(f"  [llm] {_N_CALLS[0]} calls in {(time.time() - _T0) / 60:.0f} min, {_N_TIMEOUTS[0]} timed out",
                  file=__import__("sys").stderr, flush=True)
    threading.Thread(target=_heartbeat, daemon=True).start()
    OUT = os.environ.get("GEM_OUT") or f"{ROOT}/results/llm_{JUDGE.replace('/', '_')}"
elif PROVIDER == "local":
    # GEM_PROVIDER=local: any OpenAI-compatible server (e.g. vLLM) at GEM_BASE_URL serving GEM_MODEL; no key, no cost.
    # GEM_EXTRA_BODY: optional JSON merged into every request, e.g. '{"chat_template_kwargs": {"enable_thinking": false}}'
    JUDGE = os.environ.get("GEM_MODEL", "google/gemma-4-26b-a4b-it")
    BASE_URL = os.environ.get("GEM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    OUT = os.environ.get("GEM_OUT") or f"{ROOT}/results/gemma_local"
else:
    JUDGE = "google/gemma-4-26b-a4b-it"
    OUT = f"{ROOT}/results/gemma"
OPTIM = JUDGE  # the optimizer is the judge model itself (as in ProTeGi and GEPA)
DATASETS = {"hs2": "claude_hs2", "hanna": "claude_hanna", "ell": "claude_ell", "merlin": "claude_merlin", "se": "claude_se"}
BUDGET = float(os.environ.get("GEM_BUDGET", "1e9" if PROVIDER in ("local", "llm") else "150"))  # total $ across all runs; a safety stop, not part of any method
os.makedirs(OUT, exist_ok=True)

_KEY = None
_LOCK = threading.Lock()
_DB = None
class _Tag:
    v = ""


TAG = _Tag()
SALT = _Tag()  # which method/run the paid calls belong to (for the cost log); one run per process


class Abort(Exception):
    pass


class ItemFailed(Exception):
    """One call failed for this item only (e.g. the API refused its content): the item is dropped, the run goes on."""


def _key():
    global _KEY
    if PROVIDER == "local":
        return os.environ.get("GEM_API_KEY", "EMPTY")
    if _KEY is None:
        _KEY = open(os.path.expanduser(os.environ.get("GEM_KEYFILE") or {"juapi": "~/.juapi_key", "lanyi": "~/.lanyiapi_key"}.get(
            PROVIDER, "~/.openrouter_key"))).read().strip()
    return _KEY


def _db():
    global _DB
    if _DB is None:
        _DB = sqlite3.connect(f"{OUT}/cache.sqlite", check_same_thread=False, timeout=60)
        _DB.execute("create table if not exists cache (k text primary key, txt text, in_tok int, out_tok int, cost real)")
        _DB.execute("create table if not exists spend (ts real, tag text, model text, in_tok int, out_tok int, cost real)")
        _DB.commit()
    return _DB


def spent(tag=None):
    with _LOCK:
        q = "select coalesce(sum(cost),0), count(*), coalesce(sum(in_tok),0), coalesce(sum(out_tok),0) from spend"
        return _db().execute(q + (" where tag like ?" if tag else ""), ((tag + "%",) if tag else ())).fetchone()


def chat(messages, model=JUDGE, temperature=0.0, max_tokens=2000, cache=None):
    """One OpenRouter chat call -> (text, in_tok, out_tok, cost). Cached when temperature is 0 (or cache=True)."""
    cache = (temperature == 0) if cache is None else cache
    salt = [SALT.v] if getattr(SALT, "v", "") else []  # a re-judging of the same prompt (noise control) gets its own cache
    k = hashlib.sha256(json.dumps([model, temperature, max_tokens, messages] + salt, ensure_ascii=False).encode()).hexdigest()
    if cache:
        with _LOCK:
            row = _db().execute("select txt, in_tok, out_tok, cost from cache where k=?", (k,)).fetchone()
        if row:
            return row[0], row[1], row[2], 0.0
    if spent()[0] > BUDGET:
        raise Abort(f"budget ${BUDGET} reached")
    if PROVIDER == "llm":  # the llm package does its own retries (429 / 5xx); account errors stop the run
        msg = ""
        for t in range(4):  # an empty reply or a transient error (e.g. "returned an empty response") is retried
            try:
                fut = _LLM_POOL.submit(_llm_chat_result, LLM_PROVIDER, messages, temperature=temperature, max_tokens=max_tokens)
                try:
                    res = fut.result(timeout=LLM_TIMEOUT)
                except _FutTimeout:
                    _N_TIMEOUTS[0] += 1
                    raise RuntimeError(f"no reply after {LLM_TIMEOUT:.0f}s")
                _N_CALLS[0] += 1
                if (res.content or "").strip():
                    break
                msg = "empty response"
            except Exception as e:  # noqa: BLE001  (the package's ProviderError or a transport error)
                msg = str(e)
                if re.search(r"401|403|unauthori|forbidden|quota|insufficient|billing|api[_ ]?key", msg, re.I):
                    raise Abort(f"llm package: {msg[:300]}")  # account-level problem: stop everything
            time.sleep(2 ** (t + 1))
        else:
            raise ItemFailed(msg[:300])  # still failing (e.g. content refused, empty): drop this item; not cached
        txt = res.content or ""
        u = res.usage
        it, ot = (u.input_tokens, u.output_tokens) if u is not None else (0, 0)
        with _LOCK:
            db = _db()
            db.execute("insert into spend values (?,?,?,?,?,?)", (time.time(), getattr(TAG, "v", ""), model, it, ot, 0.0))
            if cache:
                db.execute("insert or replace into cache values (?,?,?,?,?)", (k, txt, it, ot, 0.0))
            db.commit()
        return txt, it, ot, 0.0
    if PROVIDER == "juapi":  # streaming (non-streaming GPT calls are capped at 10/min); reasoning off
        body = json.dumps(dict(model=model, messages=messages, temperature=temperature, max_completion_tokens=max_tokens,
                               reasoning_effort="none", stream=True, stream_options=dict(include_usage=True))).encode()
        url = "https://www.juapi.net/v1/chat/completions"
    elif PROVIDER == "local":
        body = json.dumps(dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                               **json.loads(os.environ.get("GEM_EXTRA_BODY", "{}")))).encode()
        url = BASE_URL + "/chat/completions"
    elif PROVIDER == "lanyi":
        body = json.dumps(dict(model=model, messages=messages, temperature=temperature, max_completion_tokens=max_tokens,
                               reasoning_effort="none")).encode()
        url = "https://lanyiapi.com/v1/chat/completions"
    else:
        body = json.dumps(dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                               reasoning=dict(enabled=False), usage=dict(include=True))).encode()
        url = "https://openrouter.ai/api/v1/chat/completions"
    err = ""
    for t in range(20):  # transient outages (e.g. juapi 503 "no available channel") can last minutes: back off up to ~1 h
        if t >= 5 and not err.startswith(("HTTP 503", "HTTP 429", "HTTP 502", "HTTP 500")):
            break
        try:
            req = urllib.request.Request(url, data=body, headers={
                "Authorization": "Bearer " + _key(), "Content-Type": "application/json"})
            if PROVIDER == "juapi":
                txt, u, n = "", {}, 0
                with urllib.request.urlopen(req, timeout=600) as resp:
                    for line in resp:
                        line = line.decode(errors="replace").strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        o = json.loads(line[5:])
                        if "error" in o:
                            raise RuntimeError(str(o["error"])[:300])
                        for ch in o.get("choices") or []:
                            txt += (ch.get("delta") or {}).get("content") or ""
                        u = o.get("usage") or u
                        n += 1
                        if n > max_tokens + 50:  # the stream ignores the token cap: stop reading
                            break
                it, ot, c = u.get("prompt_tokens", 0), u.get("completion_tokens", 0), 0.0  # cost: juapi quota, see quota()
            else:
                o = json.loads(urllib.request.urlopen(req, timeout=240).read())
                if "error" in o:
                    raise RuntimeError(str(o["error"])[:300])
                txt = o["choices"][0]["message"].get("content") or ""
                u = o.get("usage", {})
                it, ot, c = u.get("prompt_tokens", 0), u.get("completion_tokens", 0), float(u.get("cost", 0) or 0)
            with _LOCK:
                db = _db()
                db.execute("insert into spend values (?,?,?,?,?,?)", (time.time(), getattr(TAG, "v", ""), model, it, ot, c))
                if cache:
                    db.execute("insert or replace into cache values (?,?,?,?,?)", (k, txt, it, ot, c))
                db.commit()
            return txt, it, ot, c
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (401, 402, 403):  # bad key / no credit / moderation or ban: stop everything
                raise Abort(f"HTTP {e.code}: {msg}")
            err = f"HTTP {e.code}: {msg}"
        except Abort:
            raise
        except Exception as e:  # noqa: BLE001
            err = str(e)[:300]
        time.sleep(3 * (t + 1) if t < 5 else 300)
    raise RuntimeError(err)


def quota():
    """juapi: total quota used so far (in juapi's USD units), from the OpenAI-style billing endpoint."""
    req = urllib.request.Request("https://www.juapi.net/v1/dashboard/billing/usage?start_date=2026-01-01&end_date=2027-01-01",
                                 headers={"Authorization": "Bearer " + _key()})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["total_usage"] / 100


def parse_scores(txt):
    for js in reversed(re.findall(r"\{[^{}]*\}", txt)):
        try:
            return {k: float(v) for k, v in json.loads(js).items() if isinstance(v, (int, float))}
        except Exception:  # noqa: BLE001
            continue
    return {}


class DS:
    """One dataset: attributes, output format, initial prompt v0, gold scores and the fit / sel / test items."""

    def __init__(self, name):
        # the dataset files live in data/<dir> (standalone package) or results/<dir> (original repository)
        self.name = name
        self.R = next((p for p in (f"{ROOT}/data/{DATASETS[name]}", f"{ROOT}/results/{DATASETS[name]}") if os.path.isdir(p)),
                      f"{ROOT}/results/{DATASETS[name]}")
        self.A = open(f"{self.R}/attrs.txt").read().split()
        self.FORMAT = open(f"{self.R}/format.txt").read().strip()
        self.LO, self.HI = map(int, re.search(r"integer from (\d+) to (\d+)", self.FORMAT).groups())
        self.v0 = open(f"{self.R}/prompts/v0.txt").read().strip()
        self.gold = pd.read_csv(f"{self.R}/gold.csv").set_index("id")
        if os.environ.get("GEM_TEACHER"):  # distillation: replace the labels of fit / sel items by teacher labels
            T = pd.read_csv(os.environ["GEM_TEACHER"]).set_index("id")
            self.gold.loc[T.index, self.A] = T[self.A].to_numpy(float)
        self.rows = {s: [json.loads(l) for f in sorted(os.listdir(f"{self.R}/chunks")) if f.startswith(s + "_")
                         for l in open(f"{self.R}/chunks/{f}")] for s in ("fit", "sel", "test")}
        self.text = {r["id"]: r["user"] for s in self.rows for r in self.rows[s]}
        # items that failed (API error, or no parsable rating from the v0 judge) are dropped from every split, for
        # every method that starts after they were recorded (scripts/screen_items.py, judge_msgs)
        self.excl_file = f"{OUT}/{name}/excluded_ids.txt"
        self.excluded = set(open(self.excl_file).read().split()) if os.path.exists(self.excl_file) else set()
        self.rows = {s: [r for r in v if r["id"] not in self.excluded] for s, v in self.rows.items()}

    def exclude(self, ids):
        new = set(ids) - self.excluded
        if new:
            with _LOCK:
                os.makedirs(os.path.dirname(self.excl_file), exist_ok=True)
                with open(self.excl_file, "a") as f:
                    f.write("".join(i + "\n" for i in sorted(new)))
            self.excluded |= new

    def keep(self, df):
        """Drop excluded items from a score table (id column)."""
        return df[~df.id.isin(self.excluded)].reset_index(drop=True)

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

    def judge_msgs(self, make, rows, workers=16, strict=True):
        """strict: stop if >20% of outputs are unparsed (outage / refusal). Optimizers scoring candidate prompts pass
        strict=False: a candidate that breaks the output format just scores badly (see worst_case)."""
        def one(r):
            try:
                txt, it, ot, c = chat(make(r))
            except Abort:
                raise
            except ItemFailed as e:
                self.exclude([r["id"]])
                return dict(id=r["id"], error="ITEM_FAILED " + str(e)[:200])
            except Exception as e:  # noqa: BLE001
                return dict(id=r["id"], error=str(e)[:200])
            sc = parse_scores(txt)
            return dict(id=r["id"], **{a: sc.get(a, np.nan) for a in self.A}, in_tok=it, out_tok=ot, reason=txt[-1500:])

        with ThreadPoolExecutor(workers) as ex:
            df = pd.DataFrame(list(ex.map(one, rows)))
        for a in self.A:
            if a not in df:
                df[a] = np.nan
        failed = df["error"].astype(str).str.startswith("ITEM_FAILED") if "error" in df else pd.Series(False, index=df.index)
        bad = df.loc[~failed, self.A].isna().any(axis=1).mean() if (~failed).any() else 0.0  # dropped items do not count
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
    """Final evaluation of a single-instruction judge: score fit, sel and test, save, return the test Spearman."""
    for s in ("fit", "sel", "test"):
        save_scores(ds, method, seed, s, ds.judge(instruction, ds.rows[s], workers))
    open(f"{ds.out(method, seed)}/final_prompt.txt", "w").write(instruction)
