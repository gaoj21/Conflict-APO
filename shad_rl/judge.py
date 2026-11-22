"""The LLM judge used as the RL reward on TL;DR, and the ShAD decoder applied to its scores.

The judge is Gemma 4 26B A4B behind any OpenAI-compatible server (vLLM on the GPU machine), with exactly the prompt
under which the decoder was fitted (decoder_tldr.json; the v0 prompt of the ShAD paper's TL;DR experiments).

  raw reward  = the judge's "overall" score (1-7)
  ShAD reward = the decoded "overall": all four judge scores + log length -> the human scale

Both are returned for every summary, so each run logs the reward it does NOT optimize as a monitor.
"""
import json, os, re, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPT = """You are a careful, calibrated evaluator. Your goal is to produce an accurate evaluation by following the Instructions below.

## Task
Evaluate the Summary of the Reddit post using the Instructions below.
1. Consider every strength and flaw you find when making your evaluation.
2. Based on the number and severity of the strengths and flaws, assign a value.

## Instructions:
- overall: Evaluate how good the summary is overall as a summary of the post. Rate from 1 to 7.
- accuracy: Evaluate to what extent the statements in the summary are stated or implied by the post. Rate from 1 to 7.
- coverage: Evaluate how well the summary covers the important information in the post. Rate from 1 to 7.
- coherence: Evaluate how easy the summary is to understand on its own. Rate from 1 to 7.

## Output format:
Think briefly (at most a few sentences), then end your reply with exactly one line of JSON with an integer from 1 to 7 for each attribute:
{"overall": 1-7, "accuracy": 1-7, "coverage": 1-7, "coherence": 1-7}"""


def user_msg(title, post, summary):
    return f"## Reddit post\nTitle: {title.strip()}\n{post.strip()}\n\n## Summary:\n{summary.strip()}\n\n## Your evaluation:"


def parse(txt, attrs):
    """The last JSON object in the reply that holds the attributes -> list of floats (NaN where missing)."""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", txt or ""))):
        try:
            o = json.loads(m.group(0))
        except ValueError:
            continue
        if any(a in o for a in attrs):
            return [float(o[a]) if isinstance(o.get(a), (int, float)) else np.nan for a in attrs]
    return [np.nan] * len(attrs)


class Decoder:
    def __init__(self, path=f"{HERE}/decoder_tldr.json"):
        d = json.load(open(path))
        self.A, self.d = d["attrs"], d
        self.W = np.array(d["W"])
        self.k = self.A.index("overall")

    def __call__(self, X, words):
        """X: (n, K) raw judge scores (NaN allowed), words: (n,) summary word counts -> decoded ratings (n, K)."""
        d = self.d
        X = np.where(np.isnan(X), np.array(d["fill"]), X)
        F = np.c_[(X - np.array(d["x_mean"])) / np.array(d["x_std"]), (np.log1p(words) - d["z_mean"]) / d["z_std"]]
        return np.clip((F @ self.W) * np.array(d["y_std"]) + np.array(d["y_mean"]), d["lo"], d["hi"])


class Judge:
    """Thread-pooled calls to an OpenAI-compatible chat endpoint. Every call has a timeout and is retried."""

    def __init__(self, base_url=None, model=None, workers=None, timeout=None, extra_body=None):
        self.url = (base_url or os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1")).rstrip("/") + "/chat/completions"
        self.model = model or os.environ.get("JUDGE_MODEL", "google/gemma-4-26b-a4b-it")
        self.key = os.environ.get("JUDGE_API_KEY", "EMPTY")
        self.workers = int(workers or os.environ.get("JUDGE_WORKERS", "64"))
        self.timeout = float(timeout or os.environ.get("JUDGE_TIMEOUT", "180"))
        self.extra = extra_body if extra_body is not None else json.loads(os.environ.get("JUDGE_EXTRA_BODY", "{}"))
        self.dec = Decoder()
        self.A = self.dec.A

    def _one(self, msg):
        body = dict(model=self.model, temperature=0.0, max_tokens=2000,
                    messages=[dict(role="system", content=PROMPT), dict(role="user", content=msg)], **self.extra)
        for t in range(4):
            try:
                r = requests.post(self.url, json=body, timeout=self.timeout, headers={"Authorization": f"Bearer {self.key}"})
                r.raise_for_status()
                return r.json()["choices"][0]["message"].get("content") or ""
            except Exception as e:  # noqa: BLE001
                err = str(e)
                time.sleep(2 ** t)
        print(f"[judge] call failed 4 times: {err[:200]}", flush=True)
        return ""

    def score(self, titles, posts, summaries):
        """-> dict(raw=(n,K) judge scores, shad=(n,K) decoded, words=(n,), parsed=(n,) bool)."""
        msgs = [user_msg(t, p, s) for t, p, s in zip(titles, posts, summaries)]
        with ThreadPoolExecutor(self.workers) as ex:
            txt = list(ex.map(self._one, msgs))
        X = np.array([parse(t, self.A) for t in txt], float).reshape(len(msgs), len(self.A))
        words = np.array([len(s.split()) for s in summaries], float)
        return dict(raw=X, shad=self.dec(X, words), words=words, parsed=~np.isnan(X).any(1))
