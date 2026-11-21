"""Transport for any OpenAI-shaped host: POST {base_url}/chat/completions (standard library only).

Provider keys used: model, base_url, api_key_env, params, native_batch, and optionally
  "max_tokens_field": "max_completion_tokens"   for hosts that reject max_tokens (newer OpenAI models)
  "headers": {...}                               extra request headers
  "batch_poll_seconds": 30                       poll interval of the native batch endpoint
  "batch_window": "24h"                          completion_window of the native batch endpoint
Every option except `timeout` goes into the request body (so provider params such as reasoning_effort pass through).
With native_batch: true, batch_result uses the OpenAI Batch API (/files + /batches) of the same host.
"""
import json, time, urllib.error, urllib.request, uuid

from ..registry import ProviderError
from ..types import LLMResult, LLMUsage


def _headers(p, ctype="application/json"):
    h = {"Content-Type": ctype, **(p.extra.get("headers") or {})}
    if p.api_key:
        h["Authorization"] = "Bearer " + p.api_key
    return h


def _url(p, path):
    if not p.base_url:
        raise ProviderError(f"provider '{p.name}' (openai_compatible) needs base_url in providers.json")
    return p.base_url.rstrip("/") + path


def _request(p, path, data=None, ctype="application/json", timeout=120, method=None, raw=False):
    req = urllib.request.Request(_url(p, path), data=data, headers=_headers(p, ctype), method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:500]
        raise ProviderError(f"provider '{p.name}': HTTP {e.code}: {msg}", e.code,
                            retryable=e.code == 429 or e.code >= 500) from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise ProviderError(f"provider '{p.name}': {type(e).__name__}: {e}") from e
    return body if raw else json.loads(body)


def _body(p, messages, opts):
    body = {k: v for k, v in opts.items() if k != "timeout"}
    if "max_tokens" in body and p.extra.get("max_tokens_field"):
        body[p.extra["max_tokens_field"]] = body.pop("max_tokens")
    return dict(model=p.model, messages=messages, **body)


def _result(p, o):
    if "error" in o and o["error"]:
        raise ProviderError(f"provider '{p.name}': {str(o['error'])[:500]}")
    try:
        content = o["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError(f"provider '{p.name}': unexpected response {str(o)[:300]}") from e
    return LLMResult(content, LLMUsage.from_raw(o.get("usage")), p.name, o.get("model") or p.model)


def complete(p, messages, opts):
    data = json.dumps(_body(p, messages, opts)).encode()
    return _result(p, _request(p, "/chat/completions", data, timeout=opts.get("timeout", 120)))


def complete_batch(p, items, opts):
    """OpenAI Batch API: upload a JSONL of requests, create a batch, poll, download, return results in order."""
    ids = [f"r{i}-{uuid.uuid4().hex[:8]}" for i in range(len(items))]
    jsonl = "\n".join(json.dumps(dict(custom_id=c, method="POST", url="/v1/chat/completions", body=_body(p, m, opts)))
                      for c, m in zip(ids, items)).encode()
    boundary = uuid.uuid4().hex
    form = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"batch.jsonl\"\r\n"
            f"Content-Type: application/jsonl\r\n\r\n").encode() + jsonl + f"\r\n--{boundary}--\r\n".encode()
    f = _request(p, "/files", form, ctype=f"multipart/form-data; boundary={boundary}", timeout=600)
    b = _request(p, "/batches", json.dumps(dict(input_file_id=f["id"], endpoint="/v1/chat/completions",
                                                completion_window=p.extra.get("batch_window", "24h"))).encode())
    poll = float(p.extra.get("batch_poll_seconds", 30))
    while b.get("status") not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(poll)
        b = _request(p, f"/batches/{b['id']}", method="GET")
    if b["status"] != "completed" or not b.get("output_file_id"):
        raise ProviderError(f"provider '{p.name}': batch {b['id']} ended {b['status']}: {str(b.get('errors'))[:300]}")
    lines = _request(p, f"/files/{b['output_file_id']}/content", method="GET", raw=True, timeout=600).decode().splitlines()
    got = {}
    for line in lines:
        if line.strip():
            o = json.loads(line)
            resp = o.get("response") or {}
            if o.get("error") or resp.get("status_code", 200) != 200:
                raise ProviderError(f"provider '{p.name}': batch item {o.get('custom_id')} failed: "
                                    f"{str(o.get('error') or resp.get('body'))[:300]}")
            got[o["custom_id"]] = _result(p, resp.get("body") or {})
    missing = [c for c in ids if c not in got]
    if missing:
        raise ProviderError(f"provider '{p.name}': batch returned {len(got)} results for {len(ids)} requests")
    return [got[c] for c in ids]
