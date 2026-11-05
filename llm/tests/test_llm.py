"""Tests of the llm package. They never call a provider: the transport is patched (or served by a local fake host).
  python -m pytest llm/tests
"""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import llm  # noqa: E402
from llm import registry, client  # noqa: E402  (internals: tests only)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    def write(providers, default):
        p = tmp_path / "providers.json"
        p.write_text(json.dumps(dict(default=default, providers=providers)))
        monkeypatch.setattr(registry, "CONFIG", str(p))
        monkeypatch.setattr(registry, "REPO", str(tmp_path))
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("EAX_PROVIDER", raising=False)
    monkeypatch.setattr(client, "RETRY_BACKOFF", (0, 0))
    return write


def test_usage_names_and_none():
    u = llm.LLMUsage.from_raw({"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}})
    assert (u.input_tokens, u.output_tokens, u.total_tokens, u.cache_read_tokens, u.uncached_input_tokens) == (10, 3, 13, 4, 6)
    assert llm.LLMUsage.from_raw(None) is None and llm.LLMUsage.from_raw({}) is None


def test_tracker_prices_and_uncounted(monkeypatch):
    monkeypatch.setenv("EVO_TOKEN_INPUT_PRICE_PER_1M", "2")
    monkeypatch.setenv("EVO_TOKEN_OUTPUT_PRICE_PER_1M", "8")
    monkeypatch.setenv("EVO_TOKEN_CACHED_INPUT_RATIO", "0.5")
    t = llm.UsageTracker()
    assert t.add(llm.LLMResult("a", llm.LLMUsage(1_000_000, 1_000_000, 2_000_000, cache_read_tokens=500_000)))
    assert not t.add(llm.LLMResult("b", None))
    assert t.uncounted == 1 and t.calls == 2
    assert t.cost == pytest.approx(0.5 * 2 + 0.5 * 2 * 0.5 + 8)


def test_errors(cfg, monkeypatch):
    cfg({"a": {"type": "openai_compatible", "base_url": "http://x", "model": "m", "api_key_env": "K_A"},
         "off": {"type": "litellm", "model": "m", "enabled": False}}, "a")
    monkeypatch.delenv("K_A", raising=False)
    with pytest.raises(llm.ProviderError, match="K_A"):
        llm.chat(None, [{"role": "user", "content": "hi"}])
    with pytest.raises(llm.ProviderError, match="unknown provider 'zz'; configured: a, off"):
        llm.chat("zz", [])
    with pytest.raises(llm.ProviderError, match="disabled"):
        llm.chat("off", [])
    monkeypatch.setenv("K_A", "x")
    with pytest.raises(llm.ProviderError, match="unsupported option"):
        llm.chat(None, [], tools=[])
    rows = {r["name"]: r for r in llm.list_providers()}
    assert rows["a"]["available"] and rows["a"]["default"] and not rows["off"]["available"]


def test_switch_provider_by_env(cfg, monkeypatch):
    cfg({"a": {"type": "litellm", "model": "m"}, "b": {"type": "litellm", "model": "m"}}, "a")
    assert llm.default_provider() == "a"
    monkeypatch.setenv("LLM_PROVIDER", "b")
    assert llm.default_provider() == "b"
    monkeypatch.setenv("EAX_PROVIDER", "a")
    assert llm.default_provider() == "a"


def test_dotenv(cfg, tmp_path, monkeypatch):
    cfg({"a": {"type": "litellm", "model": "m", "api_key_env": "K_DOT"}}, "a")
    monkeypatch.delenv("K_DOT", raising=False)
    (tmp_path / ".env").write_text("export K_DOT='v'\n")
    assert llm.get_provider().api_key == "v"


class _Host(BaseHTTPRequestHandler):
    """A fake OpenAI-shaped host: echoes the last message; 'FAIL429' is rate-limited once; batch endpoints too."""
    seen, fail_once, files = [], {"n": 0}, {}

    def log_message(self, *a):
        pass

    def _send(self, code, obj, raw=None):
        b = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _answer(self, body):
        return dict(model=body["model"], choices=[dict(message=dict(content="echo:" + body["messages"][-1]["content"]))],
                    usage=dict(prompt_tokens=5, completion_tokens=2, total_tokens=7))

    def do_POST(self):
        data = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/v1/chat/completions":
            body = json.loads(data)
            _Host.seen.append(body)
            if body["messages"][-1]["content"] == "FAIL429" and _Host.fail_once["n"] == 0:
                _Host.fail_once["n"] = 1
                return self._send(429, {"error": "slow down"})
            if body["messages"][-1]["content"] == "BAD":
                return self._send(400, {"error": "bad request"})
            return self._send(200, self._answer(body))
        if self.path == "/v1/files":
            jsonl = data.split(b"\r\n\r\n", 2)[2].rsplit(b"\r\n--", 1)[0]
            _Host.files["in"] = [json.loads(l) for l in jsonl.decode().splitlines() if l.strip()]
            return self._send(200, {"id": "file-in"})
        if self.path == "/v1/batches":
            out = [dict(custom_id=r["custom_id"], response=dict(status_code=200, body=self._answer(r["body"])))
                   for r in reversed(_Host.files["in"])]  # out of order on purpose
            _Host.files["out"] = "\n".join(json.dumps(o) for o in out).encode()
            return self._send(200, {"id": "b1", "status": "in_progress"})

    def do_GET(self):
        if self.path == "/v1/batches/b1":
            return self._send(200, {"id": "b1", "status": "completed", "output_file_id": "file-out"})
        if self.path == "/v1/files/file-out/content":
            return self._send(200, None, raw=_Host.files["out"])


@pytest.fixture
def host():
    srv = HTTPServer(("127.0.0.1", 0), _Host)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_openai_compatible_calls(cfg, host, monkeypatch):
    monkeypatch.setenv("K", "secret")
    cfg({"h": {"type": "openai_compatible", "base_url": host, "model": "m1", "api_key_env": "K",
               "max_tokens_field": "max_completion_tokens", "params": {"temperature": 1.0, "reasoning_effort": "none"}}}, "h")
    r = llm.chat_result(None, [{"role": "user", "content": "hi"}], temperature=0, max_tokens=9)
    assert r.content == "echo:hi" and r.usage.total_tokens == 7 and r.provider == "h" and r.model == "m1"
    sent = _Host.seen[-1]
    assert sent["temperature"] == 0 and sent["reasoning_effort"] == "none" and sent["max_completion_tokens"] == 9
    assert "max_tokens" not in sent and "timeout" not in sent
    assert llm.chat(None, [{"role": "user", "content": "FAIL429"}]) == "echo:FAIL429"  # retried once
    with pytest.raises(llm.ProviderError, match="HTTP 400"):
        llm.chat(None, [{"role": "user", "content": "BAD"}])
    out = llm.batch(None, [[{"role": "user", "content": str(i)}] for i in range(20)])
    assert out == [f"echo:{i}" for i in range(20)]


def test_native_batch_order(cfg, host, monkeypatch):
    monkeypatch.setenv("K", "secret")
    cfg({"h": {"type": "openai_compatible", "base_url": host, "model": "m1", "api_key_env": "K", "native_batch": True,
               "batch_poll_seconds": 0}}, "h")
    rs = llm.batch_result(None, [[{"role": "user", "content": str(i)}] for i in range(7)], temperature=0)
    assert [r.content for r in rs] == [f"echo:{i}" for i in range(7)]
    assert all(r["body"]["temperature"] == 0 for r in _Host.files["in"])


def test_wrong_count_is_error(cfg, monkeypatch):
    cfg({"a": {"type": "openai_compatible", "base_url": "http://x", "model": "m", "native_batch": True}}, "a")
    from llm.adapters import openai_compatible as oc
    monkeypatch.setattr(oc, "complete_batch", lambda p, items, opts: [llm.LLMResult("x", None)])
    with pytest.raises(llm.ProviderError, match="1 results for 2 requests"):
        llm.batch_result(None, [[], []])


def test_async(cfg, monkeypatch):
    import asyncio
    cfg({"a": {"type": "openai_compatible", "base_url": "http://x", "model": "m"}}, "a")
    from llm.adapters import openai_compatible as oc
    monkeypatch.setattr(oc, "complete", lambda p, m, o: llm.LLMResult("ok", None, p.name, p.model))
    assert asyncio.run(llm.achat(None, [])) == "ok"
    assert asyncio.run(llm.abatch(None, [[], []])) == ["ok", "ok"]


def test_litellm_transport(cfg, monkeypatch):
    import types
    cfg({"d": {"type": "litellm", "model": "deepseek/x", "api_key_env": "KD", "params": {"timeout": 120}}}, "d")
    monkeypatch.setenv("KD", "k")
    seen = {}

    def completion(**kw):
        seen.update(kw)
        msg = types.SimpleNamespace(content="hi")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], model="deepseek/x",
                                     usage={"prompt_tokens": 3, "completion_tokens": 1})

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    r = llm.chat_result(None, [{"role": "user", "content": "x"}], temperature=0.3)
    assert r.content == "hi" and r.usage.input_tokens == 3 and r.usage.total_tokens == 4
    assert seen["temperature"] == 0.3 and seen["timeout"] == 120 and seen["api_key"] == "k"
