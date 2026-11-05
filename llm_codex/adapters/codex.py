"""Transport through the Codex CLI (`codex exec`), i.e. the GPT model of your Codex login (ChatGPT plan or API key).

One completion = one `codex exec --json` process in an empty temporary directory, read-only sandbox, no session files:
  - the system message(s) replace Codex's own coding-agent prompt (written to a file passed as model_instructions_file),
    preceded by a line telling the model to answer directly without tools;
  - the user / assistant messages are the prompt (stdin);
  - the reply is the last `agent_message` event; usage comes from the `turn.completed` event
    (input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens).

Provider keys (llm/providers.json):
  "adapter": "codex"
  "model": "<codex model>"                 optional; default: the model in ~/.codex/config.toml
  "codex_bin": "codex"                     optional; path of the codex executable (env CODEX_BIN overrides)
  "codex_config": {"model_reasoning_effort": "low"}   optional; each entry becomes `-c key=value`
  "params": {"timeout": 300}               seconds per call
Sampling options other than `timeout` (temperature, top_p, max_tokens, stop, seed) cannot be passed to Codex and are
ignored. Codex always adds a short sandbox / environment preamble of its own (a few hundred tokens per call).
"""
import json, os, re, shutil, subprocess, tempfile

from ..registry import ProviderError
from ..types import LLMResult, LLMUsage

PREAMBLE = ("You are used as a plain text-completion endpoint. Answer the user's message directly in your reply. "
            "Never run commands, read or write files, or call any tool.")
_RETRY = re.compile(r"\b429\b|rate.?limit|too many requests|\b5\d\d\b|overloaded|stream (disconnected|error)|"
                    r"connection (reset|refused|error)|timed? ?out", re.I)


def _bin(p):
    b = os.environ.get("CODEX_BIN") or p.extra.get("codex_bin") or "codex"
    path = shutil.which(b) or (b if os.path.exists(b) else None)
    if not path:
        raise ProviderError(f"provider '{p.name}': codex executable not found ({b}); install it with "
                            f"`npm i -g @openai/codex` and log in with `codex login`, or set codex_bin / CODEX_BIN")
    return path


def _toml(v):
    return json.dumps(v) if isinstance(v, str) else (str(v).lower() if isinstance(v, bool) else str(v))


def _split(messages):
    system = "\n\n".join(m["content"] for m in messages if m.get("role") in ("system", "developer"))
    turns = [m for m in messages if m.get("role") not in ("system", "developer")]
    if len(turns) == 1 and turns[0].get("role") == "user":
        prompt = turns[0]["content"]
    else:  # a multi-turn conversation as a transcript
        prompt = "\n\n".join(f"[{m.get('role', 'user')}]\n{m['content']}" for m in turns) + "\n\n[assistant]\n"
    return system, prompt


def parse_events(stdout):
    """codex exec --json output -> (last agent message or None, usage dict or None, error text or None)."""
    text, usage, err = None, None, None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        t = o.get("type", "")
        item = o.get("item") or {}
        if t == "item.completed" and item.get("type") in ("agent_message", "assistant_message") and item.get("text"):
            text = item["text"]
        elif t == "turn.completed" and isinstance(o.get("usage"), dict):
            usage = o["usage"]
        elif t in ("turn.failed", "error"):
            err = str(o.get("error") or o.get("message") or o)[:500]
    return text, usage, err


def complete(p, messages, opts):
    system, prompt = _split(messages)
    work = tempfile.mkdtemp(prefix="llm_codex_")
    try:
        instr, last = os.path.join(work, "instructions.md"), os.path.join(work, "last_message.txt")
        with open(instr, "w", encoding="utf-8") as f:
            f.write(PREAMBLE + ("\n\n" + system if system else ""))
        cmd = [_bin(p), "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
               "--color", "never", "-C", work, "-o", last, "-c", f"model_instructions_file={json.dumps(instr)}"]
        if p.model:
            cmd += ["-m", p.model]
        for k, v in (p.extra.get("codex_config") or {}).items():
            cmd += ["-c", f"{k}={_toml(v)}"]
        cmd.append("-")  # prompt from stdin
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=float(opts.get("timeout", 300)),
                               cwd=work)
        except subprocess.TimeoutExpired as e:
            raise ProviderError(f"provider '{p.name}': codex exec timed out after {opts.get('timeout', 300)}s",
                                retryable=True) from e
        text, usage, err = parse_events(r.stdout)
        if text is None and os.path.exists(last):
            text = open(last, encoding="utf-8").read() or None
        if r.returncode != 0 or text is None:
            why = err or (r.stderr or "").strip()[-500:] or f"exit code {r.returncode}, no reply"
            raise ProviderError(f"provider '{p.name}': codex exec failed: {why}", retryable=bool(_RETRY.search(why)))
        return LLMResult(text, LLMUsage.from_raw(usage), p.name, p.model)
    finally:
        shutil.rmtree(work, ignore_errors=True)
