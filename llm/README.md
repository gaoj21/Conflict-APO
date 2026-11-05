# `llm` — the model boundary (llm-v12)

Every model call the project makes goes through this package. It is a standalone package at the repository root: it
imports nothing from the rest of the repository, and it is the only place a provider SDK, an endpoint URL or a
provider's response shape appears.

**The rule, in one line:** nothing outside `llm/` may name a provider, import a provider SDK, or read a provider
response. Business code says *what* to send and whether it wants text, an `LLMResult` or usage, never *where* the
call goes.

## Public API

```python
from llm import (
    chat, chat_result,            # one completion: text, or text + usage
    batch, batch_result,          # many completions, in the order given
    achat, achat_result,          # the same, awaited
    abatch, abatch_result,
    LLMResult, LLMUsage,          # what a call returns
    UsageTracker,                 # totals and what they cost
    list_providers, get_provider, default_provider,
    ProviderError,                # every failure this package raises
)
```

Nothing else is public (`llm.client`, `llm.registry`, `llm.types`, `llm.usage`, `llm.adapters` are the implementation).

- `chat(None, messages)`, `chat_result(None, messages, temperature=0.2)`, `batch_result(None, [m_a, m_b])`.
  The first argument is the provider name, or `None` for the configured default. Keyword options are sampling
  settings (`temperature`, `top_p`, `max_tokens`, `stop`, `timeout`, `seed`); anything else is a `ProviderError`.
  The provider's `params` are the floor they override.
- `batch` uses the provider's real batch endpoint when it has one (`"native_batch": true`; the `openai_compatible`
  transport implements the OpenAI Batch API), otherwise parallel requests bounded by `client.MAX_PARALLEL`
  (env `LLM_MAX_PARALLEL`, default 16). Results come back in the order given; a wrong count is a `ProviderError`.
- Retries cover rate limits (429) and 5xx only (`client.RETRY_BACKOFF`, two backoffs). A bad request is raised as is.
- `LLMResult(content, usage, provider, model)`; `usage` is `None` when the provider reported nothing, never zeroes.
  `LLMUsage` has `input_tokens`, `output_tokens`, `total_tokens`, `cache_read_tokens`, `reasoning_tokens` and the
  derived `uncached_input_tokens`, whatever names the provider used.
- `UsageTracker`: `add(result)` (returns `False` for a result without usage, counted in `uncounted`), `cost`, `summary()`.
  Prices per million tokens from `EVO_TOKEN_INPUT_PRICE_PER_1M`, `EVO_TOKEN_OUTPUT_PRICE_PER_1M`,
  `EVO_TOKEN_CACHED_INPUT_RATIO`, `EVO_WEB_SEARCH_PRICE_PER_1K`.

## Providers

`llm/providers.json` is the whole configuration. Secrets never live in it: each provider names the environment
variables it needs (`api_key_env`, or `requires: [...]`), read from the process environment and the repository `.env`.

- `type`: `litellm`, or `openai_compatible` for any OpenAI-shaped `/chat/completions` host (needs `base_url`;
  optional `max_tokens_field`, `headers`, `native_batch`, `batch_poll_seconds`, `batch_window`).
- `adapter`: a module in `llm/adapters/` (or a dotted path) instead, for a transport of your own: it defines
  `complete(provider, messages, options) -> LLMResult` and optionally `complete_batch(provider, items, options)`.
- `enabled: false` marks a documented example; it is refused, not attempted.

To switch provider: change `default` in `providers.json`, or set `LLM_PROVIDER` (`EAX_PROVIDER` takes precedence).
`list_providers()` reports each provider with `available` and `missing_env`.

On a SafeChain machine only `llm/providers.json` (provider `safechain`), its transport in `llm/adapters/` and its
environment variables differ.

## Failures

`ProviderError` (with `.status` when the transport reported an HTTP status): unknown provider (it names the ones that
exist), disabled provider, missing environment variable (it names it), unsupported call or option, HTTP errors,
retries exhausted.

## Smoke test and tests

```bash
set -a; . ./.env; set +a
python llm/examples_terminal.py      # real, billed calls; prints content and usage
python -m pytest llm/tests           # never calls a provider (patched transport / local fake host)
```
