"""Calls: chat / chat_result / batch / batch_result and their awaited versions."""
import asyncio, os, time
from concurrent.futures import ThreadPoolExecutor

from .registry import ProviderError, get_provider

MAX_PARALLEL = int(os.environ.get("LLM_MAX_PARALLEL", "16"))  # bound of the parallel fallback of batch()
# seconds before the 2nd and 3rd attempt; rate limits and 5xx only (env LLM_RETRY_BACKOFF="60,180")
RETRY_BACKOFF = tuple(float(x) for x in os.environ.get("LLM_RETRY_BACKOFF", "5,20").split(","))
OPTIONS = {"temperature", "top_p", "max_tokens", "stop", "timeout", "seed"}


def _options(provider, opts):
    bad = set(opts) - OPTIONS
    if bad:
        raise ProviderError(f"unsupported option(s) {', '.join(sorted(bad))}; sampling settings are "
                            f"{', '.join(sorted(OPTIONS))}")
    return {**(provider.params or {}), **{k: v for k, v in opts.items() if v is not None}}  # params are the floor


def _retrying(fn):
    for attempt in range(len(RETRY_BACKOFF) + 1):
        try:
            return fn()
        except ProviderError as e:
            if not e.retryable or attempt == len(RETRY_BACKOFF):
                if e.retryable:
                    raise ProviderError(f"retries exhausted: {e}", e.status) from e
                raise
            time.sleep(RETRY_BACKOFF[attempt])


def chat_result(provider, messages, **options):
    """One completion -> LLMResult (content, usage or None, provider, model)."""
    p = get_provider(provider)
    t, opts = p.transport(), _options(p, options)
    return _retrying(lambda: t.complete(p, messages, opts))


def chat(provider, messages, **options):
    """One completion -> text."""
    return chat_result(provider, messages, **options).content


def batch_result(provider, items, **options):
    """Many completions -> list of LLMResult in the order given. Uses the provider's native batch endpoint when it
    has one (native_batch: true), otherwise parallel calls bounded by MAX_PARALLEL."""
    items = list(items)
    if not items:
        return []
    p = get_provider(provider)
    t, opts = p.transport(), _options(p, options)
    if p.native_batch and hasattr(t, "complete_batch"):
        out = _retrying(lambda: t.complete_batch(p, items, opts))
        if len(out) != len(items):
            raise ProviderError(f"batch endpoint returned {len(out)} results for {len(items)} requests")
        return out
    with ThreadPoolExecutor(min(MAX_PARALLEL, len(items))) as ex:
        return list(ex.map(lambda m: _retrying(lambda: t.complete(p, m, opts)), items))


def batch(provider, items, **options):
    return [r.content for r in batch_result(provider, items, **options)]


async def achat_result(provider, messages, **options):
    return await asyncio.to_thread(chat_result, provider, messages, **options)


async def achat(provider, messages, **options):
    return (await achat_result(provider, messages, **options)).content


async def abatch_result(provider, items, **options):
    return await asyncio.to_thread(batch_result, provider, items, **options)


async def abatch(provider, items, **options):
    return [r.content for r in await abatch_result(provider, items, **options)]
