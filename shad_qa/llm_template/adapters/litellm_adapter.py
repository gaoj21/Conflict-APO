"""Transport through litellm (`model` is a litellm model string such as "openai/gpt-...", "deepseek/...")."""
from ..registry import ProviderError
from ..types import LLMResult, LLMUsage


def complete(p, messages, opts):
    try:
        import litellm
    except ImportError as e:
        raise ProviderError(f"provider '{p.name}' (litellm) needs the litellm package: pip install litellm") from e
    kw = dict(opts)
    if p.api_key:
        kw["api_key"] = p.api_key
    if p.base_url:
        kw["api_base"] = p.base_url
    try:
        r = litellm.completion(model=p.model, messages=messages, **kw)
    except Exception as e:  # noqa: BLE001  litellm maps provider errors onto exceptions carrying status_code
        status = getattr(e, "status_code", None)
        raise ProviderError(f"provider '{p.name}': {type(e).__name__}: {str(e)[:500]}", status,
                            retryable=status is not None and (status == 429 or status >= 500)) from e
    try:
        content = r.choices[0].message.content or ""
    except (AttributeError, IndexError) as e:
        raise ProviderError(f"provider '{p.name}': unexpected response {str(r)[:300]}") from e
    return LLMResult(content, LLMUsage.from_raw(getattr(r, "usage", None)), p.name, getattr(r, "model", None) or p.model)
