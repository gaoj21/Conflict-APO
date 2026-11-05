"""What a call returns: LLMResult and LLMUsage."""
from dataclasses import dataclass

# provider spellings of the same counts
_IN = ("input_tokens", "prompt_tokens")
_OUT = ("output_tokens", "completion_tokens")


def _first(d, names):
    for n in names:
        v = d.get(n)
        if v is not None:
            return int(v)
    return None


def _get(d, key):
    v = d.get(key) if isinstance(d, dict) else getattr(d, key, None)
    return v if isinstance(v, dict) else (_as_dict(v) if v is not None else {})


def _as_dict(obj):
    if obj is None or isinstance(obj, dict):
        return obj
    for m in ("model_dump", "dict", "to_dict"):
        if hasattr(obj, m):
            try:
                return getattr(obj, m)()
            except Exception:  # noqa: BLE001
                pass
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_") and not callable(getattr(obj, k, None))}


@dataclass
class LLMUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def uncached_input_tokens(self) -> int:
        return max(self.input_tokens - self.cache_read_tokens, 0)

    @classmethod
    def from_raw(cls, raw):
        """Provider usage (dict or SDK object, any of the usual field names) -> LLMUsage, or None when the provider
        reported nothing. Never a row of zeroes for a missing report."""
        d = _as_dict(raw)
        if not d:
            return None
        it, ot = _first(d, _IN), _first(d, _OUT)
        if it is None and ot is None:
            return None
        it, ot = it or 0, ot or 0
        tot = _first(d, ("total_tokens",))
        cached = (_first(_get(d, "prompt_tokens_details"), ("cached_tokens",))
                  or _first(_get(d, "input_tokens_details"), ("cached_tokens",))
                  or _first(d, ("cache_read_input_tokens", "cache_read_tokens", "cached_tokens", "cached_input_tokens")) or 0)
        reasoning = (_first(_get(d, "completion_tokens_details"), ("reasoning_tokens",))
                     or _first(_get(d, "output_tokens_details"), ("reasoning_tokens",))
                     or _first(d, ("reasoning_tokens", "reasoning_output_tokens")) or 0)
        return cls(it, ot, tot if tot is not None else it + ot, cached, reasoning)


@dataclass
class LLMResult:
    content: str
    usage: "LLMUsage | None"  # None means the provider reported none
    provider: "str | None" = None
    model: "str | None" = None
