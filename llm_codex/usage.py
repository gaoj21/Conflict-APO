"""UsageTracker: adds results up and prices them (prices per million tokens, from the environment)."""
import os
import threading


def _price(name, default=0.0):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class UsageTracker:
    """Thread-safe totals over LLMResults. add() returns False for a result with no usage (counted in `uncounted`),
    so an uncounted call is visible rather than counted as free."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = self.uncounted = 0
        self.input_tokens = self.output_tokens = self.cache_read_tokens = self.reasoning_tokens = 0
        self.web_searches = 0

    def add(self, result, web_searches=0):
        with self._lock:
            self.calls += 1
            self.web_searches += web_searches
            u = getattr(result, "usage", None)
            if u is None:
                self.uncounted += 1
                return False
            self.input_tokens += u.input_tokens
            self.output_tokens += u.output_tokens
            self.cache_read_tokens += u.cache_read_tokens
            self.reasoning_tokens += u.reasoning_tokens
            return True

    @property
    def uncached_input_tokens(self):
        return max(self.input_tokens - self.cache_read_tokens, 0)

    @property
    def cost(self):
        """$ at EVO_TOKEN_INPUT_PRICE_PER_1M / EVO_TOKEN_OUTPUT_PRICE_PER_1M; cached input at
        EVO_TOKEN_CACHED_INPUT_RATIO x the input price; web searches at EVO_WEB_SEARCH_PRICE_PER_1K."""
        p_in, p_out = _price("EVO_TOKEN_INPUT_PRICE_PER_1M") / 1e6, _price("EVO_TOKEN_OUTPUT_PRICE_PER_1M") / 1e6
        ratio = _price("EVO_TOKEN_CACHED_INPUT_RATIO", 1.0)
        return (self.uncached_input_tokens * p_in + self.cache_read_tokens * p_in * ratio + self.output_tokens * p_out
                + self.web_searches * _price("EVO_WEB_SEARCH_PRICE_PER_1K") / 1e3)

    def summary(self):
        return dict(calls=self.calls, uncounted=self.uncounted, input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens, cache_read_tokens=self.cache_read_tokens,
                    reasoning_tokens=self.reasoning_tokens, cost=round(self.cost, 6))
