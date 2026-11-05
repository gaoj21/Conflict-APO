"""Transports. A transport is a module with complete(provider, messages, options) -> LLMResult and, optionally,
complete_batch(provider, items, options) -> [LLMResult] for a native batch endpoint. Raise ProviderError on failure
(retryable=True for rate limits and 5xx). A provider names its own module with "adapter": "<module>"."""
