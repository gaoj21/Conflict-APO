"""Provider configuration: llm/providers.json plus the environment (process env and the repository .env)."""
import importlib, json, os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONFIG = os.path.join(HERE, "providers.json")
BUILTIN = {"litellm": "llm.adapters.litellm_adapter", "openai_compatible": "llm.adapters.openai_compatible"}


class ProviderError(Exception):
    """Every failure this package raises. `status` is the HTTP status when the transport reported one."""

    def __init__(self, message, status=None, retryable=False):
        super().__init__(message)
        self.status, self.retryable = status, retryable


def _dotenv():
    """KEY=VALUE lines of the repository .env (the process environment wins)."""
    out, path = {}, os.path.join(REPO, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            out[k] = v.strip().strip('"').strip("'")
    return out


def env(name):
    return os.environ.get(name) or _dotenv().get(name)


@dataclass
class Provider:
    name: str
    type: "str | None" = None
    adapter: "str | None" = None
    model: "str | None" = None
    base_url: "str | None" = None
    api_key_env: "str | None" = None
    requires: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    enabled: bool = True
    native_batch: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def needed_env(self):
        return ([self.api_key_env] if self.api_key_env else []) + [r for r in self.requires if r != self.api_key_env]

    @property
    def missing_env(self):
        return [v for v in self.needed_env if not env(v)]

    @property
    def api_key(self):
        return env(self.api_key_env) if self.api_key_env else None

    def transport(self):
        mod = self.adapter or BUILTIN.get(self.type or "")
        if not mod:
            raise ProviderError(f"provider '{self.name}': unknown type {self.type!r} and no adapter "
                                f"(built-in types: {', '.join(BUILTIN)})")
        if "." not in mod:
            mod = f"llm.adapters.{mod}"
        try:
            return importlib.import_module(mod)
        except ImportError as e:
            raise ProviderError(f"provider '{self.name}': cannot load transport {mod}: {e}") from e


_KNOWN = {"type", "adapter", "model", "base_url", "api_key_env", "requires", "params", "enabled", "native_batch"}


def _load():
    if not os.path.exists(CONFIG):
        raise ProviderError(f"no provider configuration at {CONFIG}")
    with open(CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    provs = {}
    for name, p in (cfg.get("providers") or {}).items():
        provs[name] = Provider(name=name, **{k: v for k, v in p.items() if k in _KNOWN},
                               extra={k: v for k, v in p.items() if k not in _KNOWN})
    return cfg.get("default"), provs


def default_provider():
    """Name of the provider a call with provider=None goes to: EAX_PROVIDER, else LLM_PROVIDER, else `default`."""
    return os.environ.get("EAX_PROVIDER") or os.environ.get("LLM_PROVIDER") or _load()[0]


def get_provider(name=None):
    """The configured Provider `name` (None = the default), checked to be usable."""
    default, provs = _load()
    name = name or os.environ.get("EAX_PROVIDER") or os.environ.get("LLM_PROVIDER") or default
    if not name:
        raise ProviderError("no provider named and no default in providers.json")
    if name not in provs:
        raise ProviderError(f"unknown provider '{name}'; configured: {', '.join(provs) or '(none)'}")
    p = provs[name]
    if not p.enabled:
        raise ProviderError(f"provider '{name}' is disabled (enabled: false in providers.json)")
    if p.missing_env:
        raise ProviderError(f"provider '{name}' needs environment variable(s) {', '.join(p.missing_env)}")
    return p


def list_providers():
    """Every configured provider with `available` (its variables are present) and `missing_env`."""
    default, provs = _load()
    active = os.environ.get("EAX_PROVIDER") or os.environ.get("LLM_PROVIDER") or default
    return [dict(name=n, type=p.type or p.adapter, model=p.model, enabled=p.enabled, default=(n == active),
                 available=p.enabled and not p.missing_env, missing_env=p.missing_env) for n, p in provs.items()]
