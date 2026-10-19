import functools
import os
from collections.abc import Callable
from typing import Any


def _enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def traced(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Langfuse-trace the function when keys are set; otherwise a no-op."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _enabled():
                return fn(*args, **kwargs)
            from langfuse.decorators import observe

            return observe(name=name)(fn)(*args, **kwargs)

        return wrapper

    return decorator


def langchain_callbacks() -> list:
    if not _enabled():
        return []
    from langfuse.callback import CallbackHandler

    return [CallbackHandler()]
