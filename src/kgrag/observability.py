import functools
import logging
import os
from collections.abc import Callable
from typing import Any


def _enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def traced(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Langfuse-trace the function when keys are set; otherwise a no-op."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        observed: Callable[..., Any] | None = None

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal observed
            if not _enabled():
                return fn(*args, **kwargs)
            if observed is None:
                from langfuse import observe

                observed = observe(name=name)(fn)
            return observed(*args, **kwargs)

        return wrapper

    return decorator


def langchain_callbacks() -> list:
    if not _enabled():
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def tag_current_trace(**tags: str) -> None:
    """Tag the active Langfuse trace (e.g. system, question_id, run_id); no-op without keys.

    Tracing is optional, so a tagging failure is logged, never raised.
    """
    if not _enabled():
        return
    try:
        from langfuse import get_client

        get_client().update_current_trace(
            tags=[f"{k}:{v}" for k, v in tags.items()], metadata=dict(tags)
        )
    except Exception:  # noqa: BLE001 - observability must not break an eval run
        logging.getLogger(__name__).warning("could not tag Langfuse trace", exc_info=True)
