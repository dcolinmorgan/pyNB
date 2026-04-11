"""Deprecation utilities for pyGS.

Policy: deprecated features emit DeprecationWarning for 1 minor version
before removal. For example, something deprecated in 1.1.0 is removed in 1.2.0.
"""

import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(removal_version: str, alternative: str) -> Callable[[F], F]:
    """Mark a function/class as deprecated.

    Args:
        removal_version: Version when this will be removed (e.g. "1.2.0").
        alternative: What to use instead.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{func.__qualname__} is deprecated and will be removed in "
                f"v{removal_version}. Use {alternative} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
