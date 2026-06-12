"""Common utilities for the application."""

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from loguru import logger

T = TypeVar("T")


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay

    Example:
        @retry(max_attempts=3, delay=1.0)
        async def unreliable_function():
            # May fail sometimes
            pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {current_delay}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {current_delay}s..."
                    )
                    import time

                    time.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def validate_json(data: Any) -> bool:
    """Validate if data is JSON serializable.

    Args:
        data: Data to validate

    Returns:
        True if JSON serializable, False otherwise

    Example:
        is_valid = validate_json({"key": "value"})  # True
        is_valid = validate_json(object())  # False
    """
    try:
        import json

        json.dumps(data)
        return True
    except (TypeError, ValueError):
        return False


def sanitize_agent_name(name: str) -> str:
    """Sanitize agent name for use in URLs and file paths.

    Args:
        name: Raw agent name

    Returns:
        Sanitized agent name

    Example:
        clean_name = sanitize_agent_name("Agent Alpha!")  # "agent_alpha"
    """
    import re

    # Convert to lowercase and replace non-alphanumeric chars with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", name.lower())
    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    return sanitized.strip("_")
