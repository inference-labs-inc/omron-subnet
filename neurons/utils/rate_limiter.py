from typing import Callable, ParamSpec, TypeVar
from functools import wraps
import time
import asyncio

P = ParamSpec("P")
T = TypeVar("T")


class RateLimiter:
    _instances = {}

    def __init__(self, period: float):
        self.period = period
        self.last_call = 0.0

    @classmethod
    def get_limiter(cls, func_name: str, period: float) -> "RateLimiter":
        if func_name not in cls._instances:
            cls._instances[func_name] = cls(period)
        return cls._instances[func_name]


def with_rate_limit(period: float):
    """
    Rate limits a function to one call per time period.
    Works with both async and sync functions.

    Args:
        period: Time period in seconds
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        limiter = RateLimiter.get_limiter(func.__name__, period)

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                now = time.time()
                if now - limiter.last_call < period:
                    return None

                limiter.last_call = now
                return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                now = time.time()
                if now - limiter.last_call < period:
                    return None

                limiter.last_call = now
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator
