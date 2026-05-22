"""
Thread-safe rolling-window rate limiter for AlphaLens.

This module is primarily used to protect Financial Modeling Prep API calls
from exceeding the configured requests-per-minute limit.

The implementation uses a rolling time window rather than fixed calendar
minutes. This is safer because it prevents request bursts at minute boundaries.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field

from src.config import settings


class RateLimitExceededError(Exception):
    """
    Raised when a rate limit would be exceeded and blocking is disabled.
    """

    pass


@dataclass
class RollingWindowRateLimiter:
    """
    Thread-safe rolling-window rate limiter.

    Args:
        max_requests: Maximum number of requests allowed within the window.
        window_seconds: Size of the rolling time window in seconds.

    Example:
        limiter = RollingWindowRateLimiter(max_requests=700, window_seconds=60)
        limiter.acquire()
    """

    max_requests: int
    window_seconds: float = 60.0
    _timestamps: deque[float] = field(default_factory=deque, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        """
        Validate limiter settings after initialization.
        """

        if self.max_requests <= 0:
            raise ValueError("max_requests must be greater than 0.")

        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be greater than 0.")

    def acquire(self, block: bool = True) -> None:
        """
        Acquire permission to make one request.

        If the number of requests within the rolling window is below the limit,
        this method records the request timestamp and returns immediately.

        If the limit has been reached:
            - block=True: waits until a request slot becomes available.
            - block=False: raises RateLimitExceededError.

        Args:
            block: Whether to wait when the limit has been reached.

        Raises:
            RateLimitExceededError: If limit is reached and block=False.
        """

        while True:
            wait_seconds = 0.0

            with self._lock:
                now = time.monotonic()
                self._remove_expired_timestamps(now)

                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return

                if not block:
                    raise RateLimitExceededError(
                        f"Rate limit exceeded: {self.max_requests} requests "
                        f"per {self.window_seconds} seconds."
                    )

                oldest_timestamp = self._timestamps[0]
                wait_seconds = self.window_seconds - (now - oldest_timestamp)

            # Sleep outside the lock so other threads are not blocked from
            # checking the limiter while this thread waits.
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            else:
                # Defensive fallback; the next loop should remove expired items.
                time.sleep(0.001)

    def remaining_capacity(self) -> int:
        """
        Return the number of request slots currently available.

        This is useful for debugging, testing, or optional UI diagnostics.
        """

        with self._lock:
            now = time.monotonic()
            self._remove_expired_timestamps(now)
            return max(self.max_requests - len(self._timestamps), 0)

    def reset(self) -> None:
        """
        Clear all tracked request timestamps.

        This is mostly useful for tests.
        """

        with self._lock:
            self._timestamps.clear()

    def _remove_expired_timestamps(self, now: float) -> None:
        """
        Remove timestamps that are outside the rolling window.

        Args:
            now: Current monotonic timestamp.
        """

        while self._timestamps and now - self._timestamps[0] >= self.window_seconds:
            self._timestamps.popleft()


# Shared limiter instance for FMP API calls.
#
# The FMP plan limit is 750 requests/minute, but AlphaLens uses the configured
# safety buffer from src/config.py, defaulting to 700 requests/minute.
fmp_rate_limiter = RollingWindowRateLimiter(
    max_requests=settings.fmp_max_requests_per_minute,
    window_seconds=60.0,
)