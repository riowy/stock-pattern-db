"""Simple thread-safe token-bucket rate limiter.

Used primarily to keep SEC EDGAR requests conservatively under their fair-use
guidance (never above 10 req/s; we default to 3 req/s -- see
``SEC_REQUESTS_PER_SECOND``).
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._min_interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._last_call: float = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            sleep_for = self._min_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call = time.monotonic()
