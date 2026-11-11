"""Small performance helpers for the streaming inner loop.

The Monte-Carlo design requires on the order of 10^7 calibration steps, each of
which needs one order statistic of the current calibration window. Using a
numpy ring buffer and an introselect partition instead of a deque plus a full
sort removes the dominant cost without changing any result: the returned value
is bit-identical to numpy.quantile(..., method="higher").
"""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """Fixed-capacity FIFO over a preallocated numpy array."""

    __slots__ = ("cap", "_a", "_n", "_i")

    def __init__(self, capacity: int):
        self.cap = int(capacity)
        self._a = np.empty(self.cap, dtype=float)
        self._n = 0     # number of valid elements
        self._i = 0     # next write position

    def __len__(self) -> int:
        return self._n

    def append(self, x: float) -> None:
        self._a[self._i] = x
        self._i = (self._i + 1) % self.cap
        if self._n < self.cap:
            self._n += 1

    def values(self) -> np.ndarray:
        """Contents in arbitrary order (order is irrelevant for quantiles)."""
        if self._n < self.cap:
            return self._a[:self._n]
        return self._a

    def ordered(self) -> np.ndarray:
        """Contents in FIFO order (oldest first)."""
        if self._n < self.cap:
            return self._a[:self._n].copy()
        return np.concatenate([self._a[self._i:], self._a[:self._i]])

    def keep_last(self, k: int) -> None:
        """Discard all but the k most recent elements."""
        k = max(0, min(int(k), self._n))
        last = self.ordered()[-k:] if k else np.empty(0)
        self._a[:] = 0.0
        self._n, self._i = 0, 0
        for v in last:
            self.append(float(v))

    def clear(self) -> None:
        self._n, self._i = 0, 0

    def extend(self, xs) -> None:
        for v in np.asarray(xs, dtype=float):
            self.append(float(v))


def quantile_higher(a: np.ndarray, lev: float) -> float:
    """numpy.quantile(a, lev, method='higher') in O(n) via introselect."""
    n = a.size
    if n == 0:
        return np.inf
    if lev <= 0.0:
        return float(a.min())
    if lev >= 1.0:
        return float(a.max())
    idx = int(np.ceil(lev * (n - 1)))
    idx = min(max(idx, 0), n - 1)
    return float(np.partition(a, idx)[idx])


def order_stat(a: np.ndarray, k: int) -> float:
    """k-th smallest element (1-indexed), in O(n)."""
    n = a.size
    if n == 0:
        return np.inf
    k = min(max(int(k), 1), n)
    return float(np.partition(a, k - 1)[k - 1])
