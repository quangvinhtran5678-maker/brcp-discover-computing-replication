"""
Evaluation metrics for online conformal calibration under structural breaks.

Standard metrics (marginal coverage, mean width, Winkler score) describe the
*average* behaviour of a calibration layer. They are insensitive to *when* the
miscoverage happens, and online conformal methods are designed precisely to
repay a post-break deficit with a later surplus, so a method can look perfect on
average while being badly miscalibrated exactly when a break has just occurred.

We therefore add two break-conditional metrics computed on the Monte-Carlo
estimate of the instantaneous miscoverage curve

    m_j = P( y_{tau+j} not in C_{tau+j} ),   j = -J, ..., H,

estimated by averaging the miscoverage indicator across independent replications
at a fixed offset j from the break.

Coverage Debt (CD)
    CD = sum_{j=0}^{H} max(0, m_j - alpha)
    Total excess miscoverage accumulated after the break, in units of expected
    excess violations per series. CD = 0 iff the layer never under-covers.

Time-to-Recovery of coverage (TTR-C)
    Smallest j such that m_{j'} <= alpha + eps for every j' in [j, j+w].
    Reported as inf if coverage never recovers within the evaluation horizon.
"""

from __future__ import annotations

import numpy as np


def winkler_score(y, lo, hi, alpha: float):
    """Winkler (interval) score; lower is better."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    w = hi - lo
    below = lo - y
    above = y - hi
    pen = np.where(y < lo, (2.0 / alpha) * below, 0.0) + \
          np.where(y > hi, (2.0 / alpha) * above, 0.0)
    return w + pen


def miscoverage_curve(err_by_seed: np.ndarray) -> np.ndarray:
    """Monte-Carlo estimate of m_j from an array of shape (n_seeds, n_offsets)."""
    return np.nanmean(np.asarray(err_by_seed, dtype=float), axis=0)


def coverage_debt(m: np.ndarray, alpha: float, start: int, horizon: int) -> float:
    """CD over offsets [start, start + horizon)."""
    seg = np.asarray(m, float)[start:start + horizon]
    return float(np.nansum(np.maximum(0.0, seg - alpha)))


def coverage_surplus(m: np.ndarray, alpha: float, start: int, horizon: int) -> float:
    """Excess *over*-coverage (conservativeness) over the same window."""
    seg = np.asarray(m, float)[start:start + horizon]
    return float(np.nansum(np.maximum(0.0, alpha - seg)))


def time_to_recovery(m: np.ndarray, alpha: float, start: int, horizon: int,
                     eps: float = 0.02, w: int = 25) -> float:
    """TTR-C: first offset after `start` from which coverage stays on target."""
    seg = np.asarray(m, float)[start:start + horizon]
    n = seg.size
    for j in range(n - w):
        if np.all(seg[j:j + w] <= alpha + eps):
            return float(j)
    return float("inf")


def time_to_recovery_smooth(m: np.ndarray, alpha: float, start: int, horizon: int,
                            eps: float = 0.02, period: int = 24) -> float:
    """TTR-C on the period-averaged curve.

    Under a seasonal break the miscoverage curve oscillates with the seasonal
    period, so the strict criterion never fires: marginal coverage recovers
    while conditional coverage does not. Averaging over one full period
    isolates recovery of *marginal* coverage; the gap between the two variants
    is itself diagnostic.
    """
    seg = np.asarray(m, float)[start:start + horizon]
    n = seg.size - period
    for j in range(max(n, 0)):
        if np.nanmean(seg[j:j + period]) <= alpha + eps:
            return float(j)
    return float("inf")


def smooth(x: np.ndarray, k: int = 25) -> np.ndarray:
    """Centred moving average, used only for plotting."""
    x = np.asarray(x, float)
    if k <= 1:
        return x
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="same")


def summarize_run(err, width, wink, tau_idx, alpha, horizon):
    """Scalar summaries for one replication."""
    err = np.asarray(err, float)
    width = np.asarray(width, float)
    wink = np.asarray(wink, float)
    pre = slice(0, tau_idx)
    post = slice(tau_idx, min(tau_idx + horizon, err.size))
    return dict(
        cov_all=1.0 - np.mean(err),
        cov_pre=1.0 - np.mean(err[pre]) if tau_idx > 0 else np.nan,
        cov_post=1.0 - np.mean(err[post]),
        width_all=np.mean(width),
        width_post=np.mean(width[post]),
        winkler_all=np.mean(wink),
        winkler_post=np.mean(wink[post]),
    )
