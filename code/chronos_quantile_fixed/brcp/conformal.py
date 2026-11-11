"""
Online conformal calibration layers.

All methods share the same streaming interface, so they can be swapped behind
any point forecaster (surrogate or real TSFM):

    q   = method.radius()          # interval half-width BEFORE observing y_t
    err = int(score > q)           # miscoverage indicator
    method.update(score, err, obs) # state update AFTER observing y_t

The interval is C_t = [yhat_t - q_t, yhat_t + q_t] and the target is
P(y_t in C_t) = 1 - alpha.

Implemented
-----------
SplitCP        static split conformal, calibrated once on pre-break data
RollingCP      rolling-window split conformal at the nominal level
ACI            Adaptive Conformal Inference (Gibbs & Candes, 2021)
AgACI          online aggregation of ACI experts with several step sizes
ConformalPID   quantile tracker + saturating integrator (Angelopoulos et al., 2023)
BRCP           Break-Responsive Conformal Prediction (proposed)

Numerical convention
--------------------
alpha_t is clipped to [0, 1]. When the required quantile level exceeds the
resolution of the calibration window we return its empirical maximum rather
than an infinite interval. This is the standard practical implementation, keeps
mean-width comparisons finite, and an internal counter records some saturation events; these counters
are not present in all historical exported result tables.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist

from .fastbuf import RingBuffer, order_stat, quantile_higher


# --------------------------------------------------------------------------- #
# quantile helpers
# --------------------------------------------------------------------------- #
def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample corrected split-conformal quantile."""
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0:
        return np.inf
    a = min(1.0, max(0.0, float(alpha)))
    # The conformal cutoff is the k-th order statistic, k=ceil((n+1)(1-alpha)).
    # Using numpy.quantile(method='higher') with k/n is OFF BY ONE, because
    # its interpolation index is based on n-1 rather than n.
    k = int(np.ceil((n + 1) * (1.0 - a)))
    if k > n:
        # Practical finite-radius convention already used by the experiment;
        # not the +inf finite-sample conformal convention. See audit notes.
        return float(np.max(s))
    return order_stat(s, max(k, 1))


def _tolerance_order_stat(n: int, alpha: float, delta: float) -> int:
    """Smallest k such that the k-th order statistic of n i.i.d. draws is a
    distribution-free one-sided (1-alpha)-content, (1-delta)-confidence
    tolerance limit (Wilks, 1941).

    F(X_(k)) ~ Beta(k, n-k+1), so the requirement P(F(X_(k)) >= 1-alpha) >=
    1-delta is equivalent to BetaCDF(1-alpha; k, n-k+1) <= delta. The CDF is
    decreasing in k, so a bisection over k is exact.

    Returns 0 if no order statistic of a sample of size n can certify the
    requirement (n too small).
    """
    a = min(max(float(alpha), 1e-6), 1 - 1e-6)
    ks = np.arange(1, n + 1)
    cdf = beta_dist.cdf(1.0 - a, ks, n - ks + 1)   # decreasing in k
    ok = np.flatnonzero(cdf <= delta)
    return int(ks[ok[0]]) if ok.size else 0


_TOL_CACHE: dict[tuple[int, float, float], int] = {}


def tolerance_quantile(scores, alpha: float, delta: float = 0.10) -> float:
    """Tolerance-inspired small-buffer order statistic at content 1-alpha.

    Replaces the point-estimate conformal quantile while the post-break buffer
    is small. The margin over the plain conformal quantile shrinks to zero as
    the buffer fills, so no decay schedule has to be tuned. When the buffer is
    too small for any order statistic to certify the requirement, the empirical
    maximum is returned.
    """
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0:
        return np.inf
    # Stable cache: the order-statistic calculation MUST use the same alpha
    # that defines the cache key. A 3-decimal key with an unrounded value
    # made the result depend on whether BRCP or OracleCP ran first.
    alpha_key = round(float(alpha), 12)
    delta_key = round(float(delta), 12)
    key = (n, alpha_key, delta_key)
    k = _TOL_CACHE.get(key)
    if k is None:
        k = _tolerance_order_stat(n, alpha_key, delta_key)
        _TOL_CACHE[key] = k
    return float(s.max()) if k == 0 else order_stat(s, k)


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
class _Base:
    name = "base"

    def __init__(self, alpha: float = 0.1, **kw):
        self.alpha = alpha
        self.saturated = 0
        self.n_steps = 0
        self.alarms: list[int] = []

    def warm_start(self, scores, obs=None):
        raise NotImplementedError

    def radius(self) -> float:
        raise NotImplementedError

    def update(self, score: float, err: int, obs: float | None = None):
        raise NotImplementedError


class SplitCP(_Base):
    name = "SplitCP"

    def __init__(self, alpha: float = 0.1, **kw):
        super().__init__(alpha)
        self.q = np.inf

    def warm_start(self, scores, obs=None):
        self.q = conformal_quantile(scores, self.alpha)

    def radius(self) -> float:
        return self.q

    def update(self, score: float, err: int, obs: float | None = None):
        self.n_steps += 1


class RollingCP(_Base):
    name = "RollingCP"

    def __init__(self, alpha: float = 0.1, window: int = 300, **kw):
        super().__init__(alpha)
        self.buf = RingBuffer(window)

    def warm_start(self, scores, obs=None):
        self.buf.extend(np.asarray(scores, float)[-self.buf.cap:])

    def radius(self) -> float:
        return conformal_quantile(self.buf.values(), self.alpha)

    def update(self, score: float, err: int, obs: float | None = None):
        self.buf.append(float(score))
        self.n_steps += 1


class ACI(_Base):
    name = "ACI"

    def __init__(self, alpha: float = 0.1, window: int = 300, gamma: float = 0.01, **kw):
        super().__init__(alpha)
        self.gamma = gamma
        self.alpha_t = alpha
        self.buf = RingBuffer(window)

    def warm_start(self, scores, obs=None):
        self.buf.extend(np.asarray(scores, float)[-self.buf.cap:])

    def radius(self) -> float:
        arr = self.buf.values()
        if arr.size == 0:
            return np.inf
        if self.alpha_t <= 0.0:
            self.saturated += 1
            return float(arr.max())
        return conformal_quantile(arr, self.alpha_t)

    def update(self, score: float, err: int, obs: float | None = None):
        self.alpha_t = min(1.0, max(0.0, self.alpha_t + self.gamma * (self.alpha - err)))
        self.buf.append(float(score))
        self.n_steps += 1


class AgACI(_Base):
    """Online aggregation of ACI experts with different step sizes.

    Experts are weighted by an exponentially weighted average forecaster on the
    Winkler interval score, a proper scoring rule for [yhat +/- q].
    """

    name = "AgACI"

    def __init__(self, alpha: float = 0.1, window: int = 300,
                 gammas=(0.001, 0.005, 0.01, 0.05, 0.1), eta_agg: float = 2.0, **kw):
        super().__init__(alpha)
        self.experts = [ACI(alpha=alpha, window=window, gamma=g) for g in gammas]
        self.w = np.ones(len(self.experts)) / len(self.experts)
        self.eta_agg = eta_agg
        self.cum_loss = np.zeros(len(self.experts))
        self._last_q = None
        self._scale = 1.0

    def warm_start(self, scores, obs=None):
        for e in self.experts:
            e.warm_start(scores)
        s = np.asarray(scores, float)
        self._scale = float(np.median(np.abs(s))) + 1e-8

    def radius(self) -> float:
        self._last_q = np.array([e.radius() for e in self.experts])
        return float(np.sum(self.w * self._last_q))

    def update(self, score: float, err: int, obs: float | None = None):
        q = self._last_q
        loss = 2.0 * q + (2.0 / self.alpha) * np.maximum(0.0, score - q)
        self.cum_loss += loss / self._scale
        z = -self.eta_agg * (self.cum_loss - self.cum_loss.min())
        w = np.exp(np.clip(z, -700, 0))
        self.w = w / w.sum()
        for e, qi in zip(self.experts, q):
            e.update(score, int(score > qi))
        self.n_steps += 1


class ConformalPID(_Base):
    """Quantile tracker (P) plus a saturating integrator (I).

    Simplified instantiation of Conformal PID control (Angelopoulos, Candes and
    Tibshirani, 2023) without the learned scorecaster (D) term, so that the
    comparison isolates the online-control component rather than an auxiliary
    forecasting model.
    """

    name = "ConformalPID"

    def __init__(self, alpha: float = 0.1, eta_rel: float = 0.3, k_i: float = 1.0,
                 c_sat: float = 40.0, **kw):
        super().__init__(alpha)
        self.eta_rel = eta_rel
        self.k_i = k_i
        self.c_sat = c_sat
        self.q = 1.0
        self.integ = 0.0
        self.scale = 1.0

    def warm_start(self, scores, obs=None):
        s = np.asarray(scores, float)
        self.q = conformal_quantile(s, self.alpha)
        self.scale = float(np.subtract(*np.percentile(s, [75, 25]))) + 1e-8

    def radius(self) -> float:
        r = self.k_i * self.scale * np.tanh(self.integ / self.c_sat)
        return max(1e-8, self.q + r)

    def update(self, score: float, err: int, obs: float | None = None):
        self.q = max(1e-8, self.q + self.eta_rel * self.scale * (err - self.alpha))
        self.integ += (err - self.alpha)
        self.n_steps += 1


# --------------------------------------------------------------------------- #
# proposed method
# --------------------------------------------------------------------------- #
class BRCP(_Base):
    """Break-Responsive Conformal Prediction.

    Between alarms BR-CP is ordinary ACI with a small, stable step size
    `gamma_slow`. A Page-CUSUM monitors the conformity-score stream against a
    frozen anchor window. On an alarm at time tau the layer

      (R1) uses an evidence-responsive ACI step size: graded acceleration before
           an alarm and, in the evaluated implementation, a short fast phase
           after an alarm,
      (R2) discards pre-change calibration memory using Page's estimated change
           location when reset is enabled,
      (R3) uses a tolerance-inspired order-statistic heuristic while the post-reset buffer
           is small, and
      (R4) re-anchors the detector on post-alarm conformity scores before it is
           re-armed.

    The post-alarm fast phase is retained because it is part of the evaluated
    implementation reported in the main result files. The ablation
    `BRCP-noboost` evaluates its removal and shows that it is not necessary for
    the reported advantage. Other mechanisms can also be switched off for
    ablation.
    """

    name = "BRCP"

    def __init__(self, alpha: float = 0.1, window: int = 300,
                 gamma_slow: float = 0.01, gamma_fast: float = 0.10,
                 boost_len: int = 100, detector=None, tol_delta: float = 0.10,
                 min_buffer: int = 15, use_inflation: bool = True,
                 use_boost: bool = True, use_reset: bool = True,
                 gamma_mode: str = "graded", detector_space: str = "score",
                 period: int = 24, safe_reset: bool = False, **kw):
        super().__init__(alpha)
        self.safe_reset = safe_reset
        self.gamma_mode = gamma_mode
        self.period = period
        self._obs_hist: list[float] = []
        self.alpha_t = alpha
        self.window = window
        self.gamma_slow = gamma_slow
        self.gamma_fast = gamma_fast
        self.boost_len = boost_len
        self.tol_delta = tol_delta
        self.min_buffer = min_buffer
        self.use_inflation = use_inflation
        self.use_boost = use_boost
        self.use_reset = use_reset
        self.detector = detector
        self.detector_space = detector_space
        self.buf = RingBuffer(window)
        self.since_alarm = 10**9
        self._rearm: list[float] = []
        self._buffer_target = window

    # -- lifecycle ---------------------------------------------------------- #
    def _obs_signal(self, obs: float) -> float:
        """Data-space monitoring signal.

        'data'       raw observations
        'data_sdiff' seasonal differences y_t - y_{t-m}, the strongest
                     model-free way to strip the predictable seasonal component
                     before monitoring in data space
        """
        self._obs_hist.append(float(obs))
        if len(self._obs_hist) > self.period + 1:
            self._obs_hist.pop(0)
        if self.detector_space == "data_sdiff":
            if len(self._obs_hist) <= self.period:
                return 0.0
            return float(obs) - self._obs_hist[0]
        return float(obs)

    def warm_start(self, scores, obs=None):
        s = np.asarray(scores, float)
        self.buf.extend(s[-self.window:])
        if self.detector is not None:
            if self.detector_space == "score":
                ref = s
            else:
                o = np.asarray(obs, float)
                ref = (o[self.period:] - o[:-self.period]
                       if self.detector_space == "data_sdiff" else o)
                self._obs_hist = list(o[-(self.period + 1):])
            self.detector.set_anchor(ref)

    def _gamma(self) -> float:
        """Evidence-modulated step size.

        In 'graded' mode the step size interpolates between gamma_slow and
        gamma_fast according to the normalised CUSUM statistic S/h, so that
        adaptation accelerates as evidence for a break accumulates and *before*
        the formal alarm. Under the null S hovers near zero and the step size
        stays at gamma_slow, preserving steady-state stability.
        """
        if self.use_boost and self.since_alarm < self.boost_len:
            return self.gamma_fast
        if self.gamma_mode == "graded" and self.detector is not None \
                and getattr(self.detector, "armed", False):
            s = max(self.detector.S_up, self.detector.S_dn)
            frac = min(1.0, max(0.0, s / max(self.detector.h, 1e-9)))
            return self.gamma_slow + (self.gamma_fast - self.gamma_slow) * frac
        return self.gamma_slow

    # -- streaming interface ------------------------------------------------ #
    def radius(self) -> float:
        arr = self.buf.values()
        if arr.size == 0:
            return np.inf
        if self.alpha_t <= 0.0:
            self.saturated += 1
            return float(arr.max())
        fresh = (self.use_inflation and self.since_alarm < self.window
                 and arr.size < self._buffer_target)
        if fresh:
            return tolerance_quantile(arr, self.alpha_t, self.tol_delta)
        return conformal_quantile(arr, self.alpha_t)

    def update(self, score: float, err: int, obs: float | None = None):
        self.n_steps += 1
        self.since_alarm += 1
        self.alpha_t = min(1.0, max(0.0, self.alpha_t + self._gamma() * (self.alpha - err)))
        self.buf.append(float(score))
        if self.detector is None:
            return

        signal = (float(score) if self.detector_space == "score"
                  else self._obs_signal(obs))
        if self.detector.armed:
            fired = self.detector.update(signal)
        else:
            fired = False
            self._rearm.append(signal)
            if len(self._rearm) >= self.detector.min_anchor:
                self.detector.set_anchor(self._rearm)
                self._rearm = []

        if fired:
            self.alarms.append(self.n_steps)
            self.since_alarm = 0
            # A plain reset re-centres alpha_t on the nominal level, which can
            # *narrow* the interval at the very moment a break was detected and
            # is the main source of BR-CP's sub-nominal marginal coverage. The
            # monotone-safe variant only ever widens on a reset.
            self.alpha_t = (min(self.alpha_t, self.alpha) if self.safe_reset
                            else self.alpha)
            if self.use_reset:
                keep = min(max(self.min_buffer, int(self.detector.delay_est)),
                           len(self.buf))
                self.buf.keep_last(keep)
            self.detector.disarm()
            self._rearm = []


class OracleCP(_Base):
    """Unattainable reference: a genie that is told the exact break time.

    Behaves as ACI(gamma_slow) but performs BR-CP's reset the instant the break
    occurs, with zero detection delay. Its coverage debt is a benchmark for one particular known-break reset policy,
    not an optimum or a lower bound for every causal calibration rule.
    """

    name = "OracleCP"

    def __init__(self, alpha: float = 0.1, window: int = 300, gamma: float = 0.01,
                 tau_idx: int = 10**9, min_buffer: int = 15, tol_delta: float = 0.10,
                 **kw):
        super().__init__(alpha)
        self.alpha_t = alpha
        self.gamma = gamma
        self.window = window
        self.tau_idx = tau_idx
        self.min_buffer = min_buffer
        self.tol_delta = tol_delta
        self.buf = RingBuffer(window)
        self.since_alarm = 10**9
        self._last_q = None

    def warm_start(self, scores, obs=None):
        self.buf.extend(np.asarray(scores, float)[-self.window:])

    def radius(self) -> float:
        arr = self.buf.values()
        if arr.size == 0:
            # the genie knows *when* the break happened, not the new score
            # distribution: it must still learn it from post-break data.
            return self._last_q if self._last_q is not None else np.inf
        if self.alpha_t <= 0.0:
            self.saturated += 1
            q = float(arr.max())
        elif self.since_alarm < self.window and arr.size < self.window:
            q = tolerance_quantile(arr, self.alpha_t, self.tol_delta)
        else:
            q = conformal_quantile(arr, self.alpha_t)
        self._last_q = q
        return q

    def update(self, score: float, err: int, obs: float | None = None):
        if self.n_steps == self.tau_idx:
            self.alarms.append(self.n_steps)
            self.since_alarm = 0
            self.alpha_t = self.alpha
            self.buf.clear()
        self.n_steps += 1
        self.since_alarm += 1
        self.alpha_t = min(1.0, max(0.0, self.alpha_t + self.gamma * (self.alpha - err)))
        self.buf.append(float(score))


METHODS = {
    "SplitCP": SplitCP,
    "OracleCP": OracleCP,
    "RollingCP": RollingCP,
    "ACI": ACI,
    "AgACI": AgACI,
    "ConformalPID": ConformalPID,
    "BRCP": BRCP,
}
