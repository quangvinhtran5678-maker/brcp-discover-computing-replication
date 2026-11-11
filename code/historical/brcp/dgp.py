"""
Data-generating processes with controlled structural breaks.

Each DGP produces a univariate series y[0..T-1] with a single abrupt break at
index `tau`. The pre-break segment freezes the forecaster (mimicking the
"pretrained, never updated" regime of a zero-shot TSFM); the post-break segment
stresses the calibration layer.

Break types
-----------
level          additive mean shift of mag * sigma
variance       innovation scale multiplied by mag
trend          additive linear drift after tau
season         seasonal amplitude multiplied by mag
ar             AR(1) coefficient pushed towards a near-unit root

Two further types are designed to separate detection in *score space* from
detection in *data space*:

slow_drift     the marginal level moves by mag * sigma, but along a smooth ramp
               with a long time constant. One-step-ahead predictability is
               essentially unchanged, so no recalibration is needed: a data-space
               detector fires (false alarm) while a score-space detector should
               stay silent.
predictability the AR coefficient drops while the innovation scale is rescaled
               so that the *marginal* variance of y is exactly preserved. The
               observation distribution is unchanged, so a data-space detector
               is blind, while one-step-ahead residuals inflate sharply and the
               calibration layer must react.

none           no break (used for false-alarm rate / ARL_0 and steady state)
"""

from __future__ import annotations

import numpy as np

BREAK_TYPES = ("level", "variance", "trend", "season", "ar",
               "slow_drift", "predictability", "none")

# base AR coefficient per break type (predictability needs a high starting phi)
BASE_PHI = {bt: 0.5 for bt in BREAK_TYPES}
BASE_PHI["predictability"] = 0.90


def generate(
    T: int = 1800,
    tau: int = 1200,
    break_type: str = "level",
    mag: float = 3.0,
    period: int = 24,
    phi: float | None = None,
    sigma: float = 1.0,
    amp: float = 2.0,
    slope: float = 0.0,
    drift_tau: float = 120.0,
    seed: int = 0,
):
    """Generate one series with a single abrupt break at index `tau`."""
    if break_type not in BREAK_TYPES:
        raise ValueError(f"unknown break_type {break_type!r}")
    if phi is None:
        phi = BASE_PHI[break_type]

    rng = np.random.default_rng(seed)
    t = np.arange(T)
    post = t >= tau

    seas_amp = np.full(T, amp, dtype=float)
    lvl = np.zeros(T)
    trd = slope * t.astype(float)
    sig = np.full(T, sigma, dtype=float)
    phis = np.full(T, phi, dtype=float)

    if break_type == "level":
        lvl[post] = mag * sigma
    elif break_type == "variance":
        sig[post] = sigma * mag
    elif break_type == "trend":
        trd[post] = trd[post] + mag * sigma * (t[post] - tau) / period
    elif break_type == "season":
        seas_amp[post] = amp * mag
    elif break_type == "ar":
        phis[post] = min(0.98, phi + 0.15 * mag)
    elif break_type == "slow_drift":
        lvl[post] = mag * sigma * (1.0 - np.exp(-(t[post] - tau) / drift_tau))
    elif break_type == "predictability":
        # phi drops; sigma rescaled to hold the marginal variance sigma^2/(1-phi^2)
        phi1 = max(0.05, phi - 0.10 * mag)
        phis[post] = phi1
        sig[post] = sigma * np.sqrt((1.0 - phi1**2) / (1.0 - phi**2))

    seas = seas_amp * np.sin(2.0 * np.pi * t / period)

    e = rng.standard_normal(T) * sig
    x = np.empty(T)
    x[0] = e[0] / max(np.sqrt(1.0 - phis[0] ** 2), 1e-6)
    for i in range(1, T):
        x[i] = phis[i] * x[i - 1] + e[i]

    y = 10.0 + lvl + trd + seas + x
    meta = dict(break_type=break_type, mag=mag, tau=tau, period=period,
                phi=phi, sigma=sigma, amp=amp, slope=slope, seed=seed)
    return y, tau, meta
