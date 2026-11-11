"""
Surrogate zero-shot forecasters.

A time-series foundation model (TSFM) used zero-shot is, operationally, a
*fixed function of a rolling context window*: its parameters are frozen at
pretraining time and never updated on the target series, but its input does
track the most recent observations.

We reproduce exactly that interface with cheap surrogates so that the
calibration layer -- which is the object of study -- can be evaluated over
thousands of Monte-Carlo replications on CPU. The surrogates are *not* claimed
to match TSFM accuracy; they reproduce the property that matters here, namely
that the parameters are frozen while the data-generating process breaks, so the
conformity-score distribution shifts in a way the forecaster cannot absorb.

`scripts/run_tsfm_real.py` swaps these surrogates for real checkpoints
(Chronos-Bolt / TimesFM) behind the same interface.
"""

from __future__ import annotations

import numpy as np


def _design(y: np.ndarray, idx: np.ndarray, lags: int, period: int, n_fourier: int):
    """Build the feature matrix for forecast origins `idx` (predicting y[idx])."""
    cols = [y[idx - l] for l in range(1, lags + 1)]
    for k in range(1, n_fourier + 1):
        cols.append(np.sin(2 * np.pi * k * idx / period))
        cols.append(np.cos(2 * np.pi * k * idx / period))
    cols.append(np.ones_like(idx, dtype=float))
    return np.column_stack(cols)


class FrozenAR:
    """Ridge-regularised linear autoregression with Fourier seasonality.

    Parameters are estimated once on a *pre-break* fitting window and then
    frozen; prediction always uses the true rolling context, so the model is
    'zero-shot' in the TSFM sense (no adaptation of parameters, full access to
    recent observations).

    With `instance_norm=True` the context window is standardised before the
    frozen coefficients are applied and the prediction is mapped back. This
    mirrors the instance normalisation used by essentially every deployed TSFM
    (for example, mean scaling in Chronos and TimesFM) and has an
    important consequence for calibration monitoring: shifts in the *level* and
    *scale* of the series are absorbed by the normaliser and leave the
    conformity-score distribution almost unchanged, so they require no
    recalibration even though they are large, obvious changes in the data.
    """

    def __init__(self, lags: int = 24, period: int = 24, n_fourier: int = 2,
                 ridge: float = 1e-3, instance_norm: bool = False):
        self.lags = lags
        self.period = period
        self.n_fourier = n_fourier
        self.ridge = ridge
        self.instance_norm = instance_norm
        self.beta_: np.ndarray | None = None

    def _context_stats(self, y: np.ndarray, idx: np.ndarray):
        """Mean and sd of the `lags` observations preceding each origin."""
        ctx = np.column_stack([y[idx - l] for l in range(1, self.lags + 1)])
        mu = ctx.mean(axis=1)
        sd = ctx.std(axis=1) + 1e-8
        return mu, sd

    def _design_norm(self, y: np.ndarray, idx: np.ndarray, mu, sd):
        cols = [(y[idx - l] - mu) / sd for l in range(1, self.lags + 1)]
        for k in range(1, self.n_fourier + 1):
            cols.append(np.sin(2 * np.pi * k * idx / self.period))
            cols.append(np.cos(2 * np.pi * k * idx / self.period))
        cols.append(np.ones_like(idx, dtype=float))
        return np.column_stack(cols)

    def fit(self, y: np.ndarray, fit_end: int) -> "FrozenAR":
        idx = np.arange(self.lags, fit_end)
        if self.instance_norm:
            mu, sd = self._context_stats(y, idx)
            X = self._design_norm(y, idx, mu, sd)
            target = (y[idx] - mu) / sd
        else:
            X = _design(y, idx, self.lags, self.period, self.n_fourier)
            target = y[idx]
        A = X.T @ X + self.ridge * np.eye(X.shape[1]) * len(idx)
        self.beta_ = np.linalg.solve(A, X.T @ target)
        return self

    def predict_path(self, y: np.ndarray, start: int, end: int) -> np.ndarray:
        """One-step-ahead forecasts for origins start..end-1 (predicting y[t])."""
        if self.beta_ is None:
            raise RuntimeError("call fit() first")
        idx = np.arange(start, end)
        if self.instance_norm:
            mu, sd = self._context_stats(y, idx)
            return (self._design_norm(y, idx, mu, sd) @ self.beta_) * sd + mu
        X = _design(y, idx, self.lags, self.period, self.n_fourier)
        return X @ self.beta_


class SeasonalNaive:
    """y_hat[t] = y[t - period]. Parameter-free reference forecaster."""

    def __init__(self, period: int = 24):
        self.period = period

    def fit(self, y: np.ndarray, fit_end: int) -> "SeasonalNaive":
        return self

    def predict_path(self, y: np.ndarray, start: int, end: int) -> np.ndarray:
        idx = np.arange(start, end)
        return y[idx - self.period]


FORECASTERS = {"frozen_ar": FrozenAR, "seasonal_naive": SeasonalNaive}
