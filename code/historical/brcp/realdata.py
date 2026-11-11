"""
Real benchmark data and a globally pretrained zero-shot forecaster.

Protocol
--------
The point of this module is to reproduce the *deployment protocol* of a
time-series foundation model on real data, not to reproduce a specific
checkpoint. A single global forecaster is fitted once on a pretraining corpus
(electricity, traffic, weather), its parameters are frozen, and it is then
applied to target datasets it has never seen (ETT, exchange rate, influenza).
Nothing about the target series is used to fit it. That is genuine cross-domain
zero-shot transfer, and it produces the property the calibration study needs:
conformity-score streams from real data, with the heavy tails, autocorrelation
and heteroskedasticity that synthetic AR noise does not have.

The forecaster itself is a ridge regression on an instance-normalised context
window, in the spirit of DLinear and of the mean-scaling used by Chronos and
TimesFM. It is deliberately small: it must run for hundreds of thousands of
calibration steps on one CPU core. It is *not* a published foundation-model
checkpoint, and no claim about Chronos or TimesFM follows from results
obtained with it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("BRCP_DATA", PROJECT_ROOT / "data" / "Class-1"))

# (key, relative path, native period in steps)
DATASETS = {
    "ETTh1":       ("ETT-small/ETTh1.csv", 24),
    "ETTh2":       ("ETT-small/ETTh2.csv", 24),
    "ETTm1":       ("ETT-small/ETTm1.csv", 96),
    "ETTm2":       ("ETT-small/ETTm2.csv", 96),
    "exchange":    ("exchange_rate/exchange_rate.csv", 7),
    "illness":     ("illness/national_illness.csv", 52),
    "electricity": ("electricity/electricity.csv", 24),
    "traffic":     ("traffic/traffic.csv", 24),
    "weather":     ("weather/weather.csv", 144),
}

PRETRAIN = ["electricity", "traffic", "weather"]
TARGET = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange", "illness"]


def load(name: str, max_rows: int | None = None, max_cols: int | None = None):
    """Return (columns, array of shape (T, n_series)) for one dataset."""
    path = os.path.join(DATA_ROOT, DATASETS[name][0])
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.lower() == "date"])
    df = df.select_dtypes("number")
    if max_cols is not None and df.shape[1] > max_cols:
        # evenly spaced columns rather than the first k, to avoid picking a
        # block of near-duplicate sensors
        idx = np.linspace(0, df.shape[1] - 1, max_cols).astype(int)
        df = df.iloc[:, idx]
    a = df.to_numpy(dtype=float)
    if max_rows is not None:
        a = a[:max_rows]
    a = np.where(np.isfinite(a), a, np.nan)
    return list(df.columns), a


def usable_series(arr: np.ndarray, min_std_frac: float = 1e-4):
    """Column indices with no NaNs and non-degenerate variation."""
    ok = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if np.any(~np.isfinite(col)):
            continue
        s, m = np.std(col), np.mean(np.abs(col)) + 1e-12
        if s / m < min_std_frac or s == 0:
            continue
        ok.append(j)
    return ok


# --------------------------------------------------------------------------- #
class GlobalZeroShot:
    """Frozen global forecaster on an instance-normalised context window.

    For a context $c = (y_{t-L},\\dots,y_{t-1})$ the model computes
    $\\mu, \\sigma$ from the context alone, regresses the normalised next value
    on the normalised context, and maps the prediction back. Because the
    normalisation is per-window, the model transfers across series with wildly
    different levels and scales without ever seeing them -- which is exactly the
    mechanism that makes level and scale shifts *benign* for calibration, the
    effect studied in Section 7 of the paper.
    """

    def __init__(self, context: int = 96, ridge: float = 1e-2):
        self.context = context
        self.ridge = ridge
        self.beta_: np.ndarray | None = None
        self.n_train_ = 0
        self.n_series_ = 0

    # -- feature construction ------------------------------------------------ #
    def _features(self, ctx: np.ndarray):
        """ctx: (n, L) raw context windows -> (X, mu, sd)."""
        mu = ctx.mean(axis=1, keepdims=True)
        sd = ctx.std(axis=1, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        z = (ctx - mu) / sd
        X = np.hstack([z, np.ones((z.shape[0], 1))])
        return X, mu.ravel(), sd.ravel()

    # -- pretraining --------------------------------------------------------- #
    def fit_corpus(self, series_iter, windows_per_series: int = 1500,
                   seed: int = 0, verbose: bool = True):
        """Accumulate normal equations over a corpus of series."""
        rng = np.random.default_rng(seed)
        L = self.context
        p = L + 1
        A = np.zeros((p, p))
        b = np.zeros(p)
        n_tot = 0
        n_ser = 0
        for name, y in series_iter:
            T = len(y)
            if T < L + 50:
                continue
            origins = rng.choice(np.arange(L, T), size=min(windows_per_series, T - L),
                                 replace=False)
            ctx = np.stack([y[o - L:o] for o in origins])
            tgt = y[origins]
            X, mu, sd = self._features(ctx)
            t = (tgt - mu) / sd
            keep = np.isfinite(t) & np.all(np.isfinite(X), axis=1)
            X, t = X[keep], t[keep]
            A += X.T @ X
            b += X.T @ t
            n_tot += len(t)
            n_ser += 1
            if verbose and n_ser % 100 == 0:
                print(f"    corpus: {n_ser} series, {n_tot} windows", flush=True)
        self.beta_ = np.linalg.solve(A + self.ridge * n_tot * np.eye(p), b)
        self.n_train_, self.n_series_ = n_tot, n_ser
        return self

    # -- zero-shot application ----------------------------------------------- #
    def predict_path(self, y: np.ndarray, start: int, end: int) -> np.ndarray:
        """One-step-ahead forecasts for origins start..end-1."""
        if self.beta_ is None:
            raise RuntimeError("call fit_corpus() first")
        L = self.context
        origins = np.arange(start, end)
        ctx = np.stack([y[o - L:o] for o in origins])
        X, mu, sd = self._features(ctx)
        return (X @ self.beta_) * sd + mu

    def save(self, path: str):
        np.savez_compressed(path, beta=self.beta_, context=self.context,
                            ridge=self.ridge, n_train=self.n_train_,
                            n_series=self.n_series_)

    @classmethod
    def load(cls, path: str) -> "GlobalZeroShot":
        z = np.load(path)
        m = cls(context=int(z["context"]), ridge=float(z["ridge"]))
        m.beta_ = z["beta"]
        m.n_train_, m.n_series_ = int(z["n_train"]), int(z["n_series"])
        return m


class SeasonalNaiveReal:
    """y_hat[t] = y[t - period]; the standard reference forecaster."""

    def __init__(self, period: int):
        self.period = period

    def predict_path(self, y, start, end):
        o = np.arange(start, end)
        return y[o - self.period]


# --------------------------------------------------------------------------- #
def pretrain_col_index(n_cols: int, max_cols: int = 120):
    """The evenly spaced columns `load` selects; used to define the holdout."""
    if n_cols <= max_cols:
        return np.arange(n_cols)
    return np.unique(np.linspace(0, n_cols - 1, max_cols).astype(int))


def corpus_iter(names=PRETRAIN, max_cols=120, max_rows=20000, verbose=True):
    """Yield (label, series) pairs from the pretraining datasets."""
    for name in names:
        cols, arr = load(name, max_rows=max_rows, max_cols=max_cols)
        ok = usable_series(arr)
        if verbose:
            print(f"  corpus {name}: {len(ok)}/{arr.shape[1]} usable series, "
                  f"T={arr.shape[0]}", flush=True)
        for j in ok:
            yield f"{name}:{cols[j]}", arr[:, j]


def corpus_holdout_iter(names=PRETRAIN, max_cols=120, n_holdout=40,
                        max_rows=20000, verbose=True):
    """Series from the corpus datasets that were NOT used for pretraining.

    The detector threshold must be calibrated on data that is neither the
    pretraining set nor the evaluation set. These columns satisfy both: the
    frozen model never saw them, and they are never evaluated.
    """
    for name in names:
        cols_all, arr_all = load(name, max_rows=max_rows, max_cols=None)
        used = set(pretrain_col_index(arr_all.shape[1], max_cols).tolist())
        free = [j for j in range(arr_all.shape[1]) if j not in used]
        free = [j for j in free if j in set(usable_series(arr_all))]
        if not free:
            continue
        pick = free[:: max(1, len(free) // max(n_holdout, 1))][:n_holdout]
        if verbose:
            print(f"  holdout {name}: {len(pick)} unseen series", flush=True)
        for j in pick:
            yield f"{name}:{cols_all[j]}", arr_all[:, j]


def target_series(names=TARGET, max_cols=8, max_rows=None, verbose=True):
    """Yield (dataset, column, series, period) for the held-out targets."""
    for name in names:
        period = DATASETS[name][1]
        cols, arr = load(name, max_rows=max_rows, max_cols=max_cols)
        ok = usable_series(arr)
        if verbose:
            print(f"  target {name}: {len(ok)} series, T={arr.shape[0]}", flush=True)
        for j in ok:
            yield name, cols[j], arr[:, j], period
