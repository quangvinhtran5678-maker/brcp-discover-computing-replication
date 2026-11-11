"""
Experiment runner.

For every replication the point forecaster is fitted once and the resulting
conformity-score stream is shared by all calibration layers. The comparison is
therefore exactly paired: any difference between methods is attributable to the
calibration layer alone, not to forecast accuracy or to sampling noise in the
data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conformal import ACI, METHODS, BRCP, OracleCP
from .detectors import DataCUSUM, ScoreCUSUM
from .dgp import generate
from .forecasters import FrozenAR, SeasonalNaive
from .metrics import summarize_run, winkler_score


@dataclass
class Config:
    T: int = 1800
    tau: int = 1200
    fit_end: int = 600
    warm_start: int = 600      # first index of the calibration warm-up
    eval_start: int = 1000     # first evaluated forecast origin
    alpha: float = 0.10
    period: int = 24
    lags: int = 24
    window: int = 300
    horizon: int = 500         # post-break offsets retained for the curves
    pre_offsets: int = 150     # pre-break offsets retained for the curves
    forecaster: str = "frozen_ar"
    instance_norm: bool = False


def build_methods(cfg: Config, subset=None, tau_idx: int = 10**9) -> dict:
    """Instantiate one fresh calibration layer per method name."""
    a, w = cfg.alpha, cfg.window

    def brcp(**kw):
        space = kw.pop("detector_space", "score")
        det_cls = ScoreCUSUM if space == "score" else DataCUSUM
        det_kw = dict(anchor_size=250, k=kw.pop("k", 0.65),
                      h=kw.pop("h", 6.0), min_anchor=60, two_sided=True,
                      stat=kw.pop("stat", "std"))
        return BRCP(alpha=a, window=w, detector=det_cls(**det_kw),
                    detector_space=space, period=cfg.period, **kw)

    all_m = {
        "SplitCP":       lambda: METHODS["SplitCP"](alpha=a),
        "RollingCP":     lambda: METHODS["RollingCP"](alpha=a, window=w),
        "ACI(0.005)":    lambda: ACI(alpha=a, window=w, gamma=0.005),
        "ACI(0.05)":     lambda: ACI(alpha=a, window=w, gamma=0.05),
        "AgACI":         lambda: METHODS["AgACI"](alpha=a, window=w),
        "ConformalPID":  lambda: METHODS["ConformalPID"](alpha=a),
        "BRCP":          lambda: brcp(),
        "OracleCP":      lambda: OracleCP(alpha=a, window=w, gamma=0.01,
                                          tau_idx=tau_idx),
        # ---- ablations ----
        "BRCP-data":     lambda: brcp(detector_space="data"),
        "BRCP-sdiff":    lambda: brcp(detector_space="data_sdiff"),
        "BRCP-rank":     lambda: brcp(stat="rank", k=0.10, h=2.5),
        "BRCP-binary":   lambda: brcp(gamma_mode="binary"),
        "BRCP-noreset":  lambda: brcp(use_reset=False),
        "BRCP-noinfl":   lambda: brcp(use_inflation=False),
        "BRCP-noboost":  lambda: brcp(use_boost=False),
    }
    if subset is None:
        return all_m
    return {k: v for k, v in all_m.items() if k in subset}


def run_replication(cfg: Config, break_type: str, mag: float, seed: int,
                    method_names=None):
    """Run all calibration layers on one simulated series."""
    y, tau, meta = generate(T=cfg.T, tau=cfg.tau, break_type=break_type,
                            mag=mag, period=cfg.period, seed=seed)

    fc = (FrozenAR(lags=cfg.lags, period=cfg.period,
                   instance_norm=cfg.instance_norm)
          if cfg.forecaster == "frozen_ar" else SeasonalNaive(period=cfg.period))
    fc.fit(y, cfg.fit_end)

    idx = np.arange(cfg.warm_start, cfg.T)
    yhat = fc.predict_path(y, cfg.warm_start, cfg.T)
    scores = np.abs(y[idx] - yhat)

    n_warm = cfg.eval_start - cfg.warm_start
    warm_scores, warm_obs = scores[:n_warm], y[idx[:n_warm]]
    ev = slice(n_warm, len(idx))
    ev_scores, ev_obs, ev_y, ev_hat = scores[ev], y[idx[ev]], y[idx[ev]], yhat[ev]
    tau_idx = cfg.tau - cfg.eval_start          # break offset inside the eval window

    out = {}
    for name, factory in build_methods(cfg, method_names, tau_idx).items():
        m = factory()
        m.warm_start(warm_scores, warm_obs)
        n = ev_scores.size
        q = np.empty(n)
        err = np.empty(n, dtype=np.int8)
        for i in range(n):
            qi = m.radius()
            q[i] = qi
            err[i] = int(ev_scores[i] > qi)
            m.update(float(ev_scores[i]), int(err[i]), float(ev_obs[i]))
        lo, hi = ev_hat - q, ev_hat + q
        wink = winkler_score(ev_y, lo, hi, cfg.alpha)
        out[name] = dict(
            err=err, width=2.0 * q, wink=wink, tau_idx=tau_idx,
            n_alarms=len(getattr(m, "alarms", [])),
            alarms=list(getattr(m, "alarms", [])),
            summary=summarize_run(err, 2.0 * q, wink, tau_idx, cfg.alpha, cfg.horizon),
        )
    return out, tau_idx, meta


def run_grid(cfg: Config, break_types, mags, seeds, method_names=None,
             verbose: bool = True):
    """Sweep break types x magnitudes x seeds, collecting per-offset curves."""
    rows, curves = [], {}
    for bt in break_types:
        for mag in mags:
            per_method = {}
            for s in seeds:
                res, tau_idx, _ = run_replication(cfg, bt, mag, s, method_names)
                for name, r in res.items():
                    lo = max(0, tau_idx - cfg.pre_offsets)
                    hi = min(len(r["err"]), tau_idx + cfg.horizon)
                    seg = np.full(cfg.pre_offsets + cfg.horizon, np.nan)
                    seg[(lo - tau_idx) + cfg.pre_offsets:
                        (hi - tau_idx) + cfg.pre_offsets] = r["err"][lo:hi]
                    per_method.setdefault(name, []).append(seg)
                    row = dict(break_type=bt, mag=mag, seed=s, method=name,
                               n_alarms=r["n_alarms"], **r["summary"])
                    rows.append(row)
            for name, segs in per_method.items():
                curves[(bt, mag, name)] = np.vstack(segs)
            if verbose:
                print(f"  done: {bt} mag={mag} ({len(seeds)} seeds)", flush=True)
    return rows, curves
