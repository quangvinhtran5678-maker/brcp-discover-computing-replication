"""
Confirmatory study on real time-series foundation models.

This is the script referenced in Section 8 of the paper. It has NOT been run
for the results reported there; the paper's numbers all come from
`scripts/run_simulation.py`. Running this script is what turns the simulation
study into an empirical claim about foundation models.

Design
------
Stage 1 caches one-step (or h-step) zero-shot forecasts from a TSFM checkpoint
for every series, once. Stage 2 replays the resulting conformity-score stream
through every calibration layer. Splitting the two matters for three reasons:
the expensive part runs once; every calibration layer sees an *identical*
score stream, so comparisons stay exactly paired as in the simulation; and the
cache is the reproducibility artefact a reviewer can check.

Usage
-----
    # 1. cache forecasts (slow; the only step that needs the model)
    python scripts/run_tsfm_real.py cache \
        --model chronos-bolt-small --data data/ett_h1.csv --col OT \
        --context 512 --horizon 1 --out cache/ett_h1_chronos.npz

    # 2. calibrate and evaluate (fast, CPU-only, no model needed)
    python scripts/run_tsfm_real.py evaluate \
        --cache cache/ett_h1_chronos.npz --alpha 0.1 --out results/real_ett.csv

    # 2b. with externally known break times (recommended where available)
    python scripts/run_tsfm_real.py evaluate --cache ... --breaks 3120,7841

    # 3. recalibrate the detector threshold for this model family
    python scripts/run_tsfm_real.py tune-detector \
        --cache cache/ett_h1_chronos.npz --arl-min 1500

Install (CPU is sufficient for the small checkpoints)
-----------------------------------------------------
    pip install "chronos-forecasting" torch --index-url \
        https://download.pytorch.org/whl/cpu
    pip install timesfm            # optional

Notes on protocol integrity
---------------------------
* `--holdout-frac` reserves the first part of each series for detector
  calibration only. The alarm threshold must never be tuned on the evaluation
  segment; the script refuses to do so.
* Datasets that appear in a checkpoint's pretraining corpus are not zero-shot.
  Check the model card before claiming a zero-shot result.
* Where the break time is not externally known, coverage debt is undefined.
  Use `--inject` to add synthetic breaks to real series instead of guessing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brcp.conformal import ACI, METHODS, BRCP, OracleCP        # noqa: E402
from brcp.detectors import DataCUSUM, ScoreCUSUM               # noqa: E402
from brcp.metrics import (coverage_debt, miscoverage_curve,    # noqa: E402
                          time_to_recovery, time_to_recovery_smooth,
                          winkler_score)


# --------------------------------------------------------------------------- #
# Stage 1: cache zero-shot forecasts
# --------------------------------------------------------------------------- #
def load_series(path: str, col: str | None) -> np.ndarray:
    df = pd.read_csv(path)
    if col is None:
        num = df.select_dtypes("number")
        if num.shape[1] == 0:
            raise SystemExit(f"no numeric column in {path}")
        col = num.columns[-1]
        print(f"[cache] no --col given, using '{col}'")
    return df[col].astype(float).to_numpy()


def forecast_chronos(y, context, horizon, model_name):
    """One-step-ahead rolling forecasts from a Chronos / Chronos-Bolt checkpoint."""
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(
        f"amazon/{model_name}", device_map="cpu", torch_dtype=torch.float32)
    origins = np.arange(context, len(y) - horizon + 1)
    preds = np.empty(len(origins))
    batch = 64
    for i in range(0, len(origins), batch):
        idx = origins[i:i + batch]
        ctx = [torch.tensor(y[j - context:j], dtype=torch.float32) for j in idx]
        q, mean = pipe.predict_quantiles(
            context=ctx, prediction_length=horizon, quantile_levels=[0.5])
        preds[i:i + len(idx)] = mean[:, horizon - 1].numpy()
        if (i // batch) % 20 == 0:
            print(f"  {i}/{len(origins)}", flush=True)
    return origins, preds


def forecast_timesfm(y, context, horizon, model_name):
    import timesfm
    tfm = timesfm.TimesFm(hparams=timesfm.TimesFmHparams(
        backend="cpu", per_core_batch_size=32, horizon_len=horizon,
        context_len=context))
    tfm.load_from_checkpoint(timesfm.TimesFmCheckpoint(huggingface_repo_id=model_name))
    origins = np.arange(context, len(y) - horizon + 1)
    preds = np.empty(len(origins))
    batch = 32
    for i in range(0, len(origins), batch):
        idx = origins[i:i + batch]
        ctx = [y[j - context:j] for j in idx]
        pt, _ = tfm.forecast(ctx, freq=[0] * len(ctx))
        preds[i:i + len(idx)] = np.asarray(pt)[:, horizon - 1]
    return origins, preds


def forecast_naive(y, context, horizon, period):
    """Seasonal-naive fallback so the pipeline can be smoke-tested offline."""
    origins = np.arange(max(context, period), len(y) - horizon + 1)
    return origins, y[origins - period + horizon - 1]


FORECASTERS = {
    "chronos-bolt-tiny":  lambda y, c, h: forecast_chronos(y, c, h, "chronos-bolt-tiny"),
    "chronos-bolt-small": lambda y, c, h: forecast_chronos(y, c, h, "chronos-bolt-small"),
    "chronos-bolt-base":  lambda y, c, h: forecast_chronos(y, c, h, "chronos-bolt-base"),
    "timesfm-2.0":        lambda y, c, h: forecast_timesfm(y, c, h,
                                                           "google/timesfm-2.0-500m-pytorch"),
    "seasonal-naive":     lambda y, c, h: forecast_naive(y, c, h, 24),
}


def inject_breaks(y: np.ndarray, kind: str, mag: float, n: int, rng,
                  seg_len: int = 400) -> tuple:
    """Add synthetic breaks to a real series so that tau is known exactly.

    Each break is applied over a bounded segment of `seg_len` steps and then
    reverted, so the n breaks are independent events of the *same* size rather
    than a cumulative drift. That is what makes the windows around them
    stackable as replications when estimating the miscoverage curve. Breaks are
    separated by at least 2 * seg_len.
    """
    y = y.copy()
    n_pre = int(0.4 * len(y))
    lo, hi = n_pre, len(y) - seg_len - 50
    if hi <= lo:
        raise SystemExit("series too short for the requested injection")
    gap = 2 * seg_len
    cand = np.arange(lo, hi, gap)
    if len(cand) < n:
        raise SystemExit(f"series admits at most {len(cand)} separated breaks; "
                         f"lower --inject-n or --seg-len")
    taus = np.sort(rng.choice(cand, size=n, replace=False))
    sd = float(np.std(np.diff(y)))
    for tau in taus:
        end = min(tau + seg_len, len(y))
        if kind == "level":
            y[tau:end] += mag * sd
        elif kind == "variance":
            extra = np.sqrt(max(mag**2 - 1.0, 0.0)) * sd
            y[tau:end] += rng.standard_normal(end - tau) * extra
        else:
            raise SystemExit(f"unknown --inject {kind}")
    return y, list(map(int, taus))


def cmd_cache(a):
    y = load_series(a.data, a.col)
    taus = []
    if a.inject:
        rng = np.random.default_rng(a.seed)
        y, taus = inject_breaks(y, a.inject, a.inject_mag, a.inject_n, rng)
        print(f"[cache] injected {a.inject} breaks at {taus}")
    if a.model not in FORECASTERS:
        raise SystemExit(f"--model must be one of {sorted(FORECASTERS)}")
    print(f"[cache] {a.model}: {len(y)} points, context={a.context}, "
          f"horizon={a.horizon}")
    origins, preds = FORECASTERS[a.model](y, a.context, a.horizon)
    targets = y[origins + a.horizon - 1]
    scores = np.abs(targets - preds)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out, y=y, origins=origins, preds=preds, targets=targets, scores=scores,
        taus=np.array(taus, dtype=int),
        meta=json.dumps(dict(model=a.model, data=a.data, col=a.col,
                             context=a.context, horizon=a.horizon,
                             inject=a.inject, inject_mag=a.inject_mag)))
    print(f"[cache] wrote {a.out}: {len(scores)} scores, "
          f"MAE={scores.mean():.4f}")


# --------------------------------------------------------------------------- #
# Stage 2: calibrate and evaluate
# --------------------------------------------------------------------------- #
def build_layers(alpha, window, period, tau_idx, k, h):
    def brcp(**kw):
        space = kw.pop("detector_space", "score")
        cls = ScoreCUSUM if space == "score" else DataCUSUM
        det = cls(anchor_size=250, k=kw.pop("k", k), h=kw.pop("h", h),
                  min_anchor=60, two_sided=True, stat="std")
        return BRCP(alpha=alpha, window=window, detector=det,
                    detector_space=space, period=period, **kw)

    layers = {
        "SplitCP":      lambda: METHODS["SplitCP"](alpha=alpha),
        "RollingCP":    lambda: METHODS["RollingCP"](alpha=alpha, window=window),
        "AgACI":        lambda: METHODS["AgACI"](alpha=alpha, window=window),
        "ConformalPID": lambda: METHODS["ConformalPID"](alpha=alpha),
        "BRCP":         lambda: brcp(),
        "BRCP-data":    lambda: brcp(detector_space="data"),
    }
    # ACI at several step sizes: the paper's primary baseline is the *best*
    # per-dataset gamma, which is an oracle advantage BR-CP does not get.
    for g in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1):
        layers[f"ACI({g})"] = (lambda g=g: ACI(alpha=alpha, window=window, gamma=g))
    if tau_idx is not None:
        layers["OracleCP"] = lambda: OracleCP(alpha=alpha, window=window,
                                              gamma=0.01, tau_idx=tau_idx)
    return layers


def run_layer(layer, scores, preds, targets, obs, alpha, warm):
    layer.warm_start(scores[:warm], obs[:warm])
    n = len(scores) - warm
    q = np.empty(n)
    err = np.empty(n, dtype=np.int8)
    for i in range(n):
        j = warm + i
        q[i] = layer.radius()
        err[i] = int(scores[j] > q[i])
        layer.update(float(scores[j]), int(err[i]), float(obs[j]))
    lo, hi = preds[warm:] - q, preds[warm:] + q
    return dict(err=err, width=2 * q,
                wink=winkler_score(targets[warm:], lo, hi, alpha),
                alarms=list(getattr(layer, "alarms", [])))


def cmd_evaluate(a):
    z = np.load(a.cache, allow_pickle=True)
    scores, preds, targets = z["scores"], z["preds"], z["targets"]
    obs = targets
    taus = list(z["taus"]) if z["taus"].size else []
    if a.breaks:
        taus = [int(v) for v in a.breaks.split(",")]
    meta = json.loads(str(z["meta"]))
    warm = a.warm
    eps = 2.0 * np.sqrt(a.alpha * (1 - a.alpha) / max(len(taus), 1))

    # break offsets inside the evaluation window
    origins = z["origins"]
    tau_idx = []
    for t in taus:
        k = int(np.searchsorted(origins, t)) - warm
        if 0 <= k < len(scores) - warm:
            tau_idx.append(k)
    single = tau_idx[0] if len(tau_idx) == 1 else None

    layers = build_layers(a.alpha, a.window, a.period, single, a.k, a.h)
    rows, curves = [], {}
    for name, factory in layers.items():
        r = run_layer(factory(), scores, preds, targets, obs, a.alpha, warm)
        err, width, wink = r["err"], r["width"], r["wink"]
        rec = dict(model=meta["model"], data=os.path.basename(meta["data"]),
                   method=name, n_eval=len(err),
                   coverage=1 - err.mean(), width=width.mean(),
                   winkler=wink.mean(), n_alarms=len(r["alarms"]))
        if tau_idx:
            # stack the windows around every known break: replications in the
            # sense the diagnostics require
            P, H = a.pre, a.horizon_cd
            segs = []
            for k in tau_idx:
                if k - P >= 0 and k + H <= len(err):
                    segs.append(err[k - P:k + H])
            if segs:
                mat = np.vstack(segs)
                curves[name] = mat
                m = miscoverage_curve(mat)
                rec.update(
                    n_breaks=len(segs),
                    CD=float(np.sum(np.maximum(0.0, m[P:] - a.alpha - eps))),
                    TTRC=time_to_recovery(m, a.alpha, P, H, eps=eps, w=25),
                    TTRC_marg=time_to_recovery_smooth(m, a.alpha, P, H, eps=eps,
                                                      period=a.period),
                    miscov_early=float(np.mean(m[P:P + 50])))
        rows.append(rec)
        print(f"  {name:16s} cov={rec['coverage']:.3f} width={rec['width']:.3f} "
              f"CD={rec.get('CD', float('nan')):.2f} alarms={rec['n_alarms']}",
              flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    if curves:
        np.savez_compressed(a.out.replace(".csv", "_curves.npz"), **curves)
    if "CD" in df:
        aci = df[df.method.str.startswith("ACI(")]
        best = aci.loc[aci.CD.idxmin()]
        brcp = df[df.method == "BRCP"].iloc[0]
        print(f"\nbest tuned ACI: {best.method} CD={best.CD:.2f}   "
              f"BR-CP CD={brcp.CD:.2f}   "
              f"{'PASS' if brcp.CD < best.CD else 'FAIL'} "
              "(pre-specified criterion)")
    print(f"\nwrote {a.out}")


# --------------------------------------------------------------------------- #
# Stage 3: recalibrate the detector for this model family
# --------------------------------------------------------------------------- #
def cmd_tune_detector(a):
    """Choose (k, h) on a break-free holdout so ARL_0 meets the target."""
    z = np.load(a.cache, allow_pickle=True)
    scores = z["scores"]
    taus = list(z["taus"])
    cut = int(a.holdout_frac * len(scores))
    if taus and min(taus) < cut:
        raise SystemExit("holdout overlaps a known break; lower --holdout-frac")
    hold = scores[:cut]
    if len(hold) < 600:
        raise SystemExit("holdout too short to estimate ARL_0 reliably")

    rows = []
    for k in (0.4, 0.5, 0.65, 0.8):
        for h in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
            det = ScoreCUSUM(anchor_size=250, k=k, h=h, min_anchor=60, stat="std")
            det.set_anchor(hold[:250])
            alarms, rearm, steps = 0, [], 0
            for s in hold[250:]:
                steps += 1
                if det.armed:
                    if det.update(float(s)):
                        alarms += 1
                        det.disarm()
                        rearm = []
                else:
                    rearm.append(float(s))
                    if len(rearm) >= det.min_anchor:
                        det.set_anchor(rearm)
                        rearm = []
            rows.append(dict(k=k, h=h, arl0=steps / max(alarms, 0.5),
                             alarms=alarms, steps=steps))
    df = pd.DataFrame(rows)
    ok = df[df.arl0 >= a.arl_min]
    print(df.to_string(index=False))
    if len(ok):
        sel = ok.loc[ok.h.idxmin()]      # fastest detector meeting the budget
    else:
        sel = df.loc[df.arl0.idxmax()]
        print(f"\nWARNING: no (k,h) reached ARL_0 >= {a.arl_min:.0f} on this "
              f"holdout ({len(hold)} points). Either the holdout is too short "
              f"to resolve it, or it is not break-free. Falling back to the "
              f"most conservative cell; treat alarm counts with caution.")
    print(f"\nselected: k={sel.k}, h={sel.h}, ARL_0={sel.arl0:.0f}")
    print(f"pass these to `evaluate` as --k {sel.k} --h {sel.h}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="cache zero-shot TSFM forecasts")
    c.add_argument("--model", required=True, choices=sorted(FORECASTERS))
    c.add_argument("--data", required=True)
    c.add_argument("--col", default=None)
    c.add_argument("--context", type=int, default=512)
    c.add_argument("--horizon", type=int, default=1)
    c.add_argument("--inject", default=None, choices=["level", "variance"])
    c.add_argument("--inject-mag", type=float, default=3.0)
    c.add_argument("--inject-n", type=int, default=20)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_cache)

    e = sub.add_parser("evaluate", help="run all calibration layers on a cache")
    e.add_argument("--cache", required=True)
    e.add_argument("--alpha", type=float, default=0.1)
    e.add_argument("--window", type=int, default=300)
    e.add_argument("--period", type=int, default=24)
    e.add_argument("--warm", type=int, default=400)
    e.add_argument("--pre", type=int, default=150)
    e.add_argument("--horizon-cd", type=int, default=200)
    e.add_argument("--breaks", default=None,
                   help="comma-separated externally known break indices")
    e.add_argument("--k", type=float, default=0.65)
    e.add_argument("--h", type=float, default=6.0)
    e.add_argument("--out", default="results/real_eval.csv")
    e.set_defaults(func=cmd_evaluate)

    t = sub.add_parser("tune-detector", help="recalibrate (k,h) on a holdout")
    t.add_argument("--cache", required=True)
    t.add_argument("--holdout-frac", type=float, default=0.3)
    t.add_argument("--arl-min", type=float, default=1500.0)
    t.set_defaults(func=cmd_tune_detector)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
