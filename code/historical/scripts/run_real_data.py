"""
Confirmatory study on real benchmark data.

Pipeline
--------
1.  A global forecaster is pretrained once on electricity + traffic + weather
    and frozen (`brcp/realdata.py`). It never sees the target datasets.
2.  Structural breaks of known size are injected into the *raw* target series,
    so the frozen model forecasts the broken series exactly as it would in
    deployment, and the break time tau is known exactly -- which is what
    coverage debt requires and what natural breaks in these benchmarks do not
    give us.
3.  Every calibration layer replays the same conformity-score stream, so the
    comparison is exactly paired.
4.  Windows around every break, pooled across all series and datasets, are the
    replications from which the miscoverage curve is estimated.

Stages
------
    python scripts/run_real_data.py pretrain    # fit and cache the global model
    python scripts/run_real_data.py arl         # detector ARL_0 on real scores
    python scripts/run_real_data.py main        # main grid (level + variance)
    python scripts/run_real_data.py sensitivity # detector threshold h +/- 30%
    python scripts/run_real_data.py doublebreak # two breaks inside re-anchoring
    python scripts/run_real_data.py analyse     # tables
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brcp.conformal import ACI, METHODS, BRCP, OracleCP          # noqa: E402
from brcp.detectors import DataCUSUM, ScoreCUSUM                 # noqa: E402
from brcp.metrics import (coverage_debt, miscoverage_curve,      # noqa: E402
                          time_to_recovery, time_to_recovery_smooth,
                          winkler_score)
from brcp.realdata import (GlobalZeroShot, TARGET, corpus_holdout_iter,  # noqa: E402
                           corpus_iter, target_series)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cache")
MODEL = os.path.join(CACHE, "global_zeroshot.npz")

ALPHA = 0.10
WINDOW = 300
WARM = 500
PRE = 150            # offsets kept before the break
HCD = 200            # coverage-debt horizon
SEG = 400            # length of an injected break segment
GAP = 1000           # minimum spacing between injected breaks
# Operating point calibrated on held-out corpus streams (`arl` stage):
# ARL_0 = 1744 on real conformity scores. The synthetic study selected
# (0.65, 6.0), which on real data gives ARL_0 = 642 -- real scores are much
# more autocorrelated and heavier tailed, exactly as the simulation's
# limitations section anticipated.
K_DEF, H_DEF = 0.8, 8.0


# --------------------------------------------------------------------------- #
def inject(y, kind, mag, rng, seg=SEG, gap=GAP, start_frac=0.15):
    """Inject bounded, non-cumulative breaks of identical size. Returns y, taus."""
    y = y.copy()
    sd = float(np.std(np.diff(y))) + 1e-12
    lo = int(start_frac * len(y)) + WARM
    taus = list(range(lo, len(y) - seg - 50, gap))
    for tau in taus:
        end = min(tau + seg, len(y))
        if kind == "level":
            y[tau:end] += mag * sd
        elif kind == "variance":
            y[tau:end] += rng.standard_normal(end - tau) * np.sqrt(max(mag**2 - 1, 0)) * sd
        elif kind == "none":
            pass
        else:
            raise ValueError(kind)
    return y, ([] if kind == "none" else taus)


def make_layers(period, tau_idx=None, k=K_DEF, h=H_DEF, subset=None):
    def brcp(**kw):
        space = kw.pop("detector_space", "score")
        cls = ScoreCUSUM if space == "score" else DataCUSUM
        det = cls(anchor_size=250, k=kw.pop("k", k), h=kw.pop("h", h),
                  min_anchor=60, two_sided=True, stat="std")
        return BRCP(alpha=ALPHA, window=WINDOW, detector=det,
                    detector_space=space, period=period, **kw)

    all_l = {
        "SplitCP":      lambda: METHODS["SplitCP"](alpha=ALPHA),
        "RollingCP":    lambda: METHODS["RollingCP"](alpha=ALPHA, window=WINDOW),
        "ACI(0.005)":   lambda: ACI(alpha=ALPHA, window=WINDOW, gamma=0.005),
        "ACI(0.01)":    lambda: ACI(alpha=ALPHA, window=WINDOW, gamma=0.01),
        "ACI(0.05)":    lambda: ACI(alpha=ALPHA, window=WINDOW, gamma=0.05),
        "ACI(0.1)":     lambda: ACI(alpha=ALPHA, window=WINDOW, gamma=0.1),
        "AgACI":        lambda: METHODS["AgACI"](alpha=ALPHA, window=WINDOW),
        "ConformalPID": lambda: METHODS["ConformalPID"](alpha=ALPHA),
        "BRCP":         lambda: brcp(),
        "BRCP-safe":    lambda: brcp(safe_reset=True),
        "BRCP-data":    lambda: brcp(detector_space="data"),
    }
    if tau_idx is not None:
        all_l["OracleCP"] = lambda: OracleCP(alpha=ALPHA, window=WINDOW,
                                             gamma=0.01, tau_idx=tau_idx)
    return all_l if subset is None else {k_: v for k_, v in all_l.items() if k_ in subset}


def replay(layer, scores, preds, targets, obs):
    layer.warm_start(scores[:WARM], obs[:WARM])
    n = len(scores) - WARM
    q = np.empty(n)
    err = np.empty(n, dtype=np.int8)
    for i in range(n):
        j = WARM + i
        q[i] = layer.radius()
        err[i] = int(scores[j] > q[i])
        layer.update(float(scores[j]), int(err[i]), float(obs[j]))
    lo, hi = preds[WARM:] - q, preds[WARM:] + q
    return err, 2 * q, winkler_score(targets[WARM:], lo, hi, ALPHA), \
        len(getattr(layer, "alarms", []))


def stream_for(model, y, period):
    """Zero-shot forecasts, targets and scores for one (possibly broken) series."""
    s0 = max(model.context, period) + 5
    preds = model.predict_path(y, s0, len(y))
    targets = y[s0:len(y)]
    return np.abs(targets - preds), preds, targets


# --------------------------------------------------------------------------- #
def cmd_pretrain(_):
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    m = GlobalZeroShot(context=96, ridge=1e-2)
    m.fit_corpus(corpus_iter(), windows_per_series=1200, seed=0)
    m.save(MODEL)
    print(f"pretrained on {m.n_series_} series / {m.n_train_} windows "
          f"in {time.time()-t0:.0f}s -> {MODEL}")


def cmd_arl(_):
    """Detector run length on real score streams from *held-out* corpus series.

    These columns of electricity/traffic/weather were used neither for
    pretraining the forecaster nor for evaluation, so the threshold is
    calibrated without touching the evaluation data.
    """
    model = GlobalZeroShot.load(MODEL)
    streams = []
    for label, y in corpus_holdout_iter(n_holdout=25, verbose=True):
        sc, _, _ = stream_for(model, y, 24)
        if len(sc) >= WARM + 1000:
            streams.append(sc)
    print(f"  {len(streams)} held-out streams", flush=True)
    rows = []
    for k in (0.5, 0.65, 0.8, 1.0):
        for h in (4.0, 6.0, 8.0, 10.0, 14.0, 20.0):
            steps = alarms = 0
            for sc in streams:
                det = ScoreCUSUM(anchor_size=250, k=k, h=h, min_anchor=60, stat="std")
                det.set_anchor(sc[:WARM])
                rearm = []
                for s in sc[WARM:]:
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
            rows.append(dict(k=k, h=h, steps=steps, alarms=alarms,
                             arl0=steps / max(alarms, 0.5)))
            print(f"  k={k} h={h}: ARL_0={rows[-1]['arl0']:.0f} "
                  f"({alarms} alarms / {steps} steps)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/real_arl0.csv", index=False)
    ok = df[df.arl0 >= 1500]
    sel = ok.loc[ok.h.idxmin()] if len(ok) else df.loc[df.arl0.idxmax()]
    print(f"\nselected for real data: k={sel.k}, h={sel.h}, ARL_0={sel.arl0:.0f}")
    print("Real series contain genuine regime changes, so this is an upper bound "
          "on the\nfalse-alarm rate rather than a pure null; the threshold it "
          "implies is conservative.")


def run_grid(kinds, mags, k=K_DEF, h=H_DEF, subset=None, tag="main",
             seg=SEG, gap=GAP, max_cols=8, max_rows=20000, extra_taus=None):
    model = GlobalZeroShot.load(MODEL)
    rng = np.random.default_rng(7)
    rows, curves = [], {}
    for kind in kinds:
        for mag in mags:
            acc = {}
            n_windows = 0
            for ds, col, y0, period in target_series(TARGET, max_cols=max_cols,
                                                     max_rows=max_rows, verbose=False):
                y, taus = inject(y0, kind, mag, rng, seg=seg, gap=gap)
                if extra_taus is not None:
                    y, taus = extra_taus(y0, mag, rng)
                sc, preds, targets = stream_for(model, y, period)
                s0 = max(model.context, period) + 5
                if len(sc) < WARM + PRE + HCD + 200:
                    continue
                idx = [t - s0 - WARM for t in taus]
                idx = [i for i in idx if i - PRE >= 0 and i + HCD <= len(sc) - WARM]
                if not idx and kind != "none":
                    continue   # the no-break arm has no windows but still
                               # contributes marginal coverage, width and alarms
                layers = make_layers(period, tau_idx=None, k=k, h=h, subset=subset)
                for name, factory in layers.items():
                    err, width, wink, nal = replay(factory(), sc, preds, targets, targets)
                    for i in idx:
                        acc.setdefault(name, []).append(err[i - PRE:i + HCD])
                    rows.append(dict(kind=kind, mag=mag, dataset=ds, series=col,
                                     method=name, n_breaks=len(idx),
                                     coverage=1 - err.mean(), width=width.mean(),
                                     winkler=wink.mean(), n_alarms=nal,
                                     n_steps=len(err)))
                n_windows += len(idx)
            for name, segs in acc.items():
                curves[f"{kind}|{mag}|{name}"] = np.vstack(segs)
            print(f"  {kind} mag={mag}: {n_windows} break windows", flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/real_rows_{tag}.csv", index=False)
    np.savez_compressed(f"{RES}/real_curves_{tag}.npz", **curves)
    return rows, curves


def cmd_main(argv):
    """Run one break family per call so each stage stays short."""
    kinds = argv if argv else ["level", "variance", "none"]
    for kind in kinds:
        t0 = time.time()
        run_grid([kind], [3.0], tag=f"main_{kind}", max_rows=12000, max_cols=7)
        print(f"  {kind} done in {time.time()-t0:.0f}s", flush=True)


def cmd_sensitivity(_):
    """Detector threshold robustness: h +/- 30%."""
    out = []
    for h in (5.6, 8.0, 10.4):
        rows, curves = run_grid(["variance"], [3.0], k=K_DEF, h=h,
                                subset=["BRCP"], tag=f"h{h}",
                                max_rows=12000, max_cols=7)
        df = pd.DataFrame(rows)
        m = miscoverage_curve(curves["variance|3.0|BRCP"])
        n = curves["variance|3.0|BRCP"].shape[0]
        eps = 2 * np.sqrt(ALPHA * (1 - ALPHA) / n)
        out.append(dict(h=h, n_windows=n,
                        CD=float(np.sum(np.maximum(0, m[PRE:] - ALPHA - eps))),
                        TTRC_marg=time_to_recovery_smooth(m, ALPHA, PRE, HCD,
                                                          eps=eps, period=24),
                        coverage=df.coverage.mean(), width=df.width.mean(),
                        alarms=df.n_alarms.mean()))
        print(f"  h={h}: CD={out[-1]['CD']:.2f} width={out[-1]['width']:.3f}",
              flush=True)
    pd.DataFrame(out).to_csv(f"{RES}/real_sensitivity.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False))


def cmd_doublebreak(_):
    """Two breaks 40 steps apart -- inside the 60-step re-anchoring window."""
    def two(y0, mag, rng):
        y = y0.copy()
        sd = float(np.std(np.diff(y))) + 1e-12
        lo = int(0.15 * len(y)) + WARM
        taus = []
        for tau in range(lo, len(y) - SEG - 50, GAP):
            e1 = min(tau + 40, len(y))
            y[tau:e1] += rng.standard_normal(e1 - tau) * np.sqrt(8.0) * sd
            e2 = min(tau + 40 + SEG, len(y))
            y[tau + 40:e2] += mag * sd          # second, different break
            taus.append(tau)
        return y, taus

    rows, curves = run_grid(["variance"], [3.0],
                            subset=["ACI(0.05)", "ConformalPID", "BRCP", "BRCP-safe"],
                            tag="double", extra_taus=two,
                            max_rows=12000, max_cols=7)
    df = pd.DataFrame(rows)
    out = []
    for name in ["ACI(0.05)", "ConformalPID", "BRCP", "BRCP-safe"]:
        arr = curves[f"variance|3.0|{name}"]
        m = miscoverage_curve(arr)
        eps = 2 * np.sqrt(ALPHA * (1 - ALPHA) / arr.shape[0])
        s = df[df.method == name]
        out.append(dict(method=name, n_windows=arr.shape[0],
                        CD=float(np.sum(np.maximum(0, m[PRE:] - ALPHA - eps))),
                        TTRC_marg=time_to_recovery_smooth(m, ALPHA, PRE, HCD,
                                                          eps=eps, period=24),
                        coverage=s.coverage.mean(), width=s.width.mean(),
                        alarms=s.n_alarms.mean()))
    pd.DataFrame(out).to_csv(f"{RES}/real_doublebreak.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False))


# --------------------------------------------------------------------------- #
def cmd_analyse(_):
    rows = pd.concat([pd.read_csv(f"{RES}/real_rows_main_{k}.csv")
                      for k in ["level", "variance", "none"]
                      if os.path.exists(f"{RES}/real_rows_main_{k}.csv")],
                     ignore_index=True)
    rows.to_csv(f"{RES}/real_rows_main.csv", index=False)
    z = {}
    for k in ["level", "variance", "none"]:
        f = f"{RES}/real_curves_main_{k}.npz"
        if os.path.exists(f):
            zz = np.load(f)
            z.update({kk: zz[kk] for kk in zz.files})
    np.savez_compressed(f"{RES}/real_curves_main.npz", **z)
    recs = []
    for key in z:
        kind, mag, name = key.split("|")
        arr = z[key]
        n = arr.shape[0]
        eps = 2 * np.sqrt(ALPHA * (1 - ALPHA) / n)
        m = miscoverage_curve(arr)
        sub = rows[(rows.kind == kind) & (rows.mag == float(mag))
                   & (rows.method == name)]
        # series-weighted means
        w = sub.n_steps.values
        recs.append(dict(
            kind=kind, method=name, n_windows=n, eps=eps,
            CD=float(np.sum(np.maximum(0.0, m[PRE:PRE + HCD] - ALPHA - eps))),
            TTRC=time_to_recovery(m, ALPHA, PRE, HCD, eps=eps, w=25),
            TTRC_marg=time_to_recovery_smooth(m, ALPHA, PRE, HCD, eps=eps, period=24),
            miscov_early=float(np.mean(m[PRE:PRE + 50])),
            coverage=float(np.average(sub.coverage, weights=w)),
            width=float(np.average(sub.width, weights=w)),
            winkler=float(np.average(sub.winkler, weights=w)),
            alarms_per_1k=float(1000 * sub.n_alarms.sum() / sub.n_steps.sum()),
        ))
    tab = pd.DataFrame(recs)
    tab.to_csv(f"{RES}/real_table_main.csv", index=False)

    # paired tests across (dataset, series): BR-CP vs each baseline
    tests = []
    for kind in ["level", "variance"]:
        a = rows[(rows.kind == kind) & (rows.method == "BRCP")].sort_values(
            ["dataset", "series"])
        for name in sorted(rows.method.unique()):
            if name == "BRCP":
                continue
            b = rows[(rows.kind == kind) & (rows.method == name)].sort_values(
                ["dataset", "series"])
            if len(a) != len(b):
                continue
            dw = b.winkler.values - a.winkler.values
            dd = b.width.values - a.width.values
            try:
                _, pw = wilcoxon(dw, alternative="two-sided")
            except ValueError:
                pw = 1.0
            tests.append(dict(kind=kind, baseline=name, n_series=len(a),
                              d_winkler=float(np.mean(dw)),
                              d_winkler_pct=float(100 * np.mean(dw / a.winkler.values)),
                              p_winkler=float(pw),
                              d_width_pct=float(100 * np.mean(dd / a.width.values))))
    pd.DataFrame(tests).to_csv(f"{RES}/real_tests.csv", index=False)

    pd.set_option("display.width", 240)
    order = ["SplitCP", "RollingCP", "ACI(0.005)", "ACI(0.01)", "ACI(0.05)",
             "ACI(0.1)", "AgACI", "ConformalPID", "BRCP", "BRCP-safe", "BRCP-data"]
    tab["method"] = pd.Categorical(tab.method, order, ordered=True)
    for v in ["CD", "TTRC_marg", "coverage", "width", "winkler", "alarms_per_1k"]:
        print(f"\n=== {v} ===")
        print(tab.pivot_table(index="method", columns="kind", values=v,
                              observed=True).round(3).to_string())
    print("\n=== paired tests (variance) ===")
    print(pd.DataFrame(tests).query("kind=='variance'").round(4).to_string(index=False))


CMDS = dict(pretrain=cmd_pretrain, arl=cmd_arl, main=cmd_main,
            sensitivity=cmd_sensitivity, doublebreak=cmd_doublebreak,
            analyse=cmd_analyse)

if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "main"
    CMDS[cmd](sys.argv[2:])
