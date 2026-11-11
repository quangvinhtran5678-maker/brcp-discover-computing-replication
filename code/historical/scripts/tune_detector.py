"""
Calibrate the Page-CUSUM operating point used by BR-CP.

Two quantities are estimated empirically rather than from an asymptotic
approximation, because conformity scores are neither exactly uniform nor
exactly independent:

    ARL_0   average run length to a false alarm under the no-break DGP,
            i.e. the expected number of steps between spurious resets;
    EDD     expected detection delay after a genuine break.

The operating point is the (k, h) pair with the smallest mean EDD subject to
ARL_0 >= ARL_MIN and a miss rate of at most 5% on every break family. The
selection uses only synthetic no-break and break data and is never re-tuned per
evaluation configuration.

Usage
-----
    python scripts/tune_detector.py std     # standardised increment (default)
    python scripts/tune_detector.py rank    # distribution-free rank increment

Writes results/detector_tuning_<stat>.json.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brcp.detectors import ScoreCUSUM          # noqa: E402
from brcp.dgp import generate                  # noqa: E402
from brcp.experiment import Config             # noqa: E402
from brcp.forecasters import FrozenAR          # noqa: E402

ARL_MIN = 1500.0
RES = os.path.join(os.path.dirname(__file__), "..", "results")

GRIDS = {
    "std":  (( 0.25, 0.40, 0.50, 0.65, 0.80), (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)),
    "rank": (( 0.05, 0.10, 0.15, 0.20, 0.25), (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)),
}


def score_stream(cfg: Config, break_type: str, mag: float, seed: int):
    """Warm-up scores, evaluation scores and the break offset for one series."""
    y, _, _ = generate(T=cfg.T, tau=cfg.tau, break_type=break_type,
                       mag=mag, period=cfg.period, seed=seed)
    fc = FrozenAR(lags=cfg.lags, period=cfg.period).fit(y, cfg.fit_end)
    idx = np.arange(cfg.warm_start, cfg.T)
    s = np.abs(y[idx] - fc.predict_path(y, cfg.warm_start, cfg.T))
    n_warm = cfg.eval_start - cfg.warm_start
    return s[:n_warm], s[n_warm:], cfg.tau - cfg.eval_start


def run_detector(warm, ev, k, h, stat):
    """Alarm offsets and stream length. The detector re-anchors after an alarm."""
    det = ScoreCUSUM(anchor_size=250, k=k, h=h, min_anchor=60,
                     two_sided=True, stat=stat)
    det.set_anchor(warm)
    alarms, rearm = [], []
    for i, s in enumerate(ev):
        if det.armed:
            if det.update(float(s)):
                alarms.append(i)
                det.disarm()
                rearm = []
        else:
            rearm.append(float(s))
            if len(rearm) >= det.min_anchor:
                det.set_anchor(rearm)
                rearm = []
    return alarms, len(ev)


def main(stat: str = "std"):
    cfg = Config()
    ks, hs = GRIDS[stat]
    n_null, n_alt = 40, 40

    null_streams = [score_stream(cfg, "none", 0.0, s) for s in range(n_null)]
    alt_cases = [("level", 3.0), ("variance", 2.0), ("predictability", 4.0)]
    alt_streams = {c: [score_stream(cfg, c[0], c[1], 1000 + s) for s in range(n_alt)]
                   for c in alt_cases}

    results = []
    for k, h in itertools.product(ks, hs):
        steps = alarms = 0
        for warm, ev, _ in null_streams:
            al, n = run_detector(warm, ev, k, h, stat)
            steps += n
            alarms += len(al)
        row = dict(k=k, h=h, arl0=steps / max(alarms, 0.5))
        for case, streams in alt_streams.items():
            d = []
            for warm, ev, ti in streams:
                al, _ = run_detector(warm, ev, k, h, stat)
                after = [a for a in al if a >= ti]
                d.append(after[0] - ti if after else np.nan)
            row[case[0]] = float(np.nanmean(d)) if np.any(~np.isnan(d)) else float("inf")
            row[case[0] + "_missrate"] = float(np.mean(np.isnan(d)))
        results.append(row)

    feasible = [r for r in results
                if r["arl0"] >= ARL_MIN
                and max(r[c + "_missrate"] for c, _ in alt_cases) <= 0.05]
    key = lambda r: float(np.mean([r[c] for c, _ in alt_cases]))
    best = min(feasible, key=key) if feasible else min(results, key=key)

    print(f"statistic = {stat}   (EDD is only interpretable where ARL_0 is high:\n"
          f"  at a low ARL_0 a spurious alarm just after tau masquerades as fast\n"
          f"  detection, which is why the constraint is applied before selection)\n")
    print(f"{'k':>6} {'h':>6} {'ARL_0':>9} {'EDD lvl':>9} {'EDD var':>9} {'EDD prd':>9}")
    for r in results:
        star = "  <-- selected" if r is best else ""
        print(f"{r['k']:6.2f} {r['h']:6.2f} {r['arl0']:9.0f} "
              f"{r['level']:9.1f} {r['variance']:9.1f} "
              f"{r['predictability']:9.1f}{star}")
    if not feasible:
        print(f"\nWARNING: no cell met ARL_0 >= {ARL_MIN:.0f} with miss rate <= 5%.")
    print("\nselected:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                     for k, v in best.items()}))

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"detector_tuning_{stat}.json"), "w") as f:
        json.dump(dict(stat=stat, grid=results, selected=best,
                       arl_min=ARL_MIN, n_null=n_null, n_alt=n_alt), f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "std")
