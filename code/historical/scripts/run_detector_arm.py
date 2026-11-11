"""
Detector-space experiment: does it matter whether the change detector watches
the data or the model's own conformity scores?

Two regimes isolate the two failure modes.

benign        Under instance normalisation -- which every deployed TSFM applies
              in some form -- a pure level shift is absorbed by the normaliser.
              The observation distribution moves by more than 1.7 standard
              deviations while the conformity-score distribution is essentially
              unchanged (ratio 1.04). Recalibration is unnecessary, so a
              data-space detector produces pure false alarms.

invisible     In the `predictability` DGP the AR coefficient falls while the
              innovation scale is rescaled to preserve the marginal variance of
              y exactly. The observation distribution is unchanged, so a
              data-space detector is blind, while one-step-ahead errors inflate
              by a factor of 2.2 and the calibration layer must react.

Outputs: results/table_detector.csv, results/curves_detector.npz
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brcp.experiment import Config, run_grid                    # noqa: E402
from brcp.metrics import (coverage_debt, miscoverage_curve,     # noqa: E402
                          time_to_recovery)

RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)
N_SEEDS = 200
CD_HORIZON = 200
METHODS = ["ACI(0.05)", "BRCP", "BRCP-data", "BRCP-sdiff", "OracleCP"]

ARMS = [
    # (label, instance_norm, break types, magnitude)
    ("benign(RevIN)", True, ["level", "none"], 3.0),
    ("invisible", False, ["predictability"], 3.0),
]


def main():
    rows_all, curves_all, recs = [], {}, []
    for label, inorm, breaks, mag in ARMS:
        cfg = Config(instance_norm=inorm)
        eps = 2.0 * np.sqrt(cfg.alpha * (1 - cfg.alpha) / N_SEEDS)
        mags = [mag if b != "none" else 0.0 for b in breaks]
        for b, m in zip(breaks, mags):
            rows, curves = run_grid(cfg, [b], [m], range(N_SEEDS),
                                    method_names=METHODS, verbose=False)
            df = pd.DataFrame(rows)
            df["arm"] = label
            rows_all.append(df)
            for (bt, mg, name), arr in curves.items():
                curves_all[f"{label}|{bt}|{mg}|{name}"] = arr
                mc = miscoverage_curve(arr)
                P = cfg.pre_offsets
                sub = df[df.method == name]
                recs.append(dict(
                    arm=label, break_type=bt, method=name,
                    CD=coverage_debt(mc, cfg.alpha, P, CD_HORIZON) -
                       coverage_debt(np.full_like(mc, cfg.alpha + eps),
                                     cfg.alpha, P, CD_HORIZON),
                    CD_tol=float(np.sum(np.maximum(
                        0.0, mc[P:P + CD_HORIZON] - cfg.alpha - eps))),
                    TTRC=time_to_recovery(mc, cfg.alpha, P, cfg.horizon,
                                          eps=eps, w=25),
                    alarms=float(sub.n_alarms.mean()),
                    width_all=float(sub.width_all.mean()),
                    width_post=float(sub.width_post.mean()),
                    winkler_post=float(sub.winkler_post.mean()),
                    cov_all=float(sub.cov_all.mean()),
                ))
            print(f"  {label}/{b} done", flush=True)

    out = pd.DataFrame(recs).drop(columns=["CD"])
    out.to_csv(f"{RES}/table_detector.csv", index=False)
    np.savez_compressed(f"{RES}/curves_detector.npz", **curves_all)
    pd.concat(rows_all, ignore_index=True).to_csv(f"{RES}/raw_detector.csv",
                                                  index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    main()
