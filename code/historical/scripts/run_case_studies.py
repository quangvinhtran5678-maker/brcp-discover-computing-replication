"""
Case studies on externally dated structural breaks.

The main real-data study injects breaks because coverage debt needs a known
tau. This script does the complementary thing: it takes two breaks whose dates
are fixed by events outside the data, and reports what each calibration layer
does around them. There is one window per event, so these are illustrations
rather than estimates -- no coverage debt is computed, because a single
realisation cannot estimate P(y not in C).

Events
------
2008-09-15  Lehman Brothers files for bankruptcy. Daily exchange rates of eight
            currencies (Lai et al. dataset, 1990-2010) move into a
            high-volatility regime.
2009-04-26  The United States declares a public health emergency for H1N1.
            Weekly influenza-like-illness rates (CDC, 2002-2020) break their
            seasonal pattern with an out-of-season wave.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brcp.realdata import DATA_ROOT, GlobalZeroShot
from brcp.metrics import winkler_score
import scripts.run_real_data as R

RES = os.path.join(os.path.dirname(__file__), "..", "results")
EVENTS = [
    ("exchange_rate/exchange_rate.csv", "2008-09-15", "Lehman bankruptcy",
     ["0", "1", "2", "3", "4", "5", "6", "OT"], 300, 260),
    ("illness/national_illness.csv", "2009-04-26", "H1N1 emergency declared",
     ["% WEIGHTED ILI", "%UNWEIGHTED ILI", "ILITOTAL", "OT"], 200, 60),
]
LAYERS = ["SplitCP", "ACI(0.005)", "ACI(0.05)", "AgACI", "ConformalPID",
          "BRCP", "BRCP-safe"]


def main():
    model = GlobalZeroShot.load(os.path.join(os.path.dirname(__file__), "..",
                                             "cache", "global_zeroshot.npz"))
    rows = []
    for rel, date, label, cols, warm, horiz in EVENTS:
        df = pd.read_csv(os.path.join(DATA_ROOT, rel))
        dt = pd.to_datetime(df["date"], format="mixed")
        tau_raw = int((dt < date).sum())
        for col in cols:
            y = df[col].astype(float).to_numpy()
            if not np.all(np.isfinite(y)):
                continue
            s0 = model.context + 5
            preds = model.predict_path(y, s0, len(y))
            targets = y[s0:]
            sc = np.abs(targets - preds)
            tau = tau_raw - s0 - warm          # offset inside the replay window
            if tau < 60 or tau + horiz > len(sc) - warm:
                continue
            R_WARM = R.WARM
            R.WARM = warm
            try:
                for name in LAYERS:
                    layer = R.make_layers(24, subset=[name])[name]()
                    err, width, wink, nal = R.replay(layer, sc, preds, targets, targets)
                    post = slice(tau, tau + horiz)
                    rows.append(dict(
                        event=label, date=date, series=f"{os.path.basename(rel)}:{col}",
                        method=name,
                        miscov_post=float(err[post].mean()),
                        miscov_pre=float(err[max(0, tau - horiz):tau].mean()),
                        width_post=float(width[post].mean()),
                        width_pre=float(width[max(0, tau - horiz):tau].mean()),
                        winkler_post=float(wink[post].mean()),
                        coverage_all=float(1 - err.mean()),
                        alarms=nal,
                        alarm_lag=min([a - warm - tau for a in
                                       getattr(layer, "alarms", []) if a - warm - tau >= 0],
                                      default=np.nan)))
            finally:
                R.WARM = R_WARM
    out = pd.DataFrame(rows)
    out.to_csv(f"{RES}/real_case_studies.csv", index=False)
    pd.set_option("display.width", 220)
    for ev in out.event.unique():
        d = out[out.event == ev]
        g = d.groupby("method")[["miscov_pre", "miscov_post", "width_pre",
                                 "width_post", "winkler_post", "alarm_lag"]].mean()
        g = g.reindex(LAYERS)
        print(f"\n=== {ev} ({d.date.iloc[0]}), {d.series.nunique()} series ===")
        print(g.round(3).to_string())


if __name__ == "__main__":
    main()
