"""Figures for the real-data confirmatory study."""
from __future__ import annotations
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brcp.metrics import miscoverage_curve, smooth

RES, FIG = "results", "figures"
ALPHA, PRE = 0.10, 150
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 6.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
    "savefig.bbox": "tight", "lines.linewidth": 1.1})
ST = {"SplitCP": dict(color="#9e9e9e", ls=":"),
      "RollingCP": dict(color="#b07aa1", ls=":"),
      "ACI(0.005)": dict(color="#4e79a7", ls="--"),
      "ACI(0.05)": dict(color="#59a14f", ls="--"),
      "AgACI": dict(color="#edc948", ls="-."),
      "ConformalPID": dict(color="#f28e2b", ls="-."),
      "BRCP": dict(color="#e15759", ls="-", lw=1.8),
      "BRCP-safe": dict(color="#111111", ls=(0, (1, 1)), lw=1.2)}
LAB = {"BRCP": "BR-CP (ours)", "BRCP-safe": "BR-CP-safe"}

z = np.load(f"{RES}/real_curves_main.npz")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.0))
for ax, (kind, title) in zip(axes[:2], [("variance", "Variance break, real data"),
                                        ("level", "Level break, real data")]):
    for m in ST:
        k = f"{kind}|3.0|{m}"
        if k not in z.files: continue
        mc = smooth(miscoverage_curve(z[k]), 11)
        x = np.arange(len(mc)) - PRE
        s = (x >= -50) & (x <= 200)
        ax.plot(x[s], mc[s], label=LAB.get(m, m), **ST[m])
    ax.axhline(ALPHA, color="k", lw=.7, alpha=.6); ax.axvline(0, color="k", lw=.7, alpha=.35)
    ax.set_title(title); ax.set_xlabel("steps since break"); ax.set_ylim(0, .72)
axes[0].set_ylabel("miscoverage $m_j$")
axes[0].legend(frameon=False, loc="upper right", ncol=1)

# panel 3: alarms vs efficiency on the no-break arm
r = pd.read_csv(f"{RES}/real_rows_main.csv"); n = r[r.kind == "none"]
g = n.groupby("method").apply(lambda d: pd.Series({
    "alarms": 1000 * d.n_alarms.sum() / d.n_steps.sum(),
    "width": np.average(d.width, weights=d.n_steps),
    "wink": np.average(d.winkler, weights=d.n_steps)}), include_groups=False)
ax = axes[2]
base = g.loc["BRCP"]
for m, mk, c in [("BRCP", "*", "#e15759"), ("BRCP-data", "o", "#76b7b2")]:
    ax.scatter(g.loc[m, "alarms"], 100 * (g.loc[m, "wink"] / base.wink - 1),
               marker=mk, s=110 if mk == "*" else 40, color=c,
               label="BR-CP (score space)" if m == "BRCP" else "BR-CP (data space)")
ax.axhline(0, color="k", lw=.7)
ax.set_xlabel("alarms per 1000 steps (no break)")
ax.set_ylabel("Winkler score, % vs BR-CP")
ax.set_title("Cost of monitoring the data")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(f"{FIG}/fig6_realdata.pdf"); fig.savefig(f"{FIG}/fig6_realdata.png")
print("fig6 ok")
