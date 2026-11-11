"""
Produce every figure in the paper from the checkpointed simulation results.

fig1_coverage_debt.pdf   instantaneous miscoverage around the break, four DGPs
fig2_frontier.pdf        adaptivity-stability frontier of the ACI step size
fig3_detector_space.pdf  score-space vs data-space monitoring
fig4_ablation.pdf        contribution of each BR-CP mechanism
fig5_mechanism.pdf       a single run: radius, alarms and CUSUM path
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brcp.metrics import miscoverage_curve, smooth       # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ALPHA, P = 0.10, 150

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 1.1,
})

STYLE = {
    "SplitCP":      dict(color="#9e9e9e", ls=":"),
    "RollingCP":    dict(color="#b07aa1", ls=":"),
    "ACI(0.005)":   dict(color="#4e79a7", ls="--"),
    "ACI(0.05)":    dict(color="#59a14f", ls="--"),
    "AgACI":        dict(color="#edc948", ls="-."),
    "ConformalPID": dict(color="#f28e2b", ls="-."),
    "BRCP":         dict(color="#e15759", ls="-", lw=1.8),
    "OracleCP":     dict(color="#111111", ls=(0, (1, 1)), lw=1.2),
    "BRCP-data":    dict(color="#76b7b2", ls="-"),
    "BRCP-sdiff":   dict(color="#17becf", ls="--"),
}
LABEL = {"BRCP": "BR-CP (ours)", "OracleCP": "Oracle (knows $\\tau$)",
         "BRCP-data": "BR-CP, data space", "BRCP-sdiff": "BR-CP, seas. diff."}


def load(npz):
    z = np.load(npz)
    out = {}
    for k in z.files:
        parts = k.split("|")
        out[tuple(parts)] = z[k]
    return out


# --------------------------------------------------------------------------- #
def fig_coverage_debt():
    C = load(f"{RES}/curves.npz")
    panels = [("variance", "Variance break ($\\times 3$)"),
              ("predictability", "Predictability break"),
              ("level", "Level break ($3\\sigma$)"),
              ("season", "Seasonal-amplitude break")]
    methods = ["SplitCP", "ACI(0.005)", "ACI(0.05)", "AgACI",
               "ConformalPID", "BRCP", "OracleCP"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.95), sharey=True)
    for ax, (bt, title) in zip(axes, panels):
        for mth in methods:
            key = (bt, "3.0", mth)
            if key not in C:
                continue
            m = smooth(miscoverage_curve(C[key]), 15)
            x = np.arange(len(m)) - P
            sl = (x >= -60) & (x <= 220)
            ax.plot(x[sl], m[sl], label=LABEL.get(mth, mth), **STYLE[mth])
        ax.axhline(ALPHA, color="k", lw=0.7, alpha=0.6)
        ax.axvline(0, color="k", lw=0.7, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("steps since break")
        ax.set_ylim(0, 0.62)
    axes[0].set_ylabel("miscoverage $m_j$")
    axes[0].text(-55, ALPHA + 0.025, r"nominal $\alpha$", fontsize=6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", frameon=False, ncol=7,
               bbox_to_anchor=(0.5, -0.16), columnspacing=1.2, handlelength=2.0)
    fig.savefig(f"{FIG}/fig1_coverage_debt.pdf")
    fig.savefig(f"{FIG}/fig1_coverage_debt.png")
    plt.close(fig)


def fig_frontier():
    fr = pd.read_csv(f"{RES}/frontier.csv")
    tab = pd.read_csv(f"{RES}/table_main.csv")
    b = tab[(tab.break_type == "variance") & (tab.mag == 3.0)].set_index("method")
    nul = tab[(tab.break_type == "none")].set_index("method")

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.35))
    ax = axes[0]
    ax.plot(fr.gamma, fr.CD, "o-", color="#4e79a7", ms=3.2, label="ACI($\\gamma$)")
    ax.fill_between(fr.gamma, fr.CD_lo, fr.CD_hi, color="#4e79a7", alpha=0.18, lw=0)
    for mth, col in [("BRCP", "#e15759"), ("OracleCP", "#111111")]:
        ax.axhline(b.loc[mth, "CD"], color=col, ls="--", lw=1.1,
                   label=LABEL.get(mth, mth))
    ax.set_xscale("log")
    ax.set_xlabel("ACI step size $\\gamma$")
    ax.set_ylabel("coverage debt CD")
    ax.set_title("Recovery cost is U-shaped in $\\gamma$")
    ax.legend(frameon=False, loc="upper center")

    ax = axes[1]
    ax.plot(fr.null_cov_sd, fr.CD, "o-", color="#4e79a7", ms=3.2, zorder=1)
    for _, r in fr.iterrows():
        ax.annotate(f"{r.gamma:g}", (r.null_cov_sd, r.CD), fontsize=5.6,
                    xytext=(3, 3), textcoords="offset points", color="#4e79a7")
    ax.scatter([np.nan], [np.nan])
    sd_brcp = 0.0  # filled below from the ablation/no-break run
    nb = np.load(f"{RES}/curves.npz")
    blk = nb["none|0.0|BRCP"][:, P:P + 500]
    nbk = blk.shape[1] // 50
    ax.scatter([np.nanstd(blk[:, :nbk * 50].reshape(blk.shape[0], nbk, 50).mean(2))],
               [b.loc["BRCP", "CD"]], marker="*", s=95, color="#e15759",
               zorder=5, label="BR-CP (ours)")
    blk = nb["none|0.0|OracleCP"][:, P:P + 500]
    ax.scatter([np.nanstd(blk[:, :nbk * 50].reshape(blk.shape[0], nbk, 50).mean(2))],
               [b.loc["OracleCP", "CD"]], marker="D", s=22, color="#111111",
               zorder=5, label="Oracle")
    ax.set_xlabel("steady-state coverage volatility (no break)")
    ax.set_ylabel("coverage debt CD")
    ax.set_title("Adaptivity--stability frontier")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_frontier.pdf")
    fig.savefig(f"{FIG}/fig2_frontier.png")
    plt.close(fig)


def fig_detector_space():
    d = pd.read_csv(f"{RES}/table_detector.csv")
    order = ["BRCP", "BRCP-sdiff", "BRCP-data"]
    cases = [("benign(RevIN)", "none", "no break"),
             ("benign(RevIN)", "level", "benign level shift\n(absorbed by RevIN)"),
             ("invisible", "predictability", "predictability break\n(invisible in data)")]
    cols = {"BRCP": "#e15759", "BRCP-sdiff": "#17becf", "BRCP-data": "#76b7b2"}

    short = ["no break", "benign shift\n(RevIN absorbs)", "predictability\nbreak"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    metrics = [("alarms", "alarms per series", False),
               ("width_all", "mean width, % vs BR-CP", True),
               ("winkler_post", "Winkler score, % vs BR-CP", True)]
    x = np.arange(len(cases))
    w = 0.26

    def val(a, b, mth, met):
        return d[(d.arm == a) & (d.break_type == b) & (d.method == mth)][met].values[0]

    for ax, (met, ylab, rel) in zip(axes, metrics):
        for i, mth in enumerate(order):
            v = [val(a, b, mth, met) for a, b, _ in cases]
            if rel:
                base = [val(a, b, "BRCP", met) for a, b, _ in cases]
                v = [100.0 * (vi / bi - 1.0) for vi, bi in zip(v, base)]
            ax.bar(x + (i - 1) * w, v, w, color=cols[mth],
                   label=LABEL.get(mth, mth), edgecolor="none")
            if rel:
                for xi, vi in zip(x + (i - 1) * w, v):
                    if vi > 0.15:
                        ax.text(xi, vi + 0.12, f"+{vi:.1f}", ha="center", fontsize=5.4)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=6.2)
        ax.set_ylabel(ylab)
        if rel:
            ax.axhline(0, color="k", lw=0.7)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Monitoring the model's conformity scores versus monitoring the data",
                 fontsize=8.5, y=1.04)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_detector_space.pdf")
    fig.savefig(f"{FIG}/fig3_detector_space.png")
    plt.close(fig)


def fig_ablation():
    a = pd.read_csv(f"{RES}/table_ablation.csv")
    order = ["BRCP", "BRCP-binary", "BRCP-noinfl", "BRCP-noboost",
             "BRCP-rank", "BRCP-noreset"]
    names = {"BRCP": "full", "BRCP-binary": "$-$graded $\\gamma$",
             "BRCP-noinfl": "$-$tolerance limit", "BRCP-noboost": "$-$boost",
             "BRCP-rank": "rank statistic", "BRCP-noreset": "$-$memory reset"}
    breaks = ["variance", "predictability", "level"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.2))
    x = np.arange(len(order))
    w = 0.26
    cols = ["#4e79a7", "#f28e2b", "#59a14f"]
    for ax, met, ylab in zip(axes, ["CD", "TTRC"],
                             ["coverage debt CD", "TTR-C (steps)"]):
        for i, bt in enumerate(breaks):
            v = [a[(a.break_type == bt) & (a.method == m)][met].values[0] for m in order]
            v = [min(vv, 200) if np.isfinite(vv) else 200 for vv in v]
            ax.bar(x + (i - 1) * w, v, w, color=cols[i], label=bt, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels([names[m] for m in order], rotation=28,
                           ha="right", fontsize=6.4)
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, ncol=1, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_ablation.pdf")
    fig.savefig(f"{FIG}/fig4_ablation.png")
    plt.close(fig)


def fig_mechanism():
    """One representative run: what BR-CP actually does at the break."""
    from brcp.conformal import ACI, BRCP
    from brcp.detectors import ScoreCUSUM
    from brcp.dgp import generate
    from brcp.experiment import Config
    from brcp.forecasters import FrozenAR

    cfg = Config()
    y, tau, _ = generate(T=cfg.T, tau=cfg.tau, break_type="variance",
                         mag=3.0, period=cfg.period, seed=3)
    fc = FrozenAR(lags=cfg.lags, period=cfg.period).fit(y, cfg.fit_end)
    idx = np.arange(cfg.warm_start, cfg.T)
    yhat = fc.predict_path(y, cfg.warm_start, cfg.T)
    sc = np.abs(y[idx] - yhat)
    nw = cfg.eval_start - cfg.warm_start
    warm, ev = sc[:nw], sc[nw:]
    ti = cfg.tau - cfg.eval_start

    det = ScoreCUSUM(anchor_size=250, k=0.65, h=6.0, min_anchor=60, stat="std")
    m = BRCP(alpha=cfg.alpha, window=cfg.window, detector=det, period=cfg.period)
    m.warm_start(warm, y[idx[:nw]])
    a = ACI(alpha=cfg.alpha, window=cfg.window, gamma=0.005)
    a.warm_start(warm)

    qb, qa, S, g = [], [], [], []
    for j, s in enumerate(ev):
        qb.append(m.radius()); qa.append(a.radius())
        g.append(m._gamma())
        pre = max(m.detector.S_up, m.detector.S_dn) if m.detector.armed else 0.0
        m.update(float(s), int(s > qb[-1]), float(y[idx[nw + j]]))
        post = max(m.detector.S_up, m.detector.S_dn) if m.detector.armed else 0.0
        # an alarm resets the statistic, so plot the pre-reset crossing value
        S.append(m.h_cross if hasattr(m, "h_cross") else max(pre, post))
        a.update(float(s), int(s > qa[-1]))
    S = np.asarray(S)
    for al in m.alarms:            # make the threshold crossing visible
        if 0 <= al - 1 < len(S):
            S[al - 1] = m.detector.h
    qb, qa, S, g = map(np.asarray, (qb, qa, S, g))
    x = np.arange(len(ev)) - ti

    fig, axes = plt.subplots(3, 1, figsize=(4.6, 3.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1, 1]))
    sl = (x >= -80) & (x <= 200)
    axes[0].plot(x[sl], sc[nw:][sl], color="#cccccc", lw=0.5, label="score $s_t$")
    axes[0].plot(x[sl], qa[sl], color="#4e79a7", ls="--", label="ACI(0.005) radius")
    axes[0].plot(x[sl], qb[sl], color="#e15759", label="BR-CP radius")
    axes[0].set_ylabel("radius $q_t$")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].plot(x[sl], S[sl], color="#333333")
    axes[1].axhline(6.0, color="#e15759", ls=":", lw=0.9)
    axes[1].set_ylabel("CUSUM $S_t$")
    axes[2].plot(x[sl], g[sl], color="#59a14f")
    axes[2].set_ylabel("step size $\\gamma_t$")
    axes[2].set_xlabel("steps since break")
    for ax in axes:
        ax.axvline(0, color="k", lw=0.7, alpha=0.35)
        for al in m.alarms:
            if -80 <= al - ti <= 200:
                ax.axvline(al - ti, color="#e15759", lw=0.6, alpha=0.45)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig5_mechanism.pdf")
    fig.savefig(f"{FIG}/fig5_mechanism.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_coverage_debt(); print("fig1 ok")
    fig_frontier();      print("fig2 ok")
    fig_detector_space();print("fig3 ok")
    fig_ablation();      print("fig4 ok")
    fig_mechanism();     print("fig5 ok")
