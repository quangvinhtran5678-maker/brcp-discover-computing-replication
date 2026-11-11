"""
Main controlled simulation study.

Outputs (written to results/)
    raw_runs.csv            one row per (break type, magnitude, seed, method)
    curves.npz              per-offset miscoverage indicators, per configuration
    table_main.csv          CD, TTR-C, coverage and width per configuration
    table_ablation.csv      BR-CP component ablations
    frontier.csv            ACI step-size sweep: recovery vs steady state
    tests.csv               paired Wilcoxon tests, BR-CP vs each baseline
    meta.json               configuration and environment
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brcp.experiment import Config, run_grid            # noqa: E402
from brcp.metrics import (coverage_debt, coverage_surplus,  # noqa: E402
                          miscoverage_curve, time_to_recovery,
                          time_to_recovery_smooth)

RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)

N_SEEDS = 200
CD_HORIZON = 200          # offsets over which coverage debt is accumulated
EARLY = 50                # early post-break window for the paired tests
N_BOOT = 2000

BREAKS = ["level", "variance", "trend", "season", "ar", "predictability", "slow_drift"]
MAGS = [2.0, 3.0, 4.0]
MAIN = ["SplitCP", "RollingCP", "ACI(0.005)", "ACI(0.05)", "AgACI",
        "ConformalPID", "BRCP", "OracleCP"]
ABL = ["BRCP", "BRCP-data", "BRCP-rank", "BRCP-binary", "BRCP-noreset",
       "BRCP-noinfl", "BRCP-noboost"]


def mc_tolerance(n_seeds: int, alpha: float, z: float = 2.0) -> float:
    """Monte-Carlo tolerance band eps = z * SE of a per-offset miscoverage rate.

    The coverage-debt functional takes a positive part, so plugging in a noisy
    estimate of m_j biases CD upwards even when the layer is perfectly
    calibrated. Accumulating the excess only beyond eps removes that floor: with
    eps = 2 SE the expected debt of an exactly calibrated layer is negligible,
    and the same eps is applied to every method so comparisons stay fair.
    """
    return float(z * np.sqrt(alpha * (1.0 - alpha) / n_seeds))


def cd_with_ci(err_mat, alpha, P, horizon, rng, n_boot=N_BOOT, eps=0.0):
    """Coverage debt with a seed-level bootstrap percentile interval.

    The seed bootstrap is vectorised: resampling seeds with replacement is
    equivalent to reweighting rows by multinomial counts, so all `n_boot`
    resampled miscoverage curves are obtained with a single matrix product.
    """
    A = np.asarray(err_mat, float)[:, P:P + horizon]
    A = np.where(np.isnan(A), 0.0, A)
    n = A.shape[0]
    thr = alpha + eps
    point = float(np.sum(np.maximum(0.0, A.mean(axis=0) - thr)))
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)
    M = (W @ A) / n                                  # (n_boot, horizon)
    boots = np.maximum(0.0, M - thr).sum(axis=1)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize(curves, rows_df, cfg, tag, methods, rng):
    P = cfg.pre_offsets
    recs = []
    for (bt, mag, name), arr in curves.items():
        if name not in methods:
            continue
        m = miscoverage_curve(arr)
        eps = mc_tolerance(arr.shape[0], cfg.alpha)
        cd0, _, _ = cd_with_ci(arr, cfg.alpha, P, CD_HORIZON, rng, 200, eps=0.0)
        cd, lo, hi = cd_with_ci(arr, cfg.alpha, P, CD_HORIZON, rng, eps=eps)
        sub = rows_df[(rows_df.break_type == bt) & (rows_df.mag == mag)
                      & (rows_df.method == name)]
        recs.append(dict(
            break_type=bt, mag=mag, method=name,
            CD=cd, CD_lo=lo, CD_hi=hi, CD_raw=cd0, eps=eps,
            surplus=coverage_surplus(m, cfg.alpha, P, CD_HORIZON),
            TTRC=time_to_recovery(m, cfg.alpha, P, cfg.horizon, eps=eps, w=25),
            TTRC_marg=time_to_recovery_smooth(m, cfg.alpha, P, cfg.horizon,
                                              eps=eps, period=cfg.period),
            miscov_early=float(np.nanmean(m[P:P + EARLY])),
            miscov_pre=float(np.nanmean(m[:P])),
            cov_all=float(sub.cov_all.mean()),
            width_all=float(sub.width_all.mean()),
            width_post=float(sub.width_post.mean()),
            winkler_post=float(sub.winkler_post.mean()),
            n_alarms=float(sub.n_alarms.mean()),
        ))
    df = pd.DataFrame(recs)
    df.to_csv(f"{RES}/table_{tag}.csv", index=False)
    return df


def paired_tests(curves, rows_df, cfg, ref="BRCP"):
    """Per-seed paired comparisons: early miscoverage and post-break Winkler."""
    P = cfg.pre_offsets
    out = []
    for bt in BREAKS:
        for mag in MAGS:
            ref_early = np.nanmean(curves[(bt, mag, ref)][:, P:P + EARLY], axis=1)
            for name in MAIN:
                if name in (ref, "OracleCP"):
                    continue
                oth_early = np.nanmean(curves[(bt, mag, name)][:, P:P + EARLY], axis=1)
                d = oth_early - ref_early
                try:
                    stat, p = wilcoxon(d, alternative="greater")
                except ValueError:      # all differences zero
                    stat, p = np.nan, 1.0
                a = rows_df[(rows_df.break_type == bt) & (rows_df.mag == mag)
                            & (rows_df.method == ref)].sort_values("seed")
                b = rows_df[(rows_df.break_type == bt) & (rows_df.mag == mag)
                            & (rows_df.method == name)].sort_values("seed")
                try:
                    _, p_w = wilcoxon(b.winkler_post.values - a.winkler_post.values,
                                      alternative="greater")
                except ValueError:
                    p_w = 1.0
                # rank-biserial effect size for the early-miscoverage test
                nz = d[d != 0]
                rb = (np.sign(nz).mean() if nz.size else 0.0)
                dw = b.winkler_post.values - a.winkler_post.values
                dwid = b.width_all.values - a.width_all.values
                try:
                    _, p_w2 = wilcoxon(dw, alternative="two-sided")
                except ValueError:
                    p_w2 = 1.0
                out.append(dict(break_type=bt, mag=mag, baseline=name,
                                d_mean=float(np.mean(d)), rank_biserial=float(rb),
                                p_early=float(p), p_winkler=float(p_w),
                                d_winkler=float(np.mean(dw)),
                                p_winkler_2s=float(p_w2),
                                d_width=float(np.mean(dwid))))
    df = pd.DataFrame(out)
    # Holm-Bonferroni across the whole family of early-miscoverage tests
    p = df.p_early.values
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p) - i) * p[j])
        adj[j] = min(1.0, running)
    df["p_early_holm"] = adj
    df.to_csv(f"{RES}/tests.csv", index=False)
    return df


def frontier(cfg, rng):
    """Adaptivity-stability frontier: sweep the ACI step size.

    Recovery cost  = coverage debt on a break DGP.
    Steady-state   = mean width and coverage volatility on the no-break DGP.
    """
    from brcp.conformal import ACI
    from brcp.experiment import run_replication
    import brcp.experiment as E

    gammas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    seeds = range(N_SEEDS)
    P = cfg.pre_offsets
    orig = E.build_methods

    recs = []
    for g in gammas:
        def bm(c, subset=None, tau_idx=10**9, _g=g):
            return {f"ACI({_g})": lambda: ACI(alpha=c.alpha, window=c.window, gamma=_g)}
        E.build_methods = bm
        rows_b, cur_b = run_grid(cfg, ["variance"], [3.0], seeds, verbose=False)
        rows_n, cur_n = run_grid(cfg, ["none"], [0.0], seeds, verbose=False)
        E.build_methods = orig

        key_b = [k for k in cur_b if k[2].startswith("ACI")][0]
        eps = mc_tolerance(len(seeds), cfg.alpha)
        cd, lo, hi = cd_with_ci(cur_b[key_b], cfg.alpha, P, CD_HORIZON, rng, 500,
                                eps=eps)
        key_n = [k for k in cur_n if k[2].startswith("ACI")][0]
        null_err = cur_n[key_n]
        # coverage volatility: sd of block coverage (blocks of 50) under no break
        blocks = null_err[:, P:P + 500]
        nb = blocks.shape[1] // 50
        bc = blocks[:, :nb * 50].reshape(blocks.shape[0], nb, 50).mean(axis=2)
        recs.append(dict(gamma=g, CD=cd, CD_lo=lo, CD_hi=hi,
                         null_width=float(pd.DataFrame(rows_n).width_all.mean()),
                         null_cov=float(1 - np.nanmean(blocks)),
                         null_cov_sd=float(np.nanstd(bc)),
                         break_width=float(pd.DataFrame(rows_b).width_all.mean())))
        print(f"  gamma={g}: CD={cd:.2f} null_width={recs[-1]['null_width']:.3f} "
              f"null_cov_sd={recs[-1]['null_cov_sd']:.4f}", flush=True)

    E.build_methods = orig
    df = pd.DataFrame(recs)
    df.to_csv(f"{RES}/frontier.csv", index=False)
    return df


def stage_grid(bt: str):
    """Run one break type of the main grid and checkpoint it to disk."""
    cfg = Config()
    seeds = list(range(N_SEEDS))
    mags = MAGS if bt != "none" else [0.0]
    rows, curves = run_grid(cfg, [bt], mags, seeds, method_names=MAIN)
    pd.DataFrame(rows).to_csv(f"{RES}/part_rows_{bt}.csv", index=False)
    np.savez_compressed(f"{RES}/part_curves_{bt}.npz",
                        **{f"{a}|{b}|{c}": v for (a, b, c), v in curves.items()})
    print(f"checkpointed {bt}", flush=True)


def stage_ablation():
    cfg = Config()
    seeds = list(range(N_SEEDS))
    rows_a, curves_a = run_grid(cfg, ["level", "variance", "predictability",
                                      "slow_drift", "none"], [3.0], seeds,
                                method_names=ABL)
    pd.DataFrame(rows_a).to_csv(f"{RES}/raw_ablation.csv", index=False)
    np.savez_compressed(f"{RES}/curves_ablation.npz",
                        **{f"{a}|{b}|{c}": v for (a, b, c), v in curves_a.items()})
    print("checkpointed ablation", flush=True)


def stage_frontier():
    cfg = Config()
    rng = np.random.default_rng(7)
    fr = frontier(cfg, rng)
    print(fr.to_string(index=False))


def load_parts(prefix="part_curves_"):
    curves, rows = {}, []
    for bt in BREAKS + ["none"]:
        f = f"{RES}/{prefix}{bt}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f)
        for k in z.files:
            a, b, c = k.split("|")
            curves[(a, float(b), c)] = z[k]
        rows.append(pd.read_csv(f"{RES}/part_rows_{bt}.csv"))
    return pd.concat(rows, ignore_index=True), curves


def main():
    t0 = time.time()
    cfg = Config()
    rng = np.random.default_rng(12345)

    rows_df, curves = load_parts()
    rows_df.to_csv(f"{RES}/raw_runs.csv", index=False)
    np.savez_compressed(f"{RES}/curves.npz",
                        **{f"{a}|{b}|{c}": v for (a, b, c), v in curves.items()})

    print("[2/5] summarising ...", flush=True)
    tab = summarize(curves, rows_df, cfg, "main", MAIN, rng)
    print(tab[tab.break_type == "variance"][
        ["method", "mag", "CD", "TTRC", "cov_all", "width_all"]].to_string(index=False))

    print("[3/5] paired tests ...", flush=True)
    tests = paired_tests(curves, rows_df, cfg)
    print(tests.groupby("baseline")[["p_early_holm"]].max().to_string())

    print("[4/5] ablations ...", flush=True)
    rows_a_df = pd.read_csv(f"{RES}/raw_ablation.csv")
    z = np.load(f"{RES}/curves_ablation.npz")
    curves_a = {tuple([k.split("|")[0], float(k.split("|")[1]), k.split("|")[2]]): z[k]
                for k in z.files}
    summarize(curves_a, rows_a_df, cfg, "ablation", ABL, rng)

    meta = dict(n_seeds=N_SEEDS, cd_horizon=CD_HORIZON, early=EARLY,
                n_boot=N_BOOT, breaks=BREAKS, mags=MAGS, methods=MAIN,
                ablations=ABL, config=vars(cfg),
                numpy=np.__version__, python=platform.python_version(),
                runtime_s=round(time.time() - t0, 1))
    with open(f"{RES}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nTotal runtime {meta['runtime_s']}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "grid":
        stage_grid(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "ablation":
        stage_ablation()
    elif len(sys.argv) > 1 and sys.argv[1] == "frontier":
        stage_frontier()
    else:
        main()
