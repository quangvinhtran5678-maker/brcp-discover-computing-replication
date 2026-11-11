"""BR-CP GPU confirmatory arm: frozen Chronos-Bolt forecasts + paired conformal replay.

The published-checkpoint run is ADDITIONAL empirical evidence, NOT a reproduction
of the earlier global-ridge experiment and NOT proof of training-set disjointness.
An unperturbed real series is not a statistically guaranteed no-change null.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from brcp.conformal import ACI, BRCP, RollingCP, SplitCP, AgACI, ConformalPID
from brcp.detectors import ScoreCUSUM, DataCUSUM
from brcp.metrics import winkler_score, miscoverage_curve, time_to_recovery_smooth

DATASETS = {
    "ETTh1": ("ETT-small/ETTh1.csv", 24),
    "ETTh2": ("ETT-small/ETTh2.csv", 24),
    "ETTm1": ("ETT-small/ETTm1.csv", 96),
    "ETTm2": ("ETT-small/ETTm2.csv", 96),
    "exchange": ("exchange_rate/exchange_rate.csv", 7),
    "illness": ("illness/national_illness.csv", 52),
}
BASE_METHODS = ["SplitCP", "RollingCP", "ACI(0.005)", "ACI(0.05)",
                "AgACI", "ConformalPID", "BRCP", "BRCP-safe", "BRCP-data"]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def persist_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".new.json")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, default=str))
    os.replace(tmp, path)


def usable_cols(df, max_cols):
    nums = df.select_dtypes(include="number").columns.tolist()
    good = []
    for c in nums:
        y = df[c].to_numpy(dtype=float)
        if np.isfinite(y).all() and np.std(y) > 0 and np.std(y) / (np.mean(np.abs(y))+1e-12) >= 1e-4:
            good.append(c)
    if not good:
        return []
    if len(good) <= max_cols:
        return good
    return [good[i] for i in np.unique(np.linspace(0, len(good)-1, max_cols).astype(int))]


def break_series(y, kind, sd_factor, seed, context, warm, pre, horizon, segment, gap):
    """Synthetic interventions in *observed* series; tau known by construction.

    The bounded intervention's return to baseline is a second boundary; the
    design is comparable to the archived real-data intervention, but distinct.
    """
    out = np.asarray(y, dtype=float).copy()
    if kind == "none":
        return out, np.empty(0, dtype=int)
    n = len(out)
    low = max(int(0.15 * n) + warm, context + warm + pre + 30)
    high = n - segment - max(horizon, 50)
    taus = np.arange(low, high+1, gap, dtype=int)
    if not len(taus) and low + horizon <= n - 10:
        taus = np.array([low], dtype=int)
    sd = float(np.std(np.diff(out))) + 1e-12
    rng = np.random.default_rng(seed)
    for tau in taus:
        end = min(n, int(tau) + segment)
        if kind == "level":
            out[tau:end] += sd_factor * sd
        elif kind == "variance":
            out[tau:end] += rng.normal(0.0, np.sqrt(max(sd_factor**2 - 1.0, 0.0))*sd, end-tau)
        else:
            raise ValueError(kind)
    return out, taus


def make_layers(alpha, period, window, k, h_score, h_data):
    def brcp(space="score", safe=False):
        det_cls = ScoreCUSUM if space == "score" else DataCUSUM
        return BRCP(alpha=alpha, window=window, safe_reset=safe, period=period,
                    detector=det_cls(anchor_size=250, min_anchor=60,
                                     two_sided=True, stat="std", k=k,
                                     h=h_score if space == "score" else h_data),
                    detector_space=space)
    return {
        "SplitCP": lambda: SplitCP(alpha=alpha),
        "RollingCP": lambda: RollingCP(alpha=alpha, window=window),
        "ACI(0.005)": lambda: ACI(alpha=alpha, gamma=0.005, window=window),
        "ACI(0.05)": lambda: ACI(alpha=alpha, gamma=0.05, window=window),
        "AgACI": lambda: AgACI(alpha=alpha, window=window),
        "ConformalPID": lambda: ConformalPID(alpha=alpha),
        "BRCP": lambda: brcp("score"),
        "BRCP-safe": lambda: brcp("score", safe=True),
        "BRCP-data": lambda: brcp("data"),
    }


def replay(layer, scores, targets, preds, warm, alpha):
    layer.warm_start(scores[:warm], targets[:warm])
    n = len(scores) - warm
    err = np.empty(n, dtype=np.uint8)
    width = np.empty(n, dtype=float)
    wink = np.empty(n, dtype=float)
    for i in range(n):
        j = warm+i
        q = float(layer.radius())
        if not np.isfinite(q):
            raise RuntimeError("An infinite interval appeared; report separately rather than silently averaging")
        miss = int(scores[j] > q)
        err[i] = miss
        width[i] = 2 * q
        wink[i] = 2*q + (2.0/alpha)*max(0.0, scores[j]-q)
        layer.update(float(scores[j]), miss, float(targets[j]))
    return err, width, wink, np.array(layer.alarms, dtype=int), int(layer.saturated)


def empirical_alarm_rate(stream, space, k, h, warm):
    """Operational alarm rate on unperturbed historical data, NOT true FPR."""
    cls = ScoreCUSUM if space == "score" else DataCUSUM
    d = cls(anchor_size=250, k=k, h=h, min_anchor=60, two_sided=True, stat="std")
    d.set_anchor(stream[max(0, warm-250):warm])
    alarms = 0
    n = 0
    refill = []
    for v in stream[warm:]:
        n += 1
        if d.armed:
            if d.update(float(v)):
                alarms += 1
                d.disarm()
                refill = []
        else:
            refill.append(float(v))
            if len(refill) >= d.min_anchor:
                d.set_anchor(refill)
                refill = []
    return alarms, n


def model_prepare(model_id, dtype_name="auto"):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not detected. In Google Colab select Runtime > Change runtime type > H100/A100/L4/T4 GPU")
    from chronos import BaseChronosPipeline
    torch.backends.cuda.matmul.allow_tf32 = True
    if dtype_name == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    else:
        dtype = getattr(torch, dtype_name)
    print(f"[GPU] {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__} | dtype={dtype}", flush=True)
    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cuda", torch_dtype=dtype)
    from huggingface_hub import model_info
    try:
        revision = model_info(model_id).sha
    except Exception as e:
        revision = None
        print(f"[WARN] Hugging Face model revision unavailable: {e}", flush=True)
    return pipe, revision


def forecast(pipe, y, context, partial_path, tag, batch=96, check_every=2048):
    """Resume GPU inference by saving median predictions in Drive periodically."""
    import torch
    y = np.asarray(y, dtype=np.float64)
    total = len(y)-context
    if total <= 0:
        raise ValueError("Too few observations for this context length")
    partial_path = Path(partial_path)
    pred = np.full(total, np.nan, dtype=np.float64)
    start = 0
    if partial_path.exists():
        try:
            with np.load(partial_path, allow_pickle=False) as z:
                saved_tag = str(z["tag"])
                old = z["pred"]
            if saved_tag == tag and len(old) <= total and np.isfinite(old).all():
                pred[:len(old)] = old
                start = len(old)
                print(f"[RESUME] {partial_path.name}: {start}/{total}", flush=True)
            else:
                print("[WARN] ignoring incompatible partial cache", flush=True)
        except Exception:
            print("[WARN] unreadable partial; restarting this series", flush=True)
    current_batch = batch
    while start < total:
        stop = min(total, start+current_batch)
        origins = np.arange(context+start, context+stop)
        ctx = np.stack([y[t-context:t] for t in origins]).astype(np.float32)
        try:
            with torch.inference_mode():
                out = pipe.predict(torch.from_numpy(ctx), prediction_length=1)
                med = out[:, out.shape[1]//2, 0].detach().cpu().numpy()
            if med.shape != (len(origins),) or not np.isfinite(med).all():
                raise ValueError("Invalid or non-finite Chronos-Bolt forecasts")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() and current_batch > 1:
                torch.cuda.empty_cache()
                current_batch = max(1, current_batch//2)
                print(f"[OOM] Reduce batch to {current_batch}", flush=True)
                continue
            raise
        pred[start:stop] = med
        start = stop
        if start % check_every < current_batch or start == total:
            partial_path.parent.mkdir(parents=True,exist_ok=True)
            tmp = partial_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, tag=np.array(tag), pred=pred[:start])
            os.replace(tmp, partial_path)
            print(f"[GPU CACHE] {start}/{total}", flush=True)
    return pred


def cache_run(pipe, model_id, rev, ds, col, col_idx, y0, kind, input_sha, cfg, cache_dir):
    context, warm, pre, horizon, segment, gap = [cfg[x] for x in
        ("context", "warm", "pre", "horizon_cd", "segment", "gap")]
    seed = int(hashlib.sha256(f"{cfg['seed']}|{ds}|{col}|{kind}".encode()).hexdigest()[:8], 16)
    y, taus = break_series(y0, kind, cfg["magnitude"], seed, context, warm, pre, horizon, segment, gap)
    if kind != "none" and not len(taus):
        return None
    meta = dict(model=model_id, revision=rev, dataset=ds, column=col, col_idx=col_idx,
                kind=kind, context=context, input_sha256=input_sha,
                seed=seed, magnitude=cfg["magnitude"], segment=segment, gap=gap)
    tag = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    basename = f"{ds}_col{col_idx:03d}_{kind}"
    path = cache_dir/(basename+".npz")
    partial = cache_dir/(basename+".partial.npz")
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["tag"]) == tag and len(z["preds"]) == len(y)-context and np.isfinite(z["preds"]).all():
                    return path
        except Exception:
            pass
    print(f"[CACHE] {basename}, n={len(y)}, events={len(taus)}", flush=True)
    preds = forecast(pipe, y, context, partial, tag, batch=cfg["batch"])
    origins = np.arange(context, len(y))
    targets = y[origins]
    scores = abs(targets - preds)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, tag=np.array(tag), preds=preds, targets=targets,
                        scores=scores, origins=origins, taus=taus,
                        meta=np.array(json.dumps(meta,sort_keys=True)))
    os.replace(tmp, path)
    partial.unlink(missing_ok=True)
    return path


def summarize_cache(path, ds, col, col_idx, kind, cfg, h_score, h_data, period, out_dir):
    with np.load(path, allow_pickle=False) as z:
        score, preds, targets, origins, taus = [z[x] for x in ("scores", "preds", "targets", "origins", "taus")]
        meta = json.loads(str(z["meta"]))
    name = path.stem
    row_path = out_dir/(name+"_methods.csv")
    ev_path = out_dir/(name+"_events.csv")
    trace_path = out_dir/(name+"_errors.npz")
    if row_path.exists() and ev_path.exists() and trace_path.exists():
        return row_path, ev_path
    layers = make_layers(cfg["alpha"], period, cfg["window"], cfg["k"], h_score, h_data)
    rows, events, traces = [], [], {}
    P, H = cfg["pre"], cfg["horizon_cd"]
    for method, factory in layers.items():
        err, width, wink, alarms, clip = replay(factory(), score, targets, preds, cfg["warm"], cfg["alpha"])
        if not np.isfinite(width).all() or not np.isfinite(wink).all():
            raise RuntimeError(f"Non-finite interval metrics in {name}/{method}")
        traces[method] = err
        event_segs = []
        for tau in taus:
            event = int(np.searchsorted(origins, tau))-cfg["warm"]
            if event-P < 0 or event+H > len(err):
                continue
            block = err[event-P:event+H]
            event_segs.append(block)
            events.append(dict(dataset=ds, col=col, kind=kind, method=method,
                               tau=int(tau), misscov_post50=float(np.mean(block[P:P+50])),
                               misscov_postH=float(np.mean(block[P:])),
                               n_postH=len(block[P:]), series_id=f"{ds}|{col}"))
        rows.append(dict(dataset=ds, col=col, col_idx=col_idx, kind=kind,
                         method=method, n_eval=len(err), coverage=float(1-err.mean()),
                         width=float(width.mean()), winkler=float(wink.mean()),
                         n_alarms=len(alarms), clips=clip,
                         n_events=len(event_segs), source_sha256=meta["input_sha256"],
                         model_revision=meta["revision"] or "UNVERIFIED"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(row_path,index=False)
    pd.DataFrame(events, columns=["dataset", "col", "kind", "method", "tau", "misscov_post50", "misscov_postH", "n_postH", "series_id"]).to_csv(ev_path,index=False)
    np.savez_compressed(trace_path, **traces)
    return row_path, ev_path


def calibrate_h(dev_files, cfg, choices, target_per800):
    """Development-only, independent series selection; no test results used."""
    rows = []
    for h in choices:
        for space in ("score", "data"):
            alarm_total = steps = 0
            for file in dev_files:
                with np.load(file, allow_pickle=False) as z:
                    a, b = (z["scores"], z["targets"])
                al, n = empirical_alarm_rate(a if space=="score" else b, space, cfg["k"], h, cfg["warm"])
                alarm_total += al
                steps += n
            rate = 800 * alarm_total / max(1, steps)
            rows.append(dict(space=space, h=h, alarms=alarm_total, steps=steps, alarms_per800=rate))
    df = pd.DataFrame(rows)
    chosen = {}
    for space in ("score","data"):
        x = df[df.space==space].copy()
        x["distance"] = abs(x.alarms_per800-target_per800)
        choice = x.sort_values(["distance", "h"],ascending=[True,False]).iloc[0]
        chosen[space] = float(choice.h)
    return chosen, df


def aggregate(results_dir, cfg, out):
    a = [pd.read_csv(p) for p in sorted(results_dir.glob("*_methods.csv"))]
    if not a:
        raise RuntimeError("No completed result rows to aggregate")
    rows = pd.concat(a,ignore_index=True)
    rows.to_csv(out/"all_method_results.csv",index=False)
    group = rows.groupby(["kind", "method"],as_index=False).agg(
        n_series=("series", "size") if "series" in rows.columns else ("col", "size"),
        avg_coverage=("coverage", "mean"),avg_width=("width","mean"),
        avg_winkler=("winkler","mean"),mean_alarms=("n_alarms","mean"))
    group.to_csv(out/"summary_by_kind_and_method.csv",index=False)
    evs = [pd.read_csv(p) for p in sorted(results_dir.glob("*_events.csv")) if p.stat().st_size > 90]
    if evs:
        evt = pd.concat(evs, ignore_index=True)
        evt.to_csv(out/"event_level_results.csv",index=False)
        # The same post-break window is used for the paired comparison.
        joined = evt[evt.method.isin(["BRCP", "ACI(0.05)"])].pivot_table(
            index=["dataset", "col", "kind", "tau"], columns="method", values="misscov_post50").dropna()
        joined["BRCP_minus_ACI05"] = joined["BRCP"]-joined["ACI(0.05)"]
        joined.reset_index().to_csv(out/"paired_event_comparison.csv",index=False)
        # Cluster bootstrap at series level; within-series break windows are dependent.
        ci = []
        for kind, subset in joined.reset_index().groupby("kind"):
            by = subset.groupby(["dataset", "col"],sort=True)["BRCP_minus_ACI05"].mean().to_numpy()
            if len(by)<3:
                ci.append(dict(kind=kind,n_series=len(by),note="Too few independent series",mean_diff=np.nan,lower=np.nan,upper=np.nan))
                continue
            gen = np.random.default_rng(cfg["seed"])
            inds = gen.integers(0,len(by),size=(2000,len(by)))
            stats = by[inds].mean(axis=1)
            ci.append(dict(kind=kind,n_series=len(by),note="exploratory cluster bootstrap by series",mean_diff=float(by.mean()),
                           lower=float(np.quantile(stats,0.025)),upper=float(np.quantile(stats,0.975))))
        pd.DataFrame(ci).to_csv(out/"exploratory_paired_bootstrap.csv",index=False)
    return rows, group


def run(cfg):
    import torch
    dataroot = Path(cfg["data_root"])
    outroot = Path(cfg["output_root"])
    if not dataroot.exists():
        raise FileNotFoundError(f"BRCP Recovery Class-1 folder not found: {dataroot}")
    cfg_tag = hashlib.sha256(json.dumps({k:v for k,v in cfg.items() if k not in ("output_root","data_root")},sort_keys=True).encode()).hexdigest()[:12]
    out = outroot/cfg_tag
    cache_dir, results_dir = out/"forecast_cache", out/"method_results"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ds in cfg["datasets"]:
        rel,period = DATASETS[ds]
        src = dataroot/rel
        if not src.exists():
            raise FileNotFoundError(f"Required source CSV not found: {src}")
        head = pd.read_csv(src,nrows=5)
        if len(head.select_dtypes(include="number").columns)<1:
            raise RuntimeError(f"No numeric data in: {src}")
        sha = digest(src)
        df = pd.read_csv(src,nrows=cfg["max_rows"])
        cols = usable_cols(df,cfg["max_cols_per_dataset"])
        if len(cols)<2:
            print(f"[SKIP] {ds}: requires at least 2 usable columns for dev/test split",flush=True)
            continue
        for idx,col in enumerate(cols):
            # dev: exactly first eligible selected series per dataset, never evaluate
            paths.append(dict(ds=ds,col=col,col_idx=idx, period=period,
                              y=df[col].to_numpy(dtype=float),sha=sha,is_dev=(idx==0)))
    if not paths:
        raise RuntimeError("No datasets with separate development and evaluation channels")
    manifest = dict(phase="started", started_utc=now(), config=cfg, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    python=sys.version, platform=platform.platform(), input_source_hashes={p['ds']:p['sha'] for p in paths},
                    note="Published frozen Chronos-Bolt inference; model training corpus disjointness NOT independently established. Injection creates known tau; no breaks in natural series are claimed.",
                    intended_methods=BASE_METHODS)
    persist_json(out/"RUN_MANIFEST.json",manifest)
    pipe, rev = model_prepare(cfg["model"])
    manifest["model_revision"] = rev
    persist_json(out/"RUN_MANIFEST.json",manifest)
    # Fast compatibility smoke-test on GPU; no scientific results recorded.
    small = next(p for p in paths if not p['is_dev'])
    import torch
    with torch.inference_mode():
        sample = torch.tensor(small['y'][:cfg['context']][None],dtype=torch.float32)
        smoke = pipe.predict(sample,prediction_length=1)
        if smoke.ndim!=3 or smoke.shape[1]<5 or not np.isfinite(smoke.numpy()).all():
            raise RuntimeError("Chronos API smoke test failed")
    print("[PASS] GPU model smoke-test",flush=True)
    # Full unperturbed dev series first; threshold selection must not read test series.
    dev_files=[]
    for p in paths:
        if p['is_dev']:
            f=cache_run(pipe,cfg['model'],rev,p['ds'],p['col'],p['col_idx'],p['y'],"none",p['sha'],cfg,cache_dir)
            if f:dev_files.append(f)
    selected, grid = calibrate_h(dev_files,cfg,cfg["h_candidates"],cfg["target_alarms_per800"])
    grid.to_csv(out/"dev_detector_tuning.csv",index=False)
    manifest["dev_selected_h"] = selected
    manifest["dev_series"] = [f.stem for f in dev_files]
    manifest["stage"] = "thresholds_selected_on_development"
    persist_json(out/"RUN_MANIFEST.json",manifest)
    print(f"[DEV] detector h: {selected}. Operating rates on natural 'unperturbed' series only; NOT a guaranteed false-alarm null",flush=True)
    count=0
    for p in paths:
        if p['is_dev']:
            continue
        for kind in cfg['kinds']:
            f = cache_run(pipe,cfg['model'],rev,p['ds'],p['col'],p['col_idx'],p['y'],kind,p['sha'],cfg,cache_dir)
            if not f:
                continue
            summarize_cache(f,p['ds'],p['col'],p['col_idx'],kind,cfg,selected['score'],selected['data'],p['period'],results_dir)
            count+=1
            manifest.update(stage="running",n_series_scenarios_complete=count, last_complete=f.stem,last_utc=now())
            persist_json(out/"RUN_MANIFEST.json",manifest)
            print(f"[DONE] {count}: {f.stem}",flush=True)
    rows, summary = aggregate(results_dir,cfg,out)
    no = rows[rows.kind=="none"]
    if len(no):
        n_series = no[no.method=="BRCP"].shape[0]
        b = float(no[no.method=="BRCP"].n_alarms.sum())/max(1,int(no[no.method=="BRCP"].n_eval.sum()))*800
        d = float(no[no.method=="BRCP-data"].n_alarms.sum())/max(1,int(no[no.method=="BRCP-data"].n_eval.sum()))*800
        status = "UNMATCHED: cannot assert matched test false-alarm" if abs(b-d)>cfg["test_rate_tolerance_per800"] else "SIMILAR EMPIRICAL ALARM RATES (not a null FPR guarantee)"
        manifest['test_detector_rates'] = dict(score_per800=b,data_per800=d,n_series=n_series,tolerance_per800=cfg["test_rate_tolerance_per800"],status=status)
    manifest.update(stage="completed",finished_utc=now(),n_method_rows=len(rows),method_row_expected=count*len(BASE_METHODS))
    persist_json(out/"RUN_MANIFEST.json",manifest)
    print("[COMPLETE]",out, "\n",summary.to_string(index=False),flush=True)
    print("[IMPORTANT] Generated results require peer/scientific QA; no acceptance claim. Submit run directory to research team.")
    return out
