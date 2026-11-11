# BR-CP: Coverage Debt and Break-Responsive Conformal Calibration

**Research replication release candidate — 11 November 2026. Documentation and checks are being finalized; this is not an independent confirmation of all claims.**

This repository contains research-group source code, scripts, saved model-derived scores, numerical outputs, figures and a versioned record of a Major Revision to *Discover Computing*. It preserves results that are unfavourable to BR-CP as well as favourable ones.

## Structure

- `paper/`: revised LaTeX manuscript and figures; paper is not an acceptance notice.
- `code/historical/`: frozen historical global-ridge/synthetic experiment code and result tables. **It does not establish byte-identical provenance of the original global-ridge training corpus.**
- `code/chronos_quantile_fixed/`: corrected conformal quantile implementation, evaluator and tests.
- `results/chronos_forecast_cache/`: model-derived forecast caches for 108 evaluated series/scenarios and development inputs. These do not include the original THUML raw dataset CSV files.
- `results/chronos_quantile_fixed/`: corrected 972-method-row results (plus per-series traces, as available).
- `results/chronos_original_DO_NOT_CITE/`: old outputs using an off-by-one conformal quantile; **DO NOT cite these numerical results**.
- `results/historical/`: original archived synthetic/real results; see historical provenance limits below.
- `experiments/matched_detector/`: development threshold calibration, independent frozen-null experiments and reported failures/successes; not a proof of matched false alarms on stationary natural Chronos data.
- `experiments/saocp_port/`: scores-only SAOCP port and 108-stream exploratory results, adapted with attribution from Salesforce online_conformal (Apache-2.0); NOT the original package executed end-to-end.
- `experiments/pid_ar1/`: causal AR(1)-scorecaster PI/PID variant, not the original authors' ThetaModel implementation.
- `notebooks/`: Colab notebook used for GPU Chronos-Bolt study and validation tools. Notebook paths and external datasets need user-specific configuration.
- `docs/`: detailed scientific scope/limitations, evidence gate table, audit notes.

## Quick checks (CPU)

```bash
python -m pip install -r requirements.txt
python code/chronos_quantile_fixed/test_quantile.py
python experiments/matched_detector/test_detector_equivalence.py
python scripts/verify_new_baselines.py
```

Do not infer that the tests above independently reproduce full GPU inference or original external baseline packages. Some experiment scripts refer to fixed study machine paths; they are included for provenance and require adapting to a local `BRCP` workspace to rerun. `notebooks/BRCP_GPU_Chronos_Confirmatory_2026-11-11.ipynb` provides the archived GPU procedure and can regenerate model-derived forecasts after downloading the six benchmark datasets.

## Data and model provenance

- Raw time-series benchmark CSVs (ETT hourly/minute, Exchange Rate, Illness) are **not redistributed here**; use the official [THUML Time-Series-Library repository](https://github.com/thuml/Time-Series-Library) and the dataset's applicable terms.
- The Chronos checkpoint used in the GPU study was `amazon/chronos-bolt-small`, identified by model revision in the original run manifest. These results are based on a public pretrained checkpoint, **not model retraining**.
- Historical frozen global-ridge weights are archived, but exact original Electricity/Traffic/Weather training input versions have not been reconstructed. Recovered Electricity source was from an incompatible Australian market dataset, so **do not use it** for a claimed exact pretraining reproduction.
- Training overlap of the exact six tested CSV files with the pinned Chronos pretraining corpus has not been excluded byte by byte. ETT and Exchange Rate have family-level zero-shot documentation, but exact benchmark-file provenance and National Illness need separate confirmation.

## Scientific caveats

1. The original Chronos evaluation code contained an **off-by-one** quantile error; use `results/chronos_quantile_fixed`, not the initial snapshot. The original snapshot is retained solely for audit and diagnosis.
2. Initial data-space vs score-space detector comparison was **not** matched for held-out false-alarm rates. Later development-calibration experiments with new synthetic AR(1) held-outs achieved approximate parity, but they were motivated by examining the initial failure, and do not validate all real-world null regimes.
3. BR-CP improved local post-variance-break coverage debt/early miss rates against selected baselines but was **not best** for every break family, global interval score, or competing method. Do not claim universal dominance.
4. SAOCP is a **source-derived score port**, PID–AR(1) is a **custom implementation**. Original paper-level AgACI–BOA and Conformal PID–ThetaModel baselines are **not reproduced** by this package.
5. Reported paired dataset-bootstrap confidence intervals use only six source-dataset clusters for the newer TSFM experiments. They are descriptive; cross-dataset generalization claims need more independent benchmarks.
6. Extra baselines, detector calibration and stage-three results are post-hoc; the frozen stages, including negative results, are retained.

## Citation and public release

Do not cite this repository as a proof of accepted publication. A permanent release tag/DOI should be minted only after corresponding-author approval of the final released source, third-party attribution and benchmark-sharing terms. Editorial communications and reviewer reports are intentionally **not** in this public release candidate. Follow the latest revision's manuscript and response to reviewers as separate, **private submission** files.

## Tests and licence

No blanket open-source licence is asserted for the authors' original contributions pending author choice. Source-derived code retains upstream attribution; see `docs/THIRD_PARTY.md`.
