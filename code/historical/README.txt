# Coverage Debt: Break-Responsive Conformal Calibration under Structural Breaks

Anonymous replication package for peer review.

This package contains the code, derived outputs, and figure sources used by the
manuscript. Raw public benchmark CSV files are not redistributed. Download them
from the official THUML Time-Series-Library source before rerunning the real-data
arm.

## Scope of the evidence

The paper contains three empirical components.

1. Controlled simulation: a frozen ridge autoregression is evaluated under seven
   break families and a no-break control. Break times are known by construction,
   and all calibration methods consume the same conformity-score streams.
2. Controlled stress tests on real-world benchmark series: a global forecaster is
   trained on electricity, traffic, and weather, frozen, and evaluated on five
   target datasets (ETTh1, ETTh2, ETTm1, ETTm2, and exchange rate). Structural
   breaks are injected so that their times are known. This is a semi-synthetic
   stress test on real data, not a natural-experiment claim.
3. Two externally dated case studies: the Lehman bankruptcy on 15 September 2008
   for exchange-rate series and the US H1N1 public-health emergency on 26 April
   2009 for influenza-like-illness series. These are negative results: BR-CP does
   not alarm at either event, and adaptive calibration changes little.

The evaluated main forecaster is a small global model, not a published time-series
foundation-model checkpoint. The optional `scripts/run_tsfm_real.py` supports a
separate confirmatory protocol for Chronos-Bolt and TimesFM; those model-specific
results are not claimed in the manuscript.

## Directory layout

```
brcp/
  conformal.py      Split CP, Rolling CP, ACI, AgACI, Conformal PID, BR-CP, Oracle
  detectors.py      two-sided Page-CUSUM detectors
  dgp.py            seven break families and a no-break control
  experiment.py     paired experiment runner
  fastbuf.py        ring buffer and order-statistic helpers
  forecasters.py    frozen ridge forecasters
  metrics.py        coverage debt, TTR-C, and interval metrics
  realdata.py       public benchmark loading and real-data protocol
scripts/
  download_official_data.py           cross-platform data downloader
  download_official_data_windows.ps1  Windows PowerShell data downloader
  tune_detector.py
  run_simulation.py
  run_detector_arm.py
  run_real_data.py
  run_case_studies.py
  make_figures.py
  make_real_figures.py
  run_tsfm_real.py
results/            derived tables, raw run summaries, arrays, and tests
figures/            figure sources in PDF and PNG
cache/              cached global-forecaster artifact used by the real-data arm
requirements.txt    tested scientific-Python environment for this release
```

The manuscript source is intentionally excluded from this supplementary archive
because the target journal uses double-blind peer review.

## Software environment

A portability smoke test for this release used Python 3.13.5 with the versions
listed in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

## Download the official benchmark data

Official source:

https://huggingface.co/datasets/thuml/Time-Series-Library

Original ETT repository for provenance checking:

https://github.com/zhouhaoyi/ETDataset

The expected directory layout is:

```
Class-1/
  ETT-small/ETTh1.csv
  ETT-small/ETTh2.csv
  ETT-small/ETTm1.csv
  ETT-small/ETTm2.csv
  electricity/electricity.csv
  traffic/traffic.csv
  weather/weather.csv
  exchange_rate/exchange_rate.csv
  illness/national_illness.csv
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_official_data_windows.ps1
```

Python:

```bash
python scripts/download_official_data.py --out data/Class-1
```

Alternatively, set `BRCP_DATA` to an existing `Class-1` directory.

Linux/macOS:

```bash
export BRCP_DATA=/absolute/path/to/Class-1
```

Windows PowerShell:

```powershell
$env:BRCP_DATA="C:\absolute\path\to\Class-1"
```

The exchange-rate file in this collection ends on 2010-10-10. Events after this
date are not available in that benchmark.

## Reproduce the controlled simulation

```bash
python scripts/tune_detector.py std
python scripts/tune_detector.py rank

for bt in level variance trend season ar predictability slow_drift none; do
    python scripts/run_simulation.py grid $bt
done

python scripts/run_simulation.py ablation
python scripts/run_simulation.py frontier
python scripts/run_simulation.py
python scripts/run_detector_arm.py
python scripts/make_figures.py
```

The main simulation uses 200 seeds per cell. Stage-wise checkpoints are written to
`results/`, so the grid can be run piecemeal.

## Reproduce the real-data stress tests

After setting `BRCP_DATA`:

```bash
python scripts/run_real_data.py pretrain
python scripts/run_real_data.py arl
python scripts/run_real_data.py main variance level none
python scripts/run_real_data.py sensitivity
python scripts/run_real_data.py doublebreak
python scripts/run_real_data.py analyse
python scripts/run_case_studies.py
python scripts/make_real_figures.py
```

The quantitative main arm contains 322 break windows from 35 target series and
368,081 calibration steps per method. The five target datasets are ETTh1, ETTh2,
ETTm1, ETTm2, and exchange rate. The illness dataset is used only for the dated
H1N1 case study.

## Evaluated BR-CP implementation

The released implementation matches the code used to create the included result
files. It contains four operational components:

1. Evidence-responsive step size. Before an alarm, the ACI step size increases
   smoothly with normalized CUSUM evidence. In the evaluated implementation, a
   short fast-step phase also follows an alarm.
2. Memory reset. Pre-change calibration memory is removed using Page's estimated
   change location when reset is enabled.
3. Small-buffer tolerance limit. A distribution-free one-sided tolerance limit is
   used while the post-reset calibration buffer is small.
4. Re-anchoring. The detector is re-anchored on post-alarm conformity scores before
   being re-armed.

The `BRCP-noboost` ablation removes the post-alarm fast-step phase and performs
similarly or marginally better in the evaluated settings. The phase is retained
in the main implementation because it was part of the algorithm that generated
the reported main results. The manuscript reports this ablation rather than
silently changing the algorithm after observing the result.

## Headline result files

The included outputs reproduce the reported comparisons. Key files are:

- `results/table_main.csv`: controlled-simulation summary.
- `results/table_ablation.csv`: component ablations.
- `results/table_detector.csv`: detector-space comparison.
- `results/real_table_main.csv`: real-data stress-test summary.
- `results/real_tests.csv`: paired tests for the real-data arm.
- `results/real_case_studies.csv`: dated-event case studies.
- `results/paper_numbers.txt`: generated manuscript headline values.

For the main real-data variance-break condition, the included output reports
coverage debt 3.21 for BR-CP and 3.82 for ACI with gamma=0.05, with TTR-C 8 and
27 steps, respectively. The interval-score difference between these two methods
is not statistically significant in the included paired test. These qualifications
are part of the manuscript and should not be removed.

A second break 40 steps after the first is a documented failure mode during the
re-anchoring blackout: the included double-break output shows BR-CP coverage debt
above the fast-ACI comparator in that setting.

## Optional foundation-model confirmation

`scripts/run_tsfm_real.py` can cache forecasts from Chronos-Bolt or TimesFM and
then replay identical conformity-score streams through the calibration methods.
This protocol is optional and is not part of the reported main results.

Do not call an evaluation zero-shot unless the model card confirms that the target
dataset was not used for pretraining. Coverage debt also requires an externally
known or experimentally injected break time; it should not be computed around a
break date selected after looking at calibration errors.

## Reproducibility notes

- The three scripts that previously contained machine-specific absolute paths now
  resolve the project root relative to their own location.
- `brcp/realdata.py` uses `BRCP_DATA` when set and otherwise looks for
  `data/Class-1` relative to this package.
- Figures and result files are retained so reviewers can inspect outputs without
  rerunning the full grid.
- No author names, affiliations, email addresses, acknowledgements, or funding
  statements are included in this anonymous supplementary archive.
