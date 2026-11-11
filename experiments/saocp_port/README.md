# Strongly adaptive online conformal baseline — scores-only port

- Based on the **official Salesforce SAOCP** score update/OGD core, under Apache 2.0.
- This is a scores-only port; not the original Merlion/Predictor or direct installation of `online_conformal`.
- Warm init: 300 conformal scores. `scale = sqrt(3) * max(abs(warm scores))` (matching documented reference default); horizon 1; lifetime 8.
- Exactly 108 / 108 saved Chronos-Bolt Small streams; 6 datasets, 6 disjoint evaluation channels, no-break/level/variance.
- `checkpoint/001.json` ... `108.json` contain individual, stable case results and event-level early miss metrics.
- 564 matched event observations; no synthetic second-boundary effect in the [tau,tau+100) window.
- Results `SAOCP_SUMMARY.csv`, `SAOCP_PAIRED_108.csv`, `PAIRED_EARLY_100_MISSRATE.csv`, `DATASET_CLUSTER_BOOTSTRAP.csv`.
- Cluster bootstrap uses only six dataset-level blocks; do not infer population superiority.
- No new GPU run, no training-corpus disjointness proof, no prospective pre-registration, no guarantee of source-identical reference implementation.

To rerun: use `python replay_chunks.py 0 108` (checkpoint/restart possible), then `python summarize_checkpoints.py`.

Source: https://github.com/salesforce/online_conformal/tree/main/online_conformal
License: Apache-2.0 (https://github.com/salesforce/online_conformal/blob/main/LICENSE)
