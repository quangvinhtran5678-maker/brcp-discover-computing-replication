# AR(1) scorecaster PID supplementary post-hoc comparison

- Input: **unchanged** Chronos-Bolt Small forward-pass cache already supplied by authors (`BRCP_Chronos_Bolt_GPU_Confirm_2026-11-11...zip`).
- Replays 108 channel x injection streams (6 datasets x 6 channels x 3 scenarios). No GPU or model training performed.
- Development: only the 6 distinct `col000_none` streams exist in the cache. A pre-run amendment from the intended 18 streams is documented in PROTOCOL.json. No evaluation streams are used to select parameters.
- Grid: eta = [.01,.03,.10], KI = [0,.10,.25], Csat=40, AR1 300-step rolling OLS scorecaster. Param choice minimizes dev macro normalized Winkler (per-stream warm Q90 normalization).
- Unrestricted dev choice KI=0 (P + AR(1)); remains transparently archived.
- Forced KI>0 development-choice eta=.10, KI=.10 (P + I + AR(1)), a follow-up after seeing KI=0; must be labeled exploratory, not confirmatory.
- 108/108 original forecast streams evaluated without future leakage; all q_t formed before observing score_t.
- Nonnegative radius projection was invoked (>0 cases), so no unmodified theoretical PID guarantee is claimed.
- `PID_AR1_PAIRED_108.csv` and `FORCED_POSITIVE_KI/PID_POSITIVE_KI_PAIRED.csv` contain case-level comparisons to corrected ACI and BRCP.
- Main results are not replaced. This is new supplementary evidence, not source-equivalent official ThetaModel or R BOA baseline.

Re-run on a runtime containing the two original ZIPs and `numpy`, `pandas` using `python pid_ar1_replay.py` and then `python forced_integral_comparison.py`.
