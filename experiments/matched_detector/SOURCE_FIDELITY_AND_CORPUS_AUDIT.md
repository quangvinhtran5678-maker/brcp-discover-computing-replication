# BR-CP scientific source-fidelity and checkpoint-provenance audit
**Date:** 2026-11-11. **Status:** Pre-submission / partial verification.  
**Manuscript:** Discover Computing, submission b34648b6-e715-42b1-888f-1f1ddd40aa40.  
**Rule:** No numeric result, training-corpus exclusion or reference-algorithm equivalence is claimed without supporting evidence.

## Source-to-code audit

| Research comparator | Source algorithm | Actual local code | Scientific status |
|---|---|---|---|
| AgACI, Zaffran et al. (2022) | Aggregation of ACI-based interval endpoints, using online expert aggregation (BOA among evaluated approaches); interval-bound pinball/quantile losses are relevant | `brcp/conformal.py::AgACI` forms a weighted mean of ACI radii, with exponential cumulative *Winkler interval-score* weights. | **NOT a source-equivalent AgACI implementation.** Call the published algorithm a related work and historical code `EWA-ACI expert variant` in narrative; archived CSV keys remain `AgACI`. |
| Conformal PID, Angelopoulos et al. (2023) | Quantile tracking (P), nonlinear error integrator (I), scorecaster (D) | `brcp/conformal.py::ConformalPID` has proportional tracker, tanh integrator, no learned D/scorecasting term. | **NOT a source-equivalent full Conformal PID implementation.** Only a simplified PI-like comparator. |
| ACI, Gibbs and Candes (2021) | Online update of adaptive target miscoverage alpha | Implemented projected updates and finite empirical-maximum fallback; clipping and capping counters were measured separately | Depends on boundary convention; do not inherit the paper's theoretical guarantee without accounting for clipping and adaptive resets. |
| Chronos-Bolt Small | Public checkpoint `amazon/chronos-bolt-small`, pinned revision `772f3d25d38aec6d914c8949dab4462e2d46f5d8` | Published checkpoint inference saved in author-provided per-stream cache and CPU-replayed | **Checkpoint observed; exact training/evaluation dataset membership not verified.** Model card's 27-dataset zero-shot benchmark claim does not independently demonstrate absence of all six exact BR-CP evaluation CSV versions. |

## Authoritative sources checked

1. M. Zaffran et al. (2022), *Adaptive Conformal Predictions for Time Series*, ICML / PMLR: https://proceedings.mlr.press/v162/zaffran22a.html
2. A. N. Angelopoulos, E. Candès, R. J. Tibshirani (2023), *Conformal PID Control for Time Series Prediction*, NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2023/hash/47f2fad8c1111d07f83c91be7870f8db-Abstract.html
3. Official `amazon/chronos-bolt-small` model card: https://huggingface.co/amazon/chronos-bolt-small
4. Official Amazon Chronos repository: https://github.com/amazon-science/chronos-forecasting
5. Reference implementation of the two detectors: `detectors_reference.py` in this package; corresponding accelerated calculations in `fresh_detector_null_audit.py` and 150-case exact-match test.

## New frozen synthetic AR1 audit (this package)

- Stationary Gaussian AR(1) null with phi=0.6, one-step oracle residual score; 90 independent development seeds and **300 disjoint** evaluation seeds.
- Detector threshold grid *frozen before test*: `h_score=5`, `h_data=16`, target 0.5 alarms/800, tolerance 0.2 difference/800.
- New test: score **0.5300 alarms/800**, data **0.1767/800**; difference **0.3533/800**, seed paired bootstrap 95% interval **[0.2500,0.4533]**.
- **Matching failed** on the independent stationary test. No score-versus-data superiority claim is supported at equal false-alarm operation.
- Descriptive post-break 120-step hit rate: level **0.7067** (score) vs **0.9833** (data); variance **0.9800** (score) vs **0.9733** (data). The different null frequencies preclude an equal-specificity comparison.
- Conditional median time to first alarm in variance case: score 3 versus data 24 steps, but cannot be interpreted as fair head-to-head superiority.
- This is a **synthetic detector-only diagnostic**, not a second Chronos forward pass, not a BRCP interval outcome, and not a cross-domain robustness certification.

## Remaining gates (not silently marked complete)

1. Published baseline reference-code replication (BOA/pinball AgACI and full PID with scorecaster), if exact published-method performance comparisons remain claimed.
2. Validated model training-corpus membership/exclusion for the six exact benchmark versions, if strict zero-shot disjointness is claimed.
3. A further prospectively specified design with statistically defensible false-alarm matching *confirmed on new data* before interpreting relative detection power. DO NOT retune on the 300 observed held-out seeds.
4. Reviewer-visible ZIP attachments, accurate author names/ORCID/affiliation/declarations, final corresponding-author sign-off and portal upload.

## Interpretation guardrails

Do not describe the independent AR1 test as proof BR-CP succeeds/fails overall; it tests detector operating characteristics only. Do not splice its numbers into the historical zero-shot ridge or Chronos tables. Treat observed advantages and weaknesses of BR-CP by break family and interval efficiency separately. No changes to archived numerical records were made in this addendum.

## Sequential calibration stability extension — NOT a pre-registered study sequence

The failure of the first 90-development-seed / 300-test-seed audit prompted a **new exploratory** design with more development data. It is *not* legitimate to describe the combined three-stage sequence as prospectively preregistered overall. New seeds and thresholds were frozen before each new phase.

- **Second phase:** 1,200 new development seeds, 900 disjoint new stationary test seeds. Frozen `h_score=5`, `h_data=13`. Null rates: score **0.5344**, data **0.4822** per 800; gap **0.0522**, paired bootstrap 95% CI **[-0.018, 0.122]**. Passes phase-specific numerical tolerance 0.15.
- **Third phase:** the same frozen thresholds, 1,000 different test seeds for each synthetic null/level/variance condition. Null rates score **0.524**, data **0.492** per 800; gap **0.032**, paired CI **[-0.027, 0.093]**. Retains matching on this *specific known-stationary synthetic null*.
- Third-phase post-change 120-step hit rates: **variance** score 0.984 vs data 0.977, paired difference 0.007, bootstrap CI [-0.005, 0.019]; **level** score 0.687 vs data 0.974, paired difference -0.287, CI [-0.316, -0.258]. Conditional median detection delay: variance 3 vs 17, level 44 vs 17 steps. These medians exclude non-detections and must be read together with hit rates.

This does **not** retroactively turn the earlier failed 90-seed study into a success. Nor does one AR(1) synthetic null establish equivalence across the real datasets / Chronos score streams. All three stages and all new-seed CSV results are included, not just the favorable follow-up.
