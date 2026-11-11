# Scientific gate status — BR-CP, 11 November 2026

| Gate | Evidence | Status |
|---|---|---|
| Existing Chronos GPU prediction cache | 108 evaluated streams + distinct development streams; source SHA-256 recorded | REPLAYED (not a fresh holdout) |
| Fix quantile off-by-one | Previous audit and tests; cached CPU replay of all 972 rows | VERIFIED IN EXISTING AUDIT |
| Additional SAOCP on same cache | source-derived port, 108 results + 108 checkpoints; 564 early windows | COMPLETE AS EXPLORATORY PORT |
| Additional causal PID with scorecaster | rolling AR(1), past-scores-only; KI=0 tuned; KI>0 constrained supplementary run | COMPLETE AS EXPLORATORY VARIANT |
| SAOCP vs BRCP comparison | dataset-cluster bootstrap (6 groups) + per-stream ratios | DESCRIPTIVE, LOW CLUSTER COUNT |
| Real-data matched null false alarms | rates matched in adapted posthoc setting, stationary reference not established on actual Chronos series | NOT FULLY VERIFIED |
| Original R-BOA/pinball AgACI | source identified; current runtime has no original R package execution | NOT DONE |
| Original author Theta scorecaster Conformal PID | AR(1) version tested, exact package not reproduced | NOT DONE |
| Original Chronos checkpoint training membership for every CSV | paper-level ETT+exchange exclusion evidence; exact six-file membership not audited | PARTIAL |
| Exact historic ridge training Electricity schema/provenance | historical artifact weights recovered; matching original Electricity file unavailable | NOT PROVEN |
| Accepted final scientific interpretation | manuscript and responses pre-submission; author must approve | AWAITING AUTHOR REVIEW |

**Bottom line:** This packet resolves the missing *empirical demonstration of two additional types of baseline on existing cached forecasts* only to the explicitly stated implementation scope. It does not establish acceptance, full method fidelity, exact pretraining disjointness, or global BR-CP dominance. Do not label these as completed where the evidence is insufficient.
