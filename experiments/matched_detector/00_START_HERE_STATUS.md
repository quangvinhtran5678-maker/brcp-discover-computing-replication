# BR-CP Discover Computing — Scientific revision follow-up (11 Nov 2026)

**Manuscript:** Coverage Debt and Adaptive Conformal Calibration after Structural Breaks  
**Submission:** b34648b6-e715-42b1-888f-1f1ddd40aa40  
**Deadline:** 14 Nov 2026. Authors upload themselves.  
**STATUS:** PRE-SUBMISSION — reviewer-satisfactory science and editorial acceptance are NOT certified.

## What changed in this follow-up

1. Tested new-seed detector-only stationary Gaussian AR(1) null with 90 calibration seeds and 300 independent test seeds. Calibration transferred poorly: **score 0.530, data 0.177 alarms/800**, gate failed. This negative result is fully retained.
2. Motivated by that failure, ran a new **exploratory** precision-calibration study (1200 dev, 900 independent null test; selected thresholds score=5, data=13). Test rate: 0.5344 vs 0.4822 alarms/800. Not preregistered at the level of the entire sequence.
3. Held thresholds frozen for a further 1000 independent synthetic-null/level/variance test series. Null rate: 0.524 vs 0.492 alarms/800. Level detection probability 0.687 vs 0.974 (score vs data); variance detection probability 0.984 vs 0.977, with conditional median delays 3 vs 17 steps. No universal superiority claim.
4. Independently verified numerical CUSUM alarm-count implementation against the exact source Python detector for 150 seed/shift/h configurations.
5. Compared local source code with published AgACI and Conformal PID methods: the local EWA-ACI/Winkler and simplified PI-like implementations are **not faithful full published-method replications**. A source-reference audit is included.
6. Inspected Chronos-Bolt Small official model card; exact training membership for the specific six BR-CP evaluation CSVs **remains unverified**, despite model-card statements about other zero-shot benchmarks.
7. Corrected manuscript overclaim about the historical archived global ridge training corpus: evaluation was reproduced with saved weights, but exact electricity input, original source-level disjointness, and original fitting were not fully reconstructed from source-matched files.
8. Updated complete manuscript, added detector-only section and table, updated full point-by-point response, regenerated PDF and checked pages.

## Latest deliverables

- `01_MANUSCRIPT/BRCP_Revised_Manuscript_Further_Science_Audit_2026-11-11.pdf` — 24 pages
- `02_REVIEWER_RESPONSE/BRCP_Point_by_Point_Response_Further_Science_Audit_2026-11-11.docx` — 15 rendered pages
- `02_REVIEWER_RESPONSE/BRCP_Point_by_Point_Response_Further_Science_Audit_2026-11-11.pdf`
- `03_LATEX_SOURCE/main.tex` and `Fig1.pdf`–`Fig7.pdf`
- `04_NEW_DETECTOR_AUDIT/*` — all frozen protocols, source, per-seed CSVs, paired bootstrap, and raw stage outcomes
- `05_SOURCE_AND_STATUS/SOURCE_FIDELITY_AND_CORPUS_AUDIT.md`
- `06_PRIOR_EVIDENCE/BRCP_Scientific_Update_Posthoc_Chronos_2026-11-11.zip` — previous full scientifically audited package, retained as historical evidence (DO NOT replace new manuscript with its older version)
- `SHA256SUMS.txt` — file integrity list

## Still not done (do not claim otherwise)

- Published full AgACI-BOA/pinball and Conformal PID with learned scorecaster implemented and verified source-equivalently.
- Verified non-overlap of Chronos-Bolt pretraining inputs against every exact target dataset version.
- Source-identical re-execution of historical global ridge **training** and identification of correct official Electricity raw training file (the unrelated Australian electricity CSV must never be used).
- Empirical false-alarm-rate matching proved on actual Chronos/real benchmark test streams under a known no-change null; matched budget shown above applies only to the new synthetic AR(1) null.
- Author approval of affiliation/ORCID, Funding/COI/ethics/AI declarations, journal figure/material requirements and actual online resource visibility.

## Action for author now

No new Colab GPU run is needed for this follow-up. Before portal upload, review scientific scope and author identifiers, choose the data availability declaration consistent with the text, and upload latest manuscript plus Point-by-Point Response and correctly versioned supporting resources. If the team wants *full published baseline equivalence* or *strict pretraining non-overlap* as central claims, those remain separate open work requiring external provenance and source-equivalent implementations.

Scientific source URLs appear in `05_SOURCE_AND_STATUS/SOURCE_FIDELITY_AND_CORPUS_AUDIT.md`.
