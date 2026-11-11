# Public-repository release scope — 11 November 2026

**Status: research reproducibility candidate. No acceptance claim.**

The repository publishes source, numerical outputs and model-derived prediction/error caches, but **not** the original benchmark CSVs, pretrained Chronos weights, editorial letters, or reviewers' confidential reports. The historical global-ridge data lineage is not byte-identical to a verified training corpus. Some baselines are local ports/simplified implementations rather than executions of authors' original packages.

## Methodological boundaries

- The initial GPU conformal quantile implementation had an off-by-one error. Its original results are retained **only** as a separately labeled audit trail. For new comparisons use the corrected outputs.
- The detector parity is shown on post-hoc synthetic AR(1) held-outs with tuned thresholds, not established for all real-data null distributions.
- SAOCP is an attributed, scores-only port; PID with AR(1) scorecaster is a custom causal variant; AgACI-BOA and PID-ThetaModel original author's code were **not** executed.
- Real-data results use a pinned Chronos-Bolt-Small checkpoint; no claim is made that the exact benchmark CSV files are byte-disjoint from checkpoint pretraining.
- Favourable and unfavourable results are retained; avoid unconditional superiority claims.

## Licensing and citation

Original group-owned source is shared without a broad open-source license pending author approval. The SAOCP source-derived port credits the original Salesforce code; see `docs/THIRD_PARTY.md` and `third_party_licenses/APACHE-2.0.txt`. This public release is not a substitute for the private point-by-point response to peer reviewers.

## Data provenance

Obtain raw data from the upstream THUML Time-Series-Library with its terms and verify downloaded versions/checksums. No confidential original inputs have been inferred or fabricated.
