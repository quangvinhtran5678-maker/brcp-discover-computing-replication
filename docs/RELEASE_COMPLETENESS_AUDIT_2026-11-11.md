# Release completeness audit — 11 November 2026

**Release status: INCOMPLETE; not yet ready for a full reproducibility claim.** Checked directly against `SHA256SUMS.txt` and the GitHub recursive tree.

- Expected release files in the SHA-256 manifest: **958**.
- GitHub files at latest completed audit: **632** (including manifest and this audit document).
- Missing expected release files: **328**.
- Published Chronos forecast cache files: **31**; additional model-derived cache binaries and historical `.npz` files are pending.
- Missing material includes 8 PDFs (manuscript and figures) and 320 files under `results/`.
- Raw third-party benchmark CSVs are intentionally *not* part of this release.

The repository does contain the BR-CP research code and numerous CSV results. The legacy (off-by-one) Chronos quantile results must not be cited in place of corrected results. Source-derived baseline implementations are not identical to all original official packages, and neither exact original ridge training-data provenance nor precise train/test overlap of the pinned Chronos model has been fully established.

**To complete this archive:** sync the audited release ZIP from Google Drive onto the existing repository using an incremental, non-force commit; then independently verify every file against `SHA256SUMS.txt`. Only after all expected paths and hashes match should a pinned commit be inserted into the manuscript as complete code availability.

**Repository:** https://github.com/quangvinhtran5678-maker/brcp-discover-computing-replication

This document records a snapshot, not an always-live counter. Do not change scientific claims based solely on repository file counts.
