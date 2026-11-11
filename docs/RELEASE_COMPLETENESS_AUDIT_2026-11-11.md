# Repository completeness audit — 11 November 2026

This is an automatically checked release-status snapshot. **Do not use this repository as a complete replication archive until all expected files are present and their SHA-256 hashes have been verified.**

- Expected hashed release files: **958** (see `SHA256SUMS.txt`).
- Files listed by the recursive Git tree during this audit: **600** (including `SHA256SUMS.txt`).
- Expected files not present at audit time: **359**.
- Missing files include PDF figures/manuscript outputs and model-derived Chronos `.npz` forecast caches.
- Historical and exploratory evidence must remain labelled distinctly from corrected results.
- The original benchmark CSVs are intentionally excluded from distribution. Missing benchmark raw files are *not* counted among the 359 expected release files.
- The original Chronos evaluation quantile bug has been corrected in the separately labelled corrected results.
- Not all original published third-party baselines are reproduced exactly; see the scientific caveats in README.

**Action required before code availability can be asserted:** finish uploading the expected files, verify every listed SHA-256 hash, and pin the verified commit SHA in the manuscript. This status document should be updated once the checks pass.
