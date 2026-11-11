# SAOCP scores-only port tests

- Imported core expert lifecycle, prior weighting, scale-free OGD and pinball loss conventions from `salesforce/online_conformal` (Apache-2.0), ported off the Merlion/BasePredictor wrapper to a scores-only, one-step stream.
- Tested prefix invariance: perturbing all scores after index 610 left all prediction radii through 610 unchanged.
- Tested all produced radii in a sampled 800-step stream were nonnegative.
- Ran 108 of 108 saved Chronos scenario streams; all 108 per-case JSON checkpoints available.
- Matched 564 early 100-step event windows with the corrected ACI / BRCP trace archives (event index and case keys checked).
- Six-dataset cluster bootstrap (4000 replicates); estimates are exploratory and have few clusters.

Status: SOURCE-DERIVED PORT, not a byte-identical imported Salesforce package or its original model framework. Publication is an optional exploratory comparator, not an accepted-proof claim.

Source URL: https://github.com/salesforce/online_conformal/blob/main/online_conformal/saocp.py
License: Apache-2.0 (see https://github.com/salesforce/online_conformal/blob/main/LICENSE).
