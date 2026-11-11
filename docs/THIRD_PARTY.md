# Third-party components and credit

- **SAOCP**: `experiments/saocp_port/saocp_scoreport.py` is a scores-only adaptation of Salesforce's `online_conformal` source, which declares Apache-2.0; upstream https://github.com/salesforce/online_conformal . It is not an official Salesforce reproduction. Upstream notices and the Apache-2.0 license text have been packaged under `third_party_licenses/APACHE-2.0.txt` for the scores-only source-derived port.
- **Conformal PID**: `experiments/pid_ar1/` implements a new causal AR(1)-scorecaster variant citing Angelopoulos et al.; paper/code https://github.com/aangelopoulos/conformal-time-series . It is not ThetaModel and is not an official reproduction.
- **AgACI**: archived “AgACI” numerical labels are not AgACI–BOA as published by Zaffran et al.; current method should be cited as local EWA-ACI. https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries .
- **Chronos-Bolt Small**: pretrained checkpoint by Amazon, used for inference via Hugging Face. Model weights are not included here.
- **THUML dataset**: source data must be obtained directly from the provider under its license. The repository includes recorded hashes and derived prediction/error outputs only.

**Action before making the repository public**: inspect all licenses and attribution, confirm author-approved license for original code. No blanket source redistribution permissions are asserted here.
