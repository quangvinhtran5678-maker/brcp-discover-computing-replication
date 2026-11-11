# Causality and source-correctness tests for exploratory AR(1)-scorecaster PID

- Initialized from exactly 300 warm-up conformity scores.
- For each t, the scorecaster fits rolling AR(1) using only pairs available within scores[:t].
- `q_t` is issued before score[t] updates proportional and integral control states.
- Tested zero look-ahead: changing all scores at and after index 700 left all AR scorecasts before 700 unchanged.
- Tested four AR(1) OLS forecast positions (t=300,301,500,899) against independent brute-force rolling regression; agreement within 1e-10.
- Tested nonzero KI branch with KI=0.1, eta=0.1; all 108 / 108 scenarios ran.
- Full local source is `pid_ar1_replay.py`, `forced_integral_comparison.py`; method differs from paper's packaged Theta scorecaster and uses nonnegative-radius projection.

Status: EXPLORATORY / SOURCE-ARCHITECTURE-INSPIRED, not exact published PID authors' implementation.
