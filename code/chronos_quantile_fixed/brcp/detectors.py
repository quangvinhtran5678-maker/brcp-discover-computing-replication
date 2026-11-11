"""
Online change detectors used to trigger recalibration.

Two variants are provided, and the contrast between them is the central
empirical claim of the paper:

ScoreCUSUM  (proposed) monitors the *conformity-score* stream, i.e. the
            forecaster's own error signal, rank-transformed into approximate
            conformal p-values against a frozen anchor window. Under the null
            "the current regime still looks like the anchor regime" these
            p-values are approximately Uniform(0,1); a persistent shift in the
            score distribution drives a Page-CUSUM across a threshold. This
            detects changes in the *model's competence*, not in the world.

DataCUSUM   (ablation) applies the identical recursion to the observation
            stream. It fires on distributional changes the forecaster absorbs
            without any loss of calibration, and can miss changes that leave
            the marginal distribution of the observations intact.

Both detectors expose Page's classical change-point estimate: the length of the
run since the CUSUM statistic was last at zero. BR-CP uses it to decide how
much calibration memory to discard.
"""

from __future__ import annotations

import numpy as np


class _PageCUSUM:
    """Two-sided Page-CUSUM on rank-transformed values against a frozen anchor.

    Parameters
    ----------
    anchor_size : size of the reference (null) sample.
    k           : CUSUM slack (reference value), in p-value units (max 0.5).
    h           : alarm threshold.
    min_anchor  : minimum anchor size below which the detector stays dormant.
    two_sided   : also alarm on a downward shift (intervals grown too wide).
    """

    space = "generic"

    def __init__(self, anchor_size: int = 250, k: float = 0.50, h: float = 5.0,
                 min_anchor: int = 60, two_sided: bool = True,
                 stat: str = "std", winsor: float = 4.0):
        """
        stat : 'std'  standardized, winsorized increment  z = clip((x-mu)/sd, +-W)
               'rank' conformal p-value increment          z = u - 1/2
        The statistics are compared empirically. Under an exchangeable fixed
        anchor, ranks have a useful null interpretation; they lose magnitude
        information relative to standardized increments. The numeric bound
        1/2 is not itself a causal explanation of detection delay, because
        rescaling increments, slack k and threshold h together preserves
        alarm times. Both operating points require fair empirical tuning.
        """
        if stat not in ("std", "rank"):
            raise ValueError("stat must be 'std' or 'rank'")
        self.anchor_size = anchor_size
        self.k = k
        self.h = h
        self.min_anchor = min_anchor
        self.two_sided = two_sided
        self.stat = stat
        self.winsor = winsor
        self.reset_all()

    def reset_all(self):
        self.anchor = None
        self.mu = 0.0
        self.sd = 1.0
        self.S_up = 0.0
        self.S_dn = 0.0
        self.run_up = 0          # steps since S_up was last at zero
        self.run_dn = 0
        self.delay_est = 0       # Page change-point estimate at the last alarm
        self.armed = False

    def set_anchor(self, values) -> bool:
        v = np.asarray(values, dtype=float).ravel()
        if v.size < self.min_anchor:
            self.armed = False
            return False
        self.anchor = np.sort(v[-self.anchor_size:])
        self.mu = float(np.mean(self.anchor))
        self.sd = float(np.std(self.anchor)) + 1e-9
        self.S_up = self.S_dn = 0.0
        self.run_up = self.run_dn = 0
        self.armed = True
        return True

    def _increment(self, value: float) -> float:
        if self.stat == "std":
            z = (float(value) - self.mu) / self.sd
            return float(np.clip(z, -self.winsor, self.winsor))
        n = self.anchor.size
        rank = np.searchsorted(self.anchor, float(value), side="right")
        return (rank + 0.5) / (n + 1.0) - 0.5

    def disarm(self):
        self.armed = False
        self.anchor = None
        self.S_up = self.S_dn = 0.0
        self.run_up = self.run_dn = 0

    def update(self, value: float) -> bool:
        """Feed one observation; return True iff an alarm is raised."""
        if not self.armed or self.anchor is None:
            return False
        z = self._increment(value)

        self.S_up = self.S_up + z - self.k
        if self.S_up <= 0.0:
            self.S_up, self.run_up = 0.0, 0
        else:
            self.run_up += 1

        if self.two_sided:
            self.S_dn = self.S_dn - z - self.k
            if self.S_dn <= 0.0:
                self.S_dn, self.run_dn = 0.0, 0
            else:
                self.run_dn += 1

        if self.S_up >= self.h:
            self.delay_est = self.run_up
            return True
        if self.two_sided and self.S_dn >= self.h:
            self.delay_est = self.run_dn
            return True
        return False


class ScoreCUSUM(_PageCUSUM):
    """Page-CUSUM driven by the conformity-score stream (proposed)."""
    space = "score"


class DataCUSUM(_PageCUSUM):
    """Page-CUSUM driven by the raw observation stream (ablation)."""
    space = "data"


DETECTORS = {"score": ScoreCUSUM, "data": DataCUSUM}
