"""Detection-power calibration: the certified floor of any CDI estimator.

An estimate of ~0 is only meaningful if the estimator could have detected signal
that was there. This module injects a channel carrying a KNOWN amount of directed
information at graded strengths, runs the estimator, and reports the smallest
true CDI whose bootstrap lower bound clears zero -- the certified detection floor
at the given sample size. Below the floor a null is uninformative; above it, a
null is a real negative.

Estimator-agnostic: pass any callable with signature
    estimator(y, x, z, clusters) -> object with .lcb95_bits and .cdi_bits
and a sampler that returns (y, x, z, clusters, true_cdi) for a strength.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CalibrationPoint:
    strength: float
    true_cdi: float
    est_cdi: float
    lcb95: float

    @property
    def certified(self) -> bool:
        return self.lcb95 > 0.0


@dataclass
class Calibration:
    points: list[CalibrationPoint]

    @property
    def detection_floor(self) -> float | None:
        """Smallest true CDI whose bound cleared zero (None if none did)."""
        cert = [p.true_cdi for p in self.points if p.certified and p.true_cdi > 0]
        return min(cert) if cert else None

    @property
    def leak_free(self) -> bool:
        """No false certification at true CDI == 0."""
        zeros = [p for p in self.points if p.true_cdi == 0.0]
        return all(not p.certified for p in zeros) if zeros else True


def detection_floor(estimator: Callable, sampler: Callable,
                    strengths) -> Calibration:
    """Run the estimator over a sweep of injected-signal strengths.

    Args:
        estimator: (y, x, z, clusters) -> result with .cdi_bits, .lcb95_bits.
        sampler:   strength -> (y, x, z, clusters, true_cdi).
        strengths: iterable of injected coupling strengths (include 0.0).
    """
    points = []
    for s in strengths:
        y, x, z, clusters, true = sampler(s)
        res = estimator(y, x, z, clusters)
        points.append(CalibrationPoint(
            strength=float(s), true_cdi=float(true),
            est_cdi=float(res.cdi_bits), lcb95=float(res.lcb95_bits)))
    return Calibration(points)
