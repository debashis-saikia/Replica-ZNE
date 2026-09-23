"""R-ZNE and standard two-point Richardson estimators."""

from __future__ import annotations

import numpy as np


def rzne_two_point(R1: float, R3: float) -> float:
    if R1 <= 0 or R3 <= 0:
        return np.nan
    return float(np.sqrt(R1**3 / R3))


def richardson_two_point(R1: float, R3: float) -> float:
    return float((3 * R1 - R3) / 2)
