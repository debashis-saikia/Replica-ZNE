"""Analysis utilities for folding data."""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress


def log_linear_fit(scales: np.ndarray, values: np.ndarray) -> dict:
    scales = np.asarray(scales, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & (values > 0)
    if mask.sum() < 2:
        return {"slope": np.nan, "intercept": np.nan, "alpha_eff": np.nan, "R_fit": np.nan, "R2": np.nan}
    fit = linregress(scales[mask], np.log(values[mask]))
    return {
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "alpha_eff": float(np.exp(fit.slope)),
        "R_fit": float(np.exp(fit.intercept)),
        "R2": float(fit.rvalue**2),
    }
