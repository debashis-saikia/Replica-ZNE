"""Analysis utilities for paper-style folded-interference data."""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress


def rzne_two_point(R1: float, R3: float) -> float:
    if R1 <= 0 or R3 <= 0:
        return np.nan
    return float(np.sqrt(R1**3 / R3))


def alpha_two_point(R1: float, R3: float) -> float:
    """Estimate the multiplicative noise factor from R3/R1."""
    if R1 <= 0 or R3 <= 0:
        return np.nan
    return float(np.sqrt(R3 / R1))


def richardson_two_point(R1: float, R3: float) -> float:
    return float((3.0 * R1 - R3) / 2.0)


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


def exponential_diagnostics(
    scales: np.ndarray,
    values: np.ndarray,
    exact: float,
    mitigated: float,
) -> dict:
    """Return log-linearity, error, and exponential-fit diagnostics."""
    scales = np.asarray(scales, dtype=float)
    values = np.asarray(values, dtype=float)
    fit = log_linear_fit(scales, values)
    alpha_two_point_value = alpha_two_point(values[0], values[1])
    fitted_values = fit["R_fit"] * fit["alpha_eff"] ** scales
    log_residuals = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0) & np.isfinite(fitted_values) & (fitted_values > 0)
    log_residuals[valid] = np.log(values[valid]) - np.log(fitted_values[valid])
    finite_residuals = np.isfinite(log_residuals)
    noisy_errors = exact - values
    if exact == 0.0:
        relative_noisy_errors = np.full(values.shape, np.nan, dtype=float)
        relative_mitigated_error = np.nan
    else:
        relative_noisy_errors = noisy_errors / exact
        relative_mitigated_error = (exact - mitigated) / exact
    return {
        **fit,
        "fitted_values": fitted_values,
        "log_residuals": log_residuals,
        "max_abs_log_residual": (
            float(np.max(np.abs(log_residuals[finite_residuals])))
            if finite_residuals.any()
            else np.nan
        ),
        "noisy_errors": noisy_errors,
        "noisy_error": float(noisy_errors[0]),
        "noisy_error_max_scale": float(noisy_errors[-1]),
        "relative_noisy_errors": relative_noisy_errors,
        "relative_noisy_error_max_scale": float(relative_noisy_errors[-1]),
        "mitigated_error": float(exact - mitigated),
        "relative_mitigated_error": float(relative_mitigated_error),
        "absolute_relative_noisy_errors": np.abs(relative_noisy_errors),
        "absolute_relative_mitigated_error": float(abs(relative_mitigated_error)),
        "alpha_two_point": alpha_two_point_value,
        "absolute_relative_alpha_error": (
            float(abs(1.0 - alpha_two_point_value))
            if np.isfinite(alpha_two_point_value)
            else np.nan
        ),
        "exact": float(exact),
        "mitigated": float(mitigated),
    }
