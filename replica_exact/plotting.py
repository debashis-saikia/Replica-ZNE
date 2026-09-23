"""Plotting helpers for paper-style replica simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_folded_curve(
    scales: np.ndarray,
    values: np.ndarray,
    exact: float,
    channel: str,
    strength: float,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(scales, values, "o-", label=f"{channel} (p={strength})")
    ax.axhline(exact, color="black", linestyle="--", linewidth=1, label="exact")
    ax.set_xlabel("scale s = 2c + 1")
    ax.set_ylabel("replica observable")
    ax.set_title(f"Replica interference, {channel}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{channel}_paper_exact.png", dpi=300)
    plt.close(fig)


def plot_diagnostics(
    scales: np.ndarray,
    values: np.ndarray,
    diagnostics: dict,
    channel: str,
    strength: float,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    log_valid = np.isfinite(values) & (values > 0)
    axes[0].plot(scales[log_valid], np.log(values[log_valid]), "o", label="log R(s)")
    fit_valid = np.isfinite(diagnostics["fitted_values"]) & (diagnostics["fitted_values"] > 0)
    axes[0].plot(
        scales[fit_valid],
        np.log(diagnostics["fitted_values"][fit_valid]),
        "-",
        label="linear fit",
    )
    axes[0].set_xlabel("scale s = 2c + 1")
    axes[0].set_ylabel("log R(s)")
    axes[0].set_title(f"Log-linearity (R2={diagnostics['R2']:.6f})")
    axes[0].legend()

    axes[1].plot(
        scales,
        diagnostics["absolute_relative_noisy_errors"],
        "o-",
        label="relative error, noisy R(s)",
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("scale s = 2c + 1")
    axes[1].set_ylabel("absolute relative error")
    axes[1].set_title("Relative error vs. noiseless")
    axes[1].legend()

    axes[2].plot(scales, values, "o", label="R(s)")
    axes[2].plot(scales, diagnostics["fitted_values"], "-", label="A exp(k s)")
    axes[2].axhline(diagnostics["exact"], color="black", linestyle="--", label="noiseless")
    axes[2].set_xlabel("scale s = 2c + 1")
    axes[2].set_ylabel("replica observable")
    axes[2].set_title(f"Exponential fit: alpha={diagnostics['alpha_eff']:.6f}")
    axes[2].legend()

    uncorrected_error = abs(diagnostics["noisy_error_max_scale"])
    rzne_error = abs(diagnostics["mitigated_error"])
    error_scales = np.array([1.0, 9.0])
    axes[3].plot(
        error_scales,
        [uncorrected_error, uncorrected_error],
        color="tab:orange",
        label=f"without correction ({uncorrected_error:.6f})",
    )
    axes[3].plot(
        error_scales,
        [rzne_error, rzne_error],
        color="tab:green",
        label=f"RZNE ({rzne_error:.6f})",
    )
    axes[3].set_xlabel("folding scale")
    axes[3].set_ylabel(r"$|R_{\mathrm{exact}} - R|$")
    axes[3].set_title("Absolute error comparison")
    axes[3].set_xlim(error_scales)
    axes[3].set_ylim(bottom=0.0)
    axes[3].grid(axis="y", alpha=0.3)
    axes[3].legend()

    fig.suptitle(f"{channel}, p={strength}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{channel}_diagnostics.png", dpi=300)
    plt.close(fig)


def plot_rzne_relative_error(
    diagnostics: dict,
    channel: str,
    strength: float,
    out_dir: Path,
) -> None:
    """Compare uncorrected and RZNE absolute errors as horizontal lines."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    uncorrected_error = abs(diagnostics["noisy_error_max_scale"])
    rzne_error = abs(diagnostics["mitigated_error"])
    scales = np.array([1.0, 9.0])
    ax.plot(
        scales,
        [uncorrected_error, uncorrected_error],
        color="tab:orange",
        label=f"without correction ({uncorrected_error:.6f})",
    )
    ax.plot(
        scales,
        [rzne_error, rzne_error],
        color="tab:green",
        label=f"RZNE ({rzne_error:.6f})",
    )
    ax.set_xlabel("folding scale")
    ax.set_ylabel(r"$|R_{\mathrm{exact}} - R|$")
    ax.set_title(f"Absolute error comparison, {channel}, p={strength}")
    ax.set_xlim(scales)
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{channel}_rzne_relative_error.png", dpi=300)
    plt.close(fig)
