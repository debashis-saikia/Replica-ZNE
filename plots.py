"""
Publication-quality plotting script for the Replica-ZNE simulation data.

Designed for PRX Quantum / PRA / Quantum-style figures:
  Fig. 1: Noise-scaling curves R(s) with exponential fits, grouped by scope.
  Fig. 2: Residual bias (R - R_exact) of raw vs R-ZNE estimator vs noise strength.
  Fig. 3: Relative mitigation error (log scale) and exponential-fit quality.
  Fig. 4: Effective decay/amplification factor alpha_eff vs theoretical alpha.

Key changes from the previous version, and why:

  * Fig. 3's error panel is now LOG-scale. The entire point of this figure is
    the multi-order-of-magnitude gap between the raw and R-ZNE estimators for
    ancilla-local (eigenoperator) noise; on a linear percentage axis that gap
    is invisible (R-ZNE sits indistinguishably close to 0%). A log axis is
    what actually makes the effect "vivid."
  * Ancilla-local (eigenoperator) vs system (operator-mixing) noise is now an
    explicit visual encoding (marker fill, panel border) rather than left
    for the reader to infer from the channel name -- this is the paper's
    central scientific contrast and it should read at a glance.
  * Fig. 4 now overlays the THEORETICAL alpha(p) for each channel as a thin
    reference curve. The deviation of alpha_eff from that line for
    system-noise channels *is* the numerical evidence for operator mixing;
    the previous version only plotted alpha_eff with no reference to compare
    against, which hid the story.
  * Fig. 2 now plots the residual (R - R_exact) rather than the raw R value.
    Two curves that both sit near 0.76 are hard to visually compare; their
    deviation from 0 is the actual quantity of interest.
  * A shared, explicit floor is applied before any log() or log-percentage
    computation, so that exact/near-exact cancellations (R-ZNE error at
    machine precision) don't raise warnings or silently vanish from a log
    plot -- they show up pinned at the floor, which is itself informative.

Input:
    *.npy files (optionally nested in p_* folders) under --data-dir, each a
    dict with keys: channel, strength, scope, scales, values, exact, R1, R3,
    RZNE, alpha_eff, R2. `scope` should be "ancilla" or "system".

Usage:
    python prx_quantum_replica_plots.py --data-dir /path/to/npy/files
    python prx_quantum_replica_plots.py --data-dir /path/to/npy/files --out-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Global publication style
# ---------------------------------------------------------------------------

def _tex_actually_works() -> bool:
    """Checking that latex/pdflatex are on PATH is not enough -- a broken
    or incomplete LaTeX install (missing packages, etc.) will fail at
    *render* time, deep inside matplotlib's savefig call, which is a much
    worse place to discover it. Do a real trial render instead."""
    if shutil.which("latex") is None or shutil.which("pdflatex") is None:
        return False
    old = mpl.rcParams["text.usetex"]
    try:
        mpl.rcParams["text.usetex"] = True
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$\alpha_{\rm eff}$")
        fig.canvas.draw()
        plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        return False
    finally:
        mpl.rcParams["text.usetex"] = old


USE_TEX = _tex_actually_works()

if USE_TEX:
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"""
            \usepackage{amsmath}
            \usepackage{amssymb}
            \usepackage{bm}
        """,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
    })
else:
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "font.size": 10,
})

# ---------------------------------------------------------------------------
# Consistent manuscript palette
# ---------------------------------------------------------------------------

COLORS = {
    "depolarizing": "#0072B2",
    "dephasing": "#009E73",
    "amplitude_damping": "#D55E00",
    "phase_flip": "#CC79A7",
}

LABELS = {
    "depolarizing": r"Depolarizing",
    "dephasing": r"Dephasing",
    "amplitude_damping": r"Amplitude damping",
    "phase_flip": r"Phase flip",
}

MARKERS = {
    "depolarizing": "o",
    "dephasing": "s",
    "amplitude_damping": "^",
    "phase_flip": "D",
}

# Theoretical alpha(p) for each channel, used as a reference line in Fig. 4.
# NOTE: adjust these if your channel parametrization differs (e.g. if
# "strength" for amplitude damping is gamma rather than p, this is already
# consistent with that -- alpha = sqrt(1-gamma)).
ALPHA_THEORY = {
    "phase_flip": lambda p: 1 - 2 * p,
    "dephasing": lambda p: 1 - 2 * p,
    "depolarizing": lambda p: 1 - 4 * p / 3,
    "amplitude_damping": lambda p: np.sqrt(np.clip(1 - p, 0, None)),
}

EXACT_COLOR = "#000000"
FIT_ALPHA = 0.55
ERROR_FLOOR = 1e-16   # numerical floor for log-scale error plots
PAPER_STRENGTHS = {0.02, 0.04, 0.06, 0.08}

CHANNEL_ORDER = ["phase_flip", "dephasing", "depolarizing", "amplitude_damping"]


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------

def load_records(data_dir: Path):
    records = []
    data_dir = data_dir.expanduser().resolve()

    for path in sorted(data_dir.rglob("*.npy")):
        try:
            obj = np.load(path, allow_pickle=True)
            d = obj.item()
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue

        required = {
            "channel", "strength", "scope", "scales", "values",
            "exact", "R1", "R3", "RZNE", "alpha_eff", "R2"
        }
        if not required.issubset(d):
            print(f"Skipping {path.name}: missing required keys")
            continue

        records.append({
            "file": path.name,
            "channel": str(d["channel"]),
            "strength": float(d["strength"]),
            "scope": str(d["scope"]).lower(),
            "scales": np.asarray(d["scales"], dtype=float),
            "values": np.asarray(d["values"], dtype=float),
            "exact": float(d["exact"]),
            "R1": float(d["R1"]),
            "R3": float(d["R3"]),
            "RZNE": float(d["RZNE"]),
            "alpha": float(d["alpha_eff"]),
            "R2": float(d["R2"]),
        })

    if not records:
        raise RuntimeError(f"No compatible .npy files found in {data_dir}")

    # Deduplicate (channel, scope, p), keeping the first unless the
    # duplicates genuinely disagree, in which case fail loudly rather than
    # silently picking one.
    unique = {}
    for r in records:
        key = (r["channel"], r["scope"], round(r["strength"], 12))
        if key not in unique:
            unique[key] = r
        else:
            old = unique[key]
            if not (
                np.allclose(old["scales"], r["scales"])
                and np.allclose(old["values"], r["values"])
                and np.isclose(old["RZNE"], r["RZNE"], equal_nan=True)
            ):
                raise ValueError(
                    f"Conflicting duplicate records for {key}: "
                    f"{old['file']} vs {r['file']}"
                )

    return list(unique.values())


def fit_curve(scales, values):
    """Fit ln R = b + m s, returning predicted R(s), alpha, and R^2."""
    if np.any(values <= 0):
        return np.full_like(values, np.nan), np.nan, np.nan

    x = np.asarray(scales)
    y = np.log(np.asarray(values))
    m, b = np.polyfit(x, y, 1)
    pred = np.exp(b + m * x)
    alpha = np.exp(m)

    ss_res = np.sum((y - (b + m * x)) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return pred, alpha, r2


def save_figure(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.svg")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def label_panel(ax, label: str, scope: str):
    tag = "ancilla-local" if scope == "ancilla" else "system (folded)"
    ax.annotate(
        f"{label}  " + r"$\it{(" + tag.replace(" ", r"\ ") + r")}$",
        xy=(0.0, 1.04), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=8.6, color="black",
    )


def _style_axes(ax, scope=None):
    ax.minorticks_on()
    ax.tick_params(which="major", direction="in", top=True, right=True, length=3, width=0.7)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.5, width=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.7)
    # Visually mark ancilla-local (eigenoperator) panels with a light green
    # tint on the axes background -- this is the "exact cancellation"
    # regime and should read as visually distinct from system-noise panels
    # even before the reader parses any labels.
    if scope == "ancilla":
        ax.set_facecolor("#F3FBF6")
    elif scope == "system":
        ax.set_facecolor("#FBF5F3")


def ordered_channels(records):
    present = {r["channel"] for r in records}
    return [c for c in CHANNEL_ORDER if c in present]


def legend_kwargs(**overrides):
    base = dict(
        frameon=True, fancybox=False, facecolor="white",
        edgecolor="0.7", framealpha=0.95, borderpad=0.35,
        handlelength=1.8, handletextpad=0.5, fontsize=8.0,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Figure 1: noise scaling
# ---------------------------------------------------------------------------

def figure_scaling(records, out_dir):
    channels = ordered_channels(records)
    n = len(channels)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(8.0, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.90,
                         wspace=0.26, hspace=0.40)

    for ax, channel in zip(axes, channels):
        rs = sorted([r for r in records if r["channel"] == channel],
                    key=lambda r: r["strength"])
        color = COLORS[channel]
        marker = MARKERS[channel]
        scope = rs[0]["scope"]

        for r in rs:
            x, y = r["scales"], r["values"]
            ax.plot(x, y, marker=marker, markersize=4.2, linewidth=0,
                     color=color, markeredgecolor="white", markeredgewidth=0.55,
                     label=rf"$p={r['strength']:.2f}$")

            pred, alpha, r2 = fit_curve(x, y)
            if np.all(np.isfinite(pred)):
                xs = np.linspace(x.min(), x.max(), 200)
                m, b = np.polyfit(x, np.log(y), 1)
                ax.plot(xs, np.exp(b + m * xs), color=color, linewidth=0.95,
                         alpha=FIT_ALPHA)

        exact = rs[0]["exact"]
        ax.axhline(exact, color=EXACT_COLOR, linewidth=0.85,
                    linestyle=(0, (3, 2)), zorder=0, label=r"$R_{\rm exact}$")

        ax.set_xlabel(r"Noise scale $s$")
        ax.set_ylabel(r"Replica signal $R(s)$")
        label_panel(ax, LABELS[channel], scope)
        _style_axes(ax, scope)
        ax.legend(**legend_kwargs(loc="best",
                                   ncol=2 if len(rs) > 2 else 1))

        vals = np.concatenate([r["values"] for r in rs])
        lo, hi = min(vals.min(), exact), max(vals.max(), exact)
        pad = 0.08 * (hi - lo if hi > lo else 0.01)
        ax.set_ylim(lo - pad, hi + pad)

    for ax in axes[len(channels):]:
        ax.set_visible(False)

    save_figure(fig, out_dir, "fig1_noise_scaling")


# ---------------------------------------------------------------------------
# Figure 2: residual bias, raw vs mitigated
# ---------------------------------------------------------------------------

def figure_mitigation(records, out_dir):
    channels = ordered_channels(records)
    n = len(channels)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(8.2, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, channel in zip(axes, channels):
        rs = sorted([r for r in records if r["channel"] == channel],
                    key=lambda r: r["strength"])
        scope = rs[0]["scope"]
        p = np.array([r["strength"] for r in rs])
        raw_resid = np.array([r["R1"] - r["exact"] for r in rs])
        mit_resid = np.array([r["RZNE"] - r["exact"] for r in rs])

        ax.axhline(0.0, color=EXACT_COLOR, linewidth=0.85,
                    linestyle=(0, (3, 2)), zorder=0, label=r"$R_{\rm exact}$ (zero bias)")
        ax.plot(p, raw_resid, color="0.35", marker=MARKERS[channel],
                 linewidth=1.0, linestyle="--", markersize=4.4,
                 markerfacecolor="white", markeredgewidth=0.9, label="Raw estimator")
        ax.plot(p, mit_resid, color=COLORS[channel], marker=MARKERS[channel],
                 linewidth=1.3, markersize=4.4, label="R-ZNE")

        ax.set_xlabel(r"Physical noise strength $p$")
        ax.set_ylabel(r"Bias  $\hat{R} - R_{\rm exact}$")
        label_panel(ax, LABELS[channel], scope)
        _style_axes(ax, scope)
        ax.legend(**legend_kwargs(loc="best"))

    for ax in axes[len(channels):]:
        ax.set_visible(False)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig2_raw_vs_mitigated")


# ---------------------------------------------------------------------------
# Figure 3: relative error (log scale) and fit quality
# ---------------------------------------------------------------------------

def figure_error_and_r2(records, out_dir):
    channels = ordered_channels(records)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.86, wspace=0.26)

    for channel in channels:
        rs = sorted([r for r in records if r["channel"] == channel],
                    key=lambda r: r["strength"])
        p = np.array([r["strength"] for r in rs])
        exact = np.array([r["exact"] for r in rs])
        raw = np.array([r["R1"] for r in rs])
        mit = np.array([r["RZNE"] for r in rs])
        r2 = np.array([r["R2"] for r in rs])

        color = COLORS[channel]
        marker = MARKERS[channel]

        raw_err = np.clip(100.0 * np.abs(raw - exact) / np.abs(exact), ERROR_FLOOR, None)
        mit_err = np.clip(100.0 * np.abs(mit - exact) / np.abs(exact), ERROR_FLOOR, None)

        ax1.plot(p, raw_err, marker=marker, markersize=4.4, linewidth=1.0,
                  color=color, linestyle="--", markerfacecolor="white",
                  markeredgewidth=0.9)
        ax1.plot(p, mit_err, marker=marker, markersize=4.4, linewidth=1.4,
                  color=color, markerfacecolor=color, markeredgewidth=0.7,
                  label=LABELS[channel])

        ax2.plot(p, r2, marker=marker, markersize=4.4, linewidth=1.15,
                  color=color, markerfacecolor=color, markeredgewidth=0.7,
                  label=LABELS[channel])

    ax1.set_yscale("log")
    ax1.set_xlabel(r"Physical noise strength $p$")
    ax1.set_ylabel(r"Relative error (\%)")
    ax1.set_title("Open markers, dashed: raw.  Filled, solid: R-ZNE.",
                   fontsize=8.0, loc="left", style="italic")
    _style_axes(ax1)
    ax1.legend(**legend_kwargs(loc="lower right"))

    ax2.axhline(1.0, color=EXACT_COLOR, linewidth=0.8, linestyle=(0, (4, 2)), zorder=0)
    ax2.set_xlabel(r"Physical noise strength $p$")
    ax2.set_ylabel(r"Exponential-fit $R^2$")
    _style_axes(ax2)
    ax2.legend(**legend_kwargs(loc="lower left"))

    save_figure(fig, out_dir, "fig3_error_and_fit_quality")


# ---------------------------------------------------------------------------
# Figure 4: alpha_eff vs theory
# ---------------------------------------------------------------------------

def figure_alpha(records, out_dir):
    channels = ordered_channels(records)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.88)

    p_dense = np.linspace(0.0, 0.1, 200)

    for channel in channels:
        rs = sorted([r for r in records if r["channel"] == channel],
                    key=lambda r: r["strength"])
        p = np.array([r["strength"] for r in rs])
        alpha = np.array([r["alpha"] for r in rs])
        color = COLORS[channel]

        if channel in ALPHA_THEORY:
            ax.plot(p_dense, ALPHA_THEORY[channel](p_dense), color=color,
                     linewidth=1.0, alpha=0.45, zorder=1)

        ax.plot(p, alpha, marker=MARKERS[channel], markersize=5.0, linewidth=0,
                 color=color, markerfacecolor=color, markeredgecolor="white",
                 markeredgewidth=0.6, zorder=3, label=LABELS[channel])

    ax.plot([], [], color="0.4", linewidth=1.0, alpha=0.6,
             label=r"Single-channel theory $\alpha(p)$")

    ax.set_xlabel(r"Physical noise strength $p$")
    ax.set_ylabel(r"Effective factor $\alpha_{\rm eff}$")
    ax.set_title(
        "Deviation from the theory line signals operator-mixing under folding",
        fontsize=8.6, loc="left", style="italic"
    )
    _style_axes(ax)
    ax.legend(**legend_kwargs(loc="best", ncol=2))

    save_figure(fig, out_dir, "fig4_alpha_eff")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path,
        default=repo_root / "results" / "data" / "paper_exact",
        help="Directory containing the simulation .npy files, including nested p_* folders."
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=repo_root / "results" / "figures" / "paper_exact_plots",
        help="Directory for PDF, SVG and PNG figures."
    )
    args = parser.parse_args()

    records = [
        r for r in load_records(args.data_dir)
        if float(r["strength"]) in PAPER_STRENGTHS
    ]

    if not records:
        raise RuntimeError(
            "No paper plots records remain after filtering to the paper's noise-strength set."
        )

    strengths = sorted({float(r["strength"]) for r in records})
    print(f"Loaded {len(records)} unique simulation records for the paper plot set.")
    print("Channels:", ", ".join(sorted({r['channel'] for r in records})))
    print("Scopes:  ", ", ".join(sorted({r['scope'] for r in records})))
    print("Noise strengths:", ", ".join(f"{p:.2f}" for p in strengths))

    figure_scaling(records, args.out_dir)
    figure_mitigation(records, args.out_dir)
    figure_error_and_r2(records, args.out_dir)
    figure_alpha(records, args.out_dir)

    print(f"\nFigures written to: {args.out_dir.resolve()}")
    print("Formats: PDF (vector), SVG (vector), PNG (300 dpi).")


if __name__ == "__main__":
    main()