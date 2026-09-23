"""Run the paper-style exact replica-interference simulation and save plots."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replica_exact import ReplicaPaperExperiment
from replica_exact.analysis import exponential_diagnostics, log_linear_fit, rzne_two_point
from replica_exact.plotting import plot_diagnostics, plot_folded_curve, plot_rzne_relative_error

RHO = np.array([[0.8, 0.2], [0.2, 0.2]], dtype=complex)


def run_single_case(channel: str, strength: float, scope: str = "system") -> dict:
    exp = ReplicaPaperExperiment(RHO, m=2)
    scales = np.array([2 * c + 1 for c in range(5)], dtype=float)
    values = np.array([exp.run_folded(c, channel=channel, strength=strength, scope=scope) for c in range(5)])
    fit = log_linear_fit(scales, values)
    rz = rzne_two_point(values[0], values[1])
    diagnostics = exponential_diagnostics(scales, values, exp.exact, rz)
    return {
        "channel": channel,
        "strength": strength,
        "scope": scope,
        "scales": scales,
        "values": values,
        "exact": exp.exact,
        "R1": values[0],
        "R3": values[1],
        "RZNE": rz,
        "diagnostics": diagnostics,
        **fit,
    }


def main() -> None:
    cases = [
        ("phase_flip", "ancilla"),
        ("dephasing", "system"),
        ("depolarizing", "system"),
        ("amplitude_damping", "system"),
    ]

    for strength in (0.06, 0.08):
        out_dir = ROOT / "results" / "figures" / "paper_exact" / f"p_{strength:.2f}"
        data_dir = ROOT / "results" / "data" / "paper_exact" / f"p_{strength:.2f}"
        out_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        for channel, scope in cases:
            result = run_single_case(channel, strength, scope)
            np.save(data_dir / f"{channel}_{scope}.npy", result, allow_pickle=True)
            plot_folded_curve(result["scales"], result["values"], result["exact"], channel, strength, out_dir)
            plot_diagnostics(result["scales"], result["values"], result["diagnostics"], channel, strength, out_dir)
            plot_rzne_relative_error(result["diagnostics"], channel, strength, out_dir)
            print(
                f"p={strength:.2f} {channel:20s} "
                f"R1={result['R1']:.10f} "
                f"R3={result['R3']:.10f} "
                f"RZNE={result['RZNE']:.10f} "
                f"alpha_eff={result['alpha_eff']:.6f} "
                f"R2={result['diagnostics']['R2']:.8f} "
                f"max_log_resid={result['diagnostics']['max_abs_log_residual']:.3e} "
                f"err_noisy_R1={result['diagnostics']['noisy_error']:.3e} "
                f"err_noisy_max={result['diagnostics']['noisy_error_max_scale']:.3e} "
                f"err_mitigated={result['diagnostics']['mitigated_error']:.3e} "
                f"abs_rel_noisy_max={abs(result['diagnostics']['relative_noisy_error_max_scale']):.3%} "
                f"abs_rel_mitigated={result['diagnostics']['absolute_relative_mitigated_error']:.3%} "
                f"alpha_sqrt_R3_R1={result['diagnostics']['alpha_two_point']:.6f} "
                f"abs_rel_alpha={result['diagnostics']['absolute_relative_alpha_error']:.3%}"
            )


if __name__ == "__main__":
    main()
