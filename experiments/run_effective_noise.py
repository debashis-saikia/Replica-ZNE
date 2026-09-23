"""Run the four current effective block-noise benchmarks.

This script reproduces the present coarse-grained model used during
code debugging. It is intentionally separate from the future gate-local
noise implementation.
"""

from pathlib import Path
import sys
import numpy as np
import qutip as qt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from replica_sim import ReplicaExperiment, rzne_two_point
from replica_sim.analysis import log_linear_fit


RHO = qt.Qobj([[0.8, 0.2], [0.2, 0.2]])
M = 2
P_MAIN = 0.02
MAX_C = 4
STRENGTHS = np.array([0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080])

CHANNELS = {
    "phase_flip": "ancilla",
    "dephasing": "system",
    "depolarizing": "system",
    "amplitude_damping": "system",
}


def run_channel(exp: ReplicaExperiment, channel: str, strength: float) -> dict:
    scales = np.array([2 * c + 1 for c in range(MAX_C + 1)], dtype=float)
    values = np.array([
        exp.run_folded(c, channel, strength, CHANNELS[channel])
        for c in range(MAX_C + 1)
    ])
    fit = log_linear_fit(scales, values)
    rz = rzne_two_point(values[0], values[1])
    return {
        "channel": channel,
        "strength": strength,
        "scales": scales,
        "values": values,
        "exact": exp.exact,
        "R1": values[0],
        "R3": values[1],
        "R_RZNE": rz,
        "raw_error": abs(values[0] - exp.exact),
        "rzne_error": abs(rz - exp.exact),
        **fit,
    }


def main() -> None:
    exp = ReplicaExperiment(RHO, m=M)
    out = ROOT / "results" / "data"
    out.mkdir(parents=True, exist_ok=True)

    for channel, scope in CHANNELS.items():
        main_result = run_channel(exp, channel, P_MAIN)
        np.save(out / f"{channel}_main.npy", main_result, allow_pickle=True)

        sweep = []
        for strength in STRENGTHS:
            result = run_channel(exp, channel, float(strength))
            sweep.append(result)
        np.save(out / f"{channel}_sweep.npy", np.array(sweep, dtype=object), allow_pickle=True)

        print(
            f"{channel:20s} "
            f"R(1)={main_result['R1']:.10f} "
            f"R(3)={main_result['R3']:.10f} "
            f"R_RZNE={main_result['R_RZNE']:.10f} "
            f"R2={main_result['R2']:.8f}"
        )


if __name__ == "__main__":
    main()
