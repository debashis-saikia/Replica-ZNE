"""Gate-local noise primitives for the paper-style replica circuit."""

from __future__ import annotations

import numpy as np
import qutip as qt


def _embed(op: qt.Qobj, target: int, total_qubits: int) -> qt.Qobj:
    ops = [qt.qeye(2) for _ in range(total_qubits)]
    ops[target] = op
    return qt.tensor(ops)


def apply_local_noise(
    rho: qt.Qobj,
    channel: str,
    strength: float,
    targets: list[int],
    total_qubits: int,
) -> qt.Qobj:
    """Apply a single-qubit channel independently to the selected qubits."""
    result = rho
    for target in targets:
        if channel in {"phase_flip", "dephasing"}:
            z = _embed(qt.sigmaz(), target, total_qubits)
            result = (1.0 - strength) * result + strength * (z * result * z)
        elif channel == "bit_flip":
            x = _embed(qt.sigmax(), target, total_qubits)
            result = (1.0 - strength) * result + strength * (x * result * x)
        elif channel == "depolarizing":
            if not 0.0 <= strength <= 1.0:
                raise ValueError("strength must be in [0, 1].")
            x = _embed(qt.sigmax(), target, total_qubits)
            y = _embed(qt.sigmay(), target, total_qubits)
            z = _embed(qt.sigmaz(), target, total_qubits)
            result = (1.0 - strength) * result + (strength / 3.0) * (
                x * result * x + y * result * y + z * result * z
            )
        elif channel == "amplitude_damping":
            if not 0.0 <= strength <= 1.0:
                raise ValueError("strength must be in [0, 1].")
            k0 = qt.Qobj([[1.0, 0.0], [0.0, np.sqrt(1.0 - strength)]])
            k1 = qt.Qobj([[0.0, np.sqrt(strength)], [0.0, 0.0]])
            K0 = _embed(k0, target, total_qubits)
            K1 = _embed(k1, target, total_qubits)
            result = K0 * result * K0.dag() + K1 * result * K1.dag()
        else:
            raise ValueError(f"Unknown channel: {channel}")
    return result
