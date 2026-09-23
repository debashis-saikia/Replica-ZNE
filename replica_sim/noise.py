"""Single-qubit channels used in the effective block-noise simulations."""

from __future__ import annotations

import numpy as np
import qutip as qt


def _embed(op: qt.Qobj, target: int, total_qubits: int) -> qt.Qobj:
    ops = [qt.qeye(2) for _ in range(total_qubits)]
    ops[target] = op
    return qt.tensor(ops)


def apply_phase_flip(rho: qt.Qobj, target: int, p: float, total_qubits: int) -> qt.Qobj:
    z = _embed(qt.sigmaz(), target, total_qubits)
    return (1 - p) * rho + p * z * rho * z


def apply_dephasing(rho: qt.Qobj, target: int, p: float, total_qubits: int) -> qt.Qobj:
    """Phase-flip convention E(rho)=(1-p)rho+p Z rho Z."""
    return apply_phase_flip(rho, target, p, total_qubits)


def apply_depolarizing(rho: qt.Qobj, target: int, p: float, total_qubits: int) -> qt.Qobj:
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0,1].")
    x = _embed(qt.sigmax(), target, total_qubits)
    y = _embed(qt.sigmay(), target, total_qubits)
    z = _embed(qt.sigmaz(), target, total_qubits)
    return (
        (1 - p) * rho
        + (p / 3) * (x * rho * x + y * rho * y + z * rho * z)
    )


def apply_amplitude_damping(
    rho: qt.Qobj, target: int, gamma: float, total_qubits: int
) -> qt.Qobj:
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0,1].")
    k0 = qt.Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    k1 = qt.Qobj([[0, np.sqrt(gamma)], [0, 0]])
    K0 = _embed(k0, target, total_qubits)
    K1 = _embed(k1, target, total_qubits)
    return K0 * rho * K0.dag() + K1 * rho * K1.dag()


def apply_channel_to_targets(
    rho: qt.Qobj,
    channel: str,
    strength: float,
    targets: list[int],
    total_qubits: int,
) -> qt.Qobj:
    """Apply a chosen single-qubit channel independently to targets."""
    result = rho
    for target in targets:
        if channel in {"phase_flip", "dephasing"}:
            result = apply_dephasing(result, target, strength, total_qubits)
        elif channel == "depolarizing":
            result = apply_depolarizing(result, target, strength, total_qubits)
        elif channel == "amplitude_damping":
            result = apply_amplitude_damping(result, target, strength, total_qubits)
        else:
            raise ValueError(f"Unknown channel: {channel}")
    return result
