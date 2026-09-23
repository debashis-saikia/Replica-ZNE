"""Controlled cyclic permutation operators."""

from __future__ import annotations

import numpy as np
import qutip as qt


def cyclic_permutation(m: int, n: int) -> qt.Qobj:
    """Return V_m acting on m replicas of n qubits each.

    Convention:
        |psi_0, psi_1, ..., psi_{m-1}> ->
        |psi_{m-1}, psi_0, ..., psi_{m-2}>.
    """
    if m < 2 or n < 1:
        raise ValueError("Require m >= 2 and n >= 1.")

    total = m * n
    dim = 2**total
    matrix = np.zeros((dim, dim), dtype=complex)

    for idx in range(dim):
        bits = list(format(idx, f"0{total}b"))
        replicas = [bits[r * n:(r + 1) * n] for r in range(m)]
        permuted = [replicas[-1]] + replicas[:-1]
        new_bits = [bit for block in permuted for bit in block]
        new_idx = int("".join(new_bits), 2)
        matrix[new_idx, idx] = 1.0

    return qt.Qobj(matrix, dims=[[2] * total, [2] * total])


def controlled_cyclic_permutation(m: int, n: int) -> qt.Qobj:
    """Return C(V_m) with the first subsystem as ancilla."""
    V = cyclic_permutation(m, n)
    sys_dim = 2 ** (m * n)
    I_sys = qt.Qobj(np.eye(sys_dim, dtype=complex), dims=[[2] * (m * n), [2] * (m * n)])
    P0 = qt.Qobj(np.diag([1.0, 0.0]), dims=[[2], [2]])
    P1 = qt.Qobj(np.diag([0.0, 1.0]), dims=[[2], [2]])
    return qt.tensor(P0, I_sys) + qt.tensor(P1, V)
