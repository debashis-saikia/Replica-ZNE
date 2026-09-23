"""State construction and basic replica invariants."""

from __future__ import annotations

import numpy as np
import qutip as qt


def validate_density_matrix(rho: qt.Qobj, atol: float = 1e-10) -> None:
    if rho.isoper is False:
        raise ValueError("rho must be an operator.")
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be square.")
    if not np.isclose(rho.tr(), 1.0, atol=atol):
        raise ValueError("rho must have trace 1.")
    if not rho.isherm:
        raise ValueError("rho must be Hermitian.")
    if np.min(np.linalg.eigvalsh(rho.full())) < -atol:
        raise ValueError("rho must be positive semidefinite.")


def trace_power(rho: qt.Qobj, m: int) -> float:
    if m < 1:
        raise ValueError("m must be >= 1.")
    return float(np.real((rho ** m).tr()))


def replica_state(rho: qt.Qobj, m: int) -> qt.Qobj:
    r"""Return rho^{\otimes m}."""
    if m < 1:
        raise ValueError("m must be >= 1.")
    return qt.tensor(*([rho] * m))
