"""Spectral ensemble representation of the paper's mixed input state.

The QuTiP implementation accepts rho directly as a density operator. On a QPU,
we instead realize the same mixed state by independently sampling from a
spectral ensemble of rho on each replica. For an m-replica protocol,

    rho^{\otimes m} = E_{i_1,...,i_m}[ |psi_{i_1}><psi_{i_1}| \otimes ... ],

with probability weight prod_r lambda_{i_r}.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np


@dataclass(frozen=True)
class Eigenstate:
    index: int
    probability: float
    statevector: np.ndarray


def validate_density_matrix(rho: np.ndarray, atol: float = 1e-10) -> None:
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix.")
    if not np.isclose(np.trace(rho), 1.0, atol=atol):
        raise ValueError("rho must have trace 1.")
    if not np.allclose(rho, rho.conj().T, atol=atol):
        raise ValueError("rho must be Hermitian.")
    eigvals = np.linalg.eigvalsh(rho)
    if eigvals.min() < -atol:
        raise ValueError("rho must be positive semidefinite.")


def spectral_ensemble(rho: np.ndarray, atol: float = 1e-12) -> list[Eigenstate]:
    """Return the nonzero spectral components of rho."""
    validate_density_matrix(rho)
    eigvals, eigvecs = np.linalg.eigh(np.asarray(rho, dtype=complex))
    components: list[Eigenstate] = []
    for idx, probability in enumerate(eigvals):
        probability = float(np.real(probability))
        if probability <= atol:
            continue
        state = np.asarray(eigvecs[:, idx], dtype=complex)
        state /= np.linalg.norm(state)
        components.append(Eigenstate(idx, probability, state))
    if not components:
        raise ValueError("rho has no nonzero eigenvalues.")
    total = sum(item.probability for item in components)
    components = [
        Eigenstate(item.index, item.probability / total, item.statevector)
        for item in components
    ]
    return components


def ensemble_assignments(m: int, ensemble: list[Eigenstate]) -> list[tuple[tuple[Eigenstate, ...], float]]:
    """Enumerate all independent spectral draws for m replicas."""
    if m < 2:
        raise ValueError("m must be >= 2.")
    output: list[tuple[tuple[Eigenstate, ...], float]] = []
    for choice in product(ensemble, repeat=m):
        weight = float(np.prod([item.probability for item in choice]))
        output.append((choice, weight))
    return output


def trace_power(rho: np.ndarray, m: int) -> float:
    validate_density_matrix(rho)
    if m < 1:
        raise ValueError("m must be >= 1.")
    return float(np.real(np.trace(np.linalg.matrix_power(rho, m))))
