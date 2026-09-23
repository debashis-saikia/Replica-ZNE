"""Replica-interference circuit and effective block-noise simulation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import qutip as qt

from .noise import apply_channel_to_targets
from .permutation import controlled_cyclic_permutation
from .state import replica_state, trace_power, validate_density_matrix


@dataclass
class ReplicaExperiment:
    rho: qt.Qobj
    m: int = 2

    def __post_init__(self) -> None:
        validate_density_matrix(self.rho)
        if self.m < 2:
            raise ValueError("m must be >= 2.")
        self.n = int(np.log2(self.rho.shape[0]))
        if 2**self.n != self.rho.shape[0]:
            raise ValueError("rho dimension must be a power of two.")
        self.total_qubits = 1 + self.m * self.n
        self.U = controlled_cyclic_permutation(self.m, self.n)
        H1 = qt.Qobj(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2))
        self.H = qt.tensor(H1, *[qt.qeye(2)] * (self.m * self.n))
        self.Z_anc = qt.tensor(qt.sigmaz(), *[qt.qeye(2)] * (self.m * self.n))
        self.initial = qt.tensor(qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit()), replica_state(self.rho, self.m))

    @property
    def exact(self) -> float:
        return trace_power(self.rho, self.m)

    @property
    def system_targets(self) -> list[int]:
        return list(range(1, self.total_qubits))

    @property
    def ancilla_target(self) -> list[int]:
        return [0]

    def folded_unitary(self, c: int) -> qt.Qobj:
        if c < 0:
            raise ValueError("c must be nonnegative.")
        Udag = self.U.dag()
        result = self.U
        for _ in range(c):
            result = result * Udag * self.U
        return result

    def expectation_from_state(self, rho_state: qt.Qobj) -> float:
        final = self.H * rho_state * self.H.dag()
        return float(np.real((self.Z_anc * final).tr()))

    def apply_block_noise(
        self,
        rho_state: qt.Qobj,
        channel: str,
        strength: float,
        scope: str,
    ) -> qt.Qobj:
        if scope == "system":
            targets = self.system_targets
        elif scope == "ancilla":
            targets = self.ancilla_target
        elif scope == "all":
            targets = list(range(self.total_qubits))
        else:
            raise ValueError("scope must be 'system', 'ancilla', or 'all'.")
        return apply_channel_to_targets(
            rho_state, channel, strength, targets, self.total_qubits
        )

    def run_folded(
        self,
        c: int,
        channel: str | None = None,
        strength: float = 0.0,
        scope: str = "system",
    ) -> float:
        """Run U(U^dagger U)^c with optional noise after every block."""
        rho_state = self.initial.copy()

        def noisy_U(state: qt.Qobj) -> qt.Qobj:
            out = self.U * state * self.U.dag()
            if channel is not None and strength > 0:
                out = self.apply_block_noise(out, channel, strength, scope)
            return out

        def noisy_Udag(state: qt.Qobj) -> qt.Qobj:
            out = self.U.dag() * state * self.U
            if channel is not None and strength > 0:
                out = self.apply_block_noise(out, channel, strength, scope)
            return out

        rho_state = noisy_U(rho_state)
        for _ in range(c):
            rho_state = noisy_Udag(rho_state)
            rho_state = noisy_U(rho_state)

        return self.expectation_from_state(rho_state)
