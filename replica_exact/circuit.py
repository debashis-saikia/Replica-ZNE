"""Circuit construction for the paper-style replica-interference experiment."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import qutip as qt

from .noise import apply_local_noise


def validate_density_matrix(rho: qt.Qobj | np.ndarray, atol: float = 1e-10) -> None:
    if isinstance(rho, np.ndarray):
        rho = qt.Qobj(rho)
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


def trace_power(rho: qt.Qobj | np.ndarray, m: int) -> float:
    if isinstance(rho, np.ndarray):
        rho = qt.Qobj(rho)
    if m < 1:
        raise ValueError("m must be >= 1.")
    return float(np.real((rho ** m).tr()))


def replica_state(rho: qt.Qobj | np.ndarray, m: int) -> qt.Qobj:
    if isinstance(rho, np.ndarray):
        rho = qt.Qobj(rho)
    if m < 1:
        raise ValueError("m must be >= 1.")
    return qt.tensor(*([rho] * m))


def cyclic_permutation(m: int, n: int) -> qt.Qobj:
    """Return V_m acting on m replicas of n qubits each."""
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
    """Return C(V_m) with ancilla as the first subsystem."""
    V = cyclic_permutation(m, n)
    sys_dim = 2 ** (m * n)
    I_sys = qt.Qobj(np.eye(sys_dim, dtype=complex), dims=[[2] * (m * n), [2] * (m * n)])
    P0 = qt.Qobj(np.diag([1.0, 0.0]), dims=[[2], [2]])
    P1 = qt.Qobj(np.diag([0.0, 1.0]), dims=[[2], [2]])
    return qt.tensor(P0, I_sys) + qt.tensor(P1, V)


@dataclass
class ReplicaPaperExperiment:
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
        H1 = qt.Qobj(np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0))
        self.H = qt.tensor(H1, *[qt.qeye(2)] * (self.m * self.n))
        self.Z_anc = qt.tensor(qt.sigmaz(), *[qt.qeye(2)] * (self.m * self.n))
        self.initial = qt.tensor(
            qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit()),
            replica_state(self.rho, self.m),
        )

    @property
    def exact(self) -> float:
        return trace_power(self.rho, self.m)

    @property
    def system_targets(self) -> list[int]:
        return list(range(1, self.total_qubits))

    @property
    def ancilla_target(self) -> list[int]:
        return [0]

    def targets_for(self, scope: str) -> list[int]:
        if scope == "system":
            return self.system_targets
        if scope == "ancilla":
            return self.ancilla_target
        if scope == "all":
            return list(range(self.total_qubits))
        raise ValueError("scope must be 'system', 'ancilla', or 'all'.")

    def expectation_from_state(self, rho_state: qt.Qobj) -> float:
        final = self.H * rho_state * self.H.dag()
        return float(np.real((self.Z_anc * final).tr()))

    def run_folded(
        self,
        c: int,
        channel: str | None = None,
        strength: float = 0.0,
        scope: str = "system",
    ) -> float:
        """Run U(U^dagger U)^c with optional gate-local noise after each block."""
        if c < 0:
            raise ValueError("c must be nonnegative.")

        state = self.initial.copy()
        targets = self.targets_for(scope)

        def apply_noise(state_in: qt.Qobj) -> qt.Qobj:
            if channel is not None and strength > 0:
                return apply_local_noise(state_in, channel, strength, targets, self.total_qubits)
            return state_in

        state = self.U * state * self.U.dag()
        state = apply_noise(state)

        for _ in range(c):
            state = self.U.dag() * state * self.U
            state = apply_noise(state)
            state = self.U * state * self.U.dag()
            state = apply_noise(state)

        return self.expectation_from_state(state)
