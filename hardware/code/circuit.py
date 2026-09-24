"""Qiskit implementation of the folded replica-interference circuit."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from ensemble import Eigenstate, ensemble_assignments, spectral_ensemble


@dataclass(frozen=True)
class CircuitSpec:
    scale: int
    fold_count: int
    replica_indices: tuple[int, ...]
    weight: float
    name: str


def apply_controlled_cycle(
    circuit: QuantumCircuit,
    ancilla: int,
    replicas: list[int],
) -> None:
    """Apply a controlled cyclic permutation of the replica registers.

    For one qubit per replica, the m-cycle is decomposed as

        SWAP(0,1), SWAP(0,2), ..., SWAP(0,m-1),

    all controlled on the ancilla. The reverse ordering implements the
    inverse cycle.
    """
    if len(replicas) < 2:
        raise ValueError("At least two replicas are required.")
    for target in replicas[1:]:
        circuit.cswap(ancilla, replicas[0], target)


def apply_inverse_controlled_cycle(
    circuit: QuantumCircuit,
    ancilla: int,
    replicas: list[int],
) -> None:
    if len(replicas) < 2:
        raise ValueError("At least two replicas are required.")
    for target in reversed(replicas[1:]):
        circuit.cswap(ancilla, replicas[0], target)


def apply_controlled_register_cycle(
    circuit: QuantumCircuit,
    ancilla: int,
    replica_registers: list[list[int]],
) -> None:
    for qubit_index in range(len(replica_registers[0])):
        apply_controlled_cycle(
            circuit,
            ancilla,
            [register[qubit_index] for register in replica_registers],
        )


def apply_inverse_controlled_register_cycle(
    circuit: QuantumCircuit,
    ancilla: int,
    replica_registers: list[list[int]],
) -> None:
    for qubit_index in range(len(replica_registers[0])):
        apply_inverse_controlled_cycle(
            circuit,
            ancilla,
            [register[qubit_index] for register in replica_registers],
        )


def build_replica_circuit(
    replica_states: tuple[np.ndarray, ...],
    c: int,
    measure: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """Build U(U^dagger U)^c for the one-qubit-per-replica experiment."""
    if c < 0:
        raise ValueError("c must be nonnegative.")
    m = len(replica_states)
    if m < 2:
        raise ValueError("At least two replicas are required.")

    dimensions = [np.asarray(state, dtype=complex).size for state in replica_states]
    if len(set(dimensions)) != 1 or dimensions[0] < 2:
        raise ValueError("Replica states must have the same nontrivial dimension.")
    qubits_per_replica = int(np.log2(dimensions[0]))
    if 2**qubits_per_replica != dimensions[0]:
        raise ValueError("Replica state dimensions must be powers of two.")

    qreg = QuantumRegister(1 + m * qubits_per_replica, "q")
    creg = ClassicalRegister(1, "ancilla") if measure else None
    circuit = QuantumCircuit(qreg, creg) if creg is not None else QuantumCircuit(qreg)
    circuit.name = name or f"replica_m{m}_s{2*c+1}"

    ancilla = 0
    replica_registers = [
        list(
            range(
                1 + replica_index * qubits_per_replica,
                1 + (replica_index + 1) * qubits_per_replica,
            )
        )
        for replica_index in range(m)
    ]

    # |+>_A tensor product_r |psi_r>.
    circuit.h(ancilla)
    for register, state in zip(replica_registers, replica_states):
        state = np.asarray(state, dtype=complex)
        circuit.prepare_state(state, register)

    # U.
    apply_controlled_register_cycle(circuit, ancilla, replica_registers)

    # Folding: U^dagger U pairs.
    for _ in range(c):
        apply_inverse_controlled_register_cycle(circuit, ancilla, replica_registers)
        apply_controlled_register_cycle(circuit, ancilla, replica_registers)

    # H followed by Z measurement on the ancilla.
    circuit.h(ancilla)
    if measure:
        circuit.measure(ancilla, creg[0])
    return circuit


def make_experiment_circuits(
    rho: np.ndarray,
    m: int,
    scales: tuple[int, ...],
) -> tuple[list[QuantumCircuit], list[CircuitSpec], float]:
    """Generate all spectral-ensemble circuits used in the hardware run."""
    if any(scale < 1 or scale % 2 == 0 for scale in scales):
        raise ValueError("Scales must be positive odd integers.")
    ensemble = spectral_ensemble(rho)
    assignments = ensemble_assignments(m, ensemble)
    circuits: list[QuantumCircuit] = []
    specs: list[CircuitSpec] = []

    for scale in scales:
        c = (scale - 1) // 2
        for assignment, weight in assignments:
            replica_indices = tuple(item.index for item in assignment)
            name = "replica_" + "_".join(
                [f"s{scale}"] + [f"e{i}" for i in replica_indices]
            )
            states = tuple(item.statevector for item in assignment)
            circuits.append(build_replica_circuit(states, c=c, measure=True, name=name))
            specs.append(CircuitSpec(scale, c, replica_indices, weight, name))

    exact = float(np.real(np.trace(np.linalg.matrix_power(rho, m))))
    return circuits, specs, exact


def ideal_expectation(circuit: QuantumCircuit) -> float:
    """Evaluate the ancilla Z expectation with the Qiskit Statevector class."""
    unmeasured = circuit.remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(unmeasured)
    p0, p1 = state.probabilities([0])
    return float(np.real(p0 - p1))


def ideal_ensemble_values(
    rho: np.ndarray,
    m: int,
    scales: tuple[int, ...],
) -> dict[int, float]:
    """Noiseless Qiskit check. Each value should equal Tr(rho^m)."""
    circuits, specs, _ = make_experiment_circuits(rho, m, scales)
    values = {scale: 0.0 for scale in scales}
    for circuit, spec in zip(circuits, specs):
        values[spec.scale] += spec.weight * ideal_expectation(circuit)
    return values
