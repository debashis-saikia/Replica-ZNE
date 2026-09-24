"""Run the replica-interference method on a three-qubit GHZ state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from analysis import aggregate_scale_records, analyze_scale_data, expectation_from_counts
from circuit import build_replica_circuit
from config import RESULTS_DIR
from run_hardware import build_service_and_backend, save_diagnostics_plot


SCALES = (1, 3, 5, 7, 9)


def ghz_state(qubits: int = 3) -> np.ndarray:
    if qubits < 2:
        raise ValueError("A GHZ state requires at least two qubits.")
    state = np.zeros(2**qubits, dtype=complex)
    state[0] = 1.0 / np.sqrt(2.0)
    state[-1] = 1.0 / np.sqrt(2.0)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GHZ replica test on a Qiskit backend.")
    parser.add_argument("--backend", default="fake-sherbrooke")
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--m", type=int, default=2)
    args = parser.parse_args()
    if args.shots <= 0:
        raise SystemExit("--shots must be positive.")
    if args.qubits < 2:
        raise SystemExit("--qubits must be at least 2.")
    if args.m < 2:
        raise SystemExit("--m must be at least 2.")

    service, backend = build_service_and_backend(args.backend)
    state = ghz_state(args.qubits)
    circuits = [
        build_replica_circuit(
            tuple(state for _ in range(args.m)),
            c=(scale - 1) // 2,
            name=f"ghz{args.qubits}_m{args.m}_s{scale}",
        )
        for scale in SCALES
    ]
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled = [pass_manager.run(circuit) for circuit in circuits]
    job = Sampler(mode=backend).run(transpiled, shots=args.shots)
    result = job.result()

    records = []
    for scale, pub_result in zip(SCALES, result):
        counts = pub_result.data.ancilla.get_counts()
        expectation, shots = expectation_from_counts(counts)
        records.append(
            {
                "scale": scale,
                "weight": 1.0,
                "expectation": expectation,
                "shots": shots,
            }
        )

    scale_data = aggregate_scale_records(records)
    analysis = analyze_scale_data(scale_data, exact=1.0)
    output_dir = RESULTS_DIR / f"ghz{args.qubits}_m{args.m}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stem = f"{backend.name.lower().replace(' ', '_')}_{args.shots}"
    payload = {
        "backend": backend.name,
        "state": f"{args.qubits}_qubit_ghz",
        "m": args.m,
        "shots": args.shots,
        "exact": 1.0,
        "job_id": job.job_id(),
        "raw_records": records,
        "analysis": analysis,
    }
    np.save(output_dir / f"{run_stem}.npy", payload, allow_pickle=True)
    save_diagnostics_plot(
        scale_data,
        analysis,
        backend.name,
        output_dir / f"{run_stem}.png",
    )

    print(f"Backend: {backend.name}")
    print(f"State: {args.qubits}-qubit GHZ")
    print(f"Shots per circuit: {args.shots}")
    print(f"Mode: {'fake backend' if service is None else 'live IBM backend'}")
    for scale in SCALES:
        print(f"s={scale:2d}  R(s)={scale_data[scale]['value']:.8f} +/- {scale_data[scale]['stderr']:.3e}")
    print(f"Exact Tr(rho^{args.m}) = 1.0000000000")
    print(f"R-ZNE           = {analysis['RZNE']:.10f}")
    print(f"Richardson       = {analysis['Richardson']:.10f}")
    print(f"Job ID           = {job.job_id()}")
    print(f"Saved data       = {output_dir / f'{run_stem}.npy'}")
    print(f"Saved plot       = {output_dir / f'{run_stem}.png'}")


if __name__ == "__main__":
    main()