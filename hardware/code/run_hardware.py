"""Run the replica-interference experiment on an IBM Quantum QPU.

Authentication is intentionally not hard-coded. Use a saved QiskitRuntimeService
account or the IBM Quantum environment configured on the user's machine.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from analysis import (
    aggregate_scale_records,
    analyze_scale_data,
    expectation_from_counts,
    save_json,
    save_records_csv,
)
from circuit import make_experiment_circuits
from config import M, OPTIMIZATION_LEVEL, RHO, RESULTS_DIR, SCALES, SHOTS


def build_service_and_backend(backend_name: str | None):
    backend_name_value = (backend_name or "").strip().lower().replace("_", "-")
    if backend_name_value.startswith("fake-"):
        fake_backend_name = backend_name_value[len("fake-") :]
        fake_class_name = "Fake" + "".join(
            part.capitalize() for part in fake_backend_name.split("-") if part
        )
        module = __import__("qiskit_ibm_runtime.fake_provider", fromlist=[fake_class_name])
        try:
            fake_backend_cls = getattr(module, fake_class_name)
        except AttributeError as exc:
            available = [
                name.lower().replace("fake", "fake-")
                for name in dir(module)
                if name.startswith("Fake")
            ]
            raise ValueError(
                f"Unknown fake backend '{backend_name}'. Try one of: {', '.join(available[:10])}."
            ) from exc
        return None, fake_backend_cls()

    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("QISKIT_IBM_TOKEN") or os.environ.get("IBMQ_TOKEN")
    instance = os.environ.get("QISKIT_IBM_INSTANCE") or os.environ.get("IBMQ_INSTANCE")

    if token:
        service_kwargs = {"token": token}
        if instance:
            service_kwargs["instance"] = instance
        # IBM Quantum Platform Open instances use the CRN as the instance value.
        # Keep this compatible with the newer cloud/platform flow while still
        # allowing legacy hub/group/project accounts when available.
        service_kwargs["channel"] = "ibm_quantum_platform"
        service = QiskitRuntimeService(**service_kwargs)
    else:
        try:
            service = QiskitRuntimeService(channel="ibm_quantum_platform")
        except Exception:
            service = QiskitRuntimeService()

    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=1 + M,
        )
    return service, backend


def save_live_run_data(
    backend_name: str,
    shots: int,
    raw_records: list[dict],
    analysis: dict,
    exact: float,
    job_id: str | None,
) -> None:
    """Persist the raw hardware run payload as a standalone .npy file."""
    output_dir = RESULTS_DIR / "live_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_backend = backend_name.strip().lower().replace(" ", "_")
    filename = f"{safe_backend}_{shots}.npy"
    payload = {
        "backend": backend_name,
        "shots": shots,
        "exact": exact,
        "job_id": job_id,
        "raw_records": raw_records,
        "analysis": analysis,
    }
    np.save(output_dir / filename, payload, allow_pickle=True)


def live_run_stem(backend_name: str, shots: int) -> str:
    safe_backend = backend_name.strip().lower().replace(" ", "_")
    return f"{safe_backend}_{shots}"


def save_diagnostics_plot(
    scale_data: dict[int, dict],
    analysis: dict,
    backend_name: str,
    out_path: Path,
) -> None:
    """Save a single 2x2 diagnostic plot for the hardware replica run."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scales = np.asarray(sorted(scale_data), dtype=float)
    values = np.asarray([scale_data[int(scale)]["value"] for scale in scales], dtype=float)
    exact = float(analysis["exact"])

    valid = np.isfinite(values) & (values > 0)
    if np.count_nonzero(valid) >= 2:
        fit_coeffs = np.polyfit(scales[valid], np.log(values[valid]), 1)
        fitted = np.exp(fit_coeffs[0] * scales + fit_coeffs[1])
    else:
        fitted = np.full_like(values, np.nan, dtype=float)

    relative_error = np.abs((values - exact) / exact) if abs(exact) > 0 else np.full_like(values, np.nan)
    noisy_error = np.abs(values - exact)
    uncorrected_error = float(np.max(noisy_error))
    rzne_error = float(abs(analysis["RZNE"] - exact)) if np.isfinite(analysis["RZNE"]) else float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    axes[0].plot(scales, np.log(np.clip(values, 1e-12, None)), "o", label="log R(s)")
    axes[0].plot(scales, np.log(np.clip(fitted, 1e-12, None)), "-", label="linear fit")
    axes[0].set_xlabel("scale s = 2c + 1")
    axes[0].set_ylabel("log R(s)")
    axes[0].set_title("Log-linearity")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(scales, relative_error, "o-", label="relative error")
    axes[1].axhline(0.0, color="black", lw=1)
    axes[1].set_xlabel("scale s = 2c + 1")
    axes[1].set_ylabel("absolute relative error")
    axes[1].set_title("Relative error vs exact")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(scales, values, "o", label="R(s)")
    axes[2].plot(scales, fitted, "-", label="fit")
    axes[2].axhline(exact, color="black", linestyle="--", label="exact")
    axes[2].set_xlabel("scale s = 2c + 1")
    axes[2].set_ylabel("replica observable")
    axes[2].set_title("Folded replica signal")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    err_scales = np.array([1.0, 9.0])
    axes[3].plot(err_scales, [uncorrected_error, uncorrected_error], color="tab:orange", label=f"raw ({uncorrected_error:.6f})")
    axes[3].plot(err_scales, [rzne_error, rzne_error], color="tab:green", label=f"R-ZNE ({rzne_error:.6f})")
    axes[3].set_xlabel("folding scale")
    axes[3].set_ylabel(r"$|R_{\mathrm{exact}} - R|$")
    axes[3].set_title("Absolute error comparison")
    axes[3].set_xlim(err_scales)
    axes[3].set_ylim(bottom=0.0)
    axes[3].grid(axis="y", alpha=0.3)
    axes[3].legend()

    fig.suptitle(f"Replica diagnostic plot: {backend_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run replica-interference R-ZNE on IBM Quantum hardware.")
    parser.add_argument(
        "--backend",
        default=None,
        help="IBM backend name, or a fake backend such as 'fake-sherbrooke'. If omitted, the script selects the least-busy available IBM backend.",
    )
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--optimization-level", type=int, default=OPTIMIZATION_LEVEL)
    args = parser.parse_args()

    if args.shots <= 0:
        raise SystemExit("--shots must be positive.")

    circuits, specs, exact = make_experiment_circuits(RHO, M, SCALES)
    service, backend = build_service_and_backend(args.backend)

    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    print(f"Backend: {backend.name}")
    print(f"Qubits required: {1 + M}")
    print(f"Number of circuits: {len(circuits)}")
    print(f"Shots per circuit: {args.shots}")
    if service is None:
        print("Mode: fake backend (no IBM Runtime token required)")
    else:
        print("Mode: live IBM backend (using configured IBM Quantum credentials)")

    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=args.optimization_level,
    )
    transpiled = [pass_manager.run(circuit) for circuit in circuits]

    sampler = Sampler(mode=backend)
    job = sampler.run(transpiled, shots=args.shots)
    result = job.result()

    raw_records: list[dict] = []
    for index, (pub_result, spec) in enumerate(zip(result, specs)):
        counts = pub_result.data.ancilla.get_counts()
        expectation, shots = expectation_from_counts(counts)
        p0 = counts.get("0", 0) / shots
        p1 = counts.get("1", 0) / shots
        raw_records.append(
            {
                "scale": spec.scale,
                "fold_count": spec.fold_count,
                "replica_indices": "-".join(map(str, spec.replica_indices)),
                "weight": spec.weight,
                "expectation": expectation,
                "shots": shots,
                "p0": p0,
                "p1": p1,
                "circuit_name": spec.name,
            }
        )

    scale_data = aggregate_scale_records(raw_records)
    analysis = analyze_scale_data(scale_data, exact)
    analysis.update(
        {
            "backend": backend.name,
            "job_id": job.job_id(),
            "shots_per_circuit": args.shots,
            "optimization_level": args.optimization_level,
            "m": M,
        }
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_records_csv(RESULTS_DIR / "latest_raw_records.csv", raw_records)
    save_json(RESULTS_DIR / "latest_analysis.json", analysis)
    run_stem = live_run_stem(backend.name, args.shots)
    save_live_run_data(
        backend.name,
        args.shots,
        raw_records,
        analysis,
        exact,
        job.job_id(),
    )
    save_diagnostics_plot(
        scale_data,
        analysis,
        backend.name,
        RESULTS_DIR / "live_runs" / f"{run_stem}.png",
    )
    save_diagnostics_plot(
        scale_data,
        analysis,
        backend.name,
        RESULTS_DIR / "latest_diagnostics.png",
    )

    print("\nReplica signal")
    for scale in analysis["scales"]:
        data = scale_data[scale]
        print(f"s={scale:2d}  R_m(s)={data['value']:.8f} +/- {data['stderr']:.3e}")

    print("\nMitigation")
    print(f"Exact       = {analysis['exact']:.10f}")
    print(f"Raw R1      = {analysis['R1']:.10f}")
    print(f"R3          = {analysis['R3']:.10f}")
    print(f"R-ZNE       = {analysis['RZNE']:.10f}")
    print(f"Richardson  = {analysis['Richardson']:.10f}")
    print(f"R-ZNE error = {analysis['rzne_error']:+.3e}")
    print(f"Job ID      = {analysis['job_id']}")


if __name__ == "__main__":
    main()
