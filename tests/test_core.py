import numpy as np
import qutip as qt

from replica_sim import ReplicaExperiment, rzne_two_point
from hardware.code.run_hardware import build_service_and_backend


def test_ideal_replica_interference():
    rho = qt.Qobj([[0.8, 0.2], [0.2, 0.2]])
    exp = ReplicaExperiment(rho, m=2)
    assert np.isclose(exp.exact, 0.76)
    assert np.isclose(exp.run_folded(0), 0.76, atol=1e-12)


def test_ancilla_phase_flip_exact_rzne():
    rho = qt.Qobj([[0.8, 0.2], [0.2, 0.2]])
    exp = ReplicaExperiment(rho, m=2)
    R1 = exp.run_folded(0, "phase_flip", 0.02, "ancilla")
    R3 = exp.run_folded(1, "phase_flip", 0.02, "ancilla")
    assert np.isclose(R1, 0.76 * 0.96, atol=1e-12)
    assert np.isclose(R3, 0.76 * 0.96**3, atol=1e-12)
    assert np.isclose(rzne_two_point(R1, R3), 0.76, atol=1e-12)


def test_fake_backend_name_resolves():
    _, backend = build_service_and_backend("fake-sherbrooke")
    assert backend.name.lower().startswith("fake")
    assert "sherbrooke" in backend.name.lower()
