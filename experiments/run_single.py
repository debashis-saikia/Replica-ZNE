"""Small deterministic smoke test for the cleaned simulation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qutip as qt

from replica_sim import ReplicaExperiment, rzne_two_point

rho = qt.Qobj([[0.8, 0.2], [0.2, 0.2]])
exp = ReplicaExperiment(rho, m=2)

print(f"Exact Tr(rho^2) = {exp.exact:.12f}")
print(f"Ideal folded c=0 = {exp.run_folded(0):.12f}")

for channel, strength, scope in [
    ("dephasing", 0.02, "system"),
    ("depolarizing", 0.02, "system"),
    ("amplitude_damping", 0.02, "system"),
    ("phase_flip", 0.02, "ancilla"),
]:
    R1 = exp.run_folded(0, channel, strength, scope)
    R3 = exp.run_folded(1, channel, strength, scope)
    rz = rzne_two_point(R1, R3)
    print(f"{channel:20s} R1={R1:.12f} R3={R3:.12f} RZNE={rz:.12f}")
