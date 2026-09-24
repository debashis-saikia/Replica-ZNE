"""Qiskit-only noiseless validation of the hardware circuit construction."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from circuit import ideal_ensemble_values
from config import M, RHO, SCALES
from ensemble import trace_power


def main() -> None:
    expected = trace_power(RHO, M)
    values = ideal_ensemble_values(RHO, M, SCALES)

    print(f"Qiskit ideal check for m={M}")
    print(f"Target Tr(rho^{M}) = {expected:.12f}")
    for scale in SCALES:
        value = values[scale]
        error = value - expected
        print(f"s={scale:2d}  R_m(s)={value:.12f}  error={error:+.3e}")

    max_error = max(abs(values[s] - expected) for s in SCALES)
    if max_error > 1e-10:
        raise SystemExit(f"Ideal Qiskit validation failed: max error = {max_error:.3e}")
    print("PASS: all folded scales recover the exact noiseless replica value.")


if __name__ == "__main__":
    main()
