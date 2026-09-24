"""Configuration for the Qiskit hardware implementation."""

from __future__ import annotations

from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "hardware" / "results"

RHO = np.array(
    [[0.8, 0.2],
     [0.2, 0.2]],
    dtype=complex,
)

M = 2
SCALES = (1, 3, 5, 7, 9)
SHOTS = 10_000
OPTIMIZATION_LEVEL = 1

MEASUREMENT_REGISTER = "ancilla"
