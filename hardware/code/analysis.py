"""Analysis for Qiskit sampled hardware data."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable


def expectation_from_counts(counts: dict[str, int]) -> tuple[float, int]:
    n0 = int(counts.get("0", 0))
    n1 = int(counts.get("1", 0))
    shots = n0 + n1
    if shots <= 0:
        raise ValueError("No shots found in the ancilla measurement register.")
    return float((n0 - n1) / shots), shots


def aggregate_scale_records(records: Iterable[dict]) -> dict[int, dict]:
    """Combine spectral-ensemble circuits into the mixed-state signal R_m(s)."""
    grouped: dict[int, list[dict]] = {}
    for record in records:
        grouped.setdefault(int(record["scale"]), []).append(record)

    output: dict[int, dict] = {}
    for scale, group in sorted(grouped.items()):
        value = 0.0
        variance = 0.0
        total_weight = 0.0
        for row in group:
            weight = float(row["weight"])
            expectation = float(row["expectation"])
            shots = int(row["shots"])
            value += weight * expectation
            variance += weight * weight * max(0.0, 1.0 - expectation**2) / shots
            total_weight += weight
        output[scale] = {
            "scale": scale,
            "value": value,
            "stderr": math.sqrt(max(0.0, variance)),
            "total_weight": total_weight,
        }
    return output


def rzne_two_point(r1: float, r3: float) -> float:
    if r1 <= 0.0 or r3 <= 0.0:
        return float("nan")
    return math.sqrt(r1**3 / r3)


def richardson_two_point(r1: float, r3: float) -> float:
    return 0.5 * (3.0 * r1 - r3)


def analyze_scale_data(scale_data: dict[int, dict], exact: float) -> dict:
    scales = sorted(scale_data)
    r1 = scale_data[1]["value"]
    r3 = scale_data[3]["value"]
    rz = rzne_two_point(r1, r3)
    rich = richardson_two_point(r1, r3)
    output = {
        "exact": exact,
        "scales": scales,
        "values": [scale_data[s]["value"] for s in scales],
        "stderr": [scale_data[s]["stderr"] for s in scales],
        "R1": r1,
        "R3": r3,
        "RZNE": rz,
        "Richardson": rich,
        "raw_error_R1": r1 - exact,
        "rzne_error": rz - exact if math.isfinite(rz) else float("nan"),
        "richardson_error": rich - exact,
    }
    return output


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_records_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    fields = [
        "scale", "fold_count", "replica_indices", "weight",
        "expectation", "shots", "p0", "p1", "circuit_name",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in fields})
