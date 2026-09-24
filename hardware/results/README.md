# Hardware results

This directory is intentionally empty before the first IBM Quantum run, except for this note and `.gitkeep`.

`run_hardware.py` will write:

- `latest_raw_records.csv`: circuit-level ancilla counts converted to expectation values, with spectral-ensemble weights.
- `latest_analysis.json`: aggregated \(R_m(s)\) values, standard errors, two-point R-ZNE, two-point Richardson, exact target, backend, and job metadata.

No hardware data are fabricated or copied from the QuTiP simulation.
