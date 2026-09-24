# Qiskit hardware experiment

This directory is the Qiskit-only hardware implementation of the replica-interference protocol used in the paper.

The structure is deliberately separate from the QuTiP simulation:

```text
hardware/
├── code/
│   ├── config.py
│   ├── ensemble.py
│   ├── circuit.py
│   ├── analysis.py
│   ├── ideal_check.py
│   ├── run_hardware.py
│   └── requirements.txt
└── results/
```

## Physical implementation

The paper's test state is

\[
\rho = \begin{pmatrix}0.8&0.2\\0.2&0.2\end{pmatrix},
\qquad \mathrm{Tr}(\rho^2)=0.76.
\]

A QPU does not accept an arbitrary density matrix as a one-shot state preparation. The hardware code therefore uses the spectral ensemble

\[
\rho = \sum_i \lambda_i |\psi_i\rangle\langle\psi_i|,
\]

and constructs one circuit for each independent tuple of spectral components across the replicas. The measured ancilla expectation is then combined with the exact classical weights \(\prod_r\lambda_{i_r}\). This realizes the same \(\rho^{\otimes m}\) ensemble as the density-matrix simulation.

For \(m=2\), the controlled cyclic permutation is simply a controlled-SWAP. The folded circuits are

\[
U_s = U(U^\dagger U)^c, \qquad s=2c+1,
\]

with \(s\in\{1,3,5,7,9\}\). The final Hadamard and ancilla measurement give the replica-interference signal \(R_2(s)\).

The two-point estimator used in the paper is

\[
R_{2,\mathrm{RZNE}} = \sqrt{\frac{R_2(1)^3}{R_2(3)}}.
\]

The code also reports the two-point Richardson value for comparison.

## Workflow

1. Install the hardware dependencies from `code/requirements.txt`.
2. Configure IBM Quantum Compute credentials using `QiskitRuntimeService` according to IBM's current authentication procedure.
3. Run `python code/ideal_check.py`. Every folded scale should return 0.76 up to numerical precision.
4. Submit to a QPU with `python code/run_hardware.py --backend <backend-name> --shots 10000`.
5. The raw circuit-level data and the aggregate R-ZNE analysis are written to `hardware/results/`.

The hardware runner does not inject artificial phase-flip, dephasing, depolarizing, or amplitude-damping channels. The point of this directory is to expose the protocol to the native QPU noise so that the observed scaling is hardware data rather than a second synthetic-noise model.

Built-in IBM Runtime error suppression/mitigation should be kept controlled when producing the paper's first hardware baseline, so that the reported improvement can be attributed to the replica R-ZNE protocol rather than an unreported provider-side mitigation setting.
