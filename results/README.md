# Results directory

This folder stores outputs produced by the experiment scripts in the project root.

## Contents

- `data/`
  - numerical arrays saved by the experiments
  - contains the benchmark sweep outputs and paper-exact outputs
  - examples include subdirectories such as `data/paper_exact/p_0.02/`
- `figures/`
  - generated plots and diagnostic visualizations
  - organized by experiment family and parameter settings

## Source of truth

The files in this directory are generated artifacts, not hand-edited source files.

To regenerate them, use the scripts in `experiments/`:

```bash
python experiments/run_effective_noise.py
python experiments/run_paper_exact.py
```

The package code that defines the simulation logic lives in:

- `replica_sim/`
- `replica_exact/`

## Notes

- `results/data` should reflect the parameter sweeps and benchmark runs defined in the experiment scripts.
- `results/figures` should be regenerated when the underlying model or plotting code changes.
- `legacy/` contains older rough prototype code and is not part of the current result-generation workflow.
