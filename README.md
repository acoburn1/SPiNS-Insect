# Model Training Repository for Category Learning

Purpose: Utilities, training code, and analysis scripts for producing and visualizing neural-network category learning data

Branches
- `master` — main branch, for running locally
- `openhpc` — different `program.py` + additional bash/slurm scripts for running on openhpc
  
Repository layout (key files / directories)
- `program.py` — main orchestration for data prep, training calls, and analysis runs
- `configs/` — preset .yaml files for configuring data, model specs, and (soon) output
- `Output/` — plotting and output helpers:
  - `MatrixOutput.py` — matrix plotting utilities
  - `StatOutput.py`, `PCAOutput.py` — statistical and PCA plotting utilities
- `Model/` — model implementations (training/evaluation)
- `Prep/` — data loading and preprocessing utilities
- `Results/Data/` — per-run `.npz` result files (e.g. `p_m1.npz`)
- `Results/Analysis/` — generated plots and derived outputs
- `Data/` — input CSV files and reference matrices

