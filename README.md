AI-generated overview (to be replaced with fuller manual docs).  
working summary: https://docs.google.com/document/d/1QnrTfTMwoMHdTw7BSiUTkRQp3Au_E98HAMIwpIvE99o/edit?usp=sharing   

# SPiNS Insect: Category Learning Pipeline

This repository trains neural-network models on category-learning tasks, records model activations over training, computes analysis metrics, and renders publication-style plots from those saved analysis artifacts.

---

## End-to-end flow

```text
Train model(s)
  -> save hidden/output activations (Zarr)
  -> run evaluators on saved activations
  -> save analysis outputs (.npz)
  -> generate plots from analysis outputs
```

The important design choice is that plotting is driven by saved analysis files, not by re-running training.

---

## Repository layout (high-level)

```text
lab_tests/
  program.py                 # CLI entrypoint
  DriverUtils/               # run orchestration + helpers
  Model/                     # network definitions + training wrapper
  DataHelper/                # training/probe data loading + conversion
  Eval/                      # analysis/evaluator modules
  Output/                    # output specs + plotting entry points
  Statistics/                # shared stats utilities
  configs/
    data/
    model/
    output/
    probe/
    directory/
```

Input data is expected under configured data directories (see `configs/directory/*.yaml`), and run artifacts are written under the configured results directory.

---

## What each stage produces

### 1) Training
For each requested hidden-layer size and learning-rate pair:
- train `num_models` models
- track per-epoch loss
- collect probe hidden/output activations each epoch

### 2) Activation storage
Activation/loss tensors are stored in Zarr for efficient slicing in downstream evaluators.

### 3) Evaluation
Enabled evaluators load activation slices and produce `.npz` analysis files (typically including `raw`, `mean`, `std`, `se`, confidence intervals, and `n`).

### 4) Output generation
Output modules read analysis `.npz` files and emit plots.  
Some outputs are per-run; others aggregate across hyperparameter sweeps.

---

## Running the pipeline

From `lab_tests/`, run:

```bash
python program.py --train
python program.py --evaluate
python program.py --graph
```

Or run all enabled stages:

```bash
python program.py --all
```

Use specific configs when needed:

```bash
python program.py \
  --data-config <name-or-path> \
  --model-config <name-or-path> \
  --output-config <name-or-path> \
  --probe-config <name-or-path> \
  --directory-config <name-or-path>
```

If you pass a bare config name, it resolves to `configs/<type>/<name>.yaml`.

---

## Key configuration files

- `configs/data/*`: dataset/trial settings
- `configs/model/*`: model/training sweep settings
- `configs/probe/*`: probe composition settings (exemplar/ratio/onehot)
- `configs/output/*`: which evaluators/outputs are enabled
- `configs/directory/*`: training/reference/results directories

---

## Outputs and metadata

Run outputs are organized under the configured `results` root, typically into:
- `ActivationData/`
- `AnalysisData/`
- `Output/`
- `RunMetadata/`

Stage metadata includes timestamps, git info, stage status, and stage-specific details to support reproducibility/debugging.

---

## Development notes

- Keep evaluator/output contracts stable (`Eval/Protocol.py`, `Output/Protocol.py`).
- Add new outputs by wiring dependencies in `Output/schema/dependencies.py` and adding corresponding config entries.
- Prefer consuming saved analysis artifacts rather than recomputing training outputs in plot code.
