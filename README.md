overview by chatgpt until I get around to writing proper documentation.  
working summary: https://docs.google.com/document/d/1QnrTfTMwoMHdTw7BSiUTkRQp3Au_E98HAMIwpIvE99o/edit?usp=sharing   

# Category Learning Pipeline

This project is a Python research pipeline for training neural network models on category learning tasks, saving their activations, running post-training evaluations, and generating plots from those evaluation results.

## Overview

The pipeline is organized into five main stages:

1. Train models
2. Save hidden and output activations
3. Run evaluation modules on saved activations
4. Save evaluator results to analysis files
5. Generate plots from those analysis files

The main idea is that training and evaluation are separated. Models are trained once, activations are saved once, and many different evaluators can then reuse those saved activations without rerunning training.

## High-Level Workflow

```text
Train models
    ↓
Save activations to Zarr
    ↓
Run evaluator modules
    ↓
Save evaluation outputs to NPZ
    ↓
Generate plots from output functions
```

## Directory Structure

Typical directories look like this:

```text
Data/
    Training/
    ReferenceMatrices/

Results/
    ActivationData/   # saved Zarr activations
    AnalysisData/     # evaluator outputs (.npz)
    Output/           # generated plots
```

When running hyperparameter sweeps, each hidden-layer-size / learning-rate pair gets its own suffixed directory. For example:

```text
Results/ActivationData/..._hls10_lr0p04
Results/AnalysisData/..._hls10_lr0p04
Results/Output/..._hls10_lr0p04
```

## Main Components

### Training

Training is handled by `StandardModel`.

Its job is to:

- initialize the model
- train it over epochs
- probe the model after initialization and after each epoch
- return hidden activations, output activations, and losses

The training stage is responsible for producing the raw activation data that everything else depends on.

### Activation Storage

Activations are saved in Zarr format.

This is useful because evaluators usually only need a subset of the probe rows, and Zarr allows those subsets to be loaded efficiently without rerunning the model.

Important utility location:

- `DriverUtils/Zarr.py`

Important helper:

- `load_slice(...)`

This helper is used by evaluators to load only the probe activations they need.

### Probe System

Probe examples are organized through a `probe_index`.

The probe index maps metadata conditions to probe IDs, so evaluators can request subsets such as:

- exemplars vs ratio trials
- modular vs lattice
- specific ratio conditions like `3:3`
- specific set labels within a ratio

This lets evaluation code stay clean and modular.

### Evaluation

Evaluators live in `Eval/`.

Each evaluator reads saved activations and computes a metric from them.

Typical evaluator interface:

```python
class SomeEvaluator:
    name = "SomeEvaluator"

    def run(self, cfg, zarr_path, vis=None):
        ...
        return raw
```

Some evaluators may also return metadata:

```python
return raw, metadata
```

Examples of evaluator outputs include:

- correlation statistics
- matrix correlations
- ratio test scores
- PCA / K95 values
- loss values

Evaluator results are saved as `.npz` files in `Results/AnalysisData/`.

### Analysis Files

Each analysis file usually contains things like:

- `raw`
- `mean`
- `std`
- `se`
- `ci_lo`
- `ci_hi`
- `n`

Some files may also include metadata such as:

- ratio labels
- set labels
- trial counts
- source labels
- category labels

This makes the saved analysis files more self-describing and helps output functions avoid hardcoding axis meanings.

### Output / Plotting

Output functions live in `Output/`.

Each output function reads one or more saved analysis files and converts them into `OutputSpec` objects.

Those `OutputSpec` objects are then rendered by the plotting layer.

This keeps:

- evaluation logic
- statistical summaries
- plotting logic

separate from one another.

Typical output interface:

```python
class SomeOutput:
    name = "SomeOutput"
    hyperd = False

    def generate_output(self, spec_cfg: dict, analysis_dir: str):
        ...
        return [spec1, spec2, ...]
```

Hyperparameter-dependent outputs may instead read from the shared analysis root and combine information across multiple HLS/LR subdirectories.

## Design Goals

### Post-Training Evaluation

Evaluations operate on saved activations rather than being embedded directly into training.

This makes the pipeline easier to debug, reuse, and extend.

### Reusable Activations

Hidden and output activations are saved once and reused by many evaluators.

This avoids repeated forward passes and keeps evaluation modular.

### Separation of Concerns

Training, storage, evaluation, statistics, and plotting are intentionally separated.

That makes it easier to:

- add new evaluators
- add new output functions
- change plotting without rewriting evaluators
- compare multiple outputs from the same saved activations

### Config-Driven Organization

The pipeline is config-driven.

Different configuration groups control:

- data behavior
- model behavior
- output behavior
- probe definitions
- directory structure

This helps keep experiments organized and reproducible.

## Typical Pipeline Behavior

A normal run looks like this:

```text
Train
  -> save Zarr activations
  -> run evaluators
  -> save NPZ analysis files
  -> generate plots
```

In practice, this means:

1. a model is trained
2. activations and losses are saved
3. evaluators compute metrics from those saved activations
4. outputs read evaluator results and build plots

## Notes

- Hyperparameter-dependent outputs can combine data across multiple HLS/LR directories
- Analysis files may include metadata when the axis meaning is not obvious
- Output functions should prefer reading saved metadata instead of hardcoding labels or axis assumptions
- The plotting layer is intentionally downstream of evaluation so that metrics and visualizations remain decoupled

## Summary

The architecture is built around a simple idea:

- train once
- save activations once
- evaluate many ways
- plot many ways

This makes the pipeline easier to extend and much easier to reason about than a design where all metrics are computed directly during training.
