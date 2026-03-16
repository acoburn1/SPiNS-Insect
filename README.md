overview by chatgpt until I get around to writing proper documentation.  
working summary: https://docs.google.com/document/d/1QnrTfTMwoMHdTw7BSiUTkRQp3Au_E98HAMIwpIvE99o/edit?usp=sharing   

# Category Learning Pipeline

This project trains neural network models on category learning tasks, saves hidden and output activations, runs evaluation modules on the saved activations, and generates plots from the resulting analysis files.

## Pipeline

```text
Train models
    ↓
Save activations to Zarr
    ↓
Run evaluators
    ↓
Save analysis outputs to NPZ
    ↓
Generate plots
```

## Directory Layout

```text
Data/
    Training/
    ReferenceMatrices/

Results/
    ActivationData/
    AnalysisData/
    Output/
```

For hyperparameter sweeps, results are written to suffixed directories such as:

```text
Results/ActivationData/..._hls10_lr0p04
Results/AnalysisData/..._hls10_lr0p04
Results/Output/..._hls10_lr0p04
```

## Main Parts

### Training

Training is handled by `StandardModel`.

It trains the model over epochs and records:

- hidden activations
- output activations
- losses

### Activation Storage

Activations are saved in Zarr.

Evaluators use `DriverUtils/Zarr.py` and `load_slice(...)` to load only the probe rows they need.

### Probe Index

Probe examples are organized through a `probe_index`.

This is used to select subsets such as:

- exemplar vs ratio trials
- modular vs lattice
- specific ratios
- specific set labels

### Evaluators

Evaluators are located in `Eval/`.

Each evaluator reads saved activations and returns raw evaluation output, optionally with metadata.

Typical interface:

```python
class SomeEvaluator:
    name = "SomeEvaluator"

    def run(self, cfg, zarr_path, vis=None):
        ...
        return raw
```

or

```python
return raw, metadata
```

Evaluator outputs are saved as `.npz` files in `Results/AnalysisData/`.

### Analysis Files

Analysis files usually contain:

- `raw`
- `mean`
- `std`
- `se`
- `ci_lo`
- `ci_hi`
- `n`

Some also include metadata such as axis labels, ratio labels, set labels, source labels, category labels, or trial counts.

### Output Functions

Output functions are located in `Output/`.

They read saved analysis files and build `OutputSpec` objects, which are then rendered by the plotting layer.

Typical interface:

```python
class SomeOutput:
    name = "SomeOutput"
    hyperd = False

    def generate_output(self, spec_cfg: dict, analysis_dir: str):
        ...
        return [spec1, spec2, ...]
```

Some outputs read from a single analysis directory. Others combine data across multiple HLS/LR directories.

## Run Structure

A typical run consists of:

1. training a model or sweep
2. saving activations
3. running evaluators
4. saving analysis files
5. generating plots

## Notes

- Hyperparameter-dependent outputs may combine data across multiple HLS/LR directories.
- Output functions should use saved metadata when it is available.
- Plot generation is driven from saved analysis files rather than directly from training.
