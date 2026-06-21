# SPiNS Insect: Full Guide

The following is a complete usage guide for the category learning pipeline. Contact `abrahamcoburn@gmail.com` for questions about the repository.

---

## Table of Contents

1. [Overview](#overview)  
&emsp;1.1 [Layout](#layout)  
&emsp;1.2 [Structure and flow](#structure-and-flow)  
2. [Setup](#setup)  
&emsp;2.1 [Installing requirements](#installing-requirements)  
&emsp;2.2 [Working directory](#working-directory)  
&emsp;2.3 [Configuration files](#configuration-files)  
&emsp;2.4 [Running the program](#running-the-program)  
&emsp;2.5 [Stage prerequisites](#stage-prerequisites)  
3. [Data and probes](#data-and-probes)  
&emsp;3.1 [Training data](#training-data)  
&emsp;3.2 [Probe construction](#probe-construction)  
&emsp;3.3 [Adding probe data](#adding-probe-data)  
&emsp;3.4 [Probe metadata and index](#probe-metadata-and-index)  
&emsp;3.5 [Reference matrices](#reference-matrices)  
&emsp;3.6 [Alternative training regimes](#alternative-training-regimes)  
4. [Training](#training)  
&emsp;4.1 [Model initialization](#model-initialization)  
&emsp;4.2 [Data loaders](#data-loaders)  
&emsp;4.3 [Training and probing loop](#training-and-probing-loop)  
&emsp;4.4 [Epoch zero](#epoch-zero)  
&emsp;4.5 [Saving activations](#saving-activations)  
5. [Evaluation](#evaluation)  
&emsp;5.1 [Evaluation protocol](#evaluation-protocol)  
&emsp;5.2 [Dependency selection](#dependency-selection)  
&emsp;5.3 [Evaluation loop](#evaluation-loop)  
&emsp;5.4 [Saving evaluation data](#saving-evaluation-data)  
&emsp;5.5 [Significant epochs](#significant-epochs)  
6. [Output](#output)  
&emsp;6.1 [Output protocol](#output-protocol)  
&emsp;6.2 [OutputSpec object](#outputspec-object)  
&emsp;6.3 [Output loop](#output-loop)  
&emsp;6.4 [Saving and grouping graphs](#saving-and-grouping-graphs)  
&emsp;6.5 [Exporting tabular data](#exporting-tabular-data)  
7. [Extending the pipeline](#extending-the-pipeline)  
&emsp;7.1 [Adding an evaluator](#adding-an-evaluator)  
&emsp;7.2 [Adding an output class](#adding-an-output-class)  
&emsp;7.3 [Registering dependencies](#registering-dependencies)  
&emsp;7.4 [Modifying model architecture or training](#modifying-model-architecture-or-training)  
8. [Reference](#reference)  
&emsp;8.1 [Supported evaluators](#supported-evaluators)  
&emsp;8.2 [Supported output classes](#supported-output-classes)  
&emsp;8.3 [Artifact layout](#artifact-layout)  
9. [Project status](#project-status)  
&emsp;9.1 [Known issues](#known-issues)  
&emsp;9.2 [Limitations](#limitations)  
&emsp;9.3 [Potential improvements](#potential-improvements)  
&emsp;9.4 [Legacy artifacts](#legacy-artifacts)

---

## Overview

### Layout

```text
spins-insect/       
  configs/                  # Configuration folders / files (.yaml)
    data/                     # For configuring training data
    directory/                # For configuring location of data / results
    model/                    # For configuring model hyperparameters
    output/                   # For configuring output
    probe/                    # For configuring probe content
  Data/                     # All static data
    Exemplar/                 # Exemplars (for probe)
    MissingFeature/           # Missing feature trials (for probe)
    Probes/                   # Probe source files (.parquet) + probe index file
    Ratio/                    # Ratio trials (for probe)
    ReferenceMatrices/        # Reference matrices
    Training/                 # Training data
  DataHelper/               # Helper functionality for data processing
    Probe.py                  # All functionality for building probe / probe index
    SpecialDataLoader.py      # Special dataloader class for handling uneven number of mod/lat trials
    utils.py                  # Utility functions for data handling
  DriverUtils/              # Utility files for program flow 
    Organize.py               # Functionality for organizing results
    Parser.py                 # CLI argument parsing
    RMutils.py                # Utility functions for handling/generating reference matrices
    RunConfig.py              # Program flow (configuration init | training, eval, and output generation loops)
    RunMetadata.py            # Support for writing metadata at training/evaluation/output steps
    TabularExport.py          # Functionality for exporting tabular data to csv
    Visual.py                 # Support for terminal display during execution
    Zarr.py                   # All functionality associated with saving activations
  Eval/                     # Evaluation (evaluators not shown)
    Protocol.py               # Protocol for evaluator classes
    utils.py                  # Utility functions for evaluation
  Model/                    # Model setup & training
    NeuralNetwork.py          # Architecture
    StandardModel.py          # Initialization & training loop
  Output/                   # Output generation (output classes not shown)
    schema/                   # Output flow support
      dependencies.py           # Wires evaluator prerequisites to output classes
      OutputSpec.py             # OutputSpec object definition
      PlotOutput.py             # Functionality for generating graphs from OutputSpec objects
    Protocol.py               # Protocol for output classes
    utils.py                  # Utility functions for output generation
  Statistics/               # Statistics
    StatHelper.py             # Helper functions for collecting statistics
  program.py                # CLI entrypoint
```

### Structure and flow

The primary pipeline has three stages:

| Stage | Command | What it does | Saved result |
|---|---|---|---|
| **1. Training** | `--train` | Trains an ensemble for every configured hidden-layer-size and learning-rate pair, probing each model throughout training. | Activations and losses in `ActivationData/` as Zarr stores. |
| **2. Evaluation** | `--evaluate` | Runs the evaluators required by the output configuration on saved activation data. | Model-level results and summary statistics in `AnalysisData/` as `.npz` files. |
| **3. Output generation** | `--graph` | Reads saved analysis data and generates the configured plots. | Graphs under `Output/` as `.png` files. |

`--tabular` is an additional specialized export for significant-epoch masks and regression-oriented epoch data.

Each stage saves its results so that downstream stages can be rerun without rerunning upstream work. For example, graphs can be regenerated from saved analysis files without retraining or reevaluating the models.

`program.py` creates one `RunConfig` object for the invocation. During initialization, `RunConfig`:

1. Resolves the selected YAML configuration files.
2. Builds the probe and its index.
3. Loads the training data.
4. Creates the standard or special data loader.
5. Loads or generates the modular and lattice reference matrices.
6. Determines which evaluators and output classes are required.
7. Constructs the result paths for the selected configuration.

The requested command-line stages then run in this order:

```text
--train       RunConfig.train()
--evaluate    RunConfig.evaluate()
--graph       RunConfig.generate_output()
--tabular     RunConfig.generate_tabular_output()
```

`--all` runs training, evaluation, and graph generation. It does not include tabular export.

## Setup

### Installing requirements

From the repository root, install the dependencies:

```powershell
pip install -r requirements.txt
```

The main dependencies are NumPy, PyTorch, pandas, SciPy, Matplotlib, Zarr, tqdm, xarray, and pyarrow.

### Working directory

Run the program from `spins-insect/`:

```powershell
cd spins-insect
python program.py --all
```

Several default paths are relative to `spins-insect/`, including `Data/`, `configs/`, and `Results/`. Running `program.py` from the repository root will not resolve those paths correctly without custom configuration paths.

### Configuration files

The program combines five YAML configuration groups:

| Group | Default location | Purpose |
|---|---|---|
| Data | `configs/data/` | Selects the training data and category-specific loading behavior. |
| Model | `configs/model/` | Defines model dimensions, optimizer behavior, epochs, ensemble size, and hyperparameter ranges. |
| Output | `configs/output/` | Selects output classes and output-specific options. |
| Probe | `configs/probe/` | Selects the source files used to build the probe. |
| Directory | `configs/directory/` | Defines training-data, reference-matrix, and result directories. |

A command-line value may be either a bare configuration name or a path:

```powershell
python program.py --train --model-config hls20
python program.py --train --model-config configs/model/hls20.yaml
```

A bare name is resolved as `configs/<group>/<name>.yaml`. If no option is supplied, `default.yaml` is used for that group.

#### Data configuration

| Field | Meaning |
|---|---|
| `training_name` | Training CSV filename without `.csv`. |
| `num_mod_trials` | Number of modular trials at the beginning of the training data. |
| `num_total_trials` | Expected number of training rows. |
| `special_dl` | Whether to balance unequal modular and lattice trial counts with `SpecialDataLoader`. |
| `generate_rms` | Whether to generate reference matrices from the selected training data. |
| `alt` | Enables assumptions used by the alternative lattice regime. |

`RunConfig` verifies that the loaded training-row count matches `num_total_trials`. It also requires `special_dl` when modular and lattice trial counts are unequal.

#### Model configuration

| Field | Meaning |
|---|---|
| `num_features` | Number of features in each category. Input and output dimensions are twice this value. |
| `hidden_layer_range` | Inclusive `start`, `end`, and `step` values for hidden-layer size. |
| `learning_rate` | Inclusive `start`, `end`, and `step` values for learning rate. |
| `num_epochs` | Number of training epochs. Saved data contains one additional epoch for initialization. |
| `num_models` | Number of independently initialized models per hyperparameter pair. |
| `adam` | Uses Adam when true or omitted, and SGD when false. |
| `relu` | Applies ReLU after the hidden linear layer when true or omitted. Uses a linear hidden representation when false. |

Every hidden-layer-size and learning-rate combination is run separately.

Use a distinct model-configuration filename for materially different regimes, such as `relu: false`. Result directories are keyed by configuration filenames rather than configuration contents, so editing an existing YAML file in place reuses the same artifact path.

#### Output configuration

Each top-level output key corresponds to an entry in `Output/schema/dependencies.py`:

```yaml
SeriesCorrelation:
  present: true
  group: true
  corr_type: standard
```

`present: true` enables the dependency. Output-specific fields may select:

- `corr_type`: `standard` or `wb`
- `epochs`: `range`, `sig`, or `wb-sig`
- `sige_type`: `sig` or `wb-sig`
- `range`: inclusive `start`, `stop`, and `step`
- `sets`: selected ratio-trial set labels
- `group`: whether graphs should be copied into cross-run grouping directories
- Plot labels, dimensions, bounds, and display options supported by the output class

Every top-level output key must be registered in `Output/schema/dependencies.py`. Unknown or misspelled keys raise a configuration error before any stage runs.

#### Probe configuration

The probe configuration gives the generated probe a name and selects its source files:

```yaml
name: default
data:
  exemplar: original_exemplars
  ratio: ratiotrials
  missing_feature: mf_probe_trials
  onehot: onehot
```

A source may be omitted when its corresponding evaluators are not needed, but enabled evaluators will fail if their required probe data is absent.

#### Directory configuration

The directory configuration defines:

```yaml
training_data: Data/Training
reference_matrices: Data/ReferenceMatrices
results: Results
```

Paths are interpreted relative to the working directory unless they are absolute.

### Running the program

Run all primary stages with the default configurations:

```powershell
python program.py --all
```

Run stages individually:

```powershell
python program.py --train
python program.py --evaluate
python program.py --graph
```

Export specialized tabular data:

```powershell
python program.py --tabular
```

Select configurations:

```powershell
python program.py --all `
  --data-config default `
  --model-config hls20 `
  --output-config default `
  --probe-config default `
  --directory-config default
```

Add `--visual` or `-v` to print the resolved configurations and display progress:

```powershell
python program.py --all --visual
```

Visual mode asks for confirmation after displaying the configuration. Entering anything other than `y` cancels execution.

### Stage prerequisites

Stages are intended to be rerunnable independently, but they must use the same data, model, and probe configuration names as the upstream stage.

| Stage | Required input |
|---|---|
| Training | Training CSV, probe sources, and reference matrices or enough data to generate them. |
| Evaluation | Matching `ActivationData/.../activations.zarr` directories produced by training. |
| Graph generation | Matching evaluator `.npz` files under `AnalysisData/`. |
| Tabular export | Matching correlation, K95, ratio-test, and significant-epoch analysis files. |

Changing only the output configuration does not change the result stub. This allows additional evaluators and graphs to be added to an existing trained configuration.

Although stages use saved intermediate data, every invocation currently performs full `RunConfig` initialization. Evaluation and graph-only commands therefore still load training data, rebuild the probe, and load reference matrices.

## Data and probes

### Training data

Training CSV files are stored under the configured training-data directory. The current files use:

```text
type
trial_id
input_0 ... input_21
output_0 ... output_21
```

Only rows with `type == "train"` are loaded. The configured `num_features` determines how many input and output columns are read:

```text
input dimension  = 2 * num_features
output dimension = 2 * num_features
```

The current experiments use 11 features per category. Modular features occupy indices `0` through `10`, and lattice features occupy indices `11` through `21`. Within each category, the first three features are core and the remaining features are peripheral.

Training rows must place all modular trials first because `num_mod_trials` is used to split the categories when generating reference matrices and creating the special data loader.

### Probe construction

`DataHelper/Probe.py` builds the probe whenever `RunConfig` is initialized. The existing generated probe is not treated as a cache.

Configured sources are concatenated in this order:

1. Exemplars
2. Ratio trials
3. Missing-feature trials
4. One-hot feature trials

The builder converts source CSV files into normalized Parquet data, combines their feature tensors, and writes:

```text
Data/Probes/Sources/<source>.parquet
Data/Probes/<probe-name>.parquet
Data/Probes/<probe-name>.index.json
```

The returned probe tensor is evaluated before training and after every training epoch.

| Source | Primary use |
|---|---|
| `exemplar` | Defines modular and lattice reference exemplars for ratio generalization. |
| `ratio` | Tests modular preference across feature-composition ratios and trial sets. |
| `missing_feature` | Tests whether the model activates the intended missing feature more strongly than an alternative. |
| `onehot` | Measures individual feature representations for correlation and K95 analyses. |

### Adding probe data

To add another file of an existing probe type:

1. Place the source CSV in its corresponding folder, such as `Data/Ratio/`, `Data/Exemplar/`, or `Data/MissingFeature/`.
2. Add the filename without `.csv` to a probe configuration under `configs/probe/`.
3. Run the program with that probe configuration. `DataHelper/Probe.py` will call the existing converter and rebuild the combined probe and index.

Each probe configuration currently accepts one source file per probe type. Supporting multiple files of the same type in one probe requires extending the corresponding configuration value and loading logic in `DataHelper/Probe.py`.

To introduce a new probe type, add a converter in `DataHelper/utils.py` that writes a Parquet file containing a `tensor` column plus any metadata needed by its evaluator. Every tensor must be a binary vector of length `2 * num_features`. Import and call the converter from `DataHelper/Probe.py`, add the new configuration key, normalize the result with a unique `source` name, and append it to `dfs`.

The raw source file should be committed in its source-type folder. Files under `Data/Probes/Sources/` are generated by the converter and should not be edited manually.

### Probe metadata and index

Every probe row contains a feature tensor and metadata fields from its source file. The builder adds a `source` field and normalizes metadata values to strings.

The generated index maps `column=value` keys to probe row numbers:

```json
{
  "source=onehot": [199, 200],
  "ratio=3:3": [64, 65],
  "sets=mod-core": [16, 17]
}
```

Evaluators retrieve subsets by looking up one key or intersecting several results. For example, ratio trials can be selected by intersecting:

```text
source=ratio
ratio=3:3
sets=mod-core
```

The probe Parquet file retains full row metadata. `RunConfig.probe_metadata` is used directly by the missing-feature evaluator, while `RunConfig.probe_index` is used by the other probe-based evaluators.

### Reference matrices

Correlation evaluation compares learned one-hot feature representations with modular and lattice reference matrices.

When `generate_rms: false`, matrices are loaded from CSV files in the configured reference-matrix directory.

When `generate_rms: true`, matrices are generated from the training inputs:

1. Split modular and lattice trials at `num_mod_trials`.
2. Select the corresponding category's feature columns.
3. Count feature co-occurrences.
4. Convert co-occurrence counts into pairwise Jaccard similarity.

The resulting matrices have shape `(num_features, num_features)`.

### Alternative training regimes

Alternative regimes require caution. They were used before the probe-building and stage-separation refactor and are not fully connected to the current pipeline.

Current inconsistencies include:

- `alt: true` changes expected ratio labels and selects the alternative lattice reference matrix, but it does not automatically select the available alternative exemplar, ratio, or missing-feature probe sources.
- No checked-in probe configuration currently wires together the alternative probe sources.
- Several outputs and tabular exports assume that a `3:3` ratio exists, while alternative ratio data uses `2:2` as its equal-ratio condition.
- Several S-curve bounds and midpoint values assume six-feature trials.
- The alternative lattice reference-matrix path is partly hard-coded instead of consistently using the directory configuration.

Treat alternative regimes as requiring an audit of data, probe, evaluator, and output assumptions before use. The standard configuration path is the maintained path.

## Training

### Model initialization

For every hidden-layer-size and learning-rate pair, `RunConfig.train()` creates `num_models` independent `StandardModel` instances.

The current network in `Model/NeuralNetwork.py` is:

| Layer | Operation | Size |
|---|---|---|
| Input | Binary feature vector | `2 * num_features` |
| Hidden | Fully connected layer, optionally followed by ReLU | `hidden_layer_size` |
| Output | Fully connected layer producing logits | `2 * num_features` |

The model is trained as an autoencoder: each training input is paired with its target feature vector. `StandardModel` uses `BCEWithLogitsLoss` during training and applies a sigmoid only when collecting probe output activations.

Adam is used by default. Set `adam: false` in the model configuration to use SGD.

ReLU is used by default. Set `relu: false` to replace it with an identity operation and use the hidden linear-layer output directly.

Models run on CPU unless a device is supplied directly when constructing `StandardModel`. The command-line pipeline does not currently expose a device option.

### Data loaders

The standard loader creates one shuffled full-data batch per epoch. The current default training regime therefore performs one optimizer update per epoch.

`SpecialDataLoader` is used when modular and lattice categories have unequal numbers of training trials. Each epoch:

1. Includes every trial from the smaller category.
2. Samples an equal number of trials from the larger category without replacement.
3. Favors trials that have appeared less often in previous epochs.
4. Combines and shuffles the balanced trial set into one batch.

This balances modular and lattice exposure over training while rotating through the larger category.

### Training and probing loop

For each model:

1. Evaluate the untrained model on the full probe.
2. Train for one epoch.
3. Evaluate the model on the full probe.
4. Repeat until `num_epochs` training epochs are complete.

Each probe pass records:

- Hidden-layer activations
- Sigmoid-transformed output activations
- Training loss for the corresponding epoch

The in-memory result for one model is:

```text
losses: (E,)
hidden: (E, P, H)
output: (E, P, O)
```

where:

- `E = num_epochs + 1`
- `P = number of probe trials`
- `H = hidden-layer size`
- `O = 2 * num_features`

Training currently retains every model result for one hyperparameter pair in memory and writes the pair to Zarr after the full ensemble finishes.

### Epoch zero

Epoch `0` contains probe activations from the randomly initialized model before any training.

This is intended as a diagnostic state. Evaluators should not show learned category-structure effects before learning. One known exception is bias on modular-core and lattice-core equal-ratio trials: consistent cores in the exemplars can dominate the correlation-based preference measure even before training.

There is no meaningful pre-training loss. For plotting, the code replaces the epoch-zero loss value with the epoch-one loss value. This avoids a distracting line from zero to the first training loss while preserving epoch-zero activations for analysis.

### Saving activations

Each hidden-size/learning-rate pair is saved to:

```text
<result-root>/ActivationData/<training-name>_hls<HLS>_lr<LR>/activations.zarr
```

Periods in learning-rate directory names are replaced with `p`, such as `lr0p04`.

The Zarr group contains:

| Array | Shape | Default type |
|---|---|---|
| `loss` | `(M, E)` | `float32` |
| `hidden` | `(M, E, P, H)` | `float16` |
| `output` | `(M, E, P, O)` | `float32` |

The group attributes record model, epoch, probe, hidden, and output dimensions and the stored data types.

Stage metadata is written under `RunMetadata/activation_data/` and to a `metadata.json` file in the stage directory. Metadata includes timestamps, status, configuration path, Git branch, Git commit, and training details.

## Evaluation

### Evaluation protocol

Evaluator classes follow the protocol in `Eval/Protocol.py`:

```python
class Evaluator(Protocol):
    name: str

    def run(self, cfg, zarr_path: str, vis=None) -> tuple[np.ndarray, dict]:
        ...
```

An evaluator must provide:

- A unique `name`, used as the analysis filename.
- A `run()` method that reads the required activation slices.
- A raw NumPy result whose first two axes are model and epoch whenever model-level statistics are expected.
- Optional metadata describing condition labels, trial counts, or other axes.

Evaluators should load only required probes with `DriverUtils.Zarr.load_slice()` rather than loading the full activation store when possible.

### Dependency selection

Evaluation is selected indirectly through the output configuration.

`Output/schema/dependencies.py` maps each output key to:

- Evaluator classes required to produce its analysis data.
- The output class that consumes those results.

When several enabled outputs require the same evaluator, only one evaluator instance is added. Significant-epoch evaluators are added when an enabled output requests `sig` or `wb-sig` mode.

This design allows output requirements to determine the evaluator set, but a new output or evaluator is not active until its dependency entry is registered.

### Evaluation loop

For each hidden-size/learning-rate pair:

1. Construct the path to the matching activation Zarr store.
2. Create the matching analysis directory.
3. Run every required evaluator.
4. Calculate statistics across models.
5. Save one `.npz` file per evaluator.
6. Generate significant-epoch masks when required.

The statistics helper treats model as axis `0`, ignores non-finite values, and calculates:

- Mean
- Sample standard deviation
- Standard error
- 95% t-based confidence interval
- Number of finite models contributing to each value

### Saving evaluation data

Evaluator data is saved to:

```text
<result-root>/AnalysisData/<training-name>_hls<HLS>_lr<LR>/<EvaluatorName>.npz
```

Standard evaluator files contain:

```text
raw
mean
std
se
ci_lo
ci_hi
n
metadata    # only when returned by the evaluator
```

`raw` retains model-level results. The remaining statistical arrays remove the model axis.

Evaluator metadata is stored as an object array. Load files containing metadata with:

```python
data = np.load(path, allow_pickle=True)
metadata = data["metadata"].item()
```

Significant-epoch files use a separate schema:

```text
sige.npz
  results: (M, E) binary mask

wb-sige.npz
  results: (M, E) binary mask
```

### Significant epochs

Significant-epoch modes compare models at a point where modular and lattice learning are approximately equal.

#### Standard significance

The standard mask uses hidden-layer correlations from `CorrelationEvaluator`. An epoch is selected when:

```text
modular p < 0.05
lattice p < 0.05
abs(modular r - lattice r) <= 0.05
```

Outputs using `sig` select the first matching epoch independently for each model. Models with no matching epoch are excluded from those outputs.

#### Within-vs-between significance

Within-vs-between correlation does not currently calculate a p value. The provisional `wb-sig` mask requires:

```text
modular score > 0.25
lattice score > 0.25
abs(modular score - lattice score) <= 0.05
```

The `0.25` threshold was chosen as an approximate lower bound corresponding to significant values observed with the standard correlation measure. It is not itself a formal significance test. A statistically justified test, potentially based on a t-test, should replace this heuristic.

## Output

### Output protocol

Output classes follow the protocol in `Output/Protocol.py`:

```python
class Output(Protocol):
    name: str
    hyperd: bool

    def generate_output(
        self,
        sub_cfg: dict,
        analysis_dir: str,
    ) -> list[OutputSpec]:
        ...
```

`name` matches the dependency-map entry. `hyperd` determines whether the output operates on one hyperparameter pair or discovers and compares several analysis directories.

Output classes do not save figures directly. They load saved analysis data and return one or more `OutputSpec` objects.

### OutputSpec object

`Output/schema/OutputSpec.py` defines a plot independently of the analysis that produced it.

An `OutputSpec` contains:

- Figure identifier and labels
- One or more line, scatter, or bar series, or one matrix
- Axis bounds, ticks, and reference lines
- Confidence intervals or error bars
- Colors, line styles, markers, and transparency
- Legend, grid, aspect, size, and DPI options
- Matrix color-map and color-bar options

Each `Series` specifies its plot kind and data:

```python
Series(
    kind=PlotKind.LINE,
    label="mod",
    x=[0, 1, 2],
    y=[0.1, 0.3, 0.5],
    color=Color.BLUE,
)
```

`plot_output()` requires exactly one of:

- `series_list` for line, scatter, or bar plots
- `matrix` for heatmaps

Keeping plotting separate from analysis lets several figures reuse the same evaluator artifact.

### Output loop

Output generation has two passes.

#### Hyperparameter-level outputs

Classes with `hyperd = True` receive the base analysis path and discover sibling directories whose names contain hidden-layer-size and learning-rate suffixes. These outputs compare several runs, usually grouping hidden-layer sizes by learning rate.

#### Per-run outputs

Classes with `hyperd = False` run once for every hidden-size/learning-rate pair and read that pair's analysis directory.

For each returned `OutputSpec`, `plot_output()` creates the target directory and saves the figure.

If an output class raises an exception while generating its `OutputSpec` objects, that output is reported to standard error and skipped so the remaining output classes can run. Plotting and graph-grouping exceptions are not caught and stop the output stage.

A summary failure count is printed after output generation. The output stage metadata is marked `failed` when any output-generation function fails, even though successfully generated figures are retained. Errors are not currently written to a persistent log.

### Saving and grouping graphs

Per-run figures are saved under:

```text
<result-root>/Output/<training-name>_hls<HLS>_lr<LR>/<output-name>/*.png
```

Hyperparameter-level figures are saved under:

```text
<result-root>/Output/<training-name>/<output-name>/*.png
```

Output directory names include relevant mode suffixes, such as:

```text
SeriesCorrelation_standard
SCurve_sige
Correlation-HLS_wb_wb-sige
```

When an output configuration sets `group: true`, generated graph files are also copied into:

```text
<result-root>/Output/Groupings/hls<HLS>/
<result-root>/Output/Groupings/lr<LR>/
```

Grouping directories are created only when there is more than one value on the opposite hyperparameter axis. Graphs remain in their original per-run directories.

The plotting layer currently saves PNG files only, even though the grouping helper recognizes additional image extensions.

### Exporting tabular data

`--tabular` is a specialized utility for significant-epoch analysis and possible linear regression workflows. It is not a general export of every evaluator.

For each hidden-size/learning-rate pair, it writes:

```text
tabular_sig/
  sige.csv
  wb-sige.csv
  tabular_masks/
    sige.csv
    wb-sige.csv

tabular_range/
  standard/
    tabular-e0.csv
    tabular-e5.csv
    ...
  wb/
    tabular-e0.csv
    tabular-e5.csv
    ...
```

Rows contain per-model values for:

- Modular hidden correlation
- Lattice hidden correlation
- Modular-minus-lattice hidden correlation
- Modular K95
- Lattice K95
- Modular-minus-lattice K95
- Trial-count-weighted modular preference on `3:3` trials
- Modular preference for each valid `3:3` trial set

Summary rows provide the mean, standard deviation, standard error, and confidence-interval bounds.

The mask CSV files expose every model's significant-epoch mask for direct inspection. Models without a significant epoch are omitted from significant-epoch metric tables.

Tabular export assumes that required standard correlation, within-vs-between correlation, K95, ratio-test, `sige`, and `wb-sige` files already exist. If they do not, you must first run `--evaluate` with an output configuration that triggers the necessary evaluators.

## Extending the pipeline

### Adding an evaluator

Add evaluator implementations under `Eval/`.

1. Create a class with a unique `name`.
2. Implement `run(cfg, zarr_path, vis=None)`.
3. Use the probe index or metadata to select required probe rows.
4. Use `load_slice()` to read the smallest needed activation subset.
5. Return `(raw, metadata)`.
6. Keep model and epoch as the first two axes when the standard statistics pipeline applies.
7. Update visual progress once per model/epoch when practical.
8. Register the evaluator as a dependency of at least one output configuration key.

Existing evaluators illustrate different patterns:

- `CorrelationEvaluator` selects one-hot probes and produces fixed condition axes.
- `RatioTestEvaluator` returns metadata for dynamic ratio and set axes.
- `MFA` uses probe metadata columns and returns grouped accuracy.
- `LossEvaluator` reads an existing Zarr array without probe selection.

If an evaluator result cannot be summarized across model axis `0`, extend the standard `RunConfig.evaluate()` save path rather than forcing the result into the existing contract.

### Adding an output class

Add output implementations under `Output/`.

1. Create a class with a unique `name`.
2. Set `hyperd = False` for one analysis directory or `True` for hyperparameter-level discovery.
3. Implement `generate_output(sub_cfg, analysis_dir)`.
4. Load saved `.npz` artifacts rather than recomputing evaluator logic.
5. Validate expected artifact shapes and metadata.
6. Return a list of `OutputSpec` objects.
7. Add the output to `Output/schema/dependencies.py`.
8. Add an output configuration entry with `present: true`.

Use shared helpers in `Output/utils.py` for:

- Loading and validating correlation, K95, and ratio data
- Resolving epoch ranges
- Selecting first significant epochs
- Trial-count-weighted ratio averages
- Regression lines
- Shared plot bounds
- Hyperparameter-run discovery

Keep output-specific helpers local unless they are useful to several output classes.

### Registering dependencies

`Output/schema/dependencies.py` is the central registry.

Import the evaluator and output classes, then add:

```python
dependencies = {
    "NewOutput": Dep(
        [RequiredEvaluator, AnotherEvaluator],
        NewOutputClass,
    ),
}
```

The string key must match the top-level YAML output key. The output class's `name` should use the same name unless there is a specific reason not to.

When an output supports standard and within-vs-between correlation, use `corr_type` consistently so dependency selection adds the appropriate evaluator. When an output uses a significant epoch, ensure its configuration selects `sig` or `wb-sig`.

### Modifying model architecture or training

Architecture changes belong in `Model/NeuralNetwork.py`. Preserve `forward(x, return_hidden=False)` unless the activation-collection contract is also updated.

Training behavior belongs primarily in:

- `Model/StandardModel.py`
- `DataHelper/utils.py`
- `DataHelper/SpecialDataLoader.py`
- `DriverUtils/RunConfig.py`

Important current assumptions include:

- `StandardModel.train_eval()` returns `losses`, `hidden`, and `output`.
- Probe arrays have the same epoch count.
- Probe output values are sigmoid-transformed.
- One-hot hidden activations separate into modular features followed by lattice features.
- Zarr storage has a fixed hidden dimension within one hyperparameter pair.

The current standard loader uses one full-data batch, so there is one optimizer update per epoch. To update more frequently (e.g. after every trial):

1. Add a `batch_size` model-config value and pass it from `RunConfig` to `DataHelper.utils.get_dataloader()`.
2. Give `get_dataloader()` a `batch_size` argument and use it when constructing the PyTorch `DataLoader`. Set it to `1` for one update per trial.
3. If using `SpecialDataLoader`, also replace its hard-coded full balanced-batch size with the configured batch size.
4. Save a sample-weighted mean loss if losses should remain comparable across batch sizes; the current loop sums the per-batch mean losses.

`StandardModel.train_eval()` already performs one optimizer step per returned batch, but probes only after the complete loader pass. Therefore, smaller batches produce multiple updates per saved epoch while preserving the current epoch and Zarr layout. Probing after every trial would additionally require moving `_probe()` inside the batch loop and redefining `eval_epochs` and the saved activation indices.

If the hidden representation changes location or structure, update both the model return values and every evaluator that consumes hidden activations.

## Reference

### Supported evaluators

| Evaluator | Raw shape | Purpose |
|---|---|---|
| `Loss` | `(M, E)` | Loads saved training loss. |
| `Correlation` | `(M, E, 2, 2, 2)` | Correlates hidden/output one-hot representation matrices with modular/lattice reference matrices. The final axis is Pearson `r` and `p`. |
| `MatrixCorrelation` | `(M, E, 2, 2, F, F)` | Saves pairwise one-hot representation correlation matrices. |
| `WithinVsBetweenCorrelation` | `(M, E, 2, 2)` | Measures within-structure minus between-structure similarity for modular and lattice features. |
| `K95` | `(M, E, 2, 1)` | Counts principal components required to explain 95% of hidden one-hot activation variance. |
| `RatioTest` | `(M, E, R, S)` | Measures the proportion of ratio trials whose hidden representations are closer to modular than lattice exemplars. |
| `MFA` | `(M, E, 4)` | Measures missing-feature choice accuracy for modular/lattice core/periphery groups. |

Condition-axis conventions:

```text
source axis:   0 = hidden, 1 = output
category axis: 0 = modular, 1 = lattice
```

`SignificantEpochEvaluator` and `WithinVsBetweenSignificantEpochEvaluator` are post-processors that convert correlation results into `(M, E)` masks.

### Supported output classes

| Output | Scope | Description |
|---|---|---|
| `SeriesCorrelation` | Per run | Correlation trajectories with loss; supports standard and `wb` correlation. |
| `MatrixCorrelation` | Per run | Feature-representation correlation heatmaps for selected epochs. |
| `RatioOverEpochs` | Per run | Equal-ratio modular preference over epochs by set or averaged sets. |
| `AllModels33OverEpochs` | Per run | Smoothed equal-ratio trajectories for every model. |
| `SCurve` | Per run | Mean modular preference across feature-composition ratios. |
| `AllModelsSCurve` | Per run | One ratio curve per model. |
| `SeriesK95` | Per run | Modular and lattice K95 trajectories. |
| `SeriesMFA` | Per run | Missing-feature accuracy trajectories. |
| `GeneralizationCorrelationDiff` | Per run | Modular preference versus modular-minus-lattice correlation. |
| `K95Correlation` | Per run | Category K95 versus category correlation. |
| `K95DiffCorrelationDiff` | Per run | Modular-minus-lattice K95 versus correlation differences. |
| `K95DiffGeneralization` | Per run | Modular preference versus modular-minus-lattice K95. |
| `K95-HLS` | Hyperparameter | K95 distributions and means across hidden-layer sizes. |
| `Correlation-HLS` | Hyperparameter | Correlation at first significant epoch across hidden-layer sizes. |
| `SigE-HLS` | Hyperparameter | First significant epoch across hidden-layer sizes. |
| `Epochs-HLSwK95Heatmap` | Hyperparameter | K95 summaries across sampled epochs and hidden-layer sizes. |

`MatrixCorrelation` currently supports only `range` epoch mode.

### Artifact layout

The result stub is based on selected data, model, and probe configuration filenames:

```text
d__<data-config>&m__<model-config>&p__<probe-config>
```

A typical result tree is:

```text
<results>/
  d__default&m__hls20&p__default/
    config.json
    ActivationData/
      <training-name>_hls20_lr0p04/
        activations.zarr/
    AnalysisData/
      <training-name>_hls20_lr0p04/
        Correlation.npz
        Loss.npz
        RatioTest.npz
        sige.npz
    Output/
      <training-name>_hls20_lr0p04/
        <output-name>/
          *.png
      Groupings/
    RunMetadata/
      activation_data/
      analysis_data/
      output/
```

`config.json` records all five resolved configuration groups plus Git information. It is refreshed at the beginning of every invocation so it reflects the configuration used by the current command. Stage metadata records stage-specific status and details.

## Project status

This section distinguishes confirmed behavior problems from accepted constraints and unimplemented cleanup.

### Known issues

1. **Tabular export requires both significance modes and all analysis dependencies.** Missing `sige.npz`, `wb-sige.npz`, K95, ratio-test, or correlation files cause the entire export to fail.

2. **Alternative probe sources are not wired automatically.** `alt: true` does not select alternative exemplar, ratio, or missing-feature sources, and equal-ratio assumptions remain hard-coded as `3:3` in several places.

### Potential improvements

- Add stage-specific initialization so evaluation, graphing, and tabular export load only required inputs.
- Add persistent structured logging for output failures and other stage warnings. Output failures are currently reported only to standard error.
- Replace the within-vs-between threshold heuristic with a justified significance test.
- Add focused tests for configuration resolution, probe indexing, evaluator shapes, dependency selection, and output generation.
- Stream model results into Zarr rather than retaining an entire hyperparameter pair in memory.
- Fully reconnect and validate alternative training regimes.

### Legacy artifacts

- The `per_epoch` field remains in several output configurations but is not used by current output orchestration.
