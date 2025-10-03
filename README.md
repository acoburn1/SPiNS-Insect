# lab_tests — Training and Analysis Guide

This repository trains simple neural networks to learn category structure and evaluates them against modular vs. lattice reference structure. It also aggregates results across many runs and generates analysis plots.

What you can do with this project:
- Prepare training data from tab-delimited, Python-list, or JS stimlist sources
- Train 1 or many neural networks and save per-epoch results
- Configure what to record via data parameters
- Aggregate results across runs with confidence intervals and per-epoch paired t-tests
- Plot correlations, S-curves, 3:3 analyses, activation summaries, and PCA-based dimensionality relations


## 1) Environment

- Python 3.9+
- Packages: torch, numpy, scipy, matplotlib

Visual Studio 2022 (optional):
- Create/activate a Python environment in Python Environments
- Install packages with pip
- Right-click `lab_tests\program.py` ? Set as Startup File
- Debug > Start Debugging (F5)


## 2) Data preparation

You can train from either:
- CSV with 22-dim binary inputs/outputs (recommended for training), or
- A stim list source (Python/JS) converted to CSV/tab first.

Key modules under `lab_tests\Prep`:
- `DataPreparer.py` and `DataUtils.py`: main entry points for loading/creating DataLoaders
- `FormatConverters.py` and `BaseConverter.py`: converters for tab-delimited and Python list formats
- `StimListExtractor.py`: parses the JS stimlist to a tab-delimited training file

Examples:

- Convert a tab-delimited file to CSV
```
from Prep import DataUtils

DataUtils.convert_tab_delimited_to_csv(
    input_file="Data/Old/trainPats_22dim_2cats_good.txt",
    output_file="Data/Current/stimList_gencat_hydra_forAbe.csv",
    num_features=11
)
```

- Convert a Python lists file to CSV
```
from Prep import DataUtils

DataUtils.convert_python_lists_to_csv(
    input_file="Data/Current/stimList_gencat_hydra_forAbe.py",
    output_file="Data/Current/stimList_gencat_hydra_forAbe.csv",
    num_features=11
)
```

- Extract training data from a JS stimlist (writes a tab-delimited file)
```
from Prep.StimListExtractor import StimListExtractor

extractor = StimListExtractor("Data/Current/stimList_gencat_hydra_forAbe.js", num_features=11)
extractor.save_training_data_to_file("Data/Current/extracted_training_data.txt")
# Then optionally convert that tab file to CSV via DataUtils.convert_tab_delimited_to_csv(...)
```

CSV schema (22 inputs + 22 outputs):
- Columns: `type`, `trial_id`, `input_0..input_21`, `output_0..output_21`
- `type` is `train` or `test` (optional test rows)

Reference matrices (for structure correlation evaluation):
- Place in `Data/ReferenceMatrices/`
  - `cooc-jaccard-mod.csv` (modular)
  - `cooc-jaccard-lat.csv` (lattice)

Ratio trials definition (used by ratio tests):
- `Prep.DataUtils.generate_ratio_trials()` reads `Data/Current/ratiotrials.csv`


## 3) Train a single model (end-to-end)

```
import numpy as np
from torch import nn
from Prep import DataUtils
from Model.StandardModel import StandardModel

NUM_FEATURES = 11
HIDDEN_LAYER_SIZE = 400
NUM_EPOCHS = 60
BATCH_SIZE = 96
LEARNING_RATE = 0.04
INCLUDE_E0 = True

DATA_CSV = "Data/Current/stimList_gencat_hydra_forAbe.csv"
MOD_REF = "Data/ReferenceMatrices/cooc-jaccard-mod.csv"
LAT_REF = "Data/ReferenceMatrices/cooc-jaccard-lat.csv"

DATA_PARAMS = {
  "losses": True,
  "hidden_activations": True,
  "m_output_corrs": True, "l_output_corrs": True,
  "m_hidden_corrs": True, "l_hidden_corrs": True,
  "output_matrices": False, "hidden_matrices": False,  # large; usually keep False when batching
  "output_ratio_tests": True, "hidden_ratio_tests": True,
  "output_activation_exemplar_tests": True,
  "output_activation_onehot_tests": True
}

# Data
prep = DataUtils.load_csv_data(DATA_CSV, num_features=NUM_FEATURES)
dataloader = prep.get_dataloader(batch_size=BATCH_SIZE, shuffle=True)
mod_ref, lat_ref = DataUtils.get_probability_matrices_m_l(MOD_REF, LAT_REF)

# Model
model = StandardModel(
    num_features=NUM_FEATURES,
    hidden_layer_size=HIDDEN_LAYER_SIZE,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    loss_fn=nn.BCEWithLogitsLoss()
)

# Train + evaluate each epoch
results = model.train_eval_test_P(
    dataloader,
    modular_reference_matrix=mod_ref,
    lattice_reference_matrix=lat_ref,
    data_params=DATA_PARAMS,
    print_data=False,
    include_e0=INCLUDE_E0
)

# Save for analysis
np.savez("Results/Data/run1.npz", **results)
```

Notes:
- `include_e0=True` adds a pre-training evaluation; arrays become length `num_epochs + 1`. If `False`, arrays are length `num_epochs`.


## 4) Data parameters (what gets computed/saved per epoch)

Pass a dict to `StandardModel.train_eval_test_P(..., data_params=...)`. Common keys:

- Training loss
  - `"losses"`: per-epoch training loss (sum over batches)

- Hidden activations
  - `"hidden_activations"`: hidden activations for each one-hot probe (shape: 2F × H), used by PCA/other analyses

- Structure correlations (vs reference matrices)
  - `"m_output_corrs"`, `"l_output_corrs"`: Pearson r between output-layer distributions and modular/lattice reference
  - `"m_hidden_corrs"`, `"l_hidden_corrs"`: same but using hidden similarity matrices

- Distributions (large; usually leave False when batching many runs)
  - `"output_matrices"`: full output distributions for 2F one-hot probes
  - `"hidden_matrices"`: pairwise hidden activation correlations (2F × 2F)

- Ratio tests (behavioral-style generalization)
  - `"output_ratio_tests"`, `"hidden_ratio_tests"`: evaluate predefined 0:6..6:0 sets from `ratiotrials.csv`
    - Saved structure: `dict[ratio][set_name]` with arrays of correlations vs modular/lattice exemplars

- Activation tests (category strength)
  - `"output_activation_exemplar_tests"`: per-epoch dict containing
    - `mod_avg`, `lat_avg`: category strength ((within-cat) ? (across-cat))
    - `mod_by_feature`, `lat_by_feature`: per-feature activation summaries (length 22)
  - `"output_activation_onehot_tests"`: same computation using one-hot inputs


## 5) Batch training (many models) and saving

See `lab_tests\program.py` for a typical pattern:

```
from torch import nn
import numpy as np

for i in range(1, 51):
    model = StandardModel(NUM_FEATURES, HIDDEN_LAYER_SIZE, BATCH_SIZE, NUM_EPOCHS, 0.04, nn.BCEWithLogitsLoss())
    results = model.train_eval_test_P(dataloader, mod_ref, lat_ref, DATA_PARAMS, include_e0=INCLUDE_E0)
    np.savez(f"{DATA_DIR}/p_m{i}.npz", **results)
```

Tips:
- Keep `hidden_matrices`/`output_matrices` off for speed/size
- Use a clean `DATA_DIR` per experiment to keep runs grouped for plotting


## 6) Aggregation and plotting

Plotting lives in `lab_tests\Output\StatOutput.py` (and `PCAOutput.py`). Functions expect a directory of `.npz` files (each a run) and aggregate across runs.

- Correlation and loss over epochs with CIs
```
from Output.StatOutput import plot_stats_with_confidence_intervals

CORR_DATA_PARAMS = {
  "losses": True,
  "m_output_corrs": True, "l_output_corrs": True,
  "m_hidden_corrs": True, "l_hidden_corrs": True
}

plot_stats_with_confidence_intervals(
    lr_str="04_50m",
    data_dir="Results/Data/YourExperiment",
    data_parameters=CORR_DATA_PARAMS,
    save_dir="Results/Analysis/Plots/Correlations",
    include_e0=True
)
```

- Modular ? Lattice difference with per-epoch paired t-test (non-significant segments shown in red)
```
from Output.StatOutput import plot_difference_stats_with_confidence_intervals

plot_difference_stats_with_confidence_intervals(
    lr_str="04_50m",
    data_dir="Results/Data/YourExperiment",
    data_parameters=CORR_DATA_PARAMS,
    save_dir="Results/Analysis/Plots/Diffs",
    include_e0=True,
    alpha=0.05
)
```

- Ratio trial S-curve and 3:3 analyses
```
from Output.StatOutput import plot_s_curve, plot_33s

plot_s_curve(data_dir="Results/Data/YourExperiment", hidden=True, include_e0=True)
plot_33s(data_dir="Results/Data/YourExperiment", hidden=True, include_e0=True)
```

- Activation test plots across epochs
```
from Output.StatOutput import plot_activation_tests_with_confidence_intervals

plot_activation_tests_with_confidence_intervals(
    lr_str="04_50m",
    data_dir="Results/Data/YourExperiment",
    save_dir="Results/Analysis/Plots/Activations",
    include_e0=True
)
```

- Generalization vs hidden dimensionality (PCA k95 difference)
```
from Output.PCAOutput import plot_generalization_vs_dimensionality_diff

plot_generalization_vs_dimensionality_diff(
    data_dir="Results/Data/YourExperiment",
    epoch=15,
    hidden=True,
    include_e0=True
)
```

How aggregation works under the hood:
- `Statistics\StatsProducer.py` scans a folder of `.npz` files and computes per-epoch means/SE/95% CIs across models for enabled keys (skips `hidden_matrices` and `output_matrices` by design)
- `get_difference_stats_with_ttest(modular_data, lattice_data, alpha)` computes per-epoch paired t-tests (used in diff plots)


## 7) Troubleshooting

- Paths: ensure your `Data/...` paths exist; `program.py` uses `Data/Current` and `Data/ReferenceMatrices`
- Reference CSVs: required for correlation metrics; they must be numeric CSV matrices
- Large outputs: leave `hidden_matrices`/`output_matrices` disabled for batch runs to avoid huge files
- include_e0: if `True`, pass `include_e0=True` to plotting functions for aligned x-axes
- Ratio trials: ensure `Data/Current/ratiotrials.csv` exists and follows the expected schema


## 8) File map (key scripts)

- `lab_tests\program.py` — example orchestration for training/plotting
- `lab_tests\Model\StandardModel.py` — training loop + per-epoch eval gated by data params
- `lab_tests\Prep\DataUtils.py` — load/convert helpers and probability matrix loader
- `lab_tests\Prep\StimListExtractor.py` — parse JS stimlist to a tab training file
- `lab_tests\Output\StatOutput.py` — plotting and cross-run aggregation
- `lab_tests\Output\PCAOutput.py` — PCA-based scatter plots
- `lab_tests\Statistics\StatsProducer.py` — per-epoch aggregation + paired t-tests


## 9) Minimal quickstart

1) Convert or point to a CSV at `Data/Current/*.csv`
2) Verify reference matrices in `Data/ReferenceMatrices/*.csv`
3) Run a batch in `program.py` (or the single-run script above)
4) Plot with functions in `Output/StatOutput.py` and `Output/PCAOutput.py`

Happy training!
