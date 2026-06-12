# SPiNS Insect

Neural-network modeling pipeline for training category-learning models, evaluating saved activations, and generating analysis plots.

See [GUIDE.md](GUIDE.md) for a complete repository guide.

## Quick start

Install the dependencies from the repository root:

```powershell
pip install -r requirements.txt
```

Run the program from `lab_tests/`. The default data, configuration, and result paths depend on this working directory.

```powershell
cd lab_tests
python program.py --all
```

`--all` runs the three primary stages in order:

1. Train model ensembles and save activations.
2. Evaluate the saved activations.
3. Generate plots from the saved analysis data.

The default model configuration runs 50 models for 120 epochs. Use a smaller model configuration for a short test run.

## Running individual stages

```powershell
python program.py --train
python program.py --evaluate
python program.py --graph
```

Evaluation requires matching training results, and graph generation requires matching evaluation results. Use the same data, model, and probe configurations across these stages.

Specialized tabular exports can be generated separately:

```powershell
python program.py --tabular
```

Add `--visual` or `-v` to display the resolved configuration and progress. Visual mode asks for confirmation before running.

## Selecting configurations

If no configuration options are provided, each configuration group uses its `default.yaml`.

```powershell
python program.py --all `
  --data-config default `
  --model-config hls20 `
  --output-config default `
  --probe-config default `
  --directory-config default
```

A bare name such as `hls20` resolves to `configs/model/hls20.yaml`. A direct YAML path may also be supplied.

Configuration groups are stored under:

```text
lab_tests/configs/
  data/         Training dataset and regime
  model/        Architecture and training hyperparameters
  output/       Evaluators and plots
  probe/        Probe composition
  directory/    Data and result locations
```

## Results

With the default directory configuration, results are written under `lab_tests/Results/`:

```text
ActivationData/   Saved activations and losses
AnalysisData/     Evaluator results
Output/           Generated plots
RunMetadata/      Stage status and run metadata
```

Saved intermediate results allow evaluation and graph generation to be rerun without retraining.
