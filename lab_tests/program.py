from mimetypes import suffix_map
from sys import path_importer_cache
import torch
from torch import kl_div, nn
import time
import os
import numpy as np
import argparse
import itertools
from Model.NeuralNetwork import NeuralNetwork
from Prep import DataUtils, DataPreparer
import Output
from Model.StandardModel import StandardModel
from scipy.stats import pearsonr, zscore
import Output.StatOutput as StatOutput
import Tests.RatioExemplar as RE
import Output.PCAOutput as PCAOutput
import Output.MatrixOutput as MO
import Eval.RMatrix as RM
import Model.Parameters as PAR
from configs.utils import get_config
import Prep.SpecialDataLoader as SDL

### globals ---

DATA_PARAMS = PAR.all_parameters
CORR_DATA_PARAMS = PAR.correlation_parameters

DATA_CONFIG_DIR = "configs/data"
MODEL_CONFIG_DIR = "configs/model"

### -----------

parser = argparse.ArgumentParser()
parser.add_argument("--data-cfg", required=True)
parser.add_argument("--model-cfg", required=True)
parser.add_argument("--task-id", type=int, default=None)
parser.add_argument("--print-grid-size", action="store_true")
args = parser.parse_args()

d_cfg = get_config(args.data_cfg)
m_cfg = get_config(args.model_cfg)

print("/ ---- configuration ---- \\")
for k, v in d_cfg.items():
    print(f"| {k}: {v}")
for k, v in m_cfg.items():
    print(f"| {k}: {v}")
print("\\ ----------------------- /")   

assert not (d_cfg["num_total_trials"] / d_cfg["num_mod_trials"] == 2 and d_cfg["special_dl"] or d_cfg["num_total_trials"] / d_cfg["num_mod_trials"] != 2 and not d_cfg["special_dl"]), "special data loader must be used when num mod and num lat trials ineq"
        
### config-dependent subglobals ---

ALT = d_cfg["alt"]
TRAINING_NAME = d_cfg["training_name"]
SPECIAL_DL = d_cfg["special_dl"]
NUM_MOD_TRIALS = d_cfg["num_mod_trials"]
NUM_TOTAL_TRIALS = d_cfg["num_total_trials"]
GENRATE_RMS = d_cfg["generate_rms"]

NUM_FEATURES = m_cfg["num_features"]
HIDDEN_LAYER_RANGE = np.arange(m_cfg["hidden_layer_range"]["start"], m_cfg["hidden_layer_range"]["end"] + m_cfg["hidden_layer_range"]["step"], m_cfg["hidden_layer_range"]["step"])
LEARNING_RATE_RANGE = np.linspace(m_cfg["learning_rate"]["start"], m_cfg["learning_rate"]["end"], m_cfg["learning_rate"]["num"])
NUM_EPOCHS = m_cfg["num_epochs"]
NUM_MODELS = m_cfg["num_models"]
INCLUDE_E0 = m_cfg["include_e0"]

DATA_FILENAME = f"Data/Current/{TRAINING_NAME}.csv"
MODULAR_P_M_FILENAME = "Data/ReferenceMatrices/cooc-jaccard-mod.csv"
LATTICE_P_M_FILENAME = "Data/ReferenceMatrices/cooc-jaccard-lat.csv" if not ALT else "Data/ReferenceMatrices/cooc-jaccard-lat-alt.csv"

### -----------

grid = list(itertools.product(HIDDEN_LAYER_RANGE.tolist(), LEARNING_RATE_RANGE.tolist()))

if args.print_grid_size:
    print(len(grid))
    raise SystemExit(0)

if args.task_id is not None:
    if args.task_id < 0 or args.task_id >= len(grid):
        raise ValueError(f"task-id {args.task_id} out of range 0..{len(grid)-1}")
    grid = [grid[args.task_id]]


for HLS, LR in grid:

    lr_str = f"{LR}".replace(".", "p")

    DATA_DIR = f"Results/Data/Focused_04/{TRAINING_NAME}_h{HLS}_lr{lr_str}"
    ANALYSIS_DIR = f"Results/Analysis/Plots/one_h/{TRAINING_NAME}_h{HLS}_lr{lr_str}"

    print(f"Saving data to:     {DATA_DIR}")
    print(f"Saving analysis to: {ANALYSIS_DIR}")

    for path in [DATA_DIR, ANALYSIS_DIR]:
        os.makedirs(path, exist_ok=True)

    csv_data = DataUtils.load_csv_data(DATA_FILENAME, NUM_FEATURES)

    if d_cfg["special_dl"]:
        training_inputs, training_outputs = DataUtils.training_csv_to_array(DATA_FILENAME, num_features=NUM_FEATURES)
        dataloader = SDL.SpecialDataLoader(training_inputs, training_outputs, NUM_MOD_TRIALS)
    else:
        dataloader = DataPreparer.get_dataloader(csv_data)

    if d_cfg["generate_rms"] == True:
        modular_reference_matrix, lattice_reference_matrix = RM.generate_reference_matrices(csv_data.training_inputs, NUM_MOD_TRIALS, method='jaccard')
    else:
        modular_reference_matrix, lattice_reference_matrix = DataUtils.get_probability_matrices_m_l(MODULAR_P_M_FILENAME, LATTICE_P_M_FILENAME);

    MO.save_generic_matrix(modular_reference_matrix, f"{ANALYSIS_DIR}/generated_rms", "mod.png")
    MO.save_generic_matrix(lattice_reference_matrix, f"{ANALYSIS_DIR}/generated_rms", "lat.png")

    ### -----------

    for i in range(1, NUM_MODELS + 1):
        model = StandardModel(num_features=NUM_FEATURES, hidden_layer_size=HLS, batch_size=NUM_TOTAL_TRIALS, num_epochs=NUM_EPOCHS, learning_rate=LR, loss_fn=nn.BCEWithLogitsLoss())
        results = model.train_eval_test_P(dataloader, modular_reference_matrix, lattice_reference_matrix, DATA_PARAMS, include_e0=INCLUDE_E0, alt=ALT)
        np.savez(f"{DATA_DIR}/p_m{i}.npz", **results)
        print(f"h{HLS} m{i}")

    ### -----------

    StatOutput.plot_stats_with_confidence_intervals(lr_str="4_50m", data_dir=DATA_DIR, save_dir=f"{ANALYSIS_DIR}/Correlations", data_parameters=CORR_DATA_PARAMS, include_e0=INCLUDE_E0)
    StatOutput.plot_33s(data_dir=DATA_DIR, save_dir=f"{ANALYSIS_DIR}/3-3", include_e0=INCLUDE_E0, alt=ALT)

    sig_epochs = StatOutput.get_significant_epochs(data_dir=DATA_DIR, data_parameters=CORR_DATA_PARAMS, degf=3)

    for sig_epoch in sig_epochs: 
        StatOutput.plot_s_curve(data_dir=DATA_DIR, save_dir=f"{ANALYSIS_DIR}/S-Curves", epoch=sig_epoch, include_e0=INCLUDE_E0, alt=ALT)
        PCAOutput.plot_k95_bars_epoch(data_dir=DATA_DIR, epoch=sig_epoch, save_dir=f"{ANALYSIS_DIR}/PCA/k95_bars", num_features=NUM_FEATURES, include_e0=INCLUDE_E0)

    MO.save_all_epoch_matrices(DATA_DIR, f"{ANALYSIS_DIR}/Matrices", NUM_EPOCHS, INCLUDE_E0)