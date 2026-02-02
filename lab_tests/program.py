from mimetypes import suffix_map
from sys import path_importer_cache
from turtle import st
import torch
from torch import kl_div, nn
import time
import os
import numpy as np
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

NUM_FEATURES = 11
HIDDEN_LAYER_SIZE = 400
NUM_EPOCHS = 60
NUM_MODELS = 50

INCLUDE_E0 = True

### -----------

data_config_dir = "configs/data"
data_config_filenames = [f"{data_config_dir}/mod-lat1_6-6_whole.yaml", 
                         f"{data_config_dir}/mod-lat2_5-5_whole.yaml",
                         f"{data_config_dir}/mod-lat1_6-6_missing-core.yaml",
                         f"{data_config_dir}/mod-lat2_5-5_missing-core.yaml",
                         f"{data_config_dir}/stimList_gencat_hydra.yaml"]

for d_cfg_fn in data_config_filenames[4:]:

    d_cfg = get_config(d_cfg_fn)

    print("/ ---- configuration ---- \\")
    for k, v in d_cfg.items():
        print(f"| {k}: {v}")
    print("\\ ----------------------- /")   

    assert not (d_cfg["num_total_trials"] / d_cfg["num_mod_trials"] == 2 and d_cfg["special_dl"] or d_cfg["num_total_trials"] / d_cfg["num_mod_trials"] != 2 and not d_cfg["special_dl"]), "special data loader must be used when num mod and num lat trials ineq"
        
    ### config-dependent subglobals ---

    NUM_TRAINING_TRIALS = d_cfg["num_total_trials"]

    DATA_FILENAME = f"Data/Current/{d_cfg['training_name']}.csv"
    MODULAR_P_M_FILENAME = "Data/ReferenceMatrices/cooc-jaccard-mod.csv"
    LATTICE_P_M_FILENAME = "Data/ReferenceMatrices/cooc-jaccard-lat.csv" if not d_cfg["alt"] else "Data/ReferenceMatrices/cooc-jaccard-lat-alt.csv"
    DATA_DIR = f"Results/Data/Focused_04/{d_cfg['training_name']}"
    ANALYSIS_DIR = f"Results/Analysis/Plots/one_h/{d_cfg['training_name']}"

    ### -----------

    for path in [DATA_DIR, ANALYSIS_DIR]:
        os.makedirs(path, exist_ok=True)
    
    csv_data = DataUtils.load_csv_data(DATA_FILENAME, NUM_FEATURES)

    if d_cfg["special_dl"]:
        training_inputs, training_outputs = DataUtils.training_csv_to_array(DATA_FILENAME, num_features=NUM_FEATURES)
        dataloader = SDL.SpecialDataLoader(training_inputs, training_outputs, d_cfg["num_mod_trials"])
    else:
        dataloader = DataPreparer.get_dataloader(csv_data)

    if d_cfg["generate_rms"] == True:
        modular_reference_matrix, lattice_reference_matrix = RM.generate_reference_matrices(csv_data.training_inputs, d_cfg["num_mod_trials"], method='jaccard')
    else:
        modular_reference_matrix, lattice_reference_matrix = DataUtils.get_probability_matrices_m_l(MODULAR_P_M_FILENAME, LATTICE_P_M_FILENAME);

    MO.save_generic_matrix(modular_reference_matrix, f"{ANALYSIS_DIR}/generated_rms", "mod.png")
    MO.save_generic_matrix(lattice_reference_matrix, f"{ANALYSIS_DIR}/generated_rms", "lat.png")

    ### -----------

    for i in range(1, NUM_MODELS + 1):
        model = StandardModel(num_features=NUM_FEATURES, hidden_layer_size=HIDDEN_LAYER_SIZE, batch_size=NUM_TRAINING_TRIALS, num_epochs=NUM_EPOCHS, learning_rate=.04, loss_fn=nn.BCEWithLogitsLoss())
        results = model.train_eval_test_P(dataloader, modular_reference_matrix, lattice_reference_matrix, DATA_PARAMS, include_e0=INCLUDE_E0, alt=d_cfg["alt"])
        np.savez(f"{DATA_DIR}/p_m{i}.npz", **results)
        print(i)

    ### -----------

    StatOutput.plot_stats_with_confidence_intervals(lr_str="4_50m", data_dir=DATA_DIR, save_dir=f"{ANALYSIS_DIR}/Correlations", data_parameters=CORR_DATA_PARAMS, include_e0=INCLUDE_E0)
    StatOutput.plot_33s(data_dir=DATA_DIR, save_dir=f"{ANALYSIS_DIR}/3-3", include_e0=INCLUDE_E0, alt=d_cfg['alt'])

    sig_epochs = StatOutput.get_significant_epochs(data_dir=DATA_DIR, data_parameters=CORR_DATA_PARAMS, degf=3)

    for sig_epoch in sig_epochs: 
        StatOutput.plot_s_curve(data_dir=DATA_DIR, save_dir=f"{ANALYSIS_DIR}/S-Curves", epoch=sig_epoch, include_e0=INCLUDE_E0, alt=d_cfg['alt'])
        PCAOutput.plot_k95_bars_epoch(data_dir=DATA_DIR, epoch=sig_epoch, save_dir=f"{ANALYSIS_DIR}/PCA/k95_bars", num_features=NUM_FEATURES, include_e0=INCLUDE_E0)

    MO.save_all_epoch_matrices(DATA_DIR, f"{ANALYSIS_DIR}/Matrices", NUM_EPOCHS, INCLUDE_E0)