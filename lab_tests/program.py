from sys import path_importer_cache
from turtle import st
import torch
from torch import nn
import time
import numpy as np
from Model.NeuralNetwork import NeuralNetwork
from Prep import DataUtils, DataPreparer
import Output
from Model.StandardModel import StandardModel
import Model.MultipleModels
from scipy.stats import pearsonr, zscore
import Output.StatOutput as StatOutput
import Output.CorrelationAnalyzer as CorrelationAnalyzer
import Tests.RatioExemplar as RE
import Output.RatioExemplarOutput as REO

### globals ---

DATA_FILENAME = "Data/Current/stimList_gencat_hydra_forAbe.csv"
MODULAR_P_M_FILENAME = "Data/ReferenceMatrices/cooc-jaccard-mod.csv"
LATTICE_P_M_FILENAME = "Data/ReferenceMatrices/cooc-jaccard-lat.csv"

NUM_FEATURES = 11
HIDDEN_LAYER_SIZE = 400
NUM_EPOCHS = 60
NUM_TRAINING_TRIALS = 96

INCLUDE_E0 = True

DATA_PARAMS = { "losses": True,
                "m_output_corrs": True,
                "l_output_corrs": True,
                "m_hidden_corrs": True,
                "l_hidden_corrs": True,
                "output_matrices": True,
                "hidden_matrices": True,
                "output_ratio_tests": True,
                "hidden_ratio_tests": True,
                "output_activation_exemplar_tests": True,
                "output_activation_onehot_tests": True}

CORR_DATA_PARAMS = { "losses": True,
                "m_output_corrs": True,
                "l_output_corrs": True,
                "m_hidden_corrs": True,
                "l_hidden_corrs": True,
                "output_matrices": False,
                "hidden_matrices": False,
                "output_ratio_tests": False,
                "hidden_ratio_tests": False,
                "output_activation_exemplar_tests": False,
                "hidden_activation_exemplar_tests": False,
                "output_activation_onehot_tests": False,
                "hidden_activation_onehot_tests": False }

DATA_DIR = "Results/Data/Focused_04/by_sets/one_h"

### -----------

modular_reference_matrix, lattice_reference_matrix = DataUtils.get_probability_matrices_m_l(MODULAR_P_M_FILENAME, LATTICE_P_M_FILENAME);
dataloader = DataPreparer.get_dataloader(DataUtils.load_csv_data(DATA_FILENAME, NUM_FEATURES))


"""
for i in range(1, 51):
    model = StandardModel(num_features=NUM_FEATURES, hidden_layer_size=HIDDEN_LAYER_SIZE, batch_size=NUM_TRAINING_TRIALS, num_epochs=NUM_EPOCHS, learning_rate=.04, loss_fn=nn.BCEWithLogitsLoss())
    results = model.train_eval_test_P(dataloader, modular_reference_matrix, lattice_reference_matrix, DATA_PARAMS, include_e0=INCLUDE_E0)
    np.savez(f"{DATA_DIR}/p_m{i}.npz", **results)
    print(i)

StatOutput.plot_stats_with_confidence_intervals(lr_str="4_50m", data_dir=DATA_DIR, save_dir="Results/Analysis/Plots/one_h/Correlations", data_parameters=CORR_DATA_PARAMS, include_e0=INCLUDE_E0)
StatOutput.plot_33s(data_dir=DATA_DIR, save_dir="Results/Analysis/Plots/one_h/3-3", include_e0=INCLUDE_E0)

StatOutput.plot_activation_tests_with_confidence_intervals(lr_str="4_50m", data_dir=DATA_DIR, save_dir="Results/Analysis/Plots/one_h/Activations", include_e0=INCLUDE_E0)

StatOutput.plot_difference_stats_with_confidence_intervals(lr_str="4_50m", data_dir=DATA_DIR, save_dir="Results/Analysis/Plots/one_h/Diffs", data_parameters=CORR_DATA_PARAMS, include_e0=INCLUDE_E0)


StatOutput.plot_s_curve(data_dir=DATA_DIR, save_dir="Results/Analysis/Plots/one_h/S-Curves", epoch=15, include_e0=INCLUDE_E0)

StatOutput.plot_33s(data_dir=DATA_DIR, hidden=False, save_dir="Results/Analysis/Plots/one_h/3-3", include_e0=INCLUDE_E0)

for i in range(0, 61):
    StatOutput.plot_scatter_models(data_dir=DATA_DIR, epoch=i, save_dir="Results/Analysis/Plots/one_h/scatter_models", include_e0=INCLUDE_E0)

StatOutput.plot_scatter_models_by_set(data_dir=DATA_DIR, epoch=15, save_dir="Results/Analysis/Plots/one_h/scatter_models", include_e0=INCLUDE_E0)
StatOutput.plot_scatter_activation_tests(data_dir=DATA_DIR, epoch=15, save_dir="Results/Analysis/Plots/one_h/scatter_models", include_e0=INCLUDE_E0)
StatOutput.plot_activation_vs_correlation(data_dir=DATA_DIR, category='mod', epoch=15, save_dir="Results/Analysis/Plots/one_h/category_strength-structure_learning", include_e0=INCLUDE_E0)
StatOutput.plot_activation_vs_correlation(data_dir=DATA_DIR, category='lat', epoch=15, save_dir="Results/Analysis/Plots/one_h/category_strength-structure_learning", include_e0=INCLUDE_E0)
"""

StatOutput.plot_feature_activation_over_epochs(data_dir=DATA_DIR, category='mod', save_dir="Results/Analysis/Plots/one_h/feature_activations", include_e0=INCLUDE_E0)
StatOutput.plot_feature_activation_over_epochs(data_dir=DATA_DIR, category='lat', save_dir="Results/Analysis/Plots/one_h/feature_activations", include_e0=INCLUDE_E0)