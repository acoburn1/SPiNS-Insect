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
from scipy.stats import pearsonr
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
                "m_output_corrs": False,
                "l_output_corrs": False,
                "m_hidden_corrs": True,
                "l_hidden_corrs": True,
                "output_matrices": False,
                "hidden_matrices": False,
                "output_tests": False,
                "hidden_tests": True }

### -----------

modular_reference_matrix, lattice_reference_matrix = DataUtils.get_probability_matrices_m_l(MODULAR_P_M_FILENAME, LATTICE_P_M_FILENAME);
dataloader = DataPreparer.get_dataloader(DataUtils.load_csv_data(DATA_FILENAME, NUM_FEATURES))

for lr in range(10, 65, 5):
    data_dir = f"Results/Data/Varied/0{lr}"
    StatOutput.plot_stats_with_confidence_intervals(lr, data_dir, DATA_PARAMS, include_e0=INCLUDE_E0)


"""
for i in range(1, 101):
    model = StandardModel(num_features=NUM_FEATURES, hidden_layer_size=HIDDEN_LAYER_SIZE, batch_size=NUM_TRAINING_TRIALS, num_epochs=NUM_EPOCHS, learning_rate=.04, loss_fn=nn.BCEWithLogitsLoss())
    results = model.train_eval_test_P(dataloader, modular_reference_matrix, lattice_reference_matrix, DATA_PARAMS, include_e0=INCLUDE_E0)
    np.savez(f"Results/Data/Focused_04/p_m{i}.npz", **results)
    print(i)


for e in range(0,61):
    StatOutput.plot_s_curve("Results/Data/Focused_04", hidden=True, epoch=e, include_e0=INCLUDE_E0)
"""