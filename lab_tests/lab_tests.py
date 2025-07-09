from sys import path_importer_cache
import torch
from torch import nn
import time
import numpy as np
from Model.NeuralNetwork import NeuralNetwork
from Prep import DataUtils, DataPreparer
import Output
from Model.StandardModel import StandardModel
import Model.MultipleModels

### globals ---

DATA_FILENAME = "Data/Current/stimList_gencat_hydra_forAbe.py"
MODULAR_P_M_FILENAME = "Data/ReferenceMatrices/cooc-jaccard-mod.csv"
LATTICE_P_M_FILENAME = "Data/ReferenceMatrices/cooc-jaccard-lat.csv"

NUM_FEATURES = 11
HIDDEN_LAYER_SIZE = 400
NUM_EPOCHS = 20
LEARNING_RATE=0.024
NUM_TRAINING_TRIALS = 96

### -----------


dataloader = DataPreparer.get_dataloader(DataUtils.load_python_list_data(DATA_FILENAME, NUM_FEATURES))
modular_reference_matrix, lattice_reference_matrix = DataUtils.get_probability_matrices_m_l(MODULAR_P_M_FILENAME, LATTICE_P_M_FILENAME);

model = StandardModel(num_features=NUM_FEATURES, hidden_layer_size=HIDDEN_LAYER_SIZE, batch_size=NUM_TRAINING_TRIALS, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, loss_fn=nn.BCEWithLogitsLoss())

losses, m_output_corrs, l_output_corrs, m_hidden_corrs, l_hidden_corrs, output_matrices, hidden_matrices = model.train_eval_P(dataloader, modular_reference_matrix, lattice_reference_matrix, save_models=True, subfolder_name="p_024lr_96bs_22i_400h", print_data=False)

np.savez("Results/Data/p_024lr_96bs_20e_22i_400h.npz", losses=losses, m_output_corrs=m_output_corrs, l_output_corrs=l_output_corrs, m_hidden_corrs=m_hidden_corrs, l_hidden_corrs=l_hidden_corrs, output_matrices=output_matrices, hidden_matrices=hidden_matrices)



