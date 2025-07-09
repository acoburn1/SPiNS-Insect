from sys import path_importer_cache
import torch
from torch import nn
import time
import numpy as np
from NeuralNetwork import NeuralNetwork
from DataPrep import DataPreparer, PairWalkTrainingDataGenerator, ProbabilityMatrix
import Output
from StandardModel import StandardModel
import MultipleModels

### globals

TRAINING_FILENAME = "Data/trainPats_22dim_2cats_good.txt"
TESTING_FILENAME = "Data/testPats_22dim_2cats_good.txt"
MODULAR_P_M_FILENAME = "Data/cooc-jaccard-mod.csv"
LATTICE_P_M_FILENAME = "Data/cooc-jaccard-lat.csv"

NUM_FEATURES = 11
HIDDEN_LAYER_SIZE = 400
NUM_EPOCHS = 20
LEARNING_RATE=0.024
NUM_TRAINING_TRIALS = 96

data_preparer = DataPreparer(TRAINING_FILENAME, TESTING_FILENAME, MODULAR_P_M_FILENAME, LATTICE_P_M_FILENAME, NUM_FEATURES)
data_preparer.load_data()
modular_reference_matrix, lattice_reference_matrix = data_preparer.get_probability_matrices_m_l();


model = StandardModel(num_features=NUM_FEATURES, hidden_layer_size=HIDDEN_LAYER_SIZE, batch_size=NUM_TRAINING_TRIALS, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, loss_fn=nn.BCEWithLogitsLoss())
dataloader = data_preparer.get_dataloader(NUM_TRAINING_TRIALS)

losses, m_output_corrs, l_output_corrs, m_hidden_corrs, l_hidden_corrs = model.train_eval_P(dataloader, modular_reference_matrix, lattice_reference_matrix, save_models=True, subfolder_name="p_024lr_96bs_22i_400h", print_data=False)

torch.savez

np.savez("Results/Data/p_024lr_96bs_20e_22i_400h.npz", losses=losses, m_output_corrs=m_output_corrs, l_output_corrs=l_output_corrs, m_hidden_corrs=m_hidden_corrs, l_hidden_corrs=l_hidden_corrs)


