from sys import path_importer_cache
import torch
import time
import numpy as np
from Evaluation import generate_distributions
from NeuralNetwork import NeuralNetwork
from DataPrep import DataPreparer, PairWalkTrainingDataGenerator, ProbabilityMatrix
import Output
import Evaluation
from StandardModel import StandardModel

### globals

TRAINING_FILENAME = "data/trainPats_22dim_2cats_good.txt"
TESTING_FILENAME = "data/testPats_22dim_2cats_good.txt"
MODULAR_P_M_FILENAME = "data/cooc-jaccard-mod.csv"
LATTICE_P_M_FILENAME = "data/cooc-jaccard-lat.csv"

NUM_FEATURES = 11
HIDDEN_LAYER_SIZE = 400

data_preparer = DataPreparer(TRAINING_FILENAME, TESTING_FILENAME, MODULAR_P_M_FILENAME, LATTICE_P_M_FILENAME, NUM_FEATURES)
data_preparer.load_data()
modular_reference_matrix, lattice_reference_matrix = data_preparer.get_probability_matrices_m_l();


num_epochs = 20





