from sys import path_importer_cache
import torch
import time
import numpy as np
from torch import nn
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


num_epochs = 20;
#"""
batch_sizes = [4 * (2 ** i) for i in range(6)]      # 4 to 128 in powers of two
learning_rates = [i * 0.004 for i in range(1, 26)]   # .004 to .1 in increments of .004
total_models = len(batch_sizes) * len(learning_rates)
model_count = 0

all_losses = np.zeros((len(batch_sizes), len(learning_rates), num_epochs))
all_m_avgs = np.zeros((len(batch_sizes), len(learning_rates), num_epochs))
all_l_avgs = np.zeros((len(batch_sizes), len(learning_rates), num_epochs))
all_mpms = np.zeros((len(batch_sizes), len(learning_rates)), dtype=object)
all_lpms = np.zeros((len(batch_sizes), len(learning_rates)), dtype=object)
all_gpms = np.zeros((len(batch_sizes), len(learning_rates)), dtype=object)

print(f"Training {total_models} models with batch sizes ranging from {batch_sizes[0]} to {batch_sizes[-1]} and learning rates ranging from {learning_rates[0]} to {learning_rates[-1]}:")

for b_idx, batch_size in enumerate(batch_sizes):
    for lr_idx, lr in enumerate(learning_rates):

        model_count += 1

        dataloader = data_preparer.get_dataloader(batch_size)
        standard_model = StandardModel(num_features=NUM_FEATURES, hidden_layer_size=HIDDEN_LAYER_SIZE, batch_size=batch_size, num_epochs=num_epochs, learning_rate = lr, loss_fn=nn.BCEWithLogitsLoss())

        print(f"{model_count}/{total_models}: bs={batch_size}, lr={lr}")

        losses, m_avgs, l_avgs, mpms, lpms, gpms = standard_model.train(dataloader, modular_reference_matrix, lattice_reference_matrix)

        all_m_avgs[b_idx][lr_idx] = m_avgs
        all_l_avgs[b_idx][lr_idx] = l_avgs
        all_mpms[b_idx][lr_idx] = mpms
        all_lpms[b_idx][lr_idx] = lpms
        all_gpms[b_idx][lr_idx] = gpms
        all_losses[b_idx][lr_idx] = losses

np.savez("results/avgs-losses_bs_lr_20e_400h.npz", m_avgs=all_m_avgs, l_avgs=all_l_avgs, mpms=all_mpms, lpms=all_lpms, ngpms=all_gpms, losses=all_losses, batch_sizes=batch_sizes, learning_rates=learning_rates)
#"""

data = np.load("results/avgs-losses_bs_lr_20e_400h.npz", allow_pickle=True)
all_m_avgs = data['m_avgs']
all_l_avgs = data['l_avgs']
all_losses = data['losses']
all_mpms = data['mpms']
all_lpms = data['lpms']
all_gpms = data['ngpms']
batch_sizes = data['batch_sizes']
learning_rates = data['learning_rates']

m_l_similarity_ratios = all_m_avgs / all_l_avgs

best_m_avg, (b_idx_m_a, lr_idx_m_a, e_idx_m_a) = Output.get_max_value_and_indices(all_m_avgs)
best_l_avg, (b_idx_l_a, lr_idx_l_a, e_idx_l_a) = Output.get_max_value_and_indices(all_l_avgs)
best_loss, (b_idx_l, lr_idx_l, e_idx_l) = Output.get_max_value_and_indices(all_losses, minimize=True)

print(f"Best loss: {best_loss} with batch size= {batch_sizes[b_idx_l]}, learning rate= {learning_rates[lr_idx_l]} after epoch {e_idx_l+1}\n")
print(f"Best average JS similarity (modular): {best_m_avg} with batch size= {batch_sizes[b_idx_m_a]}, learning rate= {learning_rates[lr_idx_m_a]} after epoch {e_idx_m_a+1}")

normalized_m_matrix = Evaluation.matrix_row_normalize(all_mpms[b_idx_m_a][lr_idx_m_a][e_idx_m_a])
print("\nProbability matrix at the best average similarity:\n")
Output.print_matrix(normalized_m_matrix)
print("\nRow-normalized reference matrix:\n")
Output.print_matrix(Evaluation.matrix_row_normalize(modular_reference_matrix))

print(f"\nBest average JS similarity (lattice): {best_l_avg} with batch size= {batch_sizes[b_idx_l_a]}, learning rate= {learning_rates[lr_idx_l_a]} after epoch {e_idx_l_a+1}")

normalized_l_matrix = Evaluation.matrix_row_normalize(all_mpms[b_idx_l_a][lr_idx_l_a][e_idx_l_a])
print("\nProbability matrix at the best average similarity:\n")
Output.print_matrix(normalized_l_matrix)
print("\nRow-normalized reference matrix:\n")
Output.print_matrix(Evaluation.matrix_row_normalize(lattice_reference_matrix))

Output.plot_matrices(normalized_m_matrix, modular_reference_matrix, normalized_l_matrix, lattice_reference_matrix, 
                     titles=("modular, learned", "modular, reference", "lattice, learned", "lattice, reference"))

Output.plot_matrix(Evaluation.matrix_row_normalize(all_gpms[b_idx_m_a][lr_idx_m_a][e_idx_m_a]), "Best average JS similarity (modular, whole matrix)")

Output.plot_interactive_heatmap_discrete(all_m_avgs.tolist(), batch_sizes, learning_rates, num_epochs, title="Average JS Similarities (modular)")
Output.plot_interactive_heatmap_discrete(all_l_avgs.tolist(), batch_sizes, learning_rates, num_epochs, title="Average JS Similarities (lattice)")
Output.plot_interactive_heatmap_discrete(all_losses.tolist(), batch_sizes, learning_rates, num_epochs, title="Losses (Clipped)", clip_max=5.0)
Output.plot_interactive_heatmap_discrete(m_l_similarity_ratios.tolist(), batch_sizes, learning_rates, num_epochs, title="Average JS Similarity Ratios (modular / lattice))")