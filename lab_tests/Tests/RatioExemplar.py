import Eval.PearsonEval as PEval
import Prep.DataUtils as DataUtils
from scipy.stats import pearsonr
import torch
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from Model.NeuralNetwork import NeuralNetwork

modular_exemplars = [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

lattice_exemplars = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]]

ratio_trials = DataUtils.generate_ratio_trials()

def test_ratios(model, hidden: bool=False, ratio: str="all"):
    ratios = ratio_trials.keys() if ratio == "all" else [ratio]
    test_data = {}
    exemplar_results = generate_exemplar_results(model, hidden)
    for ratio in ratios:
        trials = ratio_trials[ratio]
        trial_results = generate_results(model, trials, hidden)
        ratio_data = {}
        ratio_data["mod"] = []
        ratio_data["lat"] = []
        for trial_result in trial_results:
            ratio_data["mod"].append([pearsonr(trial_result, exemplar_results["mod"][i])[0] for i in range(len(modular_exemplars))])
            ratio_data["lat"].append([pearsonr(trial_result, exemplar_results["lat"][i])[0] for i in range(len(lattice_exemplars))])
        test_data[ratio] = ratio_data
    return test_data

def generate_exemplar_results(model, hidden: bool=False):
    results = {}
    results["mod"] = generate_results(model, modular_exemplars, hidden)
    results["lat"] = generate_results(model, lattice_exemplars, hidden)
    return results

def generate_results(model, inputs, hidden: bool=False):
    return model.get_hidden_activations(torch.tensor(inputs, dtype=torch.float32)).numpy() if hidden else torch.sigmoid(model(torch.tensor(inputs, dtype=torch.float32))).detach().numpy()







