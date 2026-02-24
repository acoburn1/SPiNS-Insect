import Eval.PearsonEval as PEval
from scipy.stats import pearsonr
import torch
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from Model.NeuralNetwork import NeuralNetwork

def test_ratios(model, hidden: bool=False, ratio: str="all", alt: bool=False):
    if alt:
        ratio_trials = alt_ratio_trials
    else:
        ratio_trials = reg_ratio_trials
    ratios = ratio_trials.keys() if ratio == "all" else [ratio]
    test_data = {}
    exemplar_results = generate_exemplar_results(model, hidden, alt=alt)
    for ratio in ratios:
        trial_sets = ratio_trials[ratio]
        ratio_data = {}
        for set_name, trials in trial_sets.items():
            set_data = {}
            trial_results = generate_results(model, trials, hidden)
            set_data["mod"] = []
            set_data["lat"] = []
            for trial_result in trial_results:
                set_data["mod"].append([pearsonr(trial_result, exemplar_results["mod"][i])[0] for i in range(len(modular_exemplars))])
                set_data["lat"].append([pearsonr(trial_result, exemplar_results["lat"][i])[0] for i in range(len(lattice_exemplars))])
            ratio_data[set_name] = set_data
        test_data[ratio] = ratio_data
    return test_data

def test_activations(model, num_features, one_hot: bool=False, alt: bool=False):
    results = generate_results(model, np.eye(2*num_features)) if one_hot else generate_exemplar_results(model, alt=alt)
    mod_results = results[:num_features] if one_hot else results["mod"]
    lat_results = results[num_features:] if one_hot else results["lat"]
    mod_avg = np.mean([np.mean(mod_result[:num_features]) - np.mean(mod_result[num_features:]) for mod_result in mod_results])
    lat_avg = np.mean([np.mean(lat_result[num_features:]) - np.mean(lat_result[:num_features]) for lat_result in lat_results])
    return {"mod_avg": mod_avg, "lat_avg": lat_avg, "mod_by_feature": np.mean(mod_results, axis=0), "lat_by_feature": np.mean(lat_results, axis=0)}

def generate_exemplar_results(model, hidden: bool=False, alt: bool=False):
    results = {}
    results["mod"] = generate_results(model, modular_exemplars, hidden) if not alt else generate_results(model, alt_mod_exemplars, hidden)
    results["lat"] = generate_results(model, lattice_exemplars, hidden) if not alt else generate_results(model, alt_lattice_exemplars, hidden)
    return results

def generate_results(model, inputs, hidden: bool=False):
    return model.get_hidden_activations(torch.tensor(inputs, dtype=torch.float32)).numpy() if hidden else torch.sigmoid(model(torch.tensor(inputs, dtype=torch.float32))).detach().numpy()