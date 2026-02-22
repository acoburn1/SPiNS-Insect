from mimetypes import suffix_map
from sys import path_importer_cache
from tkinter import HIDDEN
from turtle import st
import torch
from torch import kl_div, nn
import time
import os
import argparse
import numpy as np
from Model.NeuralNetwork import NeuralNetwork
from DataHelper import utils as DataUtils
import Output
from Model.StandardModel import StandardModel
from scipy.stats import pearsonr, zscore
import Output.StatOutput as StatOutput
import Tests.RatioExemplar as RE
import Output.PCAOutput as PCAOutput
import Output.MatrixOutput as MO
import Eval.RMatrix as RM
import Model.Parameters as PAR
from configs.utils import resolve_cfgs, print_cfgs
import DataHelper.SpecialDataLoader as SDL

parser = argparse.ArgumentParser()
parser.add_argument("--data-config", "-d", default=None)
parser.add_argument("--model-config", "-m", default=None)
parser.add_argument("--output-config", "-o", default=None)
parser.add_argument("--probe-config", "-p", default=None)
args, _unknown = parser.parse_known_args()

d_cfg, m_cfg, o_cfg, p_cfg = resolve_cfgs(args, idv=True)

assert not (d_cfg["num_total_trials"] / d_cfg["num_mod_trials"] == 2 and d_cfg["special_dl"] or d_cfg["num_total_trials"] / d_cfg["num_mod_trials"] != 2 and not d_cfg["special_dl"]), "special data loader must be used when num mod and num lat trials ineq"

print_cfgs(resolve_cfgs(args))
        
### config-dependent subglobals ---

ALT = d_cfg["alt"]
TRAINING_NAME = d_cfg["training_name"]
SPECIAL_DL = d_cfg["special_dl"]
NUM_MOD_TRIALS = d_cfg["num_mod_trials"]
NUM_TOTAL_TRIALS = d_cfg["num_total_trials"]
GENRATE_RMS = d_cfg["generate_rms"]

NUM_FEATURES = m_cfg["num_features"]
HIDDEN_LAYER_RANGE = np.arange(m_cfg["hidden_layer_range"]["start"], m_cfg["hidden_layer_range"]["end"], m_cfg["hidden_layer_range"]["step"])
LEARNING_RATE_RANGE = np.linspace(m_cfg["learning_rate"]["start"], m_cfg["learning_rate"]["end"], m_cfg["learning_rate"]["num"])
NUM_EPOCHS = m_cfg["num_epochs"]
NUM_MODELS = m_cfg["num_models"]
INCLUDE_E0 = m_cfg["include_e0"]

TRAINING_DATA_FILENAME = f"Data/Current/{TRAINING_NAME}.csv"
MODULAR_P_M_FILENAME = "Data/ReferenceMatrices/cooc-jaccard-mod.csv"
LATTICE_P_M_FILENAME = "Data/ReferenceMatrices/cooc-jaccard-lat.csv" if not ALT else "Data/ReferenceMatrices/cooc-jaccard-lat-alt.csv"
DATA_DIR = f"Results/Data/Focused_04/{TRAINING_NAME}"
ANALYSIS_DIR = f"Results/Analysis/Plots/one_h/{TRAINING_NAME}"

### -----------

