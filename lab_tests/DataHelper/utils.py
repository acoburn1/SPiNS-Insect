from typing import Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import csv
import os

def get_dataloader(training_inputs, training_outputs) -> DataLoader:
    tensor_in = torch.tensor(training_inputs, dtype=torch.float32)
    tensor_out = torch.tensor(training_outputs, dtype=torch.float32)
    dataset = TensorDataset(tensor_in, tensor_out)
    return DataLoader(dataset, batch_size=len(training_inputs), shuffle=True)

def csv_training_data_to_numpy(filename: str, num_features: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    """ Reads training data from a CSV file and converts it to binary numpy arrays """
    train_inputs, train_outputs = [], []
        
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            inp = [int(row[f'input_{i}']) for i in range(2*num_features)]
            out = [int(row[f'output_{i}']) for i in range(2*num_features)]
                
            if row['type'] == 'train':
                train_inputs.append(inp)
                train_outputs.append(out)
        
    return train_inputs, train_outputs

def csv_ratio_data_to_parquet(filename: str, output_filename: str, num_features: int=11):
    """ Converts ratio data from a ratio CSV file to a Parquet file """
    df = pd.read_csv(filename, engine="python", quotechar="'", skipinitialspace=True)

    feature_cols = [c for c in df.columns if c.lower().startswith("f")]
    if not feature_cols:
        raise ValueError(f"No feature columns found in {filename}. Expected columns like f1,f2,...")

    def fid_to_index(fid: int) -> int | None:
        if fid == 100:
            return None
        if 101 <= fid <= 100 + num_features:
            return fid - 101
        if 201 <= fid <= 200 + num_features:
            return num_features + (fid - 201)
        return None

    def row_to_tensor(row) -> list[int]:
        v = [0] * (2 * num_features)
        for c in feature_cols:
            fid = int(row[c])
            idx = fid_to_index(fid)
            if idx is not None:
                v[idx] = 1
        return v

    out = df[["ratio", "label", "sets"]].copy()
    out["tensor"] = df.apply(row_to_tensor, axis=1)

    out.to_parquet(output_filename, index=False)
    return out

def generate_onehot_parquet(output_filename: str, num_features: int = 11):
    """ Generates a Parquet file containing one-hot encoded feature vectors. Unnecessary but support existing structure and labeling """
    
    rows = []
    d = 2 * num_features
    for i in range(d):
        v = [0] * d
        v[i] = 1
        rows.append({"tensor": v})

    out = pd.DataFrame(rows, columns=["tensor"])
    out.to_parquet(output_filename, index=False)
    return out

def generate_exemplar_parquet(filename: str, output_filename: str):
    """ 
    Generates a Parquet file containing exemplar vectors
    Convention: label1, label2, ..., labelN, 'tensor'
    """
    raw = pd.read_csv(filename, header=None, dtype=str)

    header = raw.iloc[0].tolist()

    try:
        tensor_col_idx = header.index("tensor")
    except ValueError:
        raise ValueError("Exemplar file: first row must contain a 'tensor' marker column.")

    label_names = header[:tensor_col_idx]

    data = raw.iloc[1:].reset_index(drop=True)

    out = pd.DataFrame()
    for i, name in enumerate(label_names):
        out[name] = data.iloc[:, i].astype(str)

    tensor_vals = data.iloc[:, tensor_col_idx + 1 :].astype(int).values.tolist()
    out["tensor"] = tensor_vals

    out.to_parquet(output_filename, index=False)
    return out