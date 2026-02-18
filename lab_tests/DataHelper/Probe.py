import os
import json
import pandas as pd
import torch
from configs.utils import get_config

from DataHelper.utils import (
    csv_ratio_data_to_parquet,
    generate_onehot_parquet,
    generate_exemplar_parquet,
)

def build_probe(cfg_filename, probe_folder: str = "Data/Probes"):
    """
    Always rebuilds probe from sources to avoid stale data.

    Outputs:
        Data/Probes/<cfg_stem>.parquet
        Data/Probes/<cfg_stem>.index.json

    Returns:
        probe_tensor (torch.FloatTensor)
        metadata_df (pandas DataFrame)
        index (dict)
    """
    cfg = get_config(cfg_filename)
    cfg_stem = _stem(cfg_filename)

    sources_folder = os.path.join(probe_folder, "Sources")
    _ensure_dir(probe_folder)
    _ensure_dir(sources_folder)

    exemplar_src = cfg.get("exemplar")
    ratio_src = cfg.get("ratio")
    onehot_src = cfg.get("onehot")
    num_features = int(cfg.get("num_features", 11))

    dfs = []

    if exemplar_src:
        exemplar_csv = os.path.join(probe_folder, "Exemplar", f"{exemplar_src}.csv")
        exemplar_parquet = os.path.join(sources_folder, f"{exemplar_src}.parquet")
        generate_exemplar_parquet(exemplar_csv, exemplar_parquet)
        dfs.append(_normalize(_read_parquet(exemplar_parquet), "exemplar"))

    if ratio_src:
        ratio_csv = os.path.join(probe_folder, "Ratio", f"{ratio_src}.csv")
        ratio_parquet = os.path.join(sources_folder, f"{ratio_src}.parquet")
        csv_ratio_data_to_parquet(ratio_csv, ratio_parquet, num_features)
        dfs.append(_normalize(_read_parquet(ratio_parquet), "ratio"))

    if onehot_src:
        onehot_parquet = os.path.join(sources_folder, f"{onehot_src}.parquet")
        generate_onehot_parquet(onehot_parquet, num_features)
        dfs.append(_normalize(_read_parquet(onehot_parquet), "onehot"))


    if not dfs:
        raise ValueError("Config must specify at least one of: exemplar, ratio, onehot")

    probe_df = pd.concat(dfs, ignore_index=True)

    probe_parquet_path = os.path.join(probe_folder, f"{cfg_stem}.parquet")
    index_json_path = os.path.join(probe_folder, f"{cfg_stem}.index.json")

    probe_df.to_parquet(probe_parquet_path, index=False)

    probe_tensor = torch.tensor(probe_df["tensor"].tolist(), dtype=torch.float32)

    metadata = probe_df.drop(columns=["tensor"])
    index = _build_index(metadata)

    _write_json(index, index_json_path)

    return probe_tensor, metadata, index

def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _write_json(obj: dict, path: str):
    with open(path, "w") as f:
        json.dump(obj, f)

def _read_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "tensor" not in df.columns:
        raise ValueError(f"Missing 'tensor' column in {path}")
    return df

def _normalize(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    meta_cols = [c for c in df.columns if c != "tensor"]
    for c in meta_cols:
        df[c] = df[c].astype(str)
    df["source"] = str(source_name)
    return df[["source"] + meta_cols + ["tensor"]]

def _build_index(metadata_df: pd.DataFrame) -> dict:
    """
    Keys are always of the form: "col=value"
    Example:
        "label=even"
        "ratio=3:3"

    Values are lists of row indices.

    Multi-condition queries are done via set intersection:
        idx = set(index["label=even"]) & set(index["sets=mod-core"])
    """
    cols = list(metadata_df.columns)
    index = {}

    rows = metadata_df.astype(str).values.tolist()
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            key = f"{col}={row[j]}"
            index.setdefault(key, []).append(i)

    return index

