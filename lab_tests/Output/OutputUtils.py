import os
import glob
import numpy as np

def get_data_stats_across_models(data_dir: str, data_key: str):
    """Aggregates data for the specified key across models and returns stats per epoch in the form of an AggregateStatsObject."""

    data = get_data_per_model(data_dir, data_key)


def get_data_per_model(data_dir: str, data_key: str):
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    data = []
    for file in npz_files:
        with np.load(file, allow_pickle=True) as npz:
            if data_key in npz:
                data.append(npz[data_key])
            else:
                print(f"Warning: {data_key} not found in {file}")
    return data

