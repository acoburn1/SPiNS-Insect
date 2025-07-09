from .DataPreparer import DataPreparer
import numpy as np


def load_tab_delimited_data(filename: str, num_features: int = 22) -> DataPreparer:
    return DataPreparer.from_tab_delimited(filename, num_features)


def load_python_list_data(filename: str, num_features: int = 22) -> DataPreparer:
    return DataPreparer.from_python_lists(filename, num_features)


def load_csv_data(filename: str, num_features: int = 22) -> DataPreparer:
    return DataPreparer.from_csv(filename, num_features)

def get_probability_matrices_m_l(m_filename: str, l_filename: str):
    return np.loadtxt(m_filename, delimiter=','), np.loadtxt(l_filename, delimiter=',')