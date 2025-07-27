from .DataPreparer import DataPreparer
from .FormatConverters import TabDelimitedConverter, PythonListConverter
import numpy as np
import csv
import os

def load_tab_delimited_data(filename: str, num_features: int = 11) -> DataPreparer:
    return DataPreparer.from_tab_delimited(filename, num_features)


def load_python_list_data(filename: str, num_features: int = 11) -> DataPreparer:
    return DataPreparer.from_python_lists(filename, num_features)


def load_csv_data(filename: str, num_features: int = 11) -> DataPreparer:
    return DataPreparer.from_csv(filename, num_features)

def convert_tab_delimited_to_csv(input_file: str, output_file: str, num_features: int=11):
    converter = TabDelimitedConverter(num_features)
    inputs, outputs = converter.load_training_data(input_file)
    converter.save_to_csv(inputs, outputs, output_file)

def convert_python_lists_to_csv(input_file: str, output_file: str, num_features: int=11):
    converter = PythonListConverter(num_features)
    train_inputs, train_outputs, test_inputs = converter.load_from_python_file(input_file)
    converter.save_to_csv(train_inputs, train_outputs, output_file, test_inputs)

def get_probability_matrices_m_l(m_filename: str, l_filename: str):
    return np.loadtxt(m_filename, delimiter=','), np.loadtxt(l_filename, delimiter=',')

def generate_ratio_trials(csv_filename: str = None):
    if csv_filename is None:
        csv_filename = "Data/Current/ratiotrials.csv"
    
    test_sets = {}
    
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ratio = row['ratio'].strip().strip("'")
            
            features = []
            for col in ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']:
                value = int(row[col].strip())
                if value != 100 and value != 200:
                    features.append(value)
            
            binary_array = [0] * 22
            
            for feature in features:
                if 101 <= feature <= 111:
                    binary_array[feature - 101] = 1
                elif 201 <= feature <= 211:
                    binary_array[feature - 201 + 11] = 1
            
            if ratio not in test_sets:
                test_sets[ratio] = []
            test_sets[ratio].append(binary_array)
    
    return test_sets
