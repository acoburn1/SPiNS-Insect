import torch
from typing import List, Tuple, Dict, Optional
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

from .BaseConverter import BaseDataConverter
from .FormatConverters import TabDelimitedConverter, PythonListConverter


class CSVConverter(BaseDataConverter):
    """Concrete CSV converter that implements the abstract methods."""
    
    def load_training_data(self, filename: str) -> Tuple[List[List[int]], List[List[int]]]:
        train_inputs, train_outputs, _ = self.load_from_csv(filename)
        return train_inputs, train_outputs
    
    def load_testing_data(self, filename: str) -> List[List[int]]:
        _, _, test_inputs = self.load_from_csv(filename)
        return test_inputs


class DataPreparer:
    
    def __init__(self, num_features: int = 11):
        self.num_features = num_features
        self.training_inputs = []
        self.training_outputs = []
        self.test_inputs = []
    
    @classmethod
    def from_tab_delimited(cls, filename: str, num_features: int = 11) -> 'DataPreparer':
        preparer = cls(num_features)
        converter = TabDelimitedConverter(num_features)
        preparer.training_inputs, preparer.training_outputs = converter.load_training_data(filename)
        preparer.test_inputs = converter.load_testing_data(filename)
        return preparer
    
    @classmethod
    def from_python_lists(cls, filename: str, num_features: int = 11) -> 'DataPreparer':
        preparer = cls(num_features)
        converter = PythonListConverter(num_features)
        preparer.training_inputs, preparer.training_outputs, preparer.test_inputs = converter.load_from_python_file(filename)
        return preparer
    
    @classmethod
    def from_csv(cls, filename: str, num_features: int = 11) -> 'DataPreparer':
        preparer = cls(num_features)
        converter = CSVConverter(num_features)
        preparer.training_inputs, preparer.training_outputs, preparer.test_inputs = converter.load_from_csv(filename)
        return preparer
    
    def get_dataloader(self, batch_size: int = 96, shuffle: bool = True) -> TorchDataLoader:
        X = torch.tensor(self.training_inputs, dtype=torch.float32)
        Y = torch.tensor(self.training_outputs, dtype=torch.float32)
        dataset = TensorDataset(X, Y)
        return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_test_dataloader(self, batch_size: int = 96) -> TorchDataLoader:
        if not self.test_inputs:
            raise ValueError("No test data available")
            
        X = torch.tensor(self.test_inputs, dtype=torch.float32)
        # Create dummy outputs for test data
        Y = torch.zeros_like(X)
        dataset = TensorDataset(X, Y)
        return TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def get_raw_data(self) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        return self.training_inputs, self.training_outputs, self.test_inputs
    
    def get_statistics(self) -> Dict[str, int]:
        return {
            'num_training_samples': len(self.training_inputs),
            'num_test_samples': len(self.test_inputs),
            'num_features': self.num_features
        }
    
    def save_to_csv(self, filename: str):
        converter = CSVConverter(self.num_features)
        converter.save_to_csv(self.training_inputs, self.training_outputs, filename, self.test_inputs)