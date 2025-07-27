import csv
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class BaseDataConverter(ABC):
    def __init__(self, num_features: int = 11):
        self.num_features = num_features
        
    @abstractmethod
    def load_training_data(self, source) -> Tuple[List[List[int]], List[List[int]]]:
        pass
    
    @abstractmethod
    def load_testing_data(self, source) -> List[List[int]]:
        pass
    
    def save_to_csv(self, inputs: List[List[int]], outputs: List[List[int]], 
                    filename: str, test_inputs: Optional[List[List[int]]] = None):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            header = ['type', 'trial_id']
            header += [f'input_{i}' for i in range(2*self.num_features)]
            header += [f'output_{i}' for i in range(2*self.num_features)]
            writer.writerow(header)
            
            for i, (inp, out) in enumerate(zip(inputs, outputs)):
                row = ['train', i + 1] + inp + out
                writer.writerow(row)
            
            if test_inputs:
                for i, inp in enumerate(test_inputs):
                    row = ['test', i + 1] + inp + [0] * 2*self.num_features
                    writer.writerow(row)
    
    def load_from_csv(self, filename: str) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        train_inputs, train_outputs, test_inputs = [], [], []
        
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                inp = [int(row[f'input_{i}']) for i in range(2*self.num_features)]
                out = [int(row[f'output_{i}']) for i in range(2*self.num_features)]
                
                if row['type'] == 'train':
                    train_inputs.append(inp)
                    train_outputs.append(out)
                elif row['type'] == 'test':
                    test_inputs.append(inp)
        
        return train_inputs, train_outputs, test_inputs