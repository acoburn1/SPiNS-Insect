from typing import List, Tuple
from .BaseConverter import BaseDataConverter


class TabDelimitedConverter(BaseDataConverter):
    def load_training_data(self, filename: str) -> Tuple[List[List[int]], List[List[int]]]:
        inputs, outputs = [], []
        
        with open(filename, 'r') as file:
            for line in file:
                if not line.startswith("_D:"):
                    continue
                    
                tokens = line.strip().split('\t')
                if len(tokens) < 3 + 4 * self.num_features:
                    continue
                
                start_idx = 3 
                mid_idx = start_idx + 2*self.num_features
                end_idx = mid_idx + 2*self.num_features
                
                input_vals = [int(tokens[i]) for i in range(start_idx, mid_idx)]
                output_vals = [int(tokens[i]) for i in range(mid_idx, end_idx)]
                
                inputs.append(input_vals)
                outputs.append(output_vals)
        
        return inputs, outputs
    
    def load_testing_data(self, filename: str) -> List[List[int]]:
        inputs, _ = self.load_training_data(filename)
        return inputs


class PythonListConverter(BaseDataConverter):
    def __init__(self, num_features: int = 11, alt: bool = False, sol: bool = False):
        super().__init__(num_features)
        self.alt = alt
        self.sol = sol
    
    def _convert_trial_to_vector(self, trial: List, is_lattice: bool = False) -> List[int]:
        vector = [0] * 2*self.num_features
        features = trial[2:6] if is_lattice and self.alt else trial[2:7]
        
        for feature in features:
            if is_lattice:
                vector[feature - 1 + self.num_features] = 1
            else:
                vector[feature - 1] = 1
        
        return vector

    def _convert_output_to_vector(self, trial: List, is_lattice: bool = False) -> List[int]:
        vector = [0] * 2*self.num_features
        feature = trial[6] if is_lattice and self.alt else trial[7]
        
        if is_lattice:
            vector[feature - 1 + self.num_features] = 1
        else:
            vector[feature - 1] = 1
        
        return vector
    
    def load_training_data(self, mod_trials: List[List], lat_trials: List[List]) -> Tuple[List[List[int]], List[List[int]]]:
        inputs, outputs = [], []
        
        for trial in mod_trials:
            vector = self._convert_trial_to_vector(trial, is_lattice=False)
            inputs.append(vector)
            outputs.append(vector if not self.sol else self._convert_output_to_vector(trial, is_lattice=False)) 
        
        for trial in lat_trials:
            vector = self._convert_trial_to_vector(trial, is_lattice=True)
            inputs.append(vector)
            outputs.append(vector if not self.sol else self._convert_output_to_vector(trial, is_lattice=True)) 
        
        return inputs, outputs
    
    def load_testing_data(self, ratio_trials: List[List]) -> List[List[int]]:
        test_inputs = []
        
        for trial in ratio_trials:
            features = trial[3:]
            vector = [0] * 2*self.num_features
            
            for feature in features:
                if feature == 100:
                    continue
                    
                if feature >= 200:
                    idx = (feature - 201) + self.num_features
                else:
                    idx = feature - 101
                
                if 0 <= idx < 2*self.num_features:
                    vector[idx] = 1
            
            test_inputs.append(vector)
        
        return test_inputs
    
    def load_from_python_file(self, filename: str) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        namespace = {}
        with open(filename, 'r') as f:
            exec(f.read(), namespace)
        
        mod_trials = namespace.get('mod_mf_trials', [])
        lat_trials = namespace.get('lat_mf_trials', [])
        ratio_trials = namespace.get('ratioTrials', [])
        
        train_inputs, train_outputs = self.load_training_data(mod_trials, lat_trials)
        test_inputs = self.load_testing_data(ratio_trials)
        
        return train_inputs, train_outputs, test_inputs