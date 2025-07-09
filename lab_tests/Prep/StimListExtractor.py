import numpy as np
import re
from typing import List, Tuple, Dict


class StimListExtractor:
    """
    Extracts and processes data from stimList_gencat_hydra_forAbe.js file.
    Converts JavaScript trial data into neural network training format.
    """
    
    def __init__(self, js_file_path: str, num_features: int = 11):
        """
        Initialize the extractor.
        
        Args:
            js_file_path: Path to the JavaScript file
            num_features: Number of features per category (default 11)
        """
        self.js_file_path = js_file_path
        self.num_features = num_features
        self.mod_trials = []
        self.lat_trials = []
        self.ratio_trials = []
    
    def load_js_file(self) -> str:
        """Load the JavaScript file content."""
        with open(self.js_file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def parse_js_arrays(self, js_content: str) -> None:
        """
        Parse JavaScript arrays from the file content.
        
        Args:
            js_content: Content of the JavaScript file
        """
        # Extract mod_mf_trials array
        mod_pattern = r'var mod_mf_trials=\[(.*?)\];'
        mod_match = re.search(mod_pattern, js_content, re.DOTALL)
        if mod_match:
            self.mod_trials = self._parse_array_content(mod_match.group(1))
        
        # Extract lat_mf_trials array
        lat_pattern = r'var lat_mf_trials=\[(.*?)\];'
        lat_match = re.search(lat_pattern, js_content, re.DOTALL)
        if lat_match:
            self.lat_trials = self._parse_array_content(lat_match.group(1))
        
        # Extract ratioTrials array
        ratio_pattern = r'var ratioTrials=\[(.*?)\];'
        ratio_match = re.search(ratio_pattern, js_content, re.DOTALL)
        if ratio_match:
            self.ratio_trials = self._parse_array_content(ratio_match.group(1))
    
    def _parse_array_content(self, array_content: str) -> List[List]:
        """
        Parse the content of a JavaScript array.
        
        Args:
            array_content: String content between array brackets
            
        Returns:
            List of parsed arrays
        """
        # Remove comments and clean up
        lines = array_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove comments (everything after //)
            if '//' in line:
                line = line[:line.index('//')]
            line = line.strip()
            if line and not line.startswith('//'):
                cleaned_lines.append(line)
        
        # Join and split by array boundaries
        content = ' '.join(cleaned_lines)
        
        # Extract individual arrays using regex
        array_pattern = r'\[([^\[\]]*)\]'
        arrays = re.findall(array_pattern, content)
        
        result = []
        for array_str in arrays:
            # Split by comma and clean up
            elements = [elem.strip().strip("'\"") for elem in array_str.split(',')]
            # Convert numeric strings to integers where possible
            parsed_elements = []
            for elem in elements:
                try:
                    parsed_elements.append(int(elem))
                except ValueError:
                    parsed_elements.append(elem)
            result.append(parsed_elements)
        
        return result
    
    def extract_shown_features(self, trial: List) -> List[int]:
        """
        Extract the shown features from a trial (positions 2-6).
        
        Args:
            trial: Trial array from JavaScript
            
        Returns:
            List of shown feature numbers
        """
        if len(trial) >= 7:
            return trial[2:7]  # Features are in positions 2-6
        return []
    
    def convert_to_binary_vector(self, features: List[int], is_lattice: bool = False) -> List[int]:
        """
        Convert feature numbers to binary vector representation.
        
        Args:
            features: List of feature numbers (1-11)
            is_lattice: Whether this is a lattice category (adds 11 to feature numbers)
            
        Returns:
            Binary vector of length 2*num_features (22 for 11 features per category)
        """
        vector = [0] * (2 * self.num_features)
        
        for feature in features:
            if 1 <= feature <= self.num_features:
                if is_lattice:
                    # Lattice features occupy positions 11-21 (0-indexed: 11-21)
                    vector[feature - 1 + self.num_features] = 1
                else:
                    # Modular features occupy positions 0-10
                    vector[feature - 1] = 1
        
        return vector
    
    def get_modular_training_data(self) -> Tuple[List[List[int]], List[Dict]]:
        """
        Extract modular training data in neural network format.
        
        Returns:
            Tuple of (binary_vectors, trial_metadata)
        """
        binary_vectors = []
        metadata = []
        
        for trial in self.mod_trials:
            if len(trial) >= 7:
                features = self.extract_shown_features(trial)
                binary_vector = self.convert_to_binary_vector(features, is_lattice=False)
                binary_vectors.append(binary_vector)
                
                metadata.append({
                    'category': trial[0] if len(trial) > 0 else '',
                    'type': trial[1] if len(trial) > 1 else '',
                    'features': features,
                    'original_trial': trial
                })
        
        return binary_vectors, metadata
    
    def get_lattice_training_data(self) -> Tuple[List[List[int]], List[Dict]]:
        """
        Extract lattice training data in neural network format.
        
        Returns:
            Tuple of (binary_vectors, trial_metadata)
        """
        binary_vectors = []
        metadata = []
        
        for trial in self.lat_trials:
            if len(trial) >= 7:
                features = self.extract_shown_features(trial)
                binary_vector = self.convert_to_binary_vector(features, is_lattice=True)
                binary_vectors.append(binary_vector)
                
                metadata.append({
                    'category': trial[0] if len(trial) > 0 else '',
                    'type': trial[1] if len(trial) > 1 else '',
                    'features': features,
                    'original_trial': trial
                })
        
        return binary_vectors, metadata
    
    def get_all_training_data(self) -> Tuple[List[List[int]], List[Dict]]:
        """
        Get combined training data from both modular and lattice trials.
        
        Returns:
            Tuple of (binary_vectors, trial_metadata)
        """
        mod_vectors, mod_metadata = self.get_modular_training_data()
        lat_vectors, lat_metadata = self.get_lattice_training_data()
        
        all_vectors = mod_vectors + lat_vectors
        all_metadata = mod_metadata + lat_metadata
        
        return all_vectors, all_metadata
    
    def get_ratio_trials_data(self) -> List[Dict]:
        """
        Extract ratio trials data for testing.
        
        Returns:
            List of ratio trial dictionaries
        """
        ratio_data = []
        
        for trial in self.ratio_trials:
            if len(trial) >= 6:
                # Extract features (positions 3 onwards)
                features = trial[3:] if len(trial) > 3 else []
                
                ratio_data.append({
                    'ratio': trial[0] if len(trial) > 0 else '',
                    'type': trial[1] if len(trial) > 1 else '',
                    'structure': trial[2] if len(trial) > 2 else '',
                    'features': features,
                    'original_trial': trial
                })
        
        return ratio_data
    
    def save_training_data_to_file(self, filename: str) -> None:
        """
        Save training data in the format expected by DataPreparer.
        
        Args:
            filename: Output filename for training data
        """
        vectors, metadata = self.get_all_training_data()
        
        with open(filename, 'w') as file:
            # Write header
            header = ["_H:", "$ItemNum", "$Name"]
            header += [f'"%Input[2:{i},0]"' for i in range(2 * self.num_features)]
            header += [f'"%Output[2:{i},0]"' for i in range(2 * self.num_features)]
            file.write('\t'.join(header) + '\n')
            
            # Write data rows
            for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                name = f"EX{i+1}_{meta['category']}_{meta['type']}"
                row = ["_D:", str(i+1), name]
                row += [str(val) for val in vector]  # Input
                row += [str(val) for val in vector]  # Output (same as input for autoencoder)
                file.write('\t'.join(row) + '\n')
    
    def process_file(self) -> Dict:
        """
        Process the JavaScript file and extract all data.
        
        Returns:
            Dictionary containing all extracted data
        """
        js_content = self.load_js_file()
        self.parse_js_arrays(js_content)
        
        mod_vectors, mod_metadata = self.get_modular_training_data()
        lat_vectors, lat_metadata = self.get_lattice_training_data()
        ratio_data = self.get_ratio_trials_data()
        
        return {
            'modular_vectors': mod_vectors,
            'modular_metadata': mod_metadata,
            'lattice_vectors': lat_vectors,
            'lattice_metadata': lat_metadata,
            'ratio_trials': ratio_data,
            'total_mod_trials': len(mod_vectors),
            'total_lat_trials': len(lat_vectors),
            'total_ratio_trials': len(ratio_data)
        }


def example_usage():
    """Example of how to use the StimListExtractor."""
    # Initialize extractor
    extractor = StimListExtractor("Data/stimList_gencat_hydra_forAbe.js")
    
    # Process the file
    data = extractor.process_file()
    
    # Print summary
    print(f"Extracted {data['total_mod_trials']} modular trials")
    print(f"Extracted {data['total_lat_trials']} lattice trials")
    print(f"Extracted {data['total_ratio_trials']} ratio trials")
    
    # Example: Get first modular trial as binary vector
    if data['modular_vectors']:
        print(f"First modular trial: {data['modular_vectors'][0]}")
        print(f"Metadata: {data['modular_metadata'][0]}")
    
    # Save training data to file
    extractor.save_training_data_to_file("Data/extracted_training_data.txt")
    print("Training data saved to file")


if __name__ == "__main__":
    example_usage()