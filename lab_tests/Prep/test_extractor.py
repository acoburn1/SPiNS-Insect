"""
Test script to demonstrate the StimListExtractor functionality.
Run this to verify the extraction works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from StimListExtractor import StimListExtractor


def test_extractor():
    """Test the StimListExtractor with the actual JavaScript file."""
    
    # Path to the JavaScript file
    js_file_path = "../Data/stimList_gencat_hydra_forAbe.js"
    
    try:
        # Initialize extractor
        print("Initializing StimListExtractor...")
        extractor = StimListExtractor(js_file_path)
        
        # Process the file
        print("Processing JavaScript file...")
        data = extractor.process_file()
        
        # Print summary statistics
        print("\n=== EXTRACTION SUMMARY ===")
        print(f"Modular trials extracted: {data['total_mod_trials']}")
        print(f"Lattice trials extracted: {data['total_lat_trials']}")
        print(f"Ratio trials extracted: {data['total_ratio_trials']}")
        
        # Show example modular trial
        if data['modular_vectors']:
            print(f"\n=== EXAMPLE MODULAR TRIAL ===")
            print(f"Original trial: {data['modular_metadata'][0]['original_trial']}")
            print(f"Extracted features: {data['modular_metadata'][0]['features']}")
            print(f"Binary vector: {data['modular_vectors'][0]}")
            print(f"Category: {data['modular_metadata'][0]['category']}")
            print(f"Type: {data['modular_metadata'][0]['type']}")
        
        # Show example lattice trial
        if data['lattice_vectors']:
            print(f"\n=== EXAMPLE LATTICE TRIAL ===")
            print(f"Original trial: {data['lattice_metadata'][0]['original_trial']}")
            print(f"Extracted features: {data['lattice_metadata'][0]['features']}")
            print(f"Binary vector: {data['lattice_vectors'][0]}")
            print(f"Category: {data['lattice_metadata'][0]['category']}")
            print(f"Type: {data['lattice_metadata'][0]['type']}")
        
        # Show example ratio trial
        if data['ratio_trials']:
            print(f"\n=== EXAMPLE RATIO TRIAL ===")
            ratio_trial = data['ratio_trials'][0]
            print(f"Original trial: {ratio_trial['original_trial']}")
            print(f"Ratio: {ratio_trial['ratio']}")
            print(f"Type: {ratio_trial['type']}")
            print(f"Structure: {ratio_trial['structure']}")
            print(f"Features: {ratio_trial['features']}")
        
        # Save training data
        output_file = "../Data/extracted_training_data.txt"
        print(f"\n=== SAVING TRAINING DATA ===")
        extractor.save_training_data_to_file(output_file)
        print(f"Training data saved to: {output_file}")
        
        # Verify the binary vectors are correct length
        expected_length = 2 * extractor.num_features  # Should be 22 for 11 features per category
        if data['modular_vectors']:
            actual_length = len(data['modular_vectors'][0])
            print(f"\nBinary vector length: {actual_length} (expected: {expected_length})")
            assert actual_length == expected_length, f"Vector length mismatch!"
        
        print("\n? All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n? Error during extraction: {e}")
        return False


def test_vector_conversion():
    """Test the feature to binary vector conversion logic."""
    print("\n=== TESTING VECTOR CONVERSION ===")
    
    extractor = StimListExtractor("dummy_path.js")  # We don't need the file for this test
    
    # Test modular features
    mod_features = [2, 3, 4, 6, 7]  # Example from the JS file
    mod_vector = extractor.convert_to_binary_vector(mod_features, is_lattice=False)
    print(f"Modular features {mod_features} -> {mod_vector}")
    
    # Verify modular vector
    expected_mod = [0] * 22
    for f in mod_features:
        expected_mod[f-1] = 1  # Features 1-11 map to indices 0-10
    assert mod_vector == expected_mod, "Modular vector conversion failed!"
    
    # Test lattice features
    lat_features = [2, 3, 4, 5, 6]  # Example from the JS file
    lat_vector = extractor.convert_to_binary_vector(lat_features, is_lattice=True)
    print(f"Lattice features {lat_features} -> {lat_vector}")
    
    # Verify lattice vector
    expected_lat = [0] * 22
    for f in lat_features:
        expected_lat[f-1+11] = 1  # Features 1-11 map to indices 11-21
    assert lat_vector == expected_lat, "Lattice vector conversion failed!"
    
    print("? Vector conversion tests passed!")


if __name__ == "__main__":
    print("Running StimListExtractor tests...\n")
    
    # Test vector conversion logic
    test_vector_conversion()
    
    # Test full extraction
    test_extractor()