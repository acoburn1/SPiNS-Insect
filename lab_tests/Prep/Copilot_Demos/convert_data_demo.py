from DataConverter import TabDelimitedConverter, PythonListConverter, UnifiedDataLoader
import os


def convert_tab_delimited_to_csv(input_file: str, output_file: str, num_features: int = 11):
    """Convert tab-delimited format to CSV"""
    converter = TabDelimitedConverter(num_features)
    
    # Load training data
    inputs, outputs = converter.load_training_data(input_file)
    
    # Save to CSV
    converter.save_to_csv(inputs, outputs, output_file)
    print(f"Converted {input_file} to {output_file}")
    print(f"Training samples: {len(inputs)}")


def convert_python_lists_to_csv(input_file: str, output_file: str, num_features: int = 11):
    """Convert Python list format to CSV"""
    converter = PythonListConverter(num_features)
    
    # Load data from Python file
    train_inputs, train_outputs, test_inputs = converter.load_from_python_file(input_file)
    
    # Save to CSV
    converter.save_to_csv(train_inputs, train_outputs, output_file, test_inputs)
    print(f"Converted {input_file} to {output_file}")
    print(f"Training samples: {len(train_inputs)}")
    print(f"Test samples: {len(test_inputs)}")


def demonstrate_unified_loader(csv_file: str):
    """Demonstrate the unified data loader"""
    loader = UnifiedDataLoader(csv_file)
    
    # Get statistics
    stats = loader.get_statistics()
    print(f"Data statistics: {stats}")
    
    # Get DataLoaders
    train_loader = loader.get_training_dataloader(batch_size=16)
    print(f"Training DataLoader created with {len(train_loader)} batches")
    
    if stats['num_test_samples'] > 0:
        test_loader = loader.get_test_dataloader(batch_size=16)
        print(f"Test DataLoader created with {len(test_loader)} batches")
    
    # Show a sample batch
    for batch_x, batch_y in train_loader:
        print(f"Sample batch shape - X: {batch_x.shape}, Y: {batch_y.shape}")
        print(f"Sample input (first 10 features): {batch_x[0][:10].tolist()}")
        break


if __name__ == "__main__":
    # Define file paths
    data_dir = "../data"
    tab_delimited_file = os.path.join(data_dir, "testPats_11dim_2cats_good.txt")
    python_list_file = os.path.join(data_dir, "stimList_gencat_hydra_forAbe.py")
    
    csv_output_tab = "converted_tab_delimited.csv"
    csv_output_python = "converted_python_lists.csv"
    
    print("=== Data Format Converter Demo ===\n")
    
    # Convert tab-delimited format if file exists
    if os.path.exists(tab_delimited_file):
        print("1. Converting tab-delimited format:")
        convert_tab_delimited_to_csv(tab_delimited_file, csv_output_tab)
        print()
        
        print("Demonstrating unified loader with tab-delimited data:")
        demonstrate_unified_loader(csv_output_tab)
        print("\n" + "="*50 + "\n")
    
    # Convert Python list format if file exists
    if os.path.exists(python_list_file):
        print("2. Converting Python list format:")
        convert_python_lists_to_csv(python_list_file, csv_output_python)
        print()
        
        print("Demonstrating unified loader with Python list data:")
        demonstrate_unified_loader(csv_output_python)
        print("\n" + "="*50 + "\n")
    
    print("Conversion complete! Use UnifiedDataLoader with the CSV files for training.")