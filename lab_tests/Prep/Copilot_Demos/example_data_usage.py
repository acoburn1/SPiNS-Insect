"""
Example usage of the modular data preparation system
"""

# Show different import styles
print("=== Import Style Examples ===")

# Style 1: Package-level imports (cleanest)
from Prep import DataPreparer, load_python_list_data, load_tab_delimited_data
print("? Package-level imports")

# Style 2: Specific module imports  
from Prep.DataPreparer import DataPreparer as DP
from Prep.DataUtils import load_python_list_data as load_py
print("? Specific module imports")

# Style 3: Legacy compatibility (still works!)
from Prep.DataConverter import DataPreparer as LegacyDP, load_python_list_data as legacy_load
print("? Legacy compatibility imports")

import os


def example_modular_usage():
    """Demonstrate the new modular system"""
    
    print("\n=== Modular Data Preparation System Example ===\n")
    
    # Define file paths
    data_dir = "data"
    tab_file = os.path.join(data_dir, "testPats_22dim_2cats_good.txt")
    python_file = os.path.join(data_dir, "stimList_gencat_hydra_forAbe.py")
    
    # Example 1: Package-level convenience functions (recommended)
    if os.path.exists(python_file):
        print("1. Using package-level convenience function:")
        data_prep = load_python_list_data(python_file)
        
        stats = data_prep.get_statistics()
        print(f"   Statistics: {stats}")
        
        train_loader = data_prep.get_dataloader(batch_size=16, shuffle=True)
        print(f"   Training DataLoader: {len(train_loader)} batches")
        print()
    
    # Example 2: Main class with class methods
    if os.path.exists(python_file):
        print("2. Using DataPreparer class methods:")
        data_prep = DataPreparer.from_python_lists(python_file)
        
        stats = data_prep.get_statistics()
        print(f"   Statistics: {stats}")
        
        if stats['num_test_samples'] > 0:
            test_loader = data_prep.get_test_dataloader(batch_size=16)
            print(f"   Test DataLoader: {len(test_loader)} batches")
        
        print()
    
    # Example 3: Show that legacy imports still work
    if os.path.exists(python_file):
        print("3. Legacy compatibility (DataConverter import):")
        legacy_data_prep = LegacyDP.from_python_lists(python_file)
        legacy_stats = legacy_data_prep.get_statistics()
        print(f"   Legacy statistics: {legacy_stats}")
        print("   ? All legacy code continues to work unchanged!")
        print()


def demonstrate_advanced_usage():
    """Show advanced usage with specific modules"""
    data_dir = "data"
    python_file = os.path.join(data_dir, "stimList_gencat_hydra_forAbe.py")
    
    if not os.path.exists(python_file):
        return
    
    print("=== Advanced Module Usage ===")
    
    # Import specific converter for custom processing
    from Prep.FormatConverters import PythonListConverter
    from Prep.BaseConverter import BaseDataConverter
    
    print("1. Using converters directly:")
    converter = PythonListConverter(num_features=22)
    train_inputs, train_outputs, test_inputs = converter.load_from_python_file(python_file)
    print(f"   Loaded {len(train_inputs)} training samples directly")
    
    # Save to CSV using base converter
    base_converter = BaseDataConverter(num_features=22)
    base_converter.save_to_csv(train_inputs, train_outputs, "advanced_example.csv", test_inputs)
    print("   Saved to CSV using BaseConverter")
    
    # Load back using DataPreparer
    data_prep = DataPreparer.from_csv("advanced_example.csv")
    reload_stats = data_prep.get_statistics()
    print(f"   Reloaded statistics: {reload_stats}")


def show_import_flexibility():
    """Demonstrate different ways to import the same functionality"""
    print("\n=== Import Flexibility ===")
    
    print("All these imports give you the same DataPreparer class:")
    print("1. from Prep import DataPreparer")
    print("2. from Prep.DataPreparer import DataPreparer") 
    print("3. from Prep.DataConverter import DataPreparer  # Legacy")
    
    print("\nAll these give you the same convenience function:")
    print("1. from Prep import load_python_list_data")
    print("2. from Prep.DataUtils import load_python_list_data")
    print("3. from Prep.DataConverter import load_python_list_data  # Legacy")
    
    print("\nChoose the style that fits your project!")


if __name__ == "__main__":
    example_modular_usage()
    demonstrate_advanced_usage()
    show_import_flexibility()
    
    print("\n?? Benefits of the Modular System:")
    print("   ? Better code organization")
    print("   ? Easier to maintain and extend")  
    print("   ? Import only what you need")
    print("   ? All existing code still works")
    print("   ? Clear separation of concerns")