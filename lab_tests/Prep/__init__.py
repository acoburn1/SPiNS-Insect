"""
Data Preparation Package

This package provides data conversion and loading functionality for neural network training.

Main Components:
- BaseConverter: Abstract base class for data converters
- FormatConverters: Specific converters for different data formats
- DataPreparer: Main data loading class with PyTorch integration
- DataUtils: Convenience functions for quick data loading

Usage:
    # Quick start with convenience functions
    from Prep import load_python_list_data, load_tab_delimited_data
    
    data_prep = load_python_list_data("data/stimList_gencat_hydra_forAbe.py")
    train_loader = data_prep.get_dataloader(batch_size=32)
    
    # Or use the main class directly
    from Prep import DataPreparer
    
    data_prep = DataPreparer.from_python_lists("data/stimList_gencat_hydra_forAbe.py")
    train_loader = data_prep.get_dataloader(batch_size=32)
"""

# Main exports for backward compatibility
from .DataPreparer import DataPreparer
from .DataUtils import load_tab_delimited_data, load_python_list_data, load_csv_data

# Advanced exports for custom usage
from .BaseConverter import BaseDataConverter
from .FormatConverters import TabDelimitedConverter, PythonListConverter

__all__ = [
    # Main interface
    'DataPreparer',
    
    # Convenience functions
    'load_tab_delimited_data',
    'load_python_list_data', 
    'load_csv_data',
    
    # Advanced components
    'BaseDataConverter',
    'TabDelimitedConverter',
    'PythonListConverter'
]