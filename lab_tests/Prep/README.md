# Data Preparation System

This directory contains a modular data preparation system that handles multiple data formats and converts them for use with PyTorch.

## Overview

The system is now organized into focused, readable modules:

1. **BaseConverter.py** - Abstract base class and common CSV functionality
2. **FormatConverters.py** - Specific converters for tab-delimited and Python list formats
3. **DataPreparer.py** - Main data loading class with PyTorch integration
4. **DataUtils.py** - Convenience functions for quick data loading
5. **DataConverter.py** - Backward compatibility module (re-exports everything)

## Supported Formats

### Input Formats
1. **Tab-delimited format** (like `testPats_22dim_2cats_good.txt`)
   - Format: `_D: ItemNum Name Input[0] Input[1] ... Output[0] Output[1] ...`
   - Binary vectors with tab separation

2. **Python list format** (like `stimList_gencat_hydra_forAbe.py`)
   - Contains `mod_mf_trials`, `lat_mf_trials`, and `ratioTrials` lists
   - Feature numbers that get converted to binary vectors
   - Modular features: 1-11 ? indices 0-10
   - Lattice features: 1-11 ? indices 11-21 (offset by 11)

3. **CSV format** - Common intermediate format with columns:
   - `type`: 'train' or 'test'
   - `trial_id`: Sequential trial number
   - `input_0` to `input_21`: Input feature binary values
   - `output_0` to `output_21`: Output feature binary values

## Quick Start

### Method 1: Package-level imports (Recommended)from Prep import DataPreparer, load_python_list_data

# Quick loading with convenience function
data_prep = load_python_list_data("data/stimList_gencat_hydra_forAbe.py")

# Or use the main class
data_prep = DataPreparer.from_python_lists("data/stimList_gencat_hydra_forAbe.py")

# Get PyTorch DataLoader
train_loader = data_prep.get_dataloader(batch_size=32, shuffle=True)
### Method 2: Specific module importsfrom Prep.DataPreparer import DataPreparer
from Prep.DataUtils import load_tab_delimited_data

data_prep = load_tab_delimited_data("data/testPats_22dim_2cats_good.txt")
train_loader = data_prep.get_dataloader(batch_size=32)
### Method 3: Legacy compatibility (still works!)from Prep.DataConverter import DataPreparer  # Works exactly as before

data_prep = DataPreparer.from_python_lists("data/stimList_gencat_hydra_forAbe.py")
train_loader = data_prep.get_dataloader(batch_size=32)
## Module Details

### BaseConverter.py
Contains the abstract base class and shared functionality:
- `BaseDataConverter` - Abstract base with common CSV methods
- `save_to_csv()` - Export data to CSV format
- `load_from_csv()` - Import data from CSV format

### FormatConverters.py
Contains format-specific converter implementations:
- `TabDelimitedConverter` - Handles `.txt` files with tab-separated binary vectors
- `PythonListConverter` - Handles `.py` files with trial lists and feature numbers

### DataPreparer.py
Main data loading class with PyTorch integration:
- `DataPreparer` - Primary interface for data loading
- Class methods: `from_tab_delimited()`, `from_python_lists()`, `from_csv()`
- PyTorch methods: `get_dataloader()`, `get_test_dataloader()`
- Utility methods: `get_statistics()`, `save_to_csv()`, `get_raw_data()`

### DataUtils.py
Convenience functions for quick data loading:
- `load_tab_delimited_data()` - Quick tab-delimited loading
- `load_python_list_data()` - Quick Python list loading  
- `load_csv_data()` - Quick CSV loading

## Usage in Training
from Prep import load_python_list_data

# Load data
data_prep = load_python_list_data("data/stimList_gencat_hydra_forAbe.py")
train_loader = data_prep.get_dataloader(batch_size=16, shuffle=True)

# Standard training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Your training code here
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
## API Reference

### DataPreparer Class
**Class Methods:**
- `DataPreparer.from_tab_delimited(filename, num_features=22)` - Load from .txt file
- `DataPreparer.from_python_lists(filename, num_features=22)` - Load from .py file  
- `DataPreparer.from_csv(filename, num_features=22)` - Load from .csv file

**Instance Methods:**
- `get_dataloader(batch_size=32, shuffle=True)` - Get training DataLoader
- `get_test_dataloader(batch_size=32)` - Get test DataLoader  
- `get_raw_data()` - Get raw data as lists
- `get_statistics()` - Get data statistics
- `save_to_csv(filename)` - Save current data to CSV

**Legacy Compatibility:**
- `load_data()` - No-op (data already loaded)
- `get_probability_matrices_m_l()` - Returns (None, None)

### Convenience Functions
- `load_tab_delimited_data(filename, num_features=22)` - Quick tab-delimited loading
- `load_python_list_data(filename, num_features=22)` - Quick Python list loading
- `load_csv_data(filename, num_features=22)` - Quick CSV loading

### Converter Classes (for advanced usage)
- `TabDelimitedConverter` - For .txt files
- `PythonListConverter` - For .py files
- `BaseDataConverter` - Abstract base class

## Files

- `BaseConverter.py` - Abstract base class and CSV functionality
- `FormatConverters.py` - Format-specific converter implementations
- `DataPreparer.py` - Main data loading class with PyTorch integration
- `DataUtils.py` - Convenience functions
- `DataConverter.py` - Backward compatibility module
- `__init__.py` - Package initialization and exports
- `LegacyDataPrep.py` - Original data preparation (kept for reference)

## Migration Guide

### From DataConverter.py (no changes needed!)# This still works exactly the same
from Prep.DataConverter import DataPreparer
data_prep = DataPreparer.from_python_lists("data/stimList_gencat_hydra_forAbe.py")
### To New Modular Imports (optional, for cleaner code)# Cleaner package-level import
from Prep import DataPreparer
data_prep = DataPreparer.from_python_lists("data/stimList_gencat_hydra_forAbe.py")

# Or use convenience functions
from Prep import load_python_list_data
data_prep = load_python_list_data("data/stimList_gencat_hydra_forAbe.py")
## Benefits of Modular System
- ? **Improved readability** - Each file has a single, clear responsibility
- ? **Better maintainability** - Easier to modify specific functionality
- ? **Reduced complexity** - Smaller, focused files
- ? **Backward compatibility** - All existing code continues to work
- ? **Flexible imports** - Import only what you need
- ? **Clear separation of concerns** - Abstract base, format handling, data loading, utilities