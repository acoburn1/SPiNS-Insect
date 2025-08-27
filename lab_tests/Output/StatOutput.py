import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, pearsonr
from Statistics.StatsProducer import StatsProducer, AggregateStatsObject

def plot_stats_with_confidence_intervals(lr_str, data_dir, data_parameters, save_dir="Results/Analysis/Plots/Correlations", show_plots=False, include_e0=False):
    os.makedirs(save_dir, exist_ok=True)
    
    stats_producer = StatsProducer(data_parameters)
    stats_objects_dict = stats_producer.get_stats(data_dir)
    
    if not stats_objects_dict:
        print("No statistics objects generated. Check data directory and parameters.")
        return
    
    color_map = {
        'losses': 'black',
        'm_output_corrs': 'blue', 
        'l_output_corrs': 'red',
        'm_hidden_corrs': 'green',
        'l_hidden_corrs': 'magenta',
        'output_tests': 'orange',
        'hidden_tests': 'brown'
    }
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    first_stats_obj = next(iter(stats_objects_dict.values()))
    num_epochs = len(first_stats_obj.means)
    epochs = range(num_epochs) if include_e0 else range(1, num_epochs + 1)
    
    ax2 = None
    correlation_lines = []
    loss_lines = []
    
    for i, (param_name, stats_obj) in enumerate(stats_objects_dict.items()):
        color = color_map.get(param_name, f'C{i}')
        
        if param_name == 'losses':
            if ax2 is None:
                ax2 = ax1.twinx()
            
            line = ax2.plot(epochs, stats_obj.means, color=color, linewidth=2, 
                           label=param_name.replace('_', ' ').title(), marker='o', markersize=4)
            loss_lines.extend(line)
            
            for epoch, mean, ci_lower, ci_upper in zip(epochs, stats_obj.means, 
                                                      stats_obj.ci_lowers, stats_obj.ci_uppers):
                ax2.plot([epoch, epoch], [ci_lower, ci_upper], color=color, 
                        linewidth=1, alpha=0.7)
        else:
            line = ax1.plot(epochs, stats_obj.means, color=color, linewidth=2, 
                           label=param_name.replace('_', ' ').title(), marker='o', markersize=4)
            correlation_lines.extend(line)
            
            for epoch, mean, ci_lower, ci_upper in zip(epochs, stats_obj.means, 
                                                      stats_obj.ci_lowers, stats_obj.ci_uppers):
                ax1.plot([epoch, epoch], [ci_lower, ci_upper], color=color, 
                        linewidth=1, alpha=0.7)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Correlation Value', fontsize=12, color='black')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    if ax2 is not None:
        ax2.set_ylabel('Loss', fontsize=12, color='black')
        ax2.set_ylim(0, 2)
        ax2.tick_params(axis='y', labelcolor='black')
    
    max_epoch = max(epochs)
    min_epoch = min(epochs)
    
    major_ticks = list(range(10, max_epoch + 1, 10))
    if min_epoch not in major_ticks:
        major_ticks = [min_epoch] + major_ticks
    if max_epoch not in major_ticks and max_epoch % 10 != 0:
        major_ticks.append(max_epoch)
    
    medium_ticks = [x for x in range(5, max_epoch + 1, 5) if x not in major_ticks and x >= min_epoch]
    minor_ticks = [x for x in epochs if x not in major_ticks and x not in medium_ticks]
    
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(medium_ticks, minor=False)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.tick_params(which='major', length=8, width=2, labelsize=10)
    ax1.tick_params(which='minor', length=4, width=1)
    
    for tick in medium_ticks:
        ax1.axvline(x=tick, ymin=0, ymax=0.02, color='black', linewidth=1.5, clip_on=False)
    
    ax1.set_xlim(min_epoch, max_epoch)
    
    all_lines = correlation_lines + loss_lines
    all_labels = [l.get_label() for l in all_lines]
    ax1.legend(all_lines, all_labels, loc='best', fontsize=10)
    
    plt.title(f'Means Across Epochs, LR = 0.0{lr_str}', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/p_graph_stats_0{lr_str}.png", dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_s_curve(data_dir, hidden=True, save_dir="Results/Analysis/Plots/S-Curves", show_plots=False, epoch=-1, include_e0=False):
    os.makedirs(save_dir, exist_ok=True)
    
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if not npz_files:
        print(f"No NPZ files found in directory: {data_dir}")
        return
    
    x_positions = [0, 1, 2, 3, 4, 5, 6]
    x_labels = ['0:6\n(all-lat)', '1:5\n(lat-heavy)', '2:4\n(lat-heavy)', '3:3\n(even)', '4:2\n(mod-heavy)', '5:1\n(mod-heavy)', '6:0\n(all-mod)']
    
    ratio_to_position = {
        '0:6': 0,
        '1:5': 1, 
        '2:4': 2,
        '3:3': 3,
        '4:2': 4,
        '5:1': 5,
        '6:0': 6
    }
    
    category_accuracies = {pos: [] for pos in x_positions}
    
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        
        test_data_key = 'hidden_tests' if hidden else 'output_tests'
        
        if test_data_key not in data:
            print(f"Warning: {test_data_key} not found in {npz_file}")
            continue
            
        test_data = data[test_data_key][epoch]
        
        for ratio, position in ratio_to_position.items():
            if ratio in test_data:
                ratio_data = test_data[ratio]
                
                all_mod_correlations = []
                all_lat_correlations = []
                
                for set_name, set_data in ratio_data.items():
                    mod_correlations = np.array(set_data["mod"])
                    lat_correlations = np.array(set_data["lat"])
                    
                    all_mod_correlations.extend(mod_correlations)
                    all_lat_correlations.extend(lat_correlations)
                
                all_mod_correlations = np.array(all_mod_correlations)
                all_lat_correlations = np.array(all_lat_correlations)
                
                avg_mod_corr_per_trial = np.mean(all_mod_correlations, axis=1)
                avg_lat_corr_per_trial = np.mean(all_lat_correlations, axis=1)
                
                mod_preferred_trials = np.sum(avg_mod_corr_per_trial > avg_lat_corr_per_trial)
                total_trials = len(avg_mod_corr_per_trial)
                mod_preference_rate = mod_preferred_trials / total_trials if total_trials > 0 else 0
                
                category_accuracies[position].append(mod_preference_rate)
    
    means = []
    stderrs = []
    
    for pos in x_positions:
        accuracies = category_accuracies[pos]
        if accuracies:
            mean_acc = np.mean(accuracies)
            stderr_acc = np.std(accuracies) / np.sqrt(len(accuracies))
        else:
            mean_acc = 0
            stderr_acc = 0
            print(f"Warning: No data for position {pos}")
            
        means.append(mean_acc)
        stderrs.append(stderr_acc)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.errorbar(x_positions, means, yerr=stderrs, marker='o', markersize=8, 
                linewidth=2, capsize=5, capthick=2, color='blue')
    
    ax.set_xlabel('# mod feats', fontsize=14)
    ax.set_ylabel('% mod resp', fontsize=14)
    layer = 'Hidden' if hidden else 'Output'
    ax.set_title(f'Modular Response by Feature Composition - {layer}', fontsize=14)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(i) for i in x_positions])
    
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=3, color='gray', linestyle='--', alpha=0.7)
    
    l = 'h' if hidden else 'o'
    plt.tight_layout()
    plt.savefig(f"{save_dir}/s_curve_{l}_e{epoch if include_e0 else epoch+1}.png", dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_difference_stats_with_confidence_intervals(lr_str, data_dir, data_parameters, save_dir="Results/Analysis/Plots/Diffs/w_ttest", show_plots=False, include_e0=False, alpha=0.05):
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data from NPZ files
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if not npz_files:
        print(f"No NPZ files found in directory: {data_dir}")
        return
    
    # Find available modular/lattice pairs
    modular_params = ['m_output_corrs', 'm_hidden_corrs']
    lattice_params = ['l_output_corrs', 'l_hidden_corrs']
    
    param_pairs = []
    for mod_param in modular_params:
        for lat_param in lattice_params:
            if (data_parameters.get(mod_param, False) and 
                data_parameters.get(lat_param, False) and
                mod_param.split('_')[1] == lat_param.split('_')[1]):
                param_pairs.append((mod_param, lat_param))
    
    if not param_pairs:
        print("No modular/lattice pairs found in data parameters.")
        return
    
    # Create stats producer
    stats_producer = StatsProducer(data_parameters, ci=0.95)
    difference_results = {}
    
    # Process each pair
    for mod_param, lat_param in param_pairs:
        mod_data_list = []
        lat_data_list = []
        
        # Load data for this pair
        for npz_file in npz_files:
            data = np.load(npz_file, allow_pickle=True)
            if (mod_param in data and data[mod_param] is not None and
                lat_param in data and data[lat_param] is not None):
                mod_data_list.append(data[mod_param])
                lat_data_list.append(data[lat_param])
        
        # If we have data for both, compute difference stats with t-test
        if mod_data_list and lat_data_list:
            mod_array = np.array(mod_data_list)
            lat_array = np.array(lat_data_list)
            
            layer_type = mod_param.split('_')[1]
            diff_key = f'mod_vs_lat_{layer_type}'
            
            # Use the new method to compute difference stats with t-test
            difference_results[diff_key] = stats_producer.get_difference_stats_with_ttest(
                mod_array, lat_array, alpha=alpha
            )
    
    if not difference_results:
        print("No difference statistics objects generated. Check data directory and parameters.")
        return
    
    color_map = {
        'mod_vs_lat_output': 'purple',
        'mod_vs_lat_hidden': 'darkorange'
    }
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    first_result = next(iter(difference_results.values()))
    num_epochs = len(first_result['stats'].means)
    epochs = range(num_epochs) if include_e0 else range(1, num_epochs + 1)
    
    for i, (param_name, result) in enumerate(difference_results.items()):
        color = color_map.get(param_name, f'C{i}')
        stats_obj = result['stats']
        significant = result['significant']
        
        # Plot the main line
        line = ax1.plot(epochs, stats_obj.means, color=color, linewidth=2, 
                       label=param_name.replace('_', ' ').title(), marker='o', markersize=4)
        
        # Plot confidence intervals
        for epoch, mean, ci_lower, ci_upper in zip(epochs, stats_obj.means, 
                                                  stats_obj.ci_lowers, stats_obj.ci_uppers):
            ax1.plot([epoch, epoch], [ci_lower, ci_upper], color=color, 
                    linewidth=1, alpha=0.7)
        
        # Overlay red segments for non-significant parts (p >= alpha)
        epoch_idx_offset = 0 if include_e0 else 1
        for epoch_idx, epoch in enumerate(epochs):
            data_epoch_idx = epoch_idx + epoch_idx_offset
            if data_epoch_idx < len(significant) and not significant[data_epoch_idx]:
                # This epoch is NOT significant, draw red segment
                if epoch_idx < len(epochs) - 1:
                    # Draw line segment to next point in red
                    next_epoch = epochs[epoch_idx + 1]
                    next_data_idx = epoch_idx + 1 + epoch_idx_offset
                    if next_data_idx < len(stats_obj.means):
                        ax1.plot([epoch, next_epoch], 
                                [stats_obj.means[data_epoch_idx], stats_obj.means[next_data_idx]], 
                                color='red', linewidth=3, alpha=0.8)
                
                # Draw red marker for non-significant point
                ax1.plot(epoch, stats_obj.means[data_epoch_idx], 
                        marker='o', color='red', markersize=6, markeredgecolor='darkred')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Modular - Lattice Correlation Difference', fontsize=12, color='black')
    ax1.set_ylim(-0.1, 0.5)  # Bound the y-axis between -0.1 and 0.5
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Add reference line at y=0 (no difference between modular and lattice)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='No Difference')
    
    max_epoch = max(epochs)
    min_epoch = min(epochs)
    
    major_ticks = list(range(10, max_epoch + 1, 10))
    if min_epoch not in major_ticks:
        major_ticks = [min_epoch] + major_ticks
    if max_epoch not in major_ticks and max_epoch % 10 != 0:
        major_ticks.append(max_epoch)
    
    medium_ticks = [x for x in range(5, max_epoch + 1, 5) if x not in major_ticks and x >= min_epoch]
    minor_ticks = [x for x in epochs if x not in major_ticks and x not in medium_ticks]
    
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(medium_ticks, minor=False)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.tick_params(which='major', length=8, width=2, labelsize=10)
    ax1.tick_params(which='minor', length=4, width=1)
    
    for tick in medium_ticks:
        ax1.axvline(x=tick, ymin=0, ymax=0.02, color='black', linewidth=1.5, clip_on=False)
    
    ax1.set_xlim(min_epoch, max_epoch)
    
    # Add legend entry for non-significant segments
    ax1.plot([], [], color='red', linewidth=3, label=f'Non-significant (p ≥ {alpha})')
    ax1.legend(loc='best', fontsize=10)
    
    plt.title(f'Modular vs Lattice Correlation Differences (with t-test), LR = 0.0{lr_str}', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/p_graph_diff_stats_ttest_0{lr_str}.png", dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_33s(data_dir, hidden=True, save_dir="Results/Analysis/Plots/3-3", 
                        show_plots=False, start_epoch=0, num_epochs=None, include_e0=False):
    """
    Plot 3:3 ratio trial data across epochs with different colored lines for each set label.
    
    Parameters:
    - data_dir: Directory containing NPZ files
    - hidden: If True, use hidden_tests data; if False, use output_tests data
    - save_dir: Directory to save plots
    - show_plots: Whether to display plots
    - start_epoch: Starting epoch for the plot range
    - num_epochs: Number of epochs to include (if None, use all available)
    - include_e0: Whether to include epoch 0 in the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))  # Sort for consistency
    
    if not npz_files:
        print(f"No NPZ files found in directory: {data_dir}")
        return
    
    test_data_key = 'hidden_tests' if hidden else 'output_tests'
    
    # First pass: determine available epochs and set names
    all_epochs = set()
    all_set_names = set()
    
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        
        if test_data_key not in data:
            print(f"Warning: {test_data_key} not found in {npz_file}")
            continue
        
        test_data = data[test_data_key]
        all_epochs.update(range(len(test_data)))
        
        # Check what set names are available in 3:3 data
        for epoch_idx in range(len(test_data)):
            epoch_data = test_data[epoch_idx]
            if '3:3' in epoch_data:
                ratio_data = epoch_data['3:3']
                all_set_names.update(ratio_data.keys())
    
    # Sort set names for consistency
    all_set_names = sorted(list(all_set_names))
    
    # Determine epoch range
    if not all_epochs:
        print("No epoch data found")
        return
    
    max_available_epoch = max(all_epochs)
    
    if num_epochs is None:
        end_epoch = max_available_epoch
    else:
        end_epoch = min(start_epoch + num_epochs - 1, max_available_epoch)
    
    epoch_range = list(range(start_epoch, end_epoch + 1))
    display_epochs = epoch_range if include_e0 else [e + 1 for e in epoch_range]
    
    print(f"Processing epochs {start_epoch} to {end_epoch}")
    print(f"Available set names: {sorted(all_set_names)}")
    
    # Collect data organized by set_name for StatsProducer
    # Structure: {set_name: [file1_data, file2_data, ...]} where each file_data is [epoch1_mod_pref, epoch2_mod_pref, ...]
    set_data_across_files = {set_name: [] for set_name in all_set_names}
    
    # Second pass: collect data for each file
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        
        if test_data_key not in data:
            continue
            
        test_data = data[test_data_key]
        
        # For this file, collect data for each set across epochs
        file_data_by_set = {set_name: [] for set_name in all_set_names}
        
        for epoch_idx in epoch_range:
            if epoch_idx >= len(test_data):
                # If epoch doesn't exist in this file, append NaN for all sets
                for set_name in all_set_names:
                    file_data_by_set[set_name].append(np.nan)
                continue
                
            epoch_data = test_data[epoch_idx]
            
            if '3:3' not in epoch_data:
                # If 3:3 data doesn't exist for this epoch, append NaN for all sets
                for set_name in all_set_names:
                    file_data_by_set[set_name].append(np.nan)
                continue
                
            ratio_data = epoch_data['3:3']
            
            for set_name in all_set_names:
                if set_name in ratio_data:
                    set_data = ratio_data[set_name]
                    
                    # Extract mod and lat correlations directly (corrected structure)
                    mod_correlations = np.array(set_data["mod"])
                    lat_correlations = np.array(set_data["lat"])
                    
                    # Calculate preference rate per trial
                    avg_mod_corr_per_trial = np.mean(mod_correlations, axis=1)
                    avg_lat_corr_per_trial = np.mean(lat_correlations, axis=1)
                    
                    mod_preferred_trials = np.sum(avg_mod_corr_per_trial > avg_lat_corr_per_trial)
                    total_trials = len(avg_mod_corr_per_trial)
                    mod_preference_rate = mod_preferred_trials / total_trials if total_trials > 0 else 0
                    
                    file_data_by_set[set_name].append(mod_preference_rate)
                else:
                    file_data_by_set[set_name].append(np.nan)
        
        # Add this file's data to the overall collection
        for set_name in all_set_names:
            set_data_across_files[set_name].append(file_data_by_set[set_name])
    
    # Use StatsProducer to calculate statistics for each set
    stats_producer = StatsProducer(ci=0.95)
    set_stats = {}
    
    for set_name in all_set_names:
        file_data_list = set_data_across_files[set_name]
        if file_data_list:
            # Convert to numpy array and filter out files with all NaN values
            data_array = np.array(file_data_list)
            valid_files = ~np.all(np.isnan(data_array), axis=1)
            
            if np.any(valid_files):
                filtered_data = data_array[valid_files]
                epoch_stats_list = stats_producer._get_epoch_stats(filtered_data)
                set_stats[set_name] = AggregateStatsObject(epoch_stats_list)
            else:
                print(f"Warning: No valid data for set {set_name}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map for different set names
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_map = {set_name: colors[i % len(colors)] for i, set_name in enumerate(sorted(all_set_names))}
    
    # Plot each set
    for set_name in sorted(all_set_names):  # Sort for consistency
        if set_name not in set_stats:
            continue
            
        stats_obj = set_stats[set_name]
        color = color_map[set_name]
        
        # Create label without correlation values (simplified)
        label = set_name.replace('_', ' ').title()
        
        # Plot main line
        ax.plot(display_epochs, stats_obj.means, color=color, linewidth=2, 
               label=label, marker='o', markersize=4)
        
        # Plot confidence intervals
        for epoch, mean, ci_lower, ci_upper in zip(display_epochs, stats_obj.means, 
                                                  stats_obj.ci_lowers, stats_obj.ci_uppers):
            ax.plot([epoch, epoch], [ci_lower, ci_upper], color=color, 
                   linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('% Modular Response', fontsize=12)
    layer = 'Hidden' if hidden else 'Output'
    ax.set_title(f'3:3 Ratio Modular Response Across Epochs - {layer}', fontsize=14)
    
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=3, color='gray', linestyle='--', alpha=0.7)
    
    # Set up x-axis ticks
    if display_epochs:
        min_epoch = min(display_epochs)
        max_epoch = max(display_epochs)
        
        major_ticks = list(range(10, max_epoch + 1, 10))
        if min_epoch not in major_ticks:
            major_ticks = [min_epoch] + major_ticks
        if max_epoch not in major_ticks and max_epoch % 10 != 0:
            major_ticks.append(max_epoch)
        
        medium_ticks = [x for x in range(5, max_epoch + 1, 5) if x not in major_ticks and x >= min_epoch]
        minor_ticks = [x for x in display_epochs if x not in major_ticks and x not in medium_ticks]
        
        ax.set_xticks(major_ticks)
        ax.set_xticks(medium_ticks, minor=False)
        ax.set_xticks(minor_ticks, minor=True)
        ax.tick_params(which='major', length=8, width=2, labelsize=10)
        ax.tick_params(which='minor', length=4, width=1)
        
        for tick in medium_ticks:
            ax.axvline(x=tick, ymin=0, ymax=0.02, color='black', linewidth=1.5, clip_on=False)
        
        ax.set_xlim(min_epoch, max_epoch)
    
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    layer_suffix = 'h' if hidden else 'o'
    epoch_suffix = f"_e{start_epoch}-{end_epoch}" if include_e0 else f"_e{start_epoch+1}-{end_epoch+1}"
    plt.savefig(f"{save_dir}/33_ratio_epochs_{layer_suffix}{epoch_suffix}.png", dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Generate the correlation plot as well
    plot_33s_correlations(data_dir, hidden, save_dir, show_plots, start_epoch, num_epochs, include_e0)
    
    # Generate the scatter plot as well
    plot_33s_scatter(data_dir, hidden, save_dir, show_plots, start_epoch, num_epochs, include_e0)

def plot_33s_correlations(data_dir, hidden=True, save_dir="Results/Analysis/Plots/3-3", 
                         show_plots=False, start_epoch=0, num_epochs=None, include_e0=False, alpha=0.05):
    """
    Plot correlation values for 3:3 ratio trial sets with modular and lattice patterns.
    
    Parameters:
    - data_dir: Directory containing NPZ files
    - hidden: If True, use hidden_tests data; if False, use output_tests data
    - save_dir: Directory to save plots
    - show_plots: Whether to display plots
    - start_epoch: Starting epoch for the plot range
    - num_epochs: Number of epochs to include (if None, use all available)
    - include_e0: Whether to include epoch 0 in the plot
    - alpha: Significance level for correlation p-values
    """
    os.makedirs(save_dir, exist_ok=True)
    
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))  # Sort for consistency
    
    if not npz_files:
        print(f"No NPZ files found in directory: {data_dir}")
        return
    
    test_data_key = 'hidden_tests' if hidden else 'output_tests'
    
    # First pass: determine available epochs and set names
    all_epochs = set()
    all_set_names = set()
    
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        
        if test_data_key not in data:
            print(f"Warning: {test_data_key} not found in {npz_file}")
            continue
        
        test_data = data[test_data_key]
        all_epochs.update(range(len(test_data)))
        
        # Check what set names are available in 3:3 data
        for epoch_idx in range(len(test_data)):
            epoch_data = test_data[epoch_idx]
            if '3:3' in epoch_data:
                ratio_data = epoch_data['3:3']
                all_set_names.update(ratio_data.keys())
    
    # Sort set names for consistency
    all_set_names = sorted(list(all_set_names))
    
    # Determine epoch range
    if not all_epochs:
        print("No epoch data found")
        return
    
    max_available_epoch = max(all_epochs)
    
    if num_epochs is None:
        end_epoch = max_available_epoch
    else:
        end_epoch = min(start_epoch + num_epochs - 1, max_available_epoch)
    
    epoch_range = list(range(start_epoch, end_epoch + 1))
    
    # Collect data organized by set_name for StatsProducer
    # Structure: {set_name: [file1_data, file2_data, ...]} where each file_data is [epoch1_mod_pref, epoch2_mod_pref, ...]
    set_data_across_files = {set_name: [] for set_name in all_set_names}
    
    # Second pass: collect data for each file
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        
        if test_data_key not in data:
            continue
            
        test_data = data[test_data_key]
        
        # For this file, collect data for each set across epochs
        file_data_by_set = {set_name: [] for set_name in all_set_names}
        
        for epoch_idx in epoch_range:
            if epoch_idx >= len(test_data):
                for set_name in all_set_names:
                    file_data_by_set[set_name].append(np.nan)
                continue
                
            epoch_data = test_data[epoch_idx]
            
            if '3:3' not in epoch_data:
                for set_name in all_set_names:
                    file_data_by_set[set_name].append(np.nan)
                continue
                
            ratio_data = epoch_data['3:3']
            
            for set_name in all_set_names:
                if set_name in ratio_data:
                    set_data = ratio_data[set_name]
                    
                    mod_correlations = np.array(set_data["mod"])
                    lat_correlations = np.array(set_data["lat"])
                    
                    avg_mod_corr_per_trial = np.mean(mod_correlations, axis=1)
                    avg_lat_corr_per_trial = np.mean(lat_correlations, axis=1)
                    
                    mod_preferred_trials = np.sum(avg_mod_corr_per_trial > avg_lat_corr_per_trial)
                    total_trials = len(avg_mod_corr_per_trial)
                    mod_preference_rate = mod_preferred_trials / total_trials if total_trials > 0 else 0
                    
                    file_data_by_set[set_name].append(mod_preference_rate)
                else:
                    file_data_by_set[set_name].append(np.nan)
        
        # Add this file's data to the overall collection
        for set_name in all_set_names:
            set_data_across_files[set_name].append(file_data_by_set[set_name])
    
    # Use StatsProducer to calculate statistics for each set
    stats_producer = StatsProducer(ci=0.95)
    set_stats = {}
    
    for set_name in all_set_names:
        file_data_list = set_data_across_files[set_name]
        if file_data_list:
            data_array = np.array(file_data_list)
            valid_files = ~np.all(np.isnan(data_array), axis=1)
            
            if np.any(valid_files):
                filtered_data = data_array[valid_files]
                epoch_stats_list = stats_producer._get_epoch_stats(filtered_data)
                set_stats[set_name] = AggregateStatsObject(epoch_stats_list)
    
    # Get modular and lattice correlation data
    correlation_data_params = {"m_hidden_corrs": True, "l_hidden_corrs": True}
    correlation_stats_producer = StatsProducer(correlation_data_params, ci=0.95)
    correlation_stats = correlation_stats_producer.get_stats(data_dir)
    
    if 'm_hidden_corrs' not in correlation_stats or 'l_hidden_corrs' not in correlation_stats:
        print("Error: Could not load modular and lattice correlation data")
        return
    
    mod_means = correlation_stats['m_hidden_corrs'].means
    lat_means = correlation_stats['l_hidden_corrs'].means
    
    # Trim to match the epoch range
    if len(mod_means) > len(epoch_range):
        start_idx = start_epoch
        end_idx = start_idx + len(epoch_range)
        mod_means_trimmed = mod_means[start_idx:end_idx]
        lat_means_trimmed = lat_means[start_idx:end_idx]
    else:
        mod_means_trimmed = mod_means
        lat_means_trimmed = lat_means
    
    # Calculate correlations and significance for each set
    correlation_results = {}
    for set_name in sorted(set_stats.keys()):  # Sort for consistency
        set_means = set_stats[set_name].means
        if len(set_means) == len(mod_means_trimmed):
            # Original correlations (signed)
            mod_corr, mod_p = pearsonr(set_means, mod_means_trimmed)
            lat_corr, lat_p = pearsonr(set_means, lat_means_trimmed)
            
            # Modular-lattice operations (removed mult and div)
            mod_lat_diff = np.array(mod_means_trimmed) - np.array(lat_means_trimmed)
            diff_corr, diff_p = pearsonr(set_means, mod_lat_diff)
            
            mod_lat_sum = np.array(mod_means_trimmed) + np.array(lat_means_trimmed)
            sum_corr, sum_p = pearsonr(set_means, mod_lat_sum)
            
            correlation_results[set_name] = {
                'mod_corr': mod_corr, 'mod_p': mod_p, 'mod_sign': np.sign(mod_corr),
                'lat_corr': lat_corr, 'lat_p': lat_p, 'lat_sign': np.sign(lat_corr),
                'diff_corr': diff_corr, 'diff_p': diff_p, 'diff_sign': np.sign(diff_corr),
                'sum_corr': sum_corr, 'sum_p': sum_p, 'sum_sign': np.sign(sum_corr)
            }
    
    # Create the correlation plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map (same as original plot)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_map = {set_name: colors[i % len(colors)] for i, set_name in enumerate(sorted(all_set_names))}
    
    # X-axis categories - updated with new operations
    categories = ['Mod Corr', 'Lat Corr', 'Mod-Lat', 'Mod+Lat']
    x_positions = [0, 1, 2, 3]
    
    # Calculate significance threshold
    from scipy.stats import t as t_dist
    n_epochs = len(epoch_range)
    df = n_epochs - 2  # degrees of freedom for correlation
    t_critical = t_dist.ppf(1 - alpha/2, df)  # two-tailed test
    r_critical = t_critical / np.sqrt(df + t_critical**2)  # critical r value
    
    # Plot each set's correlations
    for i, set_name in enumerate(sorted(correlation_results.keys())):
        if set_name not in correlation_results:
            continue
            
        results = correlation_results[set_name]
        color = color_map[set_name]
        
        # Y values for this set (absolute values) - removed mult and div
        y_values = [abs(results['mod_corr']), abs(results['lat_corr']), 
                   abs(results['diff_corr']), abs(results['sum_corr'])]
        p_values = [results['mod_p'], results['lat_p'], 
                   results['diff_p'], results['sum_p']]
        signs = [results['mod_sign'], results['lat_sign'], 
                results['diff_sign'], results['sum_sign']]
        
        # Determine dot sizes based on significance
        dot_sizes = []
        for p_val in p_values:
            if p_val < alpha:
                dot_sizes.append(100)  # Larger dots for significant correlations
            else:
                dot_sizes.append(50)   # Smaller dots for non-significant correlations
        
        # Plot points with different markers for positive/negative correlations
        for j, (x, y, sign, size) in enumerate(zip(x_positions, y_values, signs, dot_sizes)):
            if sign >= 0:
                marker = 'o'  # Circle for positive correlations
            else:
                marker = '^'  # Triangle for negative correlations
            
            ax.scatter(x, y, color=color, s=size, alpha=0.8, marker=marker, zorder=3)
        
        # Add label only once for legend
        ax.scatter([], [], color=color, s=100, alpha=0.8, 
                  label=set_name.replace('_', ' ').title(), marker='o')
    
    # Add significance threshold line
    ax.axhline(y=r_critical, color='red', linestyle=':', alpha=0.7, linewidth=2, 
               label=f'Significance threshold (α={alpha})')
    
    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Correlation Strength (|r|)', fontsize=12)
    ax.set_xlabel('Correlation Type', fontsize=12)
    layer = 'Hidden' if hidden else 'Output'
    ax.set_title(f'3:3 Ratio Set Correlation Strengths - {layer}', fontsize=14)
    
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Add explanation for markers and sizes
    ax.text(0.02, 0.98, 'Circle: positive correlation\nTriangle: negative correlation\n' +
            f'Large dots: p < {alpha}\nSmall dots: p ≥ {alpha}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    layer_suffix = 'h' if hidden else 'o'
    epoch_suffix = f"_e{start_epoch}-{end_epoch}" if include_e0 else f"_e{start_epoch+1}-{end_epoch+1}"
    plt.savefig(f"{save_dir}/33_correlations_{layer_suffix}{epoch_suffix}.png", dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return correlation_results

def plot_33s_scatter(data_dir, hidden=True, save_dir="Results/Analysis/Plots/3-3", 
                    show_plots=False, start_epoch=0, num_epochs=None, include_e0=False, ci_elipses=False):

    os.makedirs(save_dir, exist_ok=True)
    
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    
    if not npz_files:
        print(f"No NPZ files found in directory: {data_dir}")
        return
    
    test_data_key = 'hidden_tests' if hidden else 'output_tests'
    
    # First pass: determine available epochs and set names
    all_epochs = set()
    all_set_names = set()
    
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        
        if test_data_key not in data:
            print(f"Warning: {test_data_key} not found in {npz_file}")
            continue
        
        test_data = data[test_data_key]
        all_epochs.update(range(len(test_data)))
        
        # Check what set names are available in 3:3 data
        for epoch_idx in range(len(test_data)):
            epoch_data = test_data[epoch_idx]
            if '3:3' in epoch_data:
                ratio_data = epoch_data['3:3']
                all_set_names.update(ratio_data.keys())
    
    # Sort set names for consistency
    all_set_names = sorted(list(all_set_names))
    
    # Determine epoch range
    if not all_epochs:
        print("No epoch data found")
        return
    
    max_available_epoch = max(all_epochs)
    
    if num_epochs is None:
        end_epoch = max_available_epoch
    else:
        end_epoch = min(start_epoch + num_epochs - 1, max_available_epoch)
    
    epoch_range = list(range(start_epoch, end_epoch + 1))
    
    print(f"Processing epochs {start_epoch} to {end_epoch}")
    print(f"Available set names: {sorted(all_set_names)}")
    
    # Get modular and lattice correlation data for each model separately
    correlation_data_params = {"m_hidden_corrs": True, "l_hidden_corrs": True}
    
    # Collect mod-lat differences per model per epoch
    mod_lat_diffs_per_model = []  # [model1_diffs, model2_diffs, ...]
    
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        
        if 'm_hidden_corrs' in data and 'l_hidden_corrs' in data:
            mod_data = data['m_hidden_corrs']
            lat_data = data['l_hidden_corrs']
            
            # Trim to match epoch range
            if len(mod_data) > len(epoch_range):
                start_idx = start_epoch
                end_idx = start_idx + len(epoch_range)
                mod_trimmed = mod_data[start_idx:end_idx]
                lat_trimmed = lat_data[start_idx:end_idx]
            else:
                mod_trimmed = mod_data
                lat_trimmed = lat_data
            
            # Calculate mod-lat difference for this model
            model_diffs = np.array(mod_trimmed) - np.array(lat_trimmed)
            mod_lat_diffs_per_model.append(model_diffs)
    
    if not mod_lat_diffs_per_model:
        print("Error: Could not load modular and lattice correlation data")
        return
    
    # Convert to array for easier manipulation: [n_models, n_epochs]
    mod_lat_diffs_array = np.array(mod_lat_diffs_per_model)
    
    # Collect 3:3 ratio data per epoch per model per set
    # Structure: {set_name: {epoch_idx: [model1_mod_pref, model2_mod_pref, ...]}}
    set_epoch_data = {set_name: {epoch_idx: [] for epoch_idx in range(len(epoch_range))} 
                      for set_name in all_set_names}
    
    # Second pass: collect data for each file
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        
        if test_data_key not in data:
            continue
            
        test_data = data[test_data_key]
        
        for i, epoch_idx in enumerate(epoch_range):
            if epoch_idx >= len(test_data):
                continue
                
            epoch_data = test_data[epoch_idx]
            
            if '3:3' not in epoch_data:
                continue
                
            ratio_data = epoch_data['3:3']
            
            for set_name in all_set_names:
                if set_name in ratio_data:
                    set_data = ratio_data[set_name]
                    
                    mod_correlations = np.array(set_data["mod"])
                    lat_correlations = np.array(set_data["lat"])
                    
                    # Calculate preference rate per trial
                    avg_mod_corr_per_trial = np.mean(mod_correlations, axis=1)
                    avg_lat_corr_per_trial = np.mean(lat_correlations, axis=1)
                    
                    mod_preferred_trials = np.sum(avg_mod_corr_per_trial > avg_lat_corr_per_trial)
                    total_trials = len(avg_mod_corr_per_trial)
                    mod_preference_rate = mod_preferred_trials / total_trials if total_trials > 0 else 0
                    
                    set_epoch_data[set_name][i].append(mod_preference_rate)
    
    # Prepare data for scatterplot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map for different set names
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_map = {set_name: colors[i % len(colors)] for i, set_name in enumerate(sorted(all_set_names))}
    
    # For each set, plot points for each epoch
    for set_name in sorted(all_set_names):
        x_values = []  # mod-lat correlation values
        y_values = []  # mean %mod response values
        x_errors = []  # standard errors for x-axis (mod-lat correlation)
        y_errors = []  # standard errors for y-axis (%mod response)
        
        for i, epoch_idx in enumerate(epoch_range):
            if i < len(mod_lat_diffs_array[0]):
                # X-axis: mod-lat correlation difference
                x_vals_for_epoch = mod_lat_diffs_array[:, i]  # All models for this epoch
                x_val = np.mean(x_vals_for_epoch)
                x_err = np.std(x_vals_for_epoch, ddof=1) / np.sqrt(len(x_vals_for_epoch)) if len(x_vals_for_epoch) > 1 else 0
                
                # Y-axis: %mod response
                model_responses = set_epoch_data[set_name][i]
                
                if model_responses:  # If we have data for this epoch/set combination
                    y_val = np.mean(model_responses)  # Mean across models
                    y_err = np.std(model_responses, ddof=1) / np.sqrt(len(model_responses)) if len(model_responses) > 1 else 0
                    
                    x_values.append(x_val)
                    y_values.append(y_val)
                    x_errors.append(x_err)
                    y_errors.append(y_err)
        
        if x_values and y_values:  # Only plot if we have data
            color = color_map[set_name]
            
            # Main scatter points
            ax.scatter(x_values, y_values, color=color, s=60, alpha=0.7, 
                      label=set_name.replace('_', ' ').title(), zorder=3)
            
            # Add solid cross for average across all models for this set
            avg_x = np.mean(x_values)
            avg_y = np.mean(y_values)
            ax.scatter(avg_x, avg_y, color=color, s=120, alpha=1.0, marker='+', 
                      linewidth=3, zorder=4)
            
            # Add confidence interval ellipses
            if ci_elipses:
                from matplotlib.patches import Ellipse
                for x, y, x_err, y_err in zip(x_values, y_values, x_errors, y_errors):
                    if x_err > 0 or y_err > 0:  # Only draw ellipse if we have some uncertainty
                        # Create ellipse with width=2*x_err, height=2*y_err (roughly 1 standard error)
                        # Scale by 1.96 for approximate 95% confidence interval
                        ellipse = Ellipse((x, y), width=2*1.96*x_err, height=2*1.96*y_err, 
                                        facecolor=color, alpha=0.2, edgecolor=color, linewidth=1)
                        ax.add_patch(ellipse)
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1, 
               label='Chance Level (50%)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1, 
               label='No Mod-Lat Difference')
    
    # Formatting
    ax.set_xlabel('Mod-Lat Correlation Difference', fontsize=12)
    ax.set_ylabel('% Modular Response (3:3 Trials)', fontsize=12)
    layer = 'Hidden' if hidden else 'Output'
    ax.set_title(f'3:3 Modular Response vs Mod-Lat Correlation - {layer}', fontsize=14)
    
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, 0.8)  # Fixed x-axis range
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Add solid crosses to show average x and y values for each set across all epochs
    for set_name in sorted(all_set_names):
        if set_name in set_epoch_data:
            avg_x = np.mean([mod_lat_diffs_array[i, j] for i in range(len(mod_lat_diffs_array)) 
                            for j in range(len(epoch_range)) if not np.isnan(set_epoch_data[set_name][j])])
            avg_y = np.mean([set_epoch_data[set_name][j] for j in range(len(epoch_range)) 
                            if not np.isnan(set_epoch_data[set_name][j])])
            
            ax.scatter(avg_x, avg_y, color='black', s=70, alpha=0.8, marker='+', linewidth=2)
    
    plt.tight_layout()
    
    layer_suffix = 'h' if hidden else 'o'
    epoch_suffix = f"_e{start_epoch}-{end_epoch}" if include_e0 else f"_e{start_epoch+1}-{end_epoch+1}"
    plt.savefig(f"{save_dir}/33_scatter_{layer_suffix}{epoch_suffix}.png", dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return {
        'set_names': all_set_names,
        'epoch_range': epoch_range,
        'mod_lat_diffs_array': mod_lat_diffs_array,
        'set_epoch_data': set_epoch_data
    }

def plot_scatter_models(data_dir, epoch, hidden=True, save_dir="Results/Analysis/Plots/scatter_models", show_plots=False, include_e0=False):
    os.makedirs(save_dir, exist_ok=True)
    
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    
    if not npz_files:
        print(f"No NPZ files found in directory: {data_dir}")
        return
    
    test_data_key = 'hidden_tests' if hidden else 'output_tests'
    
    # First pass: determine available set names
    all_set_names = set()
    
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        
        if test_data_key not in data:
            print(f"Warning: {test_data_key} not found in {npz_file}")
            continue
        
        test_data = data[test_data_key]
        
        # Check if epoch exists and has 3:3 data
        if epoch < len(test_data):
            epoch_data = test_data[epoch]
            if '3:3' in epoch_data:
                ratio_data = epoch_data['3:3']
                all_set_names.update(ratio_data.keys())
    
    # Sort set names for consistency
    all_set_names = sorted(list(all_set_names))
    
    print(f"Processing epoch {epoch}")
    print(f"Available set names: {sorted(all_set_names)}")
    
    # Collect mod-lat differences per model for the specified epoch
    mod_lat_diffs_per_model = []  # [model1_diff, model2_diff, ...]
    model_names = []  # Keep track of model filenames
    
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        
        if 'm_hidden_corrs' in data and 'l_hidden_corrs' in data:
            mod_data = data['m_hidden_corrs']
            lat_data = data['l_hidden_corrs']
            
            # Check if epoch exists in correlation data
            if epoch < len(mod_data) and epoch < len(lat_data):
                # Calculate mod-lat difference for this model at this epoch
                model_diff = mod_data[epoch] - lat_data[epoch]
                mod_lat_diffs_per_model.append(model_diff)
                model_names.append(os.path.basename(npz_file))
    
    if not mod_lat_diffs_per_model:
        print("Error: Could not load modular and lattice correlation data for the specified epoch")
        return
    
    # Collect 3:3 ratio data per model per set for the specified epoch
    # Structure: {set_name: [model1_mod_pref, model2_mod_pref, ...]}
    set_model_data = {set_name: [] for set_name in all_set_names}
    valid_models = []  # Track which models have valid 3:3 data
    
    # Process each file (model)
    for i, npz_file in enumerate(npz_files):
        data = np.load(npz_file, allow_pickle=True)
        
        if test_data_key not in data:
            continue
            
        test_data = data[test_data_key]
        
        # Check if epoch exists
        if epoch >= len(test_data):
            continue
            
        epoch_data = test_data[epoch]
        
        if '3:3' not in epoch_data:
            continue
            
        ratio_data = epoch_data['3:3']
        model_has_data = False
        
        for set_name in all_set_names:
            if set_name in ratio_data:
                set_data = ratio_data[set_name]
                
                mod_correlations = np.array(set_data["mod"])
                lat_correlations = np.array(set_data["lat"])
                
                # Calculate preference rate per trial
                avg_mod_corr_per_trial = np.mean(mod_correlations, axis=1)
                avg_lat_corr_per_trial = np.mean(lat_correlations, axis=1)
                
                mod_preferred_trials = np.sum(avg_mod_corr_per_trial > avg_lat_corr_per_trial)
                total_trials = len(avg_mod_corr_per_trial)
                mod_preference_rate = mod_preferred_trials / total_trials if total_trials > 0 else 0
                
                set_model_data[set_name].append(mod_preference_rate)
                model_has_data = True
            else:
                set_model_data[set_name].append(np.nan)
        
        if model_has_data:
            valid_models.append(i)
    
    # Filter mod_lat_diffs to only include models with valid 3:3 data
    if len(valid_models) != len(mod_lat_diffs_per_model):
        filtered_diffs = [mod_lat_diffs_per_model[i] for i in valid_models if i < len(mod_lat_diffs_per_model)]
        filtered_model_names = [model_names[i] for i in valid_models if i < len(model_names)]
    else:
        filtered_diffs = mod_lat_diffs_per_model
        filtered_model_names = model_names
    
    # Prepare data for scatterplot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map for different set names
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_map = {set_name: colors[i % len(colors)] for i, set_name in enumerate(sorted(all_set_names))}
    
    # For each set, plot points for each model
    for set_name in sorted(all_set_names):
        x_values = []  # mod-lat correlation values
        y_values = []  # %mod response values
        
        for i, model_diff in enumerate(filtered_diffs):
            if i < len(set_model_data[set_name]):
                mod_pref_rate = set_model_data[set_name][i]
                
                # Only include if we have valid data (not NaN)
                if not np.isnan(mod_pref_rate):
                    x_values.append(model_diff)
                    y_values.append(mod_pref_rate)
        
        if x_values and y_values:  # Only plot if we have data
            color = color_map[set_name]
            
            # Main scatter points
            ax.scatter(x_values, y_values, color=color, s=60, alpha=0.7, 
                      label=set_name.replace('_', ' ').title(), zorder=3)
            
            # Add solid cross for average across all models for this set
            avg_x = np.mean(x_values)
            avg_y = np.mean(y_values)
            ax.scatter(avg_x, avg_y, color=color, s=120, alpha=1.0, marker='+', 
                      linewidth=3, zorder=4)
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1, 
               label='Chance Level (50%)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1, 
               label='No Mod-Lat Difference')
    
    # Formatting
    ax.set_xlabel('Mod-Lat Correlation Difference', fontsize=12)
    ax.set_ylabel('% Modular Response (3:3 Trials)', fontsize=12)
    layer = 'Hidden' if hidden else 'Output'
    epoch_display = epoch if include_e0 else epoch + 1
    ax.set_title(f'3:3 Modular Response vs Mod-Lat Correlation by Model - {layer} (Epoch {epoch_display})', fontsize=14)
    
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    layer_suffix = 'h' if hidden else 'o'
    epoch_suffix = f"_e{epoch}" if include_e0 else f"_e{epoch+1}"
    plt.savefig(f"{save_dir}/scatter_{layer_suffix}{epoch_suffix}.png", dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return {
        'set_names': all_set_names,
        'epoch': epoch,
        'mod_lat_diffs': filtered_diffs,
        'set_model_data': set_model_data,
        'model_names': filtered_model_names
    }
