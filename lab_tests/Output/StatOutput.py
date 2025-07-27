import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from Statistics.StatsProducer import StatsProducer

def plot_stats_with_confidence_intervals(lr_str, data_dir, data_parameters, save_dir="Results/Plots/WithStats", show_plots=False, include_e0=False):
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
                label_data = test_data[ratio]
                mod_correlations = np.array(label_data["mod"])
                lat_correlations = np.array(label_data["lat"])
                
                avg_mod_corr_per_trial = np.mean(mod_correlations, axis=1)
                avg_lat_corr_per_trial = np.mean(lat_correlations, axis=1)
                
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