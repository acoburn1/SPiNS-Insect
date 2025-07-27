import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.widgets import Slider
import Output.RatioExemplarOutput as REO
from Statistics.StatsProducer import StatsProducer


class CorrelationAnalyzer:
    def __init__(self, npz_filepath, include_e0=False):
        self.data = np.load(npz_filepath, allow_pickle=True)
        self.losses = self.data['losses']
        self.m_output_corrs = self.data['m_output_corrs']
        self.l_output_corrs = self.data['l_output_corrs']
        self.m_hidden_corrs = self.data['m_hidden_corrs']
        self.l_hidden_corrs = self.data['l_hidden_corrs']
        self.output_matrices = self.data['output_matrices']
        self.hidden_matrices = self.data['hidden_matrices']
        self.output_tests = self.data['output_tests']
        self.hidden_tests = self.data['hidden_tests']
        self.include_e0 = include_e0
        
    def find_highest_correlations(self):
        results = {
            'modular_output': {
                'max_corr': np.max(self.m_output_corrs),
                'epoch': np.argmax(self.m_output_corrs),
                'matrix': self.output_matrices[np.argmax(self.m_output_corrs)]
            },
            'lattice_output': {
                'max_corr': np.max(self.l_output_corrs),
                'epoch': np.argmax(self.l_output_corrs),
                'matrix': self.output_matrices[np.argmax(self.l_output_corrs)]
            },
            'modular_hidden': {
                'max_corr': np.max(self.m_hidden_corrs),
                'epoch': np.argmax(self.m_hidden_corrs),
                'matrix': self.hidden_matrices[np.argmax(self.m_hidden_corrs)]
            },
            'lattice_hidden': {
                'max_corr': np.max(self.l_hidden_corrs),
                'epoch': np.argmax(self.l_hidden_corrs),
                'matrix': self.hidden_matrices[np.argmax(self.l_hidden_corrs)]
            }
        }
        return results
    
    def analyze_test_accuracy(self, epoch=None):
        if epoch is None:
            epoch = len(self.output_tests) - 1
        elif epoch < 0 or epoch >= len(self.output_tests):
            print(f"Invalid epoch {epoch}. Valid range: 0 to {len(self.output_tests) - 1}")
            return None, None
            
        output_accuracy = self._calculate_accuracy(self.output_tests[epoch])
        hidden_accuracy = self._calculate_accuracy(self.hidden_tests[epoch])
        
        print(f"Output Layer Test Accuracy (Epoch {epoch if self.include_e0 else epoch+1}): {output_accuracy:.2%}")
        print(f"Hidden Layer Test Accuracy (Epoch {epoch if self.include_e0 else epoch+1}): {hidden_accuracy:.2%}")
        
        return output_accuracy, hidden_accuracy
    
    def _calculate_accuracy(self, test_data):
        target_labels = ['all-mod', 'mod-heavy', 'all-lat', 'lat-heavy']
        total_correct = 0
        total_trials = 0
        
        for label in target_labels:
            if label not in test_data:
                continue
                
            data = test_data[label]
            mod_correlations = np.array(data["mod"])
            lat_correlations = np.array(data["lat"])
            
            avg_mod_corr_per_trial = np.mean(mod_correlations, axis=1)
            avg_lat_corr_per_trial = np.mean(lat_correlations, axis=1)
            
            expected_category = 'mod' if 'mod' in label else 'lat'
            
            for i in range(len(avg_mod_corr_per_trial)):
                predicted_category = 'mod' if avg_mod_corr_per_trial[i] > avg_lat_corr_per_trial[i] else 'lat'
                
                if predicted_category == expected_category:
                    total_correct += 1
                else:
                    print(f"Incorrect: {label} trial {i+1} - Mod: {avg_mod_corr_per_trial[i]:.3f}, Lat: {avg_lat_corr_per_trial[i]:.3f}")
                
                total_trials += 1
        
        return total_correct / total_trials if total_trials > 0 else 0
    
    def plot_correlation_trends(self, learning_rate, save_dir="Results/Plots", show_plots=True):
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        epochs = range(len(self.losses)) if self.include_e0 else range(1, len(self.losses) + 1)
        max_epoch = len(epochs)
        
        # Create the plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Plot correlation lines on primary y-axis
        line1 = ax1.plot(epochs, self.m_output_corrs, 'b-', linewidth=2, label='Modular Output', marker='o', markersize=4)
        line2 = ax1.plot(epochs, self.l_output_corrs, 'r-', linewidth=2, label='Lattice Output', marker='s', markersize=4)
        line3 = ax1.plot(epochs, self.m_hidden_corrs, 'g--', linewidth=2, label='Modular Hidden', marker='^', markersize=4)
        line4 = ax1.plot(epochs, self.l_hidden_corrs, 'm--', linewidth=2, label='Lattice Hidden', marker='v', markersize=4)
        
        # Set up primary y-axis (correlations) with fixed scale
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Pearson Correlation', fontsize=12, color='black')
        ax1.set_ylim(0, 1)  # Fixed scale for correlations
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)
        
        # Create secondary y-axis for loss with fixed scale
        ax2 = ax1.twinx()
        line5 = ax2.plot(epochs, self.losses, 'k-', linewidth=2, label='Loss', marker='d', markersize=3)
        ax2.set_ylabel('Loss', fontsize=12, color='black')
        ax2.set_ylim(0, 2)  # Fixed scale for loss
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Set up sophisticated x-axis ticking
        # Major ticks every 10 epochs with labels
        major_ticks = list(range(10, max_epoch + 1, 10))
        if 1 not in major_ticks:
            major_ticks = [1] + major_ticks
        if max_epoch not in major_ticks and max_epoch % 10 != 0:
            major_ticks.append(max_epoch)
        
        # Medium ticks every 5 epochs (but not on major ticks)
        medium_ticks = [x for x in range(5, max_epoch + 1, 5) if x not in major_ticks]
        
        # Minor ticks for all other epochs
        minor_ticks = [x for x in epochs if x not in major_ticks and x not in medium_ticks]
        
        # Apply ticks
        ax1.set_xticks(major_ticks)
        ax1.set_xticks(medium_ticks, minor=False)
        ax1.set_xticks(minor_ticks, minor=True)
        
        # Customize tick appearance
        ax1.tick_params(which='major', length=8, width=2, labelsize=10)
        ax1.tick_params(which='minor', length=4, width=1)
        
        # Add medium ticks manually with custom length
        for tick in medium_ticks:
            ax1.axvline(x=tick, ymin=0, ymax=0.02, color='black', linewidth=1.5, clip_on=False)
        
        ax1.set_xlim(1, max_epoch)
        
        # Combine legends from both axes
        lines = line1 + line2 + line3 + line4 + line5
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=8)
        
        plt.title(f'Correlation Trends and Loss Over Training Epochs (LR = .0{int(learning_rate*1000)})', fontsize=14)
        
        # Add annotations for peak correlation values (only if within visible range)
        peak_results = self.find_highest_correlations()
        for key, result in peak_results.items():
            epoch = result['epoch'] + 1  # Convert to 1-based indexing
            max_corr = result['max_corr']
            
            # Only annotate if the correlation is within the visible range
            if 0 <= max_corr <= 1:
                ax1.annotate(f'Max: {max_corr:.3f}', 
                            xy=(epoch, max_corr), 
                            xytext=(10, 10), 
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{save_dir}/ptrends_0{int(learning_rate*1000)}lr.png", dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()

    def plot_matrices_at_epoch(self, epoch, save_dir="Results/Plots", show_plots=True):
        os.makedirs(save_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        output_matrix = self.output_matrices[epoch]
        sns.heatmap(output_matrix, ax=ax1, annot=False, cmap="viridis", 
                   square=True, vmin=0, vmax=1, cbar_kws={'label': 'Correlation Value'})
        ax1.set_title(f'Output Layer Matrix - Epoch {epoch if self.include_e0 else epoch+1}', fontsize=14)
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        
        hidden_matrix = self.hidden_matrices[epoch]
        sns.heatmap(hidden_matrix, ax=ax2, annot=False, cmap="viridis", 
                   square=True, vmin=0, vmax=1, cbar_kws={'label': 'Correlation Value'})
        ax2.set_title(f'Hidden Layer Matrix - Epoch {epoch if self.include_e0 else epoch+1}', fontsize=14)
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        
        fig.suptitle(f'Correlation Matrices at Epoch {epoch if self.include_e0 else epoch+1}', 
                     fontsize=16, y=1.02)
        
        plt.tight_layout()
        
        plt.savefig(f"{save_dir}/matrices_epoch_{epoch}.png", dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def print_summary(self):
        """
        Print a summary of the highest correlation values and their epochs.
        """
        peak_results = self.find_highest_correlations()
        
        print("=== Correlation Analysis Summary ===\n")
        
        for key, result in peak_results.items():
            corr_type = key.replace('_', ' ').title()
            epoch = result['epoch'] + 1  # Convert to 1-based indexing
            max_corr = result['max_corr']
            loss_at_peak = self.losses[result['epoch']]
            
            print(f"{corr_type}:")
            print(f"  Highest Correlation: {max_corr:.4f}")
            print(f"  Occurred at Epoch: {epoch}")
            print(f"  Loss at Peak: {loss_at_peak:.6f}")
            print()
            