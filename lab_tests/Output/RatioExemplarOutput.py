import numpy as np
from typing import Dict, List, Any

def print_ratio_results(results: Dict[str, Dict[str, Dict[str, List[List[float]]]]], result_type: str = "Output"):
    """
    Print the ratio trial results in a readable format.
    
    Args:
        results: Dictionary from test_ratios function with structure:
                {ratio: {set_name: {"mod": [[correlations]], "lat": [[correlations]]}}}
        result_type: String describing the type of results (e.g., "Output", "Hidden")
    """
    print(f"\n{'='*60}")
    print(f"{result_type.upper()} LAYER RATIO TRIAL RESULTS")
    print(f"{'='*60}")
    
    for ratio, ratio_data in results.items():
        print(f"\nRatio: {ratio}")
        print("=" * 40)
        
        for set_name, data in ratio_data.items():
            print(f"\nSet: {set_name}")
            print("-" * 40)
            
            num_trials = len(data["mod"])
            print(f"Number of trials: {num_trials}")
            
            # Calculate summary statistics
            mod_correlations = np.array(data["mod"])  # Shape: (num_trials, 8)
            lat_correlations = np.array(data["lat"])  # Shape: (num_trials, 8)
            
            # Average correlations across exemplars for each trial
            avg_mod_corr_per_trial = np.mean(mod_correlations, axis=1)
            avg_lat_corr_per_trial = np.mean(lat_correlations, axis=1)
            
            # Overall statistics
            print(f"Average correlation with modular exemplars: {np.mean(avg_mod_corr_per_trial):.4f} (+/- {np.std(avg_mod_corr_per_trial):.4f})")
            print(f"Average correlation with lattice exemplars: {np.mean(avg_lat_corr_per_trial):.4f} (+/- {np.std(avg_lat_corr_per_trial):.4f})")
            
            # Show which category this label correlates with more strongly
            overall_mod_avg = np.mean(avg_mod_corr_per_trial)
            overall_lat_avg = np.mean(avg_lat_corr_per_trial)
            preferred_category = "Modular" if overall_mod_avg > overall_lat_avg else "Lattice"
            difference = abs(overall_mod_avg - overall_lat_avg)
            print(f"Preferred category: {preferred_category} (difference: {difference:.4f})")
            
            # Show first few trials in detail if there are multiple trials
            if num_trials > 1:
                show_trials = min(3, num_trials)
                print(f"\nFirst {show_trials} trials (correlation with each exemplar):")
                for trial_idx in range(show_trials):
                    print(f"  Trial {trial_idx + 1}:")
                    print(f"    Modular: {[f'{corr:.3f}' for corr in mod_correlations[trial_idx]]}")
                    print(f"    Lattice: {[f'{corr:.3f}' for corr in lat_correlations[trial_idx]]}")
            else:
                print(f"\nDetailed correlations:")
                print(f"  Modular: {[f'{corr:.3f}' for corr in mod_correlations[0]]}")
                print(f"  Lattice: {[f'{corr:.3f}' for corr in lat_correlations[0]]}")
