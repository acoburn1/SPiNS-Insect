from scipy.stats import t, ttest_rel
import numpy as np
import os
import glob

class StatsProducer:
    def __init__(self, data_parameters=None, ci=0.95):
        if data_parameters:
            self.data_parameters = {k: v for k, v in data_parameters.items() if k not in ["hidden_matrices", "output_matrices"]}
        else:
            self.data_parameters = {}
        self.ci = ci
    
    def get_stats(self, data_dir):
        npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
        agg_stats_objects = {}
        
        for param in self.data_parameters.keys():
            if self.data_parameters[param]:
                param_data_list = []
                for npz_file in npz_files:
                    data = np.load(npz_file, allow_pickle=True)
                    if param in data and data[param] is not None:
                        param_data_list.append(data[param])
                
                if param_data_list:
                    data_array = np.array(param_data_list)
                    agg_stats_objects[param] = AggregateStatsObject(self._get_epoch_stats(data_array))
        
        return agg_stats_objects
    
    def get_difference_stats_with_ttest(self, modular_data, lattice_data, alpha=0.05):

        if modular_data.shape != lattice_data.shape:
            raise ValueError(f"Modular and lattice data must have same shape. Got {modular_data.shape} vs {lattice_data.shape}")
        
        difference_array = modular_data - lattice_data
        agg_stats_obj = AggregateStatsObject(self._get_epoch_stats(difference_array))
        
        # Perform paired t-test for each epoch
        num_epochs = modular_data.shape[1]
        significant = np.zeros(num_epochs, dtype=bool)
        p_values = np.zeros(num_epochs)
        
        for epoch in range(num_epochs):
            mod_epoch = modular_data[:, epoch]
            lat_epoch = lattice_data[:, epoch]
            
            # Paired t-test: H0: modular_mean = lattice_mean
            t_stat, p_value = ttest_rel(mod_epoch, lat_epoch)
            p_values[epoch] = p_value
            significant[epoch] = p_value < alpha
        
        return {
            'stats': agg_stats_obj,
            'significant': significant,
            'p_values': p_values
        }
    
    def _get_epoch_stats(self, data_array):
        stats_objects = []
        num_epochs = data_array.shape[1]
        
        for epoch in range(num_epochs):
            epoch_data = data_array[:, epoch]
            stats = self._get_stats_object(epoch_data)
            stats_objects.append(stats)
        
        return stats_objects

    def _get_stats_object(self, data):
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)
        std_err = std_dev / np.sqrt(data.shape[0])
        t_score = t.ppf((1 + self.ci) / 2, df=data.shape[0] - 1)
        ci_lower = mean - t_score * std_err
        ci_upper = mean + t_score * std_err
        return StatsObject(mean, std_dev, std_err, ci_lower, ci_upper)

class StatsObject:
    def __init__(self, mean, std_dev, std_err, ci_lower, ci_upper):
        self.mean = mean
        self.std_dev = std_dev
        self.std_err = std_err
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        
class AggregateStatsObject:
    def __init__(self, stats_objects):
        self.means = []
        self.std_devs = []
        self.std_errs = []
        self.ci_lowers = []
        self.ci_uppers = []
        
        for stat_object in stats_objects:
            self.means.append(stat_object.mean)
            self.std_devs.append(stat_object.std_dev)
            self.std_errs.append(stat_object.std_err)
            self.ci_lowers.append(stat_object.ci_lower)
            self.ci_uppers.append(stat_object.ci_upper)
