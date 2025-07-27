from scipy.stats import t
import numpy as np
import os
import glob

class StatsProducer:
    def __init__(self, data_parameters, ci=0.95):
        self.data_parameters = {k: v for k, v in data_parameters.items() if k not in ["hidden_matrices", "output_matrices"]}
        self.ci = ci
    
    def get_stats(self, data_dir):
        npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
        stats_objects = {}
        
        for param in self.data_parameters.keys():
            if self.data_parameters[param]:
                param_data_list = []
                for npz_file in npz_files:
                    data = np.load(npz_file, allow_pickle=True)
                    if param in data and data[param] is not None:
                        param_data_list.append(data[param])
                
                if param_data_list:
                    data_array = np.array(param_data_list)
                    stats_objects[param] = self._get_epoch_stats(data_array)
        
        return stats_objects
    
    def _get_epoch_stats(self, data_array):
        num_epochs = data_array.shape[1]
        means = []
        std_devs = []
        std_errs = []
        ci_lowers = []
        ci_uppers = []
        
        for epoch in range(num_epochs):
            epoch_data = data_array[:, epoch]
            stats = self._get_stats_object(epoch_data)
            means.append(stats.means)
            std_devs.append(stats.std_devs)
            std_errs.append(stats.std_errs)
            ci_lowers.append(stats.ci_lowers)
            ci_uppers.append(stats.ci_uppers)
        
        return StatsObject(means, std_devs, std_errs, ci_lowers, ci_uppers)

    def _get_stats_object(self, data):
        mean = np.mean(data)
        std_devs = np.std(data, ddof=1)
        std_errs = std_devs / np.sqrt(data.shape[0])
        t_score = t.ppf((1 + self.ci) / 2, df=data.shape[0] - 1)
        ci_lowers = mean - t_score * std_errs
        ci_uppers = mean + t_score * std_errs
        return StatsObject(mean, std_devs, std_errs, ci_lowers, ci_uppers)


class StatsObject:
    def __init__(self, means, std_devs, std_errs, ci_lowers, ci_uppers):
        self.means = means
        self.std_devs = std_devs
        self.std_errs = std_errs
        self.ci_lowers = ci_lowers
        self.ci_uppers = ci_uppers
