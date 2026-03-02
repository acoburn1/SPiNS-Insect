from scipy.stats import t, ttest_rel
import numpy as np
import os
import glob

class StatsObject:
    def __init__(self, data: list, ci=0.95):
        self.mean = np.mean(data)
        self.std_dev = np.std(data, ddof=1)
        self.std_err = self.std_dev / np.sqrt(data.shape[0])
        t_score = t.ppf((1 + ci) / 2, df=data.shape[0] - 1)
        self.ci_lower = self.mean - t_score * self.std_err
        self.ci_upper = self.mean + t_score * self.std_err
        
class AggregateStatsObject:
    def __init__(self, stats_objects: list[StatsObject]):
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
