import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
import numpy as np


class SpecialDataLoader:
    def __init__(self, training_inputs, training_outputs, num_mod_features):
        self.modin_tensor = torch.tensor(training_inputs[:num_mod_features], dtype=torch.float32)
        self.modout_tensor = torch.tensor(training_outputs[:num_mod_features], dtype=torch.float32)
        self.latin_tensor = torch.tensor(training_inputs[num_mod_features:], dtype=torch.float32)
        self.latout_tensor = torch.tensor(training_outputs[num_mod_features:], dtype=torch.float32)

        self.n_lat = self.latin_tensor.size(0)
        self.n_mod = self.modin_tensor.size(0)

        if self.n_lat <= self.n_mod:
            self.base = "lat"
            self.base_in = self.latin_tensor
            self.base_out = self.latout_tensor
            self.sample_in = self.modin_tensor
            self.sample_out = self.modout_tensor
        else:
            self.base = "mod"
            self.base_in = self.modin_tensor
            self.base_out = self.modout_tensor
            self.sample_in = self.latin_tensor
            self.sample_out = self.latout_tensor

        self.n_base = self.base_in.size(0)
        self.n_sample = self.sample_in.size(0)

        self.appearances = torch.zeros(self.n_sample, dtype=torch.float32)

    def get_special_dataloader(self) -> TorchDataLoader:
        weights = 1.0 / (self.appearances + 1)
        probs = weights / weights.sum()

        idx = torch.multinomial(probs, num_samples=self.n_sample, replacement=False)[:self.n_base]
        self.appearances[idx] += 1

        sampled_in = self.sample_in[idx]
        sampled_out = self.sample_out[idx]

        X = torch.cat([sampled_in, self.base_in], dim=0)
        Y = torch.cat([sampled_out, self.base_out], dim=0)

        dataset = TensorDataset(X, Y)
        return TorchDataLoader(dataset, self.n_base * 2, shuffle=True)

    def reset_appearances(self):
        self.appearances.zero_()