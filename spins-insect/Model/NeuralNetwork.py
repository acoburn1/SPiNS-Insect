import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, num_features, size_hidden_layer, relu: bool = True):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2*num_features, size_hidden_layer),
            nn.ReLU() if relu else nn.Identity(),
        ) 
        self.decoder = nn.Sequential(
            nn.Linear(size_hidden_layer, 2*num_features)
        )

    def forward(self, x, return_hidden=False):
        h = self.encoder(x)
        y = self.decoder(h)
        if return_hidden:
            return h, y
        return y



