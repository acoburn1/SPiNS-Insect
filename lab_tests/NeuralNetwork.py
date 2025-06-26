import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, num_features, size_hidden_layer):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2*num_features, size_hidden_layer),
            nn.ReLU(),
            nn.Linear(size_hidden_layer, size_hidden_layer),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(size_hidden_layer, 2*num_features)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))




