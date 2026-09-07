import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, num_features, size_hidden_layer, relu: bool = True, initialization: str = "default"):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2*num_features, size_hidden_layer),
            nn.ReLU() if relu else nn.Identity(),
        ) 
        self.decoder = nn.Sequential(
            nn.Linear(size_hidden_layer, 2*num_features)
        )

        if initialization == "xavier":
            for layer in (self.encoder[0], self.decoder[0]):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        elif initialization != "default":
            raise ValueError(f"Unsupported initialization: {initialization}")

    def forward(self, x, return_hidden=False):
        h = self.encoder(x)
        y = self.decoder(h)
        if return_hidden:
            return h, y
        return y


