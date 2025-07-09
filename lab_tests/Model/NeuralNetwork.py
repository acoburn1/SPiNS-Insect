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
        self.hidden_activations = None
        self._hook_handle = None

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def register_hooks(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
        def hook_fn(module, input, output):
            self.hidden_activations = output.detach()
        self._hook_handle = self.encoder.register_forward_hook(hook_fn)
    
    def get_hidden_activations(self, x):
        self.register_hooks()
        with torch.no_grad():
            _ = self.forward(x)
        return self.hidden_activations




