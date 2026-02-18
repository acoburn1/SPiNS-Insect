from numpy import save
from scipy import special
from Model.NeuralNetwork import NeuralNetwork
import torch
import os
import Eval.PearsonEval as PearsonEval
import Tests.RatioExemplar as RE
import DataHelper.SpecialDataLoader as SDL
from functools import partial

class StandardModel:
    def __init__(self, num_features, hidden_layer_size, batch_size, num_epochs, learning_rate, loss_fn, first_h=False, device=None):
        self.device = device or torch.device("cpu")
        self.model = NeuralNetwork(num_features, hidden_layer_size, first_h).to(self.device)
        self.num_features = num_features
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

    def train_eval(self, dataloader, X_probe, include_e0=False, alt=False):
        results = { "losses": [], "hidden": [], "output": []}
        Xp = X_probe.to(self.device)

        def _probe():
            self.model.eval()
            with torch.no_grad():
                hid, out = self.model(Xp, return_hidden=True)
            return hid.cpu(), out.cpu()

        if include_e0:
            hid, out = _probe()
            results["losses"].append(0)
            results["hidden"].append(hid)
            results["output"].append(out)

        special_dl = isinstance(dataloader, SDL.SpecialDataLoader)

        if special_dl:
            dataloader.reset_appearances()
        for epoch in range(self.num_epochs):
            total_loss = 0
            if special_dl:
                epoch_loader = dataloader.get_special_dataloader()
            else:
                epoch_loader = dataloader
            for batch_X, batch_Y in epoch_loader:
                #flat = batch_X.cpu().numpy()       # uncomment to debug batch data
                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)
                pred = self.model(batch_X)
                loss = self.loss_fn(pred, batch_Y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            hid, out = _probe()
            results["losses"].append(total_loss)
            results["hidden"].append(hid)
            results["output"].append(out)

        if include_e0 and len(results["losses"]) > 1:
            results["losses"][0] = results["losses"][1]
        
        return results

