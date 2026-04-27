from numpy import save
from scipy import special
from Model.NeuralNetwork import NeuralNetwork
import torch
import os
import sys
import time
import DataHelper.SpecialDataLoader as SDL
from functools import partial
import DriverUtils.Visual as Visual

class StandardModel:
    def __init__(self, num_features, hidden_layer_size, batch_size, num_epochs, learning_rate, loss_fn, adam: bool = True, device=None):
        self.device = device or torch.device("cpu")
        self.model = NeuralNetwork(num_features, hidden_layer_size).to(self.device)
        self.num_features = num_features
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate) if adam else torch.optim.SGD(self.model.parameters(), self.learning_rate)

    def train_eval(self, dataloader, X_probe, vis=None):
        results = { "losses": [], "hidden": [], "output": []}
        Xp = X_probe.to(self.device)

        def _probe():       # note: no train/eval mode switch since no dropout or batchnorm 
            self.model.eval()
            with torch.no_grad():
                hid, out = self.model(Xp, return_hidden=True)
            return hid.cpu(), torch.sigmoid(out).cpu()

        # add probe data from initialization
        hid, out = _probe()
        results["losses"].append(0)
        results["hidden"].append(hid)
        results["output"].append(out)

        special_dl = isinstance(dataloader, SDL.SpecialDataLoader)
        if special_dl:
            dataloader.reset_appearances()

        t0 = time.time()
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
            
            if vis is not None:
                vis.progress_line(epoch_i=epoch, loss=total_loss)

        if len(results["losses"]) > 1:
            results["losses"][0] = results["losses"][1]
        
        return results

