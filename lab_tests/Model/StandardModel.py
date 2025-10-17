from numpy import save
from Model.NeuralNetwork import NeuralNetwork
import torch
import os
import Eval.PearsonEval as PearsonEval
import Eval.Outdated.RowWiseJSEval as RowWiseJSEval
import Tests.RatioExemplar as RE

class StandardModel:
    def __init__(self, num_features, hidden_layer_size, batch_size, num_epochs, learning_rate, loss_fn, first_h=False):
        self.model = NeuralNetwork(num_features, hidden_layer_size, first_h)
        self.num_features = num_features
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

    def train(self, dataloader):
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_X, batch_Y in dataloader:
                pred = self.model(batch_X)
                loss = self.loss_fn(pred, batch_Y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

    def train_eval_test_P(self, dataloader, modular_reference_matrix, lattice_reference_matrix, data_params, print_data=False, include_e0=False):
        result_data = {
            "losses": [],
            "hidden_activations": [],
            "m_output_corrs": [],
            "l_output_corrs": [],
            "m_hidden_corrs": [],
            "l_hidden_corrs": [],
            "output_matrices": [],
            "hidden_matrices": [],
            "output_ratio_tests": [],
            "hidden_ratio_tests": [],
            "output_activation_exemplar_tests": [],
            "output_activation_onehot_tests": []
        }

        if include_e0:
            initial_results = self._evaluate_model(modular_reference_matrix, lattice_reference_matrix, data_params)
            
            for key in result_data.keys():
                if key == "losses":
                    result_data[key].append(0)
                else:
                    eval_key = key[:-1] if key.endswith('s') else key[:-1]
                    result_data[key].append(initial_results.get(eval_key, None))

        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_X, batch_Y in dataloader:
                pred = self.model(batch_X)
                loss = self.loss_fn(pred, batch_Y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            epoch_results = self._evaluate_model(modular_reference_matrix, lattice_reference_matrix, data_params)
            
            for key in result_data.keys():
                if key == "losses":
                    result_data[key].append(total_loss)
                else:
                    eval_key = key[:-1] if key.endswith('s') else key[:-1]
                    result_data[key].append(epoch_results.get(eval_key, None))

            if print_data and data_params.get("m_output_corrs", False) and data_params.get("l_output_corrs", False):
                m_corr = epoch_results.get("m_output_corr", "N/A")
                l_corr = epoch_results.get("l_output_corr", "N/A")
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss:.7f}, Pearson correlation (modular): {m_corr:.3f}, Pearson correlation (lattice): {l_corr:.3f}")

        if include_e0 and data_params.get("losses", False) and len(result_data["losses"]) > 1:
            result_data["losses"][0] = result_data["losses"][1]

        filtered_results = {}
        for key, value in result_data.items():
            if data_params.get(key, False):
                filtered_results[key] = value
        
        return filtered_results

    def _evaluate_model(self, modular_reference_matrix, lattice_reference_matrix, data_params):
        results = {}
        
        if data_params.get("hidden_activations", False):
            results["hidden_activation"] = PearsonEval.generate_hidden_activations(self.model, 2*self.num_features)

        if data_params.get("m_output_corrs", False):
            results["m_output_corr"] = PearsonEval.test_and_compare_modular(self.model, self.num_features, modular_reference_matrix, hidden=False)
        
        if data_params.get("l_output_corrs", False):
            results["l_output_corr"] = PearsonEval.test_and_compare_lattice(self.model, self.num_features, lattice_reference_matrix, hidden=False)
        
        if data_params.get("m_hidden_corrs", False):
            results["m_hidden_corr"] = PearsonEval.test_and_compare_modular(self.model, self.num_features, modular_reference_matrix, hidden=True)
        
        if data_params.get("l_hidden_corrs", False):
            results["l_hidden_corr"] = PearsonEval.test_and_compare_lattice(self.model, self.num_features, lattice_reference_matrix, hidden=True)

        if data_params.get("output_matrices", False):
            results["output_matrix"] = PearsonEval.generate_output_distributions(self.model, 2 * self.num_features)
        
        if data_params.get("hidden_matrices", False):
            results["hidden_matrix"] = PearsonEval.generate_hidden_distributions(self.model, 2 * self.num_features)

        if data_params.get("output_ratio_tests", False):
            results["output_ratio_test"] = RE.test_ratios(self.model, hidden=False)
        
        if data_params.get("hidden_ratio_tests", False):
            results["hidden_ratio_test"] = RE.test_ratios(self.model, hidden=True)
        
        if data_params.get("output_activation_exemplar_tests", False):
            results["output_activation_exemplar_test"] = RE.test_activations(self.model, self.num_features, one_hot=False)
        
        if data_params.get("output_activation_onehot_tests", False):
            results["output_activation_onehot_test"] = RE.test_activations(self.model, self.num_features, one_hot=True)
        
        return results

    def test_model(self, raw_inputs):
        test_inputs = torch.tensor(raw_inputs, dtype=torch.float32)
        test_outputs = torch.sigmoid(self.model(test_inputs))
        return test_outputs.detach().numpy()

    ### not in use

    def train_eval_JS(self, dataloader, modular_reference_matrix, lattice_reference_matrix, print_data=False):
        losses, m_avgs, l_avgs, mpms, lpms, gpms = [], [], [], [], [], []
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_X, batch_Y in dataloader:
                pred = self.model(batch_X)
                loss = self.loss_fn(pred, batch_Y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            m_avg, mpm = RowWiseJSEval.test_and_compare_modular(self.model, self.num_features, modular_reference_matrix)
            l_avg, lpm = RowWiseJSEval.test_and_compare_lattice(self.model, self.num_features, lattice_reference_matrix)
            gpm = RowWiseJSEval.generate_distributions(self.model, 2*self.num_features)

            m_avgs.append(m_avg)
            l_avgs.append(l_avg)
            mpms.append(mpm)
            lpms.append(lpm)
            gpms.append(gpm)
            losses.append(total_loss)

            if (print_data):
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss:.7f}, Average row-wise JS similarity (modular): {m_avg:.3f}, Average row-wise JS similarity (lattice): {l_avg:.3f}")
        return losses, m_avgs, l_avgs, mpms, lpms, gpms

