from NeuralNetwork import NeuralNetwork
import torch
import PearsonEval
import RowWiseJSEval

class StandardModel:
    def __init__(self, num_features, hidden_layer_size, batch_size, num_epochs, learning_rate, loss_fn):
        self.model = NeuralNetwork(num_features, hidden_layer_size)
        self.num_features = num_features
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

    def train_eval_P(self, dataloader, modular_reference_matrix, lattice_reference_matrix, print_data=False):
        losses, m_output_corrs, l_output_corrs, m_hidden_corrs, l_hidden_corrs = [], [], [], [], []
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_X, batch_Y in dataloader:
                pred = self.model(batch_X)
                loss = self.loss_fn(pred, batch_Y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            m_output_corr = PearsonEval.test_and_compare_modular(self.model, self.num_features, modular_reference_matrix, hidden=False)
            l_output_corr = PearsonEval.test_and_compare_lattice(self.model, self.num_features, lattice_reference_matrix, hidden=False)
            m_hidden_corr = PearsonEval.test_and_compare_modular(self.model, self.num_features, modular_reference_matrix, hidden=True)
            l_hidden_corr = PearsonEval.test_and_compare_lattice(self.model, self.num_features, lattice_reference_matrix, hidden=True)

            m_output_corrs.append(m_output_corr)
            l_output_corrs.append(l_output_corr)
            m_hidden_corrs.append(m_hidden_corr)
            l_hidden_corrs.append(l_hidden_corr)
            losses.append(total_loss)

            if (print_data):
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss:.7f}, Pearson correlation (modular): {m_output_corr:.3f}, Pearson correlation (lattice): {l_output_corr:.3f}")
        return losses, m_output_corrs, l_output_corrs, m_hidden_corrs, l_hidden_corrs
    

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
    

