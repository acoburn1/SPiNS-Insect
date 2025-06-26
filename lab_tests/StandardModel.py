from NeuralNetwork import NeuralNetwork
import torch
import Evaluation

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

    def train(self, dataloader, modular_reference_matrix, lattice_reference_matrix, print_data=False):
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

            m_avg, mpm = Evaluation.test_and_compare_modular(self.model, self.num_features, modular_reference_matrix)
            l_avg, lpm = Evaluation.test_and_compare_lattice(self.model, self.num_features, lattice_reference_matrix)
            gpm = Evaluation.generate_distributions(self.model, 2*self.num_features)

            m_avgs.append(m_avg)
            l_avgs.append(l_avg)
            mpms.append(mpm)
            lpms.append(lpm)
            gpms.append(gpm)
            losses.append(total_loss)

            if (print_data):
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss:.7f}, Average row-wise JS similarity: {avg:.3f}")
        return losses, m_avgs, l_avgs, mpms, lpms, gpms

    def test_model(self, raw_inputs):
        test_inputs = torch.tensor(raw_inputs, dtype=torch.float32)
        test_outputs = torch.sigmoid(self.model(test_inputs))
        return test_outputs.detach().numpy()

