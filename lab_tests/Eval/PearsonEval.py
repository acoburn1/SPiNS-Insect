from scipy.stats import pearsonr
import torch
import numpy as np


def test_and_compare_modular(model, num_features, reference_matrix, hidden=False):     
    output_matrix = generate_output_distributions(model, 2*num_features) if hidden == False else generate_hidden_distributions(model, 2*num_features)                                                                                         
    return flatten_and_eval(output_matrix[:num_features, :num_features], reference_matrix, hidden)   
def test_and_compare_lattice(model, num_features, reference_matrix, hidden=False): 
    output_matrix = generate_output_distributions(model, 2*num_features) if hidden == False else generate_hidden_distributions(model, 2*num_features)                                                                                             
    return flatten_and_eval(output_matrix[num_features:, num_features:], reference_matrix, hidden)      
def flatten_and_eval(m1, m2, triangle: bool=False):
    corr, p = pearsonr(cut(m1, triangle), cut(m2, triangle))
    return corr
def cut(m, triangle: bool):
        m_cut = []
        for i in range(len(m)):
            for j in range(len(m[0])):
                if (i < j if triangle else i != j):
                    m_cut.append(m[i][j])
        return m_cut
def generate_output_distributions(model, num_features):
    inputs = torch.eye(num_features, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(inputs))
def generate_hidden_distributions(model, num_features):
    correlations = []
    hidden_activations = generate_hidden_activations(model, num_features)
    for ha_base in hidden_activations:
        base_correlations = []
        for ha_comp in hidden_activations:
            corr, p = pearsonr(ha_base, ha_comp)
            base_correlations.append(corr)
        correlations.append(base_correlations)
    return np.array(correlations)
def generate_hidden_activations(model, num_features):
    inputs = torch.eye(num_features, dtype=torch.float32)
    return model.get_hidden_activations(inputs).numpy()

