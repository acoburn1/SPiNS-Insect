from scipy.spatial.distance import jensenshannon
import torch
import numpy

def test_and_compare_modular(model, num_features, reference_matrix):                                                                                 
    output_matrix = generate_distributions(model, 2*num_features)                                                                                    
    return eval_matrix_average_similarity(output_matrix[:num_features, :num_features], reference_matrix), output_matrix[:num_features, :num_features]
def test_and_compare_lattice(model, num_features, reference_matrix):                                                                                 
    output_matrix = generate_distributions(model, 2*num_features)                                                                                    
    return eval_matrix_average_similarity(output_matrix[num_features:, num_features:], reference_matrix), output_matrix[num_features:, num_features:]
def eval_matrix_average_similarity(m1, m2):
    avg, distances = eval_matrix_average_js(m1, m2)
    similarities = []
    for d in distances:
        similarities.append(1/(1+d))
    return 1 / (1+avg) #, similarities
def eval_matrix_average_js(m1, m2):
    distances = []
    for i in range(len(m1)):
        distances.append(eval_row_js_distance(m1[i], m2[i]))
    return sum(distances) / len(distances), distances
def eval_row_js_distance(row1, row2):
    n_row1 = row_normalize(row1)
    n_row2 = row_normalize(row2)
    return jensenshannon(n_row1, n_row2)
def matrix_row_normalize(matrix):
    for i in range(len(matrix)):
        matrix[i] = row_normalize(matrix[i])
    return matrix
def row_normalize(row):
    return row / sum(row) if sum(row) > 0 else row
def generate_distributions(model, num_features):
    inputs = torch.eye(num_features, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(inputs))
