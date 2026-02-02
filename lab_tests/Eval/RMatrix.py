import numpy as np

def generate_reference_matrices(training_data, num_mod_samples, method='jaccard', show=False):
    td = np.array(training_data)
    mod_rm = generate_reference_matrix_jaccard(td[:num_mod_samples,:11])
    lat_rm = generate_reference_matrix_jaccard(td[num_mod_samples:,11:])
    if method == 'cosine':
        mod_rm = generate_reference_matrix_cosine(td[:num_mod_samples,:11])
        lat_rm = generate_reference_matrix_cosine(td[num_mod_samples:,11:])
    if show:
        print("Modular Reference Matrix:\n")
        print_matrix(mod_rm)
        print("Lattice Reference Matrix:\n")
        print_matrix(lat_rm)
    return mod_rm, lat_rm

def generate_reference_matrix_cosine(data):
    co = data.T @ data
    diag = np.diag(co)
    norms = np.sqrt(diag)
    ref = co / np.outer(norms, norms)
    ref[np.isnan(ref)] = 0 
    return ref

def generate_reference_matrix_jaccard(data):
    co = data.T @ data
    diag = np.diag(co)
    union = diag[:,None] + diag[None,:] - co
    jacc = co / union
    jacc[np.isnan(jacc)] = 0 
    return jacc

def print_matrix(matrix):
    for row in matrix:
        print("\t".join([f"{val:.2f}" for val in row]))