import numpy as np

def get_reference_matrices_m_l(mod_filename: str, lat_filename: str, num_mod_trials: int, training_inputs, generate_rms: bool):
    if generate_rms:
        return _generate_reference_matrices(training_inputs, num_mod_trials, method='jaccard')
    else:
        return _get_probability_matrices_m_l(mod_filename, lat_filename)

def _generate_reference_matrices(training_data, num_mod_samples):
    td = np.array(training_data)
    mod_rm = _generate_reference_matrix_jaccard(td[:num_mod_samples,:11])
    lat_rm = _generate_reference_matrix_jaccard(td[num_mod_samples:,11:])
    return mod_rm, lat_rm

def _generate_reference_matrix_jaccard(data):
    co = data.T @ data
    diag = np.diag(co)
    union = diag[:,None] + diag[None,:] - co
    jacc = co / union
    jacc[np.isnan(jacc)] = 0 
    return jacc

def _get_probability_matrices_m_l(m_filename: str, l_filename: str):
    return np.loadtxt(m_filename, delimiter=','), np.loadtxt(l_filename, delimiter=',')