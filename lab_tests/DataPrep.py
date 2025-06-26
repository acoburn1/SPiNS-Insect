from turtle import distance
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class DataPreparer:
    def __init__(self, training_filename, testing_filename, modular_p_m_filename, lattice_p_m_filename, num_features):
        self.training_filename = training_filename
        self.testing_filename = testing_filename
        self.modular_p_m_filename = modular_p_m_filename
        self.lattice_p_m_filename = lattice_p_m_filename
        self.num_features = num_features
        self.training_inputs = []
        self.training_outputs = []
        self.test_inputs = []
    def load_data(self):
        start = 3
        split = 3 + 2*self.num_features
        end = split + 2*self.num_features
        with open(self.training_filename, "r") as file:
            for line in file:
                if not line.startswith("_D:"):
                    continue
                tokens = line.strip().split('\t')
                if len(tokens) >= end:
                    input_vals = list(map(int, tokens[start:split]))
                    output_vals = list(map(int, tokens[split:end]))
                    self.training_inputs.append(input_vals)
                    self.training_outputs.append(output_vals)
        with open(self.testing_filename, "r") as file:
            for line in file:
                if not line.startswith("_D:"):
                    continue
                tokens = line.strip().split('\t')
                if len(tokens) >= end:
                    input_vals = list(map(int, tokens[start:split]))
                    self.test_inputs.append(input_vals)
    def get_dataloader(self, batch_size):
        X = torch.tensor(self.training_inputs, dtype=torch.float32)
        Y = torch.tensor(self.training_outputs, dtype=torch.float32)
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
    def get_probability_matrices_m_l(self):
        return np.loadtxt(self.modular_p_m_filename, delimiter=','), np.loadtxt(self.lattice_p_m_filename, delimiter=',')


class PairWalkTrainingDataGenerator:
    def __init__(self, probability_matrix_1, probability_matrix_2, num_training_trials=550):
        self.probability_matrix_1 = probability_matrix_1
        self.probability_matrix_2 = probability_matrix_2
        self.num_features = len(probability_matrix_1[0])
        self.num_training_trials = num_training_trials
        self.training_inputs = []
        self.training_outputs = []
        self.test_inputs = []
        self.test_outputs = []
    def generate_and_write_training_data(self, training_data_filename):
        self.generate_training_data()
        self.write_data(self.num_features, self.training_inputs, self.training_outputs, training_data_filename)
    def write_data(self, num_features, inputs, outputs, filename):
        with open(filename, "w") as file:
            header = ["_H:", "$ItemNum", "$Name"]
            header += [f'"%Input[2:{i},0]"' for i in range(2*num_features)]
            header += [f'"%Output[2:{i},0]"' for i in range(2*num_features)]
            for i, inp in enumerate(inputs):
                name = f"EX{i+1}"
                row = ["_D:", str(i+1), name]
                row += [str(val) for val in inp]
                row += [str(val) for val in outputs[i]]
                file.write('\t'.join(row) + '\n')
    def generate_training_data(self):
        p_m = self.probability_matrix_1
        x, y = self.generate_initial_x_y()
        s_h = False
        for i in range(self.num_training_trials):
            if (i == self.num_training_trials / 2):
                s_h = True;
                p_m = self.probability_matrix_2
                x, y = self.generate_initial_x_y(second_half=s_h)
            line = np.zeros(2*self.num_features, dtype=int)
            line[x] = 1
            line[y] = 1
            self.training_inputs.append(line)
            self.training_outputs.append(line)
            x = y
            y = self.step(x, p_m, s_h)
    def generate_initial_x_y(self, second_half=False):
        rng = np.random.default_rng()
        x = rng.choice(self.num_features)
        y = x
        while (x == y):
            y = rng.choice(self.num_features)
        if (second_half):
            x += self.num_features
            y += self.num_features
        return x, y
    def step(self, feature_num, probability_matrix, second_half=False):
        rng = np.random.default_rng()
        feature_num = feature_num % self.num_features
        choice = rng.choice(self.num_features, p = probability_matrix[feature_num])
        return choice if not second_half else choice + self.num_features


### TODO: Implement test data generators
class TestDataGenerators():
    def generate_one_missing_test_data():
        pass


class ProbabilityMatrix:
    def generate_probability_matrix(core_size=3, num_features=11, num_modules=2, structure="modular", cc_w=10, cp_w=4, mm_w=2, l1_w=2, l2_w=1): ### cc_w = core-core weight, cp_w = core-periphery weight, mm_w = module-module weight, l1_w = lattice 1st neighbor weight, l2_w = lattice 2nd neighbor weight
        p_m = np.zeros((num_features, num_features), dtype=float)

        if (structure == "modular"):
            if ((num_features - core_size) % num_modules != 0):
                raise Exception("Number of peripheral features minus must be divisible by number of modules.")
            # TODO: rewrite logic to allow for more than 2 modules
            module_size = (num_features - core_size) // num_modules
            m2_start = core_size + module_size
            for i in range(num_features):
                for j in range(num_features):
                    if (
                        i == j
                        or core_size <= i < m2_start and m2_start <= j
                        or core_size <= j < m2_start and m2_start <= i
                       ):
                        continue
                    elif (i < core_size and j < core_size):
                        p_m[i][j] = cc_w
                    elif (i < core_size or j < core_size):
                        p_m[i][j] = cp_w
                    else:
                        p_m[i][j] = mm_w

        elif (structure == "lattice"):
            periphery_range = num_features - core_size
            for i in range(num_features):
                for j in range(num_features):
                    diff = abs(i - j) % periphery_range
                    distance = min(diff, periphery_range - diff)
                    if (i == j):
                        continue
                    elif (i < core_size and j < core_size):
                        p_m[i][j] = cc_w
                    elif (i < core_size or j < core_size):
                        p_m[i][j] = cp_w
                    elif (distance == 1):
                        p_m[i][j] = l1_w
                    elif (distance == 2):
                        p_m[i][j] = l2_w

        elif (structure == "random"):
            ### TODO: Implement random
            pass

        else:
            raise Exception("Invalid structure type. Please choose 'modular', 'lattice', or 'random'.")

        return p_m / p_m.sum(axis=1, keepdims=True)
