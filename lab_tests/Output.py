import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns

def output_results_5_6(inputs, outputs):
    clean_test_outputs = binary_clean(inputs, outputs)
    print("Test Outputs (Test <num>: <input_features> | <output_features>):")
    for test_num, (inp, output) in enumerate(zip(inputs, clean_test_outputs), start=1):
        input_features = get_feature_nums(inp)
        output_features = get_feature_nums(output)
        print(f"Test {test_num}: {input_features} | {output_features}")

def output_results_2(test_inputs, test_outputs):
    for test_num, (t_in, t_out) in enumerate(zip(test_inputs, test_outputs), start=1):
        print(f"Test {test_num}: Input Layer | Output Layer\n")
        for i in range(len(t_in)):
            space = " " if i < 10 else ""
            print(f"F{i}{space}            {t_in[i]} | {t_out[i]}")
        print("\n")
        print(f"Max 6 values: Input Layer | Output Layer\n")
        print(f"{get_max_6(t_in)} | {get_max_6(t_out)}\n")
        print("\n--------------------------------------------------\n")

def binary_clean(model_input, model_output):
    for i in range(len(model_input)):
        model_output[i] = clean_line(model_input[i], model_output[i])
    return model_output
def clean_line(model_input_line, model_output_line):
    given = []
    for i in range(len(model_input_line)): 
        if model_input_line[i] == 1: given.append(i)
    max = 0
    for i in range(len(model_output_line)):
        if model_output_line[i] > max and i not in given: max = i
    for i in range(len(model_output_line)):
        if i in given or i == max:
            model_output_line[i] = 1
        else: 
            model_output_line[i] = 0
    return model_output_line
def get_feature_nums(clean_line):
    output = ""
    for i in range(len(clean_line)):
        if clean_line[i] == 1: output += str(i+1) + " "
    return output.strip()
def get_max_6(line):
    return sorted(range(len(line)), key=lambda i: line[i], reverse=True)[:6]

def print_matrix(matrix):
        for i in range(len(matrix[0])):
            for j in range(len(matrix[0])):
                print(f"{matrix[i][j]:.4f}", end="\t")
            print("\n")

def get_max_value_and_indices(data, minimize=False):
    best_val = float('inf') if minimize else float('-inf')
    best_indices = (None, None, None)  # (batch_idx, lr_idx, epoch_idx)
    for b_idx, row in enumerate(data):
        for lr_idx, epoch_vals in enumerate(row):
            for e_idx, val in enumerate(epoch_vals):
                if (minimize and val < best_val) or (not minimize and val > best_val):
                    best_val = val
                    best_indices = (b_idx, lr_idx, e_idx)
    return best_val, best_indices

def plot_matrix(matrix, title="Probability Matrix Heatmap"):
        plt.figure(figsize=(10, 6))
        sns.heatmap(matrix, annot=False, cmap="viridis", cbar=True, square=False)
        plt.title(title)
        plt.xlabel("Feature Index")
        plt.ylabel("Row / Sample Index")
        plt.tight_layout()
        plt.show()

def plot_matrices(m1, m2, m3, m4, titles=("Matrix 1", "Matrix 2", "Matrix 3", "Matrix 4")):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    matrices = [m1, m2, m3, m4]
    for ax, matrix, title in zip(axes.flat, matrices, titles):
        sns.heatmap(matrix, ax=ax, annot=False, cmap="viridis", cbar=True, square=False)
        ax.set_title(title)
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Row / Sample Index")

    plt.tight_layout()
    plt.show()

def plot_interactive_heatmap_discrete(data, batch_sizes, learning_rates, num_epochs, title="Heatmap", clip_max=None):
    pio.renderers.default = 'browser'

    data = np.array(data)

    if clip_max is not None:
        data = np.clip(data, None, clip_max)

    x_labels = [str(lr) for lr in learning_rates]
    y_labels = [str(bs) for bs in batch_sizes]

    zmin, zmax = data.min(), data.max()

    matrix_0 = data[:, :, 0]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_0,
            x=x_labels,
            y=y_labels,
            colorscale='Viridis',
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Value")
        ),
        layout=go.Layout(
            title=title,
            xaxis=dict(title="Learning Rate", type="category"),
            yaxis=dict(title="Batch Size", type="category"),
            sliders=[{
                "steps": [
                    {
                        "args": [[f"frame{e}"],
                                 {"frame": {"duration": 0, "redraw": True},
                                  "mode": "immediate"}],
                        "label": f"Epoch {e+1}",
                        "method": "animate"
                    }
                    for e in range(num_epochs)
                ],
                "transition": {"duration": 0},
                "x": 0.1,
                "xanchor": "left",
                "y": -0.15,
                "yanchor": "top"
            }]
        ),
        frames=[
            go.Frame(
                data=go.Heatmap(
                    z=data[:, :, e],
                    x=x_labels,
                    y=y_labels,
                    colorscale='Viridis',
                    zmin=zmin,
                    zmax=zmax
                ),
                name=f"frame{e}"
            )
            for e in range(num_epochs)
        ]
    )

    fig.show()

