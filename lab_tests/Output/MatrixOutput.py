from asyncio.windows_events import NULL
import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt

def save_all_epoch_matrices(data_dir, save_dir, num_epochs, include_E0):
    if include_E0:
        num_epochs += 1
    for epoch in range(num_epochs):
        os.makedirs(save_dir, exist_ok=True)
        npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        matrices = []
        for path in npz_files:
            with np.load(path, allow_pickle=True) as data:
                arr = data["hidden_matrices"]
                matrices.append(arr[epoch])
        avg_matrix = np.mean(np.stack(matrices, axis=0), axis=0)
        save_matrix(avg_matrix, save_dir, epoch)
    _flatten_mod_lat_folders(save_dir)

def save_epoch_matrix(data_dir, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    matrices = []
    for path in npz_files:
        with np.load(path, allow_pickle=True) as data:
            arr = data["hidden_matrices"]
            matrices.append(arr[epoch])
    avg_matrix = np.mean(np.stack(matrices, axis=0), axis=0)
    save_matrix(avg_matrix, save_dir)

def save_matrix(matrix, save_dir, epoch=-1):
    os.makedirs(save_dir, exist_ok=True)
    n = matrix.shape[0] // 2
    mod = matrix[:n, :n]
    lat = matrix[n:, n:]
    epoch_str = f"{epoch:03d}" if epoch != -1 else ""
    for name, m in [("e" + epoch_str + "_" + "mod", mod), ("e" + epoch_str + "_" + "lat", lat)]:
        plt.figure()
        plt.imshow(m, vmin=0, vmax=1, origin="upper")
        plt.colorbar()
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300)
        plt.close()

def save_generic_matrix(matrix, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.imshow(matrix, vmin=0, vmax=1, origin="upper")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300)
    plt.close()

def _flatten_mod_lat_folders(outer_dir):
    mod_dir = os.path.join(outer_dir, "mod_all")
    lat_dir = os.path.join(outer_dir, "lat_all")

    os.makedirs(mod_dir, exist_ok=True)
    os.makedirs(lat_dir, exist_ok=True)

    for root, _, files in os.walk(outer_dir):
        if root in [mod_dir, lat_dir]:
            continue

        for file in files:
            src = os.path.join(root, file)

            if "mod" in file.lower():
                dst = os.path.join(mod_dir, file)
                shutil.move(src, dst)

            elif "lat" in file.lower():
                dst = os.path.join(lat_dir, file)
                shutil.move(src, dst)
