import os
import json
import numpy as np
import zarr

def init_zarr_run(
    zarr_path: str,
    *,
    num_models: int,
    num_epochs: int,
    num_probes: int,
    hidden_dim: int,
    out_dim: int,
    hidden_dtype=np.float16,
    out_dtype=np.float32,
    loss_dtype=np.float32,
    compressor=None,
    epoch_chunk: int = 1,
):
    os.makedirs(os.path.dirname(zarr_path) or ".", exist_ok=True)
    root = zarr.open_group(zarr_path, mode="w")

    root.attrs.update(
        dict(
            num_models=int(num_models),
            num_epochs=int(num_epochs),
            num_probes=int(num_probes),
            hidden_dim=int(hidden_dim),
            out_dim=int(out_dim),
            hidden_dtype=str(np.dtype(hidden_dtype)),
            out_dtype=str(np.dtype(out_dtype)),
            loss_dtype=str(np.dtype(loss_dtype)),
        )
    )

    root.create_dataset(
        "loss",
        shape=(num_models, num_epochs),
        chunks=(1, min(epoch_chunk, num_epochs)),
        dtype=loss_dtype,
        compressor=compressor,
        fill_value=np.nan,
        overwrite=True,
    )

    root.create_dataset(
        "hidden",
        shape=(num_models, num_epochs, num_probes, hidden_dim),
        chunks=(1, min(epoch_chunk, num_epochs), num_probes, hidden_dim),
        dtype=hidden_dtype,
        compressor=compressor,
        fill_value=0,
        overwrite=True,
    )

    root.create_dataset(
        "output",
        shape=(num_models, num_epochs, num_probes, out_dim),
        chunks=(1, min(epoch_chunk, num_epochs), num_probes, out_dim),
        dtype=out_dtype,
        compressor=compressor,
        fill_value=0,
        overwrite=True,
    )

    return root


def write_model_result(root, model_i, result):
    n_epochs = root.attrs["num_epochs"]

    loss = np.asarray(result["losses"], dtype=root["loss"].dtype)
    root["loss"][model_i, :] = loss

    hid_model = np.stack(
        [h.detach().cpu().numpy() if hasattr(h, "detach") else np.asarray(h) for h in result["hidden"]],
        axis=0,
    ).astype(root["hidden"].dtype, copy=False)  # (E, P, H)

    out_model = np.stack(
        [o.detach().cpu().numpy() if hasattr(o, "detach") else np.asarray(o) for o in result["output"]],
        axis=0,
    ).astype(root["output"].dtype, copy=False)  # (E, P, O)

    root["hidden"][model_i, :, :, :] = hid_model
    root["output"][model_i, :, :, :] = out_model


def save_probe_index_json(path: str, probe_index: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(probe_index, f, indent=2, ensure_ascii=False)


def open_run(zarr_path: str, mode: str = "r"):
    return zarr.open_group(zarr_path, mode=mode)


def load_slice(zarr_path: str, *, model_slice=slice(None), epoch_slice=slice(None), probe_ids=None):
    root = open_run(zarr_path, mode="r")
    hid = root["hidden"]
    out = root["output"]
    loss = root["loss"]

    if probe_ids is None:
        return (
            hid[model_slice, epoch_slice, :, :],
            out[model_slice, epoch_slice, :, :],
            loss[model_slice, epoch_slice],
        )

    probe_ids = np.asarray(probe_ids, dtype=np.int64)
    return (
        hid.oindex[model_slice, epoch_slice, probe_ids, :],
        out.oindex[model_slice, epoch_slice, probe_ids, :],
        loss[model_slice, epoch_slice],
    )


def build_zarr_from_results(
    zarr_path: str,
    results_per_model: list,
    *,
    hidden_dtype=np.float16,
    out_dtype=np.float32,
    loss_dtype=np.float32,
    compressor=None,
    epoch_chunk: int = 10,
):
    num_models = len(results_per_model)
    if num_models == 0:
        raise ValueError("results_per_model is empty")

    r0 = results_per_model[0]
    num_epochs = len(r0["losses"])

    h0 = r0["hidden"][0]
    o0 = r0["output"][0]

    if hasattr(h0, "detach"):
        h0 = h0.detach().cpu().numpy()
    else:
        h0 = np.asarray(h0)

    if hasattr(o0, "detach"):
        o0 = o0.detach().cpu().numpy()
    else:
        o0 = np.asarray(o0)

    num_probes = int(h0.shape[0])
    hidden_dim = int(h0.shape[1])
    out_dim = int(o0.shape[1])

    root = init_zarr_run(
        zarr_path,
        num_models=num_models,
        num_epochs=num_epochs,
        num_probes=num_probes,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        hidden_dtype=hidden_dtype,
        out_dtype=out_dtype,
        loss_dtype=loss_dtype,
        compressor=compressor,
        epoch_chunk=epoch_chunk,
    )

    for i, r in enumerate(results_per_model):
        write_model_result(root, i, r)

    return root
