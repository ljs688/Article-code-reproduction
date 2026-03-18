from pathlib import Path

import numpy as np
import scipy.io
import torch
from sklearn.preprocessing import MinMaxScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def _resolve_views(raw_views):
    if not isinstance(raw_views, np.ndarray) or raw_views.dtype != object:
        raise ValueError("Expected multi-view data stored as an object ndarray.")
    return [np.asarray(view) for view in raw_views.reshape(-1)]


def _resolve_labels(raw_labels):
    labels = raw_labels
    if isinstance(raw_labels, np.ndarray) and raw_labels.dtype == object:
        labels = np.asarray(raw_labels.reshape(-1)[0])
    labels = np.asarray(labels).reshape(-1)
    return labels


def _to_sample_feature_matrix(view, num_samples):
    view = np.asarray(view)
    if view.ndim != 2:
        raise ValueError(f"Expected 2D view matrix, got shape {view.shape}")
    if view.shape[0] != num_samples and view.shape[1] == num_samples:
        view = view.T
    elif view.shape[0] != num_samples and view.shape[1] != num_samples:
        raise ValueError(
            f"Cannot align view shape {view.shape} to sample count {num_samples}"
        )
    scaler = MinMaxScaler()
    return scaler.fit_transform(view)


def load_train_data(config):
    data_path = DATA_DIR / config["data_file"]
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    print(f"Loading {config['dataset']} for Training....")
    data_set = scipy.io.loadmat(data_path)

    datas = _resolve_views(data_set[config["feature_key"]])
    labels = _resolve_labels(data_set[config["label_key"]])
    num_samples = labels.shape[0]

    if len(datas) != config["v_num"]:
        raise ValueError(
            f"Expected {config['v_num']} views for {config['dataset']}, got {len(datas)}"
        )

    train_views = []
    for idx, view in enumerate(datas, start=1):
        scaled_view = _to_sample_feature_matrix(view, num_samples)
        expected_dim = config["Autoencoder"][f"gcnEncoder{idx}"][0]
        if scaled_view.shape[1] != expected_dim:
            raise ValueError(
                f"Unexpected input dimension for view {idx} of {config['dataset']}: "
                f"expected {expected_dim}, got {scaled_view.shape[1]}"
            )
        train_views.append(torch.tensor(scaled_view, dtype=torch.float32))

    y_train = torch.tensor(labels, dtype=torch.long)

    if train_views[0].shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Sample count mismatch for {config['dataset']}: "
            f"{train_views[0].shape[0]} samples vs {y_train.shape[0]} labels"
        )

    print(f"Loading {config['dataset']} over!!!")
    return train_views, y_train
