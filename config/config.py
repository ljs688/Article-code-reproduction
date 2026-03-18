from copy import deepcopy

import torch


DATASET_CONFIGS = {
    "Caltech101-2V": dict(
        dataset="Caltech101-2V",
        data_file="2view-caltech101-8677sample.mat",
        feature_key="X",
        label_key="gt",
        seed=0,
        v_num=2,
        topk=10,
        missing_rate=0.5,
        n_clusters=101,
        training=dict(
            epoch=200,
            lr=1e-4,
        ),
        Autoencoder=dict(
            gcnEncoder1=[4096, 1024, 1024, 1024, 512],
            gcnEncoder2=[4096, 1024, 1024, 1024, 512],
            graphEncoder1=[512, 1024, 1024, 1024, 512],
            graphEncoder2=[512, 1024, 1024, 1024, 512],
            graphEncoderf=[512, 1024, 1024, 1024, 512],
            activations1="relu",
            activations2="relu",
            activationsf="relu",
            batchnorm=True,
        ),
    ),
    "Hdigit": dict(
        dataset="Hdigit",
        data_file="Hdigit.mat",
        feature_key="data",
        label_key="truelabel",
        seed=0,
        v_num=2,
        topk=10,
        missing_rate=0.5,
        n_clusters=10,
        training=dict(
            epoch=200,
            lr=1e-3,
        ),
        Autoencoder=dict(
            gcnEncoder1=[784, 1024, 1024, 1024, 128],
            gcnEncoder2=[256, 1024, 1024, 1024, 128],
            graphEncoder1=[128, 1024, 1024, 1024, 128],
            graphEncoder2=[128, 1024, 1024, 1024, 128],
            graphEncoderf=[128, 1024, 1024, 1024, 128],
            activations1="relu",
            activations2="relu",
            activationsf="relu",
            batchnorm=True,
        ),
    ),
    "MSRCV1": dict(
        dataset="MSRCV1",
        data_file="MSRCV1.mat",
        feature_key="X",
        label_key="gt",
        seed=0,
        v_num=6,
        topk=10,
        missing_rate=0.5,
        n_clusters=7,
        training=dict(
            epoch=200,
            lr=1e-4,
        ),
        Autoencoder=dict(
            gcnEncoder1=[1302, 1024, 1024, 1024, 128],
            gcnEncoder2=[48, 1024, 1024, 1024, 128],
            gcnEncoder3=[512, 1024, 1024, 1024, 128],
            gcnEncoder4=[100, 1024, 1024, 1024, 128],
            gcnEncoder5=[256, 1024, 1024, 1024, 128],
            gcnEncoder6=[210, 1024, 1024, 1024, 128],
            graphEncoder1=[128, 1024, 1024, 1024, 128],
            graphEncoder2=[128, 1024, 1024, 1024, 128],
            graphEncoder3=[128, 1024, 1024, 1024, 128],
            graphEncoder4=[128, 1024, 1024, 1024, 128],
            graphEncoder5=[128, 1024, 1024, 1024, 128],
            graphEncoder6=[128, 1024, 1024, 1024, 128],
            graphEncoderf=[128, 1024, 1024, 1024, 128],
            activations1="relu",
            activations2="relu",
            activations3="relu",
            activations4="relu",
            activations5="relu",
            activations6="relu",
            activationsf="relu",
            batchnorm=True,
        ),
    ),
}


FLAG_TO_DATASET = {
    0: "Caltech101-2V",
    1: "Hdigit",
    2: "MSRCV1",
}


SKIPPED_DATASETS = {
    "YouTubeFace": "Skipped because this method runs out of memory on that dataset.",
}


def list_datasets():
    return list(DATASET_CONFIGS.keys())


def get_config(flag=0, dataset=None):
    if dataset is None:
        dataset = FLAG_TO_DATASET.get(flag)
    if dataset in SKIPPED_DATASETS:
        raise ValueError(SKIPPED_DATASETS[dataset])
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}")

    config = deepcopy(DATASET_CONFIGS[dataset])
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return config
