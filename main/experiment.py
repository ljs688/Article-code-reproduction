import collections
from copy import deepcopy

import numpy as np
import torch

from config.config import get_config
from models.GHICMC import GHICMC
from utils.dataloader import load_train_data
from utils.graph_adjacency import get_adjacency
from utils.util import cal_std, format_mean_std, get_logger, get_mask, setup_seed


def expand_sparse_adjacency(local_adj, observed_mask, total_size):
    observed_indices = torch.nonzero(observed_mask, as_tuple=False).view(-1)
    local_adj = local_adj.coalesce()
    local_indices = local_adj.indices()
    global_indices = observed_indices[local_indices]
    return torch.sparse_coo_tensor(
        global_indices,
        local_adj.values(),
        size=(total_size, total_size),
        dtype=local_adj.dtype,
    ).coalesce()


def build_graph_inputs(train_data, mask, config):
    train_miss = []
    adj = []
    adj_add = []

    for i in range(config["v_num"]):
        mask_column = mask[:, i].to(train_data[i].dtype).unsqueeze(1)
        train_miss_data = train_data[i] * mask_column
        train_miss.append(train_miss_data.to(torch.float32))

    for i in range(config["v_num"]):
        observed_mask = mask[:, i].bool()
        features = train_miss[i][observed_mask]
        adj_i, _ = get_adjacency(features, features.shape[0], config["topk"])
        adj.append(adj_i.coalesce())
        adj_add.append(expand_sparse_adjacency(adj_i, observed_mask, mask.shape[0]))

    return train_miss, adj, adj_add


def run_single_seed(config, train_data, train_labels, mask, logger):
    train_miss, adj, adj_add = build_graph_inputs(train_data, mask, config)
    setup_seed(config["seed"])
    accumulated_metrics = collections.defaultdict(list)
    model = GHICMC(config).to(config["device"])
    return model.run_train(
        train_miss, train_labels, adj, adj_add, mask, accumulated_metrics, logger
    )


def run_dataset_experiment(dataset, seeds, missing_rate=0.5, epochs_override=None):
    config = get_config(dataset=dataset)
    config["missing_rate"] = missing_rate
    if epochs_override is not None:
        config["training"]["epoch"] = epochs_override

    train_data, train_labels = load_train_data(config)
    logger = get_logger(config)

    fold_acc, fold_nmi, fold_ari = [], [], []

    for seed in seeds:
        logger.info(
            "%s___%.1f___seed %s",
            config["dataset"],
            config["missing_rate"],
            seed,
        )
        config["seed"] = seed
        setup_seed(seed)
        mask = torch.from_numpy(
            get_mask(train_data[0].shape[0], config["missing_rate"], config["v_num"])
        ).long()
        try:
            acc, nmi, ari = run_single_seed(config, train_data, train_labels, mask, logger)
        except RuntimeError as exc:
            is_cuda_oom = (
                config["device"].type == "cuda" and "out of memory" in str(exc).lower()
            )
            if not is_cuda_oom:
                raise
            logger.warning(
                "CUDA OOM on %s seed %s. Retrying on CPU.",
                config["dataset"],
                seed,
            )
            torch.cuda.empty_cache()
            cpu_config = deepcopy(config)
            cpu_config["device"] = torch.device("cpu")
            acc, nmi, ari = run_single_seed(cpu_config, train_data, train_labels, mask, logger)

        fold_acc.append(acc)
        fold_nmi.append(nmi)
        fold_ari.append(ari)

        logger.info(
            "%s___%.1f___seed %s RESULT: ACC:%.2f NMI:%.2f ARI:%.2f",
            config["dataset"],
            config["missing_rate"],
            seed,
            acc * 100,
            nmi * 100,
            ari * 100,
        )

    logger.info("--------------------Training over--------------------")
    cal_std(logger, fold_acc, fold_nmi, fold_ari)
    logger.handlers.clear()

    return {
        "dataset": config["dataset"],
        "ACC": format_mean_std(fold_acc),
        "NMI": format_mean_std(fold_nmi),
        "ARI": format_mean_std(fold_ari),
        "raw": {
            "acc": deepcopy(fold_acc),
            "nmi": deepcopy(fold_nmi),
            "ari": deepcopy(fold_ari),
        },
    }
