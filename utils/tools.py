import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def print_formatted_dict(d: dict) -> None:
    for key, value in d.items():
        if isinstance(value, float):
            print(f"{key}: {value}")
            # print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def select_best_metrics(
    metrics: dict, target_mode: str = "test", target_metric: str = "acc"
) -> dict:
    best_metrics = {}

    # Find the epoch with the best target metric
    target_metric_values = metrics[target_mode][target_metric]
    best_epoch_index = target_metric_values.index(max(target_metric_values))
    best_metrics["best_epoch"] = best_epoch_index + 1  # Epoch starts from 1

    # Now gather metrics from all modes for this epoch
    for mode, mode_metrics in metrics.items():
        for metric_name, metric_values in mode_metrics.items():
            best_metrics[f"{mode}_{metric_name}"] = metric_values[best_epoch_index]

    return best_metrics
