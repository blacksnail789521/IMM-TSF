import gc
import numpy as np
import sklearn as sk
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import lib.utils as utils
from lib.utils import get_device

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent


def compute_error(truth, pred_y, mask, func, reduce, norm_dict=None):
    # pred_y shape [n_traj_samples, n_batch, n_tp, n_dim]
    # truth shape  [n_bacth, n_tp, n_dim] or [B, L, n_dim]

    if len(pred_y.shape) == 3:
        pred_y = pred_y.unsqueeze(dim=0)
    n_traj_samples, n_batch, n_tp, n_dim = pred_y.size()
    truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
    mask = mask.repeat(pred_y.size(0), 1, 1, 1)

    if func == "MSE":
        error = (
            (truth_repeated - pred_y) ** 2
        ) * mask  # (n_traj_samples, n_batch, n_tp, n_dim)
    elif func == "MAE":
        error = (
            torch.abs(truth_repeated - pred_y) * mask
        )  # (n_traj_samples, n_batch, n_tp, n_dim)
    elif func == "MAPE":
        if norm_dict == None:
            mask = (truth_repeated != 0) * mask
            truth_div = truth_repeated + (truth_repeated == 0) * 1e-8
            error = torch.abs(truth_repeated - pred_y) / truth_div * mask
        else:
            data_max = norm_dict["data_max"]
            data_min = norm_dict["data_min"]
            truth_rescale = truth_repeated * (data_max - data_min) + data_min
            pred_y_rescale = pred_y * (data_max - data_min) + data_min
            mask = (truth_rescale != 0) * mask
            truth_rescale_div = truth_rescale + (truth_rescale == 0) * 1e-8
            error = torch.abs(truth_rescale - pred_y_rescale) / truth_rescale_div * mask
    else:
        raise Exception("Error function not specified")

    error_var_sum = error.reshape(-1, n_dim).sum(dim=0)  # (n_dim, )
    mask_count = mask.reshape(-1, n_dim).sum(dim=0)  # (n_dim, )

    if reduce == "mean":
        ### 1. Compute avg error of each variable first
        ### 2. Compute avg error along the variables
        error_var_avg = error_var_sum / (mask_count + 1e-8)  # (n_dim, )
        # print("error_var_avg", error_var_avg.max().item(), error_var_avg.min().item(), (1.0*error_var_avg).mean().item())
        n_avai_var = torch.count_nonzero(mask_count)
        error_avg = error_var_avg.sum() / n_avai_var  # (1, )

        return error_avg  # a scalar (1, )

    elif reduce == "sum":
        # (n_dim, ) , (n_dim, )
        return error_var_sum, mask_count

    else:
        raise Exception("Reduce argument not specified!")


def compute_all_losses(
    model, fusion, batch_dict, enable_text=True, use_text_embeddings=True
):
    # Condition on subsampled points
    # Make predictions for all the points
    # shape of pred --- [n_traj_samples=1, n_batch, n_tp, n_dim]

    pred_y = model.forecasting(
        batch_dict["tp_to_predict"],
        batch_dict["observed_data"],
        batch_dict["observed_tp"],
        batch_dict["observed_mask"],
    )
    if torch.isnan(pred_y).any():
        print(f"pred_y: {pred_y}")
        raise ValueError("pred_y contains NaN values.")

    if enable_text and fusion is not None:
        notes_input = (
            batch_dict["notes_embeddings"]
            if use_text_embeddings
            else batch_dict["notes_text"]
        )
        pred_y = fusion(
            notes_input,
            batch_dict["tau"],
            batch_dict["tp_to_predict"],
            pred_y,
        )

    # Compute avg error of each variable first, then compute avg error of all variables
    if torch.isnan(pred_y).any():
        raise ValueError("pred_y contains NaN values.")
    if torch.isnan(batch_dict["data_to_predict"]).any():
        raise ValueError("data_to_predict contains NaN values.")
    mse = compute_error(
        batch_dict["data_to_predict"],
        pred_y,
        mask=batch_dict["mask_predicted_data"],
        func="MSE",
        reduce="mean",
    )  # a scalar
    # raise Exception(
    #     pred_y.shape,
    #     batch_dict["data_to_predict"].shape,
    #     batch_dict["mask_predicted_data"].shape,
    # )
    # print(batch_dict["mask_predicted_data"])
    # Assert that every sample in batch_dict["mask_predicted_data"] is not all 0
    # for i in range(batch_dict["mask_predicted_data"].shape[0]):
    #     if batch_dict["mask_predicted_data"][i].sum() == 0:
    #         raise ValueError(
    #             f"mask_predicted_data for sample {i} is all zeros: {batch_dict['mask_predicted_data'][i]}"
    #         )
    
    # Check mask is not all zero for every sample
    for i in range(batch_dict["mask_predicted_data"].shape[0]):
        if batch_dict["mask_predicted_data"][i].sum() == 0:
            raise ValueError(
                f"mask_predicted_data for sample {i} is all zeros: {batch_dict['mask_predicted_data'][i]}"
            )

    # mse = masked_mse_nn(
    #     pred_y,
    #     batch_dict["data_to_predict"],
    #     mask=batch_dict["mask_predicted_data"],
    # )  # a scalar
    # print("mse", mse.item())
    # Check if mse is nan
    if torch.isnan(mse).any():
        print(f"pred_y: {pred_y}")
        raise ValueError("MSE is NaN")
    # rmse = torch.sqrt(mse)
    # print(mse, rmse)
    # mae = compute_error(
    #     batch_dict["data_to_predict"],
    #     pred_y,
    #     mask=batch_dict["mask_predicted_data"],
    #     func="MAE",
    #     reduce="mean",
    # )  # a scalar

    ################################
    # mse loss
    loss = mse

    results = {}
    results["loss"] = loss
    results["mse"] = mse.item()
    # results["rmse"] = rmse.item()
    # results["mae"] = mae.item()

    return results


def masked_mse_nn(pred_y: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked MSE using nn.MSELoss, with elementwise masking.

    Args:
        pred_y: [B, T, D] — predictions
        target: [B, T, D] — ground truth
        mask:   [B, T, D] — binary mask

    Returns:
        Scalar masked MSE
    """
    mse_loss = nn.MSELoss(reduction='mean')

    # Flatten all to 1D
    pred_flat = pred_y.reshape(-1)
    target_flat = target.reshape(-1)
    mask_flat = mask.reshape(-1).bool()

    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=pred_y.device)

    return mse_loss(pred_flat[mask_flat], target_flat[mask_flat])


def evaluation(model, fusion, dataloader, enable_text=True, use_text_embeddings=True):

    n_eval_samples = 0
    n_eval_samples_mape = 0
    total_results = {}
    total_results["loss"] = 0
    total_results["mse"] = 0
    total_results["mae"] = 0
    total_results["rmse"] = 0
    total_results["mape"] = 0

    # for _ in range(n_batches):
    for step, batch_dict in enumerate(tqdm(dataloader)):
        pred_y = model.forecasting(
            batch_dict["tp_to_predict"],
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            batch_dict["observed_mask"],
        )

        if enable_text and fusion is not None:
            notes_input = (
                batch_dict["notes_embeddings"]
                if use_text_embeddings
                else batch_dict["notes_text"]
            )
            pred_y = fusion(
                notes_input,
                batch_dict["tau"],
                batch_dict["tp_to_predict"],
                pred_y,
            )

        # (n_dim, ) , (n_dim, )
        se_var_sum, mask_count = compute_error(
            batch_dict["data_to_predict"],
            pred_y,
            mask=batch_dict["mask_predicted_data"],
            func="MSE",
            reduce="sum",
        )  # a vector

        ae_var_sum, _ = compute_error(
            batch_dict["data_to_predict"],
            pred_y,
            mask=batch_dict["mask_predicted_data"],
            func="MAE",
            reduce="sum",
        )  # a vector

        # norm_dict = {"data_max": batch_dict["data_max"], "data_min": batch_dict["data_min"]}
        ape_var_sum, mask_count_mape = compute_error(
            batch_dict["data_to_predict"],
            pred_y,
            mask=batch_dict["mask_predicted_data"],
            func="MAPE",
            reduce="sum",
        )  # a vector

        # add a tensor (n_dim, )
        total_results["loss"] += se_var_sum
        total_results["mse"] += se_var_sum
        total_results["mae"] += ae_var_sum
        total_results["mape"] += ape_var_sum
        n_eval_samples += mask_count
        n_eval_samples_mape += mask_count_mape

    n_avai_var = torch.count_nonzero(n_eval_samples)
    n_avai_var_mape = torch.count_nonzero(n_eval_samples_mape)

    ### 1. Compute avg error of each variable first
    ### 2. Compute avg error along the variables
    total_results["loss"] = (
        total_results["loss"] / (n_eval_samples + 1e-8)
    ).sum() / n_avai_var
    total_results["mse"] = (
        total_results["mse"] / (n_eval_samples + 1e-8)
    ).sum() / n_avai_var
    total_results["mae"] = (
        total_results["mae"] / (n_eval_samples + 1e-8)
    ).sum() / n_avai_var
    total_results["rmse"] = torch.sqrt(total_results["mse"])
    total_results["mape"] = (
        total_results["mape"] / (n_eval_samples_mape + 1e-8)
    ).sum() / n_avai_var_mape

    for key, var in total_results.items():
        if isinstance(var, torch.Tensor):
            var = var.item()
        total_results[key] = var

    return total_results
