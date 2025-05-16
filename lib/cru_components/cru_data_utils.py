# Modeling Irregular Time Series with Continuous Recurrent Units (CRUs)
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import os


# new code component
def subsample(data, sample_rate, imagepred=False, random_state=0):
    train_obs, train_targets, test_obs, test_targets = (
        data["train_obs"],
        data["train_targets"],
        data["test_obs"],
        data["test_targets"],
    )
    seq_length = train_obs.shape[1]
    train_time_points = []
    test_time_points = []
    n = int(sample_rate * seq_length)

    if imagepred:
        train_obs_valid = data["train_obs_valid"]
        test_obs_valid = data["test_obs_valid"]
        data_components = (
            train_obs,
            train_targets,
            test_obs,
            test_targets,
            train_obs_valid,
            test_obs_valid,
        )
        (
            train_obs_sub,
            train_targets_sub,
            test_obs_sub,
            test_targets_sub,
            train_obs_valid_sub,
            test_obs_valid_sub,
        ) = [np.zeros_like(x[:, :n, ...]) for x in data_components]
    else:
        data_components = train_obs, train_targets, test_obs, test_targets
        train_obs_sub, train_targets_sub, test_obs_sub, test_targets_sub = [
            np.zeros_like(x[:, :n, ...]) for x in data_components
        ]

    for i in range(train_obs.shape[0]):
        rng_train = np.random.default_rng(random_state + i + train_obs.shape[0])
        choice = np.sort(rng_train.choice(seq_length, n, replace=False))
        train_time_points.append(choice)
        train_obs_sub[i, ...], train_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [train_obs, train_targets]
        ]
        if imagepred:
            train_obs_valid_sub[i, ...] = train_obs_valid[i, choice, ...]

    for i in range(test_obs.shape[0]):
        rng_test = np.random.default_rng(random_state + i)
        choice = np.sort(rng_test.choice(seq_length, n, replace=False))
        test_time_points.append(choice)
        test_obs_sub[i, ...], test_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [test_obs, test_targets]
        ]
        if imagepred:
            test_obs_valid_sub[i, ...] = test_obs_valid[i, choice, ...]

    train_time_points, test_time_points = np.stack(train_time_points, 0), np.stack(
        test_time_points, 0
    )

    if imagepred:
        return (
            train_obs_sub,
            train_targets_sub,
            train_time_points,
            train_obs_valid_sub,
            test_obs_sub,
            test_targets_sub,
            test_time_points,
            test_obs_valid_sub,
        )
    else:
        return (
            train_obs_sub,
            train_targets_sub,
            test_obs_sub,
            test_targets_sub,
            train_time_points,
            test_time_points,
        )


# new code component
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# new code component
def discretize_data(
    obs, targets, time_points, obs_valid, n_bins=10, take_always_closest=True
):
    N = obs.shape[0]
    T_max = time_points.max()
    bin_length = T_max / n_bins
    obs_valid = np.squeeze(obs_valid)

    # define the bins
    _, bin_edges = np.histogram(time_points, bins=n_bins)

    # get the center of each bin
    bin_length = bin_edges[1] - bin_edges[0]
    bin_center = bin_edges + bin_length / 2

    # get the timepoint, obs etc that is closest to the bin center
    tp_all = []
    obs_valid_all = []
    obs_all = np.zeros((N, n_bins, 24, 24, 1), dtype="uint8")
    targets_all = np.zeros((N, n_bins, 24, 24, 1), dtype="uint8")
    for i in range(N):
        tp_list = []
        obs_valid_list = []
        for j in range(n_bins):
            sample_tp = time_points[i, :]
            center = bin_center[j]
            idx = find_nearest(sample_tp, center)
            if (
                bin_edges[j] <= sample_tp[idx] <= bin_edges[j + 1]
            ) or take_always_closest:
                tp_list.append(sample_tp[idx])
                obs_valid_list.append(obs_valid[i, idx])
                obs_all[i, j, ...] = obs[i, idx, ...]
                targets_all[i, j, ...] = targets[i, idx, ...]
            else:
                tp_list.append(np.nan)
                obs_valid_list.append(False)
                obs_all[i, j, ...] = 0
                targets_all[i, j, ...] = 0

        tp_all.append(tp_list)
        obs_valid_all.append(obs_valid_list)

    return obs_all, targets_all, np.array(tp_all), np.array(obs_valid_all)


# new code component
def create_unobserved_mask(n_col, T, seed=0):
    # subsamples features (used to experiment with partial observability on USHCN)
    rng = np.random.RandomState(seed)
    mask = []
    for i in range(T):
        mask_t = np.full(n_col, False, dtype=bool)
        n_unobserved_dimensions = rng.choice(n_col, 1, p=[0.6, 0.1, 0.1, 0.1, 0.1])
        unobserved_dimensions = rng.choice(
            n_col, n_unobserved_dimensions, replace=False
        )
        mask_t[unobserved_dimensions] = True
        mask.append(mask_t)
    return np.array(mask)


# new code component
def align_output_and_target(output_mean, output_var, targets, mask_targets):
    # removes last time point of output and first time point of target for one-step-ahead prediction
    output_mean = output_mean[:, :-1, ...]
    output_var = output_var[:, :-1, ...]
    targets = targets[:, 1:, ...]
    mask_targets = mask_targets[:, 1:, ...]
    return output_mean, output_var, targets, mask_targets


# new code component
def adjust_obs_for_extrapolation(
    obs, obs_valid, obs_times=None, mask_truth=None, cut_time=None
):
    obs_valid_extrap = obs_valid.clone()
    obs_extrap = obs.clone()
    mask_truth_eval = mask_truth.clone()  # for evaluation

    # zero out last half of observation (used for USHCN)
    if cut_time is None:
        n_observed_time_points = obs.shape[1] // 2
        obs_valid_extrap[:, n_observed_time_points:, ...] = False
        obs_extrap[:, n_observed_time_points:, ...] = 0

    # zero out observations at > cut_time (used for Physionet)
    else:
        # print(obs_times[0])
        # mask_before_cut_time = obs_times < cut_time
        mask_before_cut_time = torch.lt(obs_times, cut_time)
        obs_valid_extrap *= mask_before_cut_time
        # print(obs_valid_extrap)
        obs_extrap = torch.where(obs_valid_extrap[:, :, None], obs_extrap, 0.0)
        mask_truth_eval = torch.where(
            obs_valid_extrap[:, :, None], 0.0, mask_truth_eval
        )

    return obs_extrap, obs_valid_extrap, mask_truth_eval
