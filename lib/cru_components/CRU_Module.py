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
#
# This source code is derived from Pytorch RKN Implementation (https://github.com/ALRhub/rkn_share)
# Copyright (c) 2021 Philipp Becker (Autonomous Learning Robots Lab @ KIT)
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import os
import sys

sys.path.append("..")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time as t
from datetime import datetime
import os
from typing import Tuple

# from torch.utils.tensorboard import SummaryWriter
from lib.cru_components.cru_lib_utils import (
    TimeDistributed,
    log_to_tensorboard,
    make_dir,
)
from lib.cru_components.cru_encoder import Encoder
from lib.cru_components.cru_decoder import SplitDiagGaussianDecoder, BernoulliDecoder
from lib.cru_components.CRULayer import CRULayer
from lib.cru_components.CRUCell import var_activation, var_activation_inverse
from lib.cru_components.cru_losses import rmse, mse, GaussianNegLogLik, bernoulli_nll
from lib.cru_components.cru_data_utils import (
    align_output_and_target,
    adjust_obs_for_extrapolation,
)

import lib.utils as utils
import lib.evaluation as evaluation


# taken from https://github.com/ALRhub/rkn_share/ and modified
class CRU_Module(nn.Module):

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def __init__(
        self,
        target_dim: int,
        lsd: int,
        args,
        use_cuda_if_available: bool = True,
        bernoulli_output: bool = False,
    ):
        """
        :param target_dim: output dimension
        :param lsd: latent state dimension
        :param args: parsed arguments
        :param use_cuda_if_available: if to use cuda or cpu
        :param use_bernoulli_output: if to use a convolutional decoder (for image data)
        """
        super().__init__()
        self._device = args.device
        # target_dim = args.input_dim
        # lsd = args.latent_state_dim

        self._lsd = lsd
        if self._lsd % 2 == 0:
            self._lod = int(self._lsd / 2)
        else:
            raise Exception("Latent state dimension must be even number.")
        self.args = args

        # parameters TODO: Make configurable
        self._enc_out_normalization = "pre"
        self._initial_state_variance = 10.0
        self._learning_rate = self.args.lr
        self.bernoulli_output = bernoulli_output
        # main model

        self._cru_layer = CRULayer(latent_obs_dim=self._lod, args=args).to(self._device)

        Encoder._build_hidden_layers = self._build_enc_hidden_layers
        enc = Encoder(
            self._lod,
            output_normalization=self._enc_out_normalization,
            enc_var_activation=args.enc_var_activation,
        ).to(dtype=torch.float32)

        if bernoulli_output:
            BernoulliDecoder._build_hidden_layers = self._build_dec_hidden_layers
            self._dec = TimeDistributed(
                BernoulliDecoder(self._lod, out_dim=target_dim, args=args).to(
                    self._device, dtype=torch.float32
                ),
                num_outputs=1,
                low_mem=True,
            )
            self._enc = TimeDistributed(enc, num_outputs=2, low_mem=True).to(
                self._device
            )

        else:
            SplitDiagGaussianDecoder._build_hidden_layers_mean = (
                self._build_dec_hidden_layers_mean
            )
            SplitDiagGaussianDecoder._build_hidden_layers_var = (
                self._build_dec_hidden_layers_var
            )
            self._dec = TimeDistributed(
                SplitDiagGaussianDecoder(
                    self._lod,
                    out_dim=target_dim,
                    dec_var_activation=args.dec_var_activation,
                ).to(dtype=torch.float32),
                num_outputs=2,
            ).to(self._device)
            self._enc = TimeDistributed(enc, num_outputs=2).to(self._device)

        # build (default) initial state
        self._initial_mean = torch.zeros(1, self._lsd).to(
            self._device, dtype=torch.float32
        )
        log_ic_init = var_activation_inverse(self._initial_state_variance)
        self._log_icu = torch.nn.Parameter(
            log_ic_init * torch.ones(1, self._lod).to(self._device, dtype=torch.float32)
        )
        self._log_icl = torch.nn.Parameter(
            log_ic_init * torch.ones(1, self._lod).to(self._device, dtype=torch.float32)
        )
        self._ics = torch.zeros(1, self._lod).to(self._device, dtype=torch.float32)

        # params and optimizer
        self._params = list(self._enc.parameters())
        self._params += list(self._cru_layer.parameters())
        self._params += list(self._dec.parameters())
        self._params += [self._log_icu, self._log_icl]

        self._optimizer = optim.Adam(self._params, lr=self.args.lr)
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_enc_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_dec_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_dec_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for variance decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def forward(
        self,
        obs_batch: torch.Tensor,
        time_points: torch.Tensor = None,
        obs_valid: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass on a batch
        :param obs_batch: (B, L, D) batch of observation sequences
        :param time_points: (B, L) timestamps of observations
        :param obs_valid: (B, L) boolean if timestamp contains valid observation
        """
        # print(obs_batch.shape, time_points.shape, obs_valid.shape)
        y, y_var = self._enc(obs_batch)
        # print(y.shape, y_var.shape)

        post_mean, post_cov, prior_mean, prior_cov, kalman_gain = self._cru_layer(
            y,
            y_var,
            self._initial_mean,
            [var_activation(self._log_icu), var_activation(self._log_icl), self._ics],
            obs_valid=obs_valid,
            time_points=time_points,
        )
        # output an image
        if self.bernoulli_output:
            out_mean = self._dec(post_mean)
            out_var = None

        # # output prediction for the next time step
        # elif self.args.task == "one_step_ahead_prediction":
        #     out_mean, out_var = self._dec(prior_mean, torch.cat(prior_cov, dim=-1))

        # output filtered observation
        else:
            out_mean, out_var = self._dec(post_mean, torch.cat(post_cov, dim=-1))

        intermediates = {
            "post_mean": post_mean,
            "post_cov": post_cov,
            "prior_mean": prior_mean,
            "prior_cov": prior_cov,
            "kalman_gain": kalman_gain,
            "y": y,
            "y_var": y_var,
        }

        return out_mean, out_var, intermediates

    # new code component
    def interpolation(self, data, track_gradient=True):
        """Computes loss on interpolation task

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, outputs, inputs, intermediate variables, metrics on imputed points
        """
        if self.bernoulli_output:
            obs, truth, obs_valid, obs_times, mask_truth = [
                j.to(self._device) for j in data
            ]
            mask_obs = None
        else:
            obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
                j.to(self._device) for j in data
            ]

        obs_times = self.args.ts * obs_times

        with torch.set_grad_enabled(track_gradient):
            output_mean, output_var, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid
            )

            if self.bernoulli_output:
                loss = bernoulli_nll(truth, output_mean, uint8_targets=False)
                mask_imput = (~obs_valid[..., None, None, None]) * mask_truth
                imput_loss = np.nan  # TODO: compute bernoulli loss on imputed points
                imput_mse = mse(
                    truth.flatten(start_dim=2),
                    output_mean.flatten(start_dim=2),
                    mask=mask_imput.flatten(start_dim=2),
                )

            else:
                loss = GaussianNegLogLik(
                    output_mean, truth, output_var, mask=mask_truth
                )
                # compute metric on imputed points only
                mask_imput = (~obs_valid[..., None]) * mask_truth
                imput_loss = GaussianNegLogLik(
                    output_mean, truth, output_var, mask=mask_imput
                )
                imput_mse = mse(truth, output_mean, mask=mask_imput)

        return (
            loss,
            output_mean,
            output_var,
            obs,
            truth,
            mask_obs,
            mask_truth,
            intermediates,
            imput_loss,
            imput_mse,
        )

    # new code component
    # def extrapolation(self, data, track_gradient=True):
    #     """Computes loss on extrapolation task

    #     :param data: batch of data
    #     :param track_gradient: if to track gradient for backpropagation
    #     :return: loss, outputs, inputs, intermediate variables, metrics on imputed points
    #     """
    #     obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
    #         j.to(self._device) for j in data]
    #     obs, obs_valid = adjust_obs_for_extrapolation(
    #         obs, obs_valid, obs_times, self.args.cut_time)
    #     obs_times = self.args.ts * obs_times

    #     with torch.set_grad_enabled(track_gradient):
    #         output_mean, output_var, intermediates = self.forward(
    #             obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)

    #         loss = GaussianNegLogLik(
    #             output_mean, truth, output_var, mask=mask_truth)

    #         # compute metric on imputed points only
    #         mask_imput = (~obs_valid[..., None]) * mask_truth
    #         imput_loss = GaussianNegLogLik(
    #             output_mean, truth, output_var, mask=mask_imput)
    #         imput_mse = mse(truth, output_mean, mask=mask_imput)

    #     return loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse

    def forecasting(self, batch_dict, track_gradient=True):
        """Computes loss on forecasting task

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, outputs, inputs, intermediate variables, metrics on imputed points
        """
        # obs, targets, obs_valid, time_points, mask_targets, mask_obs
        # print(batch_dict.keys())
        obs = batch_dict["obs"]
        truth = batch_dict["targets"]
        obs_valid = batch_dict["obs_valid"]
        obs_times = batch_dict["time_points"]
        mask_truth = batch_dict["mask_targets"]
        mask_obs = batch_dict["mask_obs"]

        obs, obs_valid, mask_truth_eval = adjust_obs_for_extrapolation(
            obs, obs_valid, obs_times, mask_truth, self.args.history
        )
        obs_times = self.args.ts * obs_times

        # print(mask_truth.sum(), mask_truth_eval.sum())

        with torch.set_grad_enabled(track_gradient):
            output_mean, output_var, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid
            )
            # print(2222, output_mean.shape, output_var.shape)

            loss = GaussianNegLogLik(output_mean, truth, output_var, mask=mask_truth)

            # compute metric on imputed points only
            mask_imput = (~obs_valid[..., None]) * mask_truth
            imput_loss = GaussianNegLogLik(
                output_mean, truth, output_var, mask=mask_imput
            )
            imput_mse = mse(truth, output_mean, mask=mask_imput)

        return (
            loss,
            output_mean,
            output_var,
            obs,
            truth,
            mask_obs,
            mask_truth,
            intermediates,
            imput_loss,
            imput_mse,
            mask_truth_eval,
        )

    # new code component
    def regression(self, data, track_gradient=True):
        """Computes loss on regression task

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, input, intermediate variables and computed output
        """
        obs, truth, obs_times, obs_valid = [j.to(self._device) for j in data]
        mask_truth = None
        mask_obs = None
        with torch.set_grad_enabled(track_gradient):
            output_mean, output_var, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid
            )
            loss = GaussianNegLogLik(output_mean, truth, output_var, mask=mask_truth)

        return (
            loss,
            output_mean,
            output_var,
            obs,
            truth,
            mask_obs,
            mask_truth,
            intermediates,
        )

    # new code component
    def one_step_ahead_prediction(self, data, track_gradient=True):
        """Computes loss on one-step-ahead prediction

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, input, intermediate variables and computed output
        """
        obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
            j.to(self._device) for j in data
        ]
        obs_times = self.args.ts * obs_times
        with torch.set_grad_enabled(track_gradient):
            output_mean, output_var, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid
            )
            output_mean, output_var, truth, mask_truth = align_output_and_target(
                output_mean, output_var, truth, mask_truth
            )
            loss = GaussianNegLogLik(output_mean, truth, output_var, mask=mask_truth)

        return (
            loss,
            output_mean,
            output_var,
            obs,
            truth,
            mask_obs,
            mask_truth,
            intermediates,
        )

    # new code component
    def train_epoch(self, dl, optimizer, num_batches):
        """Trains model for one epoch

        :param dl: dataloader containing training data
        :param optimizer: optimizer to use for training
        :return: evaluation metrics, computed output, input, intermediate variables
        """
        epoch_ll = 0
        epoch_rmse = 0
        epoch_mse = 0

        if self.args.save_intermediates is not None:
            mask_obs_epoch = []
            intermediates_epoch = []

        if self.args.task == "extrapolation" or self.args.task == "interpolation":
            epoch_imput_ll = 0
            epoch_imput_mse = 0

        for i in range(num_batches):
            batch_dict = utils.get_next_batch(dl)

            if self.args.task == "interpolation":
                (
                    loss,
                    output_mean,
                    output_var,
                    obs,
                    truth,
                    mask_obs,
                    mask_truth,
                    intermediates,
                    imput_loss,
                    imput_mse,
                ) = self.interpolation(data)

            # elif self.args.task == 'extrapolation':
            #     loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.extrapolation(
            #         data)

            elif self.args.task == "extrapolation":
                (
                    loss,
                    output_mean,
                    output_var,
                    obs,
                    truth,
                    mask_obs,
                    mask_truth,
                    intermediates,
                    imput_loss,
                    imput_mse,
                    mask_truth_eval,
                ) = self.forecasting(batch_dict)

            elif self.args.task == "regression":
                (
                    loss,
                    output_mean,
                    output_var,
                    obs,
                    truth,
                    mask_obs,
                    mask_truth,
                    intermediates,
                ) = self.regression(data)

            elif self.args.task == "one_step_ahead_prediction":
                (
                    loss,
                    output_mean,
                    output_var,
                    obs,
                    truth,
                    mask_obs,
                    mask_truth,
                    intermediates,
                ) = self.one_step_ahead_prediction(data)

            else:
                raise Exception("Unknown task")

            # check for NaNs
            if torch.any(torch.isnan(loss)):
                print("--NAN in loss")
            for name, par in self.named_parameters():
                if torch.any(torch.isnan(par)):
                    print("--NAN before optimiser step in parameter ", name)
            torch.autograd.set_detect_anomaly(self.args.anomaly_detection)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            if self.args.grad_clip:
                nn.utils.clip_grad_norm_(self.parameters(), 1)
            optimizer.step()

            # check for NaNs in gradient
            for name, par in self.named_parameters():
                if torch.any(torch.isnan(par.grad)):
                    print("--NAN in gradient ", name)
                if torch.any(torch.isnan(par)):
                    print("--NAN after optimiser step in parameter ", name)

        return loss

    def evaluation(self, dataloader, n_batches):
        n_eval_samples = 0
        n_eval_samples_mape = 0
        total_results = {}
        total_results["loss"] = 0
        total_results["mse"] = 0
        total_results["mae"] = 0
        total_results["rmse"] = 0
        total_results["mape"] = 0

        for _ in range(n_batches):
            batch_dict = utils.get_next_batch(dataloader)
            # bs = batch_dict["observed_data"].shape[0]
            # print(batch_dict["observed_data"].shape)

            (
                loss,
                output_mean,
                output_var,
                obs,
                truth,
                mask_obs,
                mask_truth,
                intermediates,
                imput_loss,
                imput_mse,
                mask_truth_eval,
            ) = self.forecasting(batch_dict, track_gradient=False)

            pred_y = output_mean.unsqueeze(dim=0)

            # print(truth.shape, output_mean.shape, mask_truth.shape)
            # print(truth[0])
            # print(mask_truth[0])
            # print(truth[mask_truth_eval==1].sum(), mask_truth_eval.sum())

            se_var_sum, mask_count = evaluation.compute_error(
                truth, pred_y, mask=mask_truth_eval, func="MSE", reduce="sum"
            )  # a vector

            ae_var_sum, _ = evaluation.compute_error(
                truth, pred_y, mask=mask_truth_eval, func="MAE", reduce="sum"
            )  # a vector

            # norm_dict = {"data_max": batch_dict["data_max"], "data_min": batch_dict["data_min"]}
            ape_var_sum, mask_count_mape = evaluation.compute_error(
                truth, pred_y, mask=mask_truth_eval, func="MAPE", reduce="sum"
            )  # a vector

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

        # print(total_results, n_eval_samples.long(), n_avai_var)

        return total_results

    # # new code component
    # def eval_epoch(self, dl):
    #     """Evaluates model on the entire dataset

    #     :param dl: dataloader containing validation or test data
    #     :return: evaluation metrics, computed output, input, intermediate variables
    #     """
    #     epoch_ll = 0
    #     epoch_rmse = 0
    #     epoch_mse = 0

    #     if self.args.task == 'extrapolation' or self.args.task == 'interpolation':
    #         epoch_imput_ll = 0
    #         epoch_imput_mse = 0

    #     if self.args.save_intermediates is not None:
    #         mask_obs_epoch = []
    #         intermediates_epoch = []

    #     for i, data in enumerate(dl):

    #         if self.args.task == 'interpolation':
    #             loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.interpolation(
    #                 data, track_gradient=False)

    #         elif self.args.task == 'extrapolation':
    #             loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.extrapolation(
    #                 data, track_gradient=False)

    #         elif self.args.task == 'regression':
    #             loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates = self.regression(
    #                 data, track_gradient=False)

    #         elif self.args.task == 'one_step_ahead_prediction':
    #             loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates = self.one_step_ahead_prediction(
    #                 data, track_gradient=False)

    #         epoch_ll += loss
    #         epoch_rmse += rmse(truth, output_mean, mask_truth).item()
    #         epoch_mse += mse(truth, output_mean, mask_truth).item()

    #         if self.args.task == 'extrapolation' or self.args.task == 'interpolation':
    #             epoch_imput_ll += imput_loss
    #             epoch_imput_mse += imput_mse
    #             imput_metrics = [epoch_imput_ll/(i+1), epoch_imput_mse/(i+1)]
    #         else:
    #             imput_metrics = None

    #         if self.args.save_intermediates is not None:
    #             intermediates_epoch.append(intermediates)
    #             mask_obs_epoch.append(mask_obs)

    #     # save for plotting
    #     if self.args.save_intermediates is not None:
    #         torch.save(output_mean, os.path.join(
    #             self.args.save_intermediates, 'valid_output_mean.pt'))
    #         torch.save(obs, os.path.join(
    #             self.args.save_intermediates, 'valid_obs.pt'))
    #         torch.save(output_var, os.path.join(
    #             self.args.save_intermediates, 'valid_output_var.pt'))
    #         torch.save(truth, os.path.join(
    #             self.args.save_intermediates, 'valid_truth.pt'))
    #         torch.save(intermediates_epoch, os.path.join(
    #             self.args.save_intermediates, 'valid_intermediates.pt'))
    #         torch.save(mask_obs_epoch, os.path.join(
    #             self.args.save_intermediates, 'valid_mask_obs.pt'))

    #     return epoch_ll/(i+1), epoch_rmse/(i+1), epoch_mse/(i+1), [output_mean, output_var], intermediates, [obs, truth, mask_obs], imput_metrics

    # # new code component
    # def raw_train(self, train_dl, valid_dl, identifier, logger, epoch_start=0):
    #     """Trains model on trainset and evaluates on test data. Logs results and saves trained model.

    #     :param train_dl: training dataloader
    #     :param valid_dl: validation dataloader
    #     :param identifier: logger id
    #     :param logger: logger object
    #     :param epoch_start: starting epoch
    #     """

    #     optimizer = optim.Adam(self.parameters(), self.args.lr)
    #     def lr_update(epoch): return self.args.lr_decay ** epoch
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(
    #         optimizer, lr_lambda=lr_update)

    #     # make_dir(f'../results/tensorboard/{self.args.dataset}')
    #     # writer = SummaryWriter(f'../results/tensorboard/{self.args.dataset}/{identifier}')

    #     for epoch in range(epoch_start, self.args.epochs):
    #         start = datetime.now()
    #         logger.info(f'Epoch {epoch} starts: {start.strftime("%H:%M:%S")}')

    #         # train
    #         train_ll, train_rmse, train_mse, train_output, intermediates, train_input, train_imput_metrics = self.train_epoch(
    #             train_dl, optimizer)
    #         end_training = datetime.now()
    #         if self.args.tensorboard:
    #             log_to_tensorboard(self, writer=writer,
    #                             mode='train',
    #                             metrics=[train_ll, train_rmse, train_mse],
    #                             output=train_output,
    #                             input=train_input,
    #                             intermediates=intermediates,
    #                             epoch=epoch,
    #                             imput_metrics=train_imput_metrics,
    #                             log_rythm=self.args.log_rythm)

    #         # eval
    #         valid_ll, valid_rmse, valid_mse, valid_output, intermediates, valid_input, valid_imput_metrics = self.eval_epoch(
    #             valid_dl)
    #         if self.args.tensorboard:
    #             log_to_tensorboard(self, writer=writer,
    #                             mode='valid',
    #                             metrics=[valid_ll, valid_rmse, valid_mse],
    #                             output=valid_output,
    #                             input=valid_input,
    #                             intermediates=intermediates,
    #                             epoch=epoch,
    #                             imput_metrics=valid_imput_metrics,
    #                             log_rythm=self.args.log_rythm)

    #         end = datetime.now()
    #         logger.info(f'Training epoch {epoch} took: {(end_training - start).total_seconds()}')
    #         logger.info(f'Epoch {epoch} took: {(end - start).total_seconds()}')
    #         logger.info(f' train_nll: {train_ll:3f}, train_mse: {train_mse:3f}')
    #         logger.info(f' valid_nll: {valid_ll:3f}, valid_mse: {valid_mse:3f}')
    #         if self.args.task == 'extrapolation' or self.args.impute_rate is not None:
    #             if self.bernoulli_output:
    #                 logger.info(f' train_mse_imput: {train_imput_metrics[1]:3f}')
    #                 logger.info(f' valid_mse_imput: {valid_imput_metrics[1]:3f}')
    #             else:
    #                 logger.info(f' train_nll_imput: {train_imput_metrics[0]:3f}, train_mse_imput: {train_imput_metrics[1]:3f}')
    #                 logger.info(f' valid_nll_imput: {valid_imput_metrics[0]:3f}, valid_mse_imput: {valid_imput_metrics[1]:3f}')

    #         scheduler.step()

    #     make_dir(f'../results/models/{self.args.dataset}')
    #     torch.save({'epoch': epoch,
    #                 'model_state_dict': self.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'loss': train_ll,
    #                 }, f'../results/models/{self.args.dataset}/{identifier}.tar')
