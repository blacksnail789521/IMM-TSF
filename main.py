import os
import sys

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.optim as optim

from utils.tools import set_seed, print_formatted_dict, select_best_metrics
import lib.utils as utils
from lib.evaluation import compute_all_losses, evaluation

# from lib.parse_datasets_old import parse_datasets, get_input_and_pred_len

from lib.parse_datasets import parse_datasets, get_input_and_pred_len
from models.tPatchGNN import tPatchGNN
from models.TimesNet import TimesNet
from models.DLinear import DLinear
from models.PatchTST import PatchTST
from models.NeuralFlow import NeuralFlow
from models.CRU import CRU
from models.LatentODE import LatentODE  # Added LatentODE import

# from models.MTGNN import MTGNN
from models.Informer import Informer

# from models.TimeXer import TimeXer
from models.TimeMixer import TimeMixer
from models.TimeLLM import TimeLLM
from models.TTM import TTM

from fusions.FusionModel import FusionModel
from fusions.load_llm import get_context_window_size


def get_args_from_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMMTSF")

    # ── General / Execution Options ──────────────────────────────────────────────
    parser.add_argument(
        "--overwrite_args",
        action="store_true",
        help="overwrite args with fixed_params and tunable_params",
        default=False,
    )
    parser.add_argument(
        "--state",
        type=str,
        default="def",
        help='State of the experiment (e.g., "def", "train", "eval")',
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID to use")

    # ── Paths & Data Selection ───────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default="FNSPID", help="Which dataset to load"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for all data files",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=int(1e8),
        help="Max size of the dataset (number of samples)",
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default="sample",
        choices=["instance", "sample"],
        help="Method to split the dataset into train/val/test",
    )
    parser.add_argument(
        "--enable_text",
        action="store_true",
        help="Enable multimodal text data",
    )
    parser.add_argument(
        "--use_text_embeddings",
        action="store_true",
        help="Enable pre-computed text embeddings",
    )

    # ── Data Processing / Windowing ──────────────────────────────────────────────
    parser.add_argument(
        "--time_unit",
        type=str,
        default="days",
        choices=["seconds", "minutes", "hours", "days", "weeks", "custom"],
        help="Time unit for the dataset",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=24,
        help="Historical window length (hours for physionet/mimic, months for ushcn)",
    )
    parser.add_argument(
        "--pred_window",
        type=int,
        default=24,
        help="Forecast horizon (prediction window)",
    )
    parser.add_argument(
        "--stride", type=int, default=24, help="Stride between consecutive patches"
    )

    # ── Temporal Patching (t-PatchGNN-specific) ──────────────────────────────────
    parser.add_argument(
        "-ps", "--patch_size", type=int, default=24, help="Size of each temporal patch"
    )
    parser.add_argument(
        "--npatch",
        type=int,
        default=None,
        help="Number of patches (default: history/patch_size)",
    )
    parser.add_argument(
        "--patch_stride",
        type=int,
        default=None,
        help="Stride between patches (defaults to patch_size)",
    )

    # ── Model Selection & Architecture ───────────────────────────────────────────
    parser.add_argument(
        "--model", type=str, default="tPatchGNN", help="Model architecture to use"
    )
    parser.add_argument(
        "--outlayer", type=str, default="Linear", help="Type of final output layer"
    )
    parser.add_argument(
        "-hd",
        "--hid_dim",
        type=int,
        default=64,
        help="Hidden units per layer (also default for some NF/CRU/ODE params)",
    )
    parser.add_argument(
        "-td", "--te_dim", type=int, default=10, help="Units for time‐encoding vectors"
    )
    parser.add_argument(
        "-nd",
        "--node_dim",
        type=int,
        default=10,
        help="Units for node‐embedding vectors",
    )
    parser.add_argument("--hop", type=int, default=1, help="Number of GNN hops")
    parser.add_argument(
        "--tf_layer", type=int, default=1, help="Number of Transformer layers"
    )
    parser.add_argument(
        "--nlayer",
        type=int,
        default=1,
        help="Number of layers in the time‐series backbone",
    )
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=2, help="num of heads")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument(
        "--down_sampling_layers",
        type=int,
        default=3,
        help="num of down sampling layers",
    )
    parser.add_argument(
        "--down_sampling_window", type=int, default=2, help="down sampling window size"
    )
    parser.add_argument(
        "--down_sampling_method",
        type=str,
        default="avg",
        help="down sampling method, only support avg, max, conv",
    )
    parser.add_argument(
        "--decomp_method",
        type=str,
        default="moving_avg",
        help="method of series decompsition, only support moving_avg or dft_decomp",
    )
    parser.add_argument(
        "--channel_independence",
        type=int,
        default=1,
        help="0: channel dependence 1: channel independence for FreTS model",
    )
    parser.add_argument(
        "--use_norm",
        type=int,
        default=1,
        help="whether to use normalize; True 1 False 0",
    )

    # TTM
    parser.add_argument("--n_vars", type=int, default=7, help="number of variables")
    parser.add_argument(
        "--mode",
        type=str,
        default="mix_channel",
        help="allowed values: common_channel, mix_channel",
    )
    parser.add_argument(
        "--AP_levels", type=int, default=3, help="number of attention patching levels"
    )
    parser.add_argument(
        "--use_decoder", action="store_true", help="use decoder", default=True
    )
    parser.add_argument(
        "--d_mode",
        type=str,
        default="common_channel",
        help="allowed values: common_channel, mix_channel",
    )
    parser.add_argument("--d_d_model", type=int, default=64, help="d_model in decoder")

    # Time-LLM
    parser.add_argument(
        "--ts_vocab_size",
        type=int,
        default=1000,
        help="size of a small collection of text prototypes in llm",
    )
    parser.add_argument(
        "--domain_des",
        type=str,
        default="The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.",
        help="domain description",
    )
    parser.add_argument(
        "--input_token_len", type=int, default=576, help="input token length"
    )
    parser.add_argument(
        "--output_token_len", type=int, default=96, help="output token length"
    )
    parser.add_argument(
        "--llm_model_timellm",
        type=str,
        default="GPT2",
        help="LLM model (for TimeLLM), LLAMA, GPT2, BERT, OPT are supported",
    )
    parser.add_argument(
        "--llm_layers_timellm", type=int, default=6, help="number of layers in llm"
    )

    # ── NeuralFlow Specific Hyperparameters ──────────────────────────────────────
    parser.add_argument(
        "--nf_latents", type=int, default=20, help="NeuralFlow: Latent dimension"
    )
    parser.add_argument(
        "--nf_rec_dims", type=int, default=40, help="NeuralFlow: Recognition dimensions"
    )
    parser.add_argument(
        "--nf_gru_units",
        type=int,
        default=32,
        help="NeuralFlow: GRU units",
    )
    parser.add_argument(
        "--nf_hidden_layers",
        type=int,
        default=3,
        help="NeuralFlow: Number of hidden layers in ODE func",
    )
    parser.add_argument(
        "--nf_hidden_dim",
        type=int,
        default=32,
        help="NeuralFlow: Hidden dimension in ODE func",
    )
    parser.add_argument(
        "--nf_flow_model",
        type=str,
        default="coupling",
        choices=["coupling", "resnet", "gru"],
        help="NeuralFlow: Type of flow model",
    )
    parser.add_argument(
        "--nf_flow_layers",
        type=int,
        default=2,
        help="NeuralFlow: Number of flow layers",
    )
    parser.add_argument(
        "--nf_time_net",
        type=str,
        default="TimeLinear",
        help="NeuralFlow: Time network type",
    )
    parser.add_argument(
        "--nf_time_hidden_dim",
        type=int,
        default=8,
        help="NeuralFlow: Time network hidden dimension",
    )
    parser.add_argument(
        "--nf_solver", type=str, default="dopri5", help="NeuralFlow: ODE solver"
    )
    parser.add_argument(
        "--nf_solver_step",
        type=float,
        default=0.05,
        help="NeuralFlow: Solver step size",
    )
    parser.add_argument(
        "--nf_atol",
        type=float,
        default=1e-4,
        help="NeuralFlow: Absolute tolerance for solver",
    )
    parser.add_argument(
        "--nf_rtol",
        type=float,
        default=1e-3,
        help="NeuralFlow: Relative tolerance for solver",
    )
    parser.add_argument(
        "--nf_odenet", type=str, default="concat", help="NeuralFlow: ODE network type"
    )
    parser.add_argument(
        "--nf_activation",
        type=str,
        default="Tanh",
        help="NeuralFlow: Activation function",
    )
    parser.add_argument(
        "--nf_final_activation",
        type=str,
        default="Identity",
        help="NeuralFlow: Final activation function",
    )
    parser.add_argument(
        "--nf_obsrv_std",
        type=float,
        default=0.01,
        help="NeuralFlow: Observation standard deviation",
    )
    parser.add_argument(
        "--nf_weight_decay",
        type=float,
        default=0.0001,
        help="NeuralFlow: Weight decay for internal optimizer (if applicable)",
    )
    parser.add_argument(
        "--nf_quantization",
        type=float,
        default=0.0,
        help="NeuralFlow: Quantization parameter",
    )
    parser.add_argument(
        "--nf_max_t",
        type=float,
        default=5.0,
        help="NeuralFlow: Max time for ODE integration",
    )
    parser.add_argument(
        "--nf_mixing", type=float, default=0.0001, help="NeuralFlow: Mixing coefficient"
    )
    parser.add_argument(
        "--nf_gob_prep_hidden",
        type=int,
        default=10,
        help="NeuralFlow: GOB prep hidden units",
    )
    parser.add_argument(
        "--nf_gob_cov_hidden",
        type=int,
        default=50,
        help="NeuralFlow: GOB cov hidden units",
    )
    parser.add_argument(
        "--nf_gob_p_hidden", type=int, default=25, help="NeuralFlow: GOB p hidden units"
    )
    parser.add_argument(
        "--nf_invertible",
        type=int,
        default=1,
        help="NeuralFlow: Invertible flag (0 or 1)",
    )
    parser.add_argument(
        "--nf_components", type=int, default=8, help="NeuralFlow: Number of components"
    )
    parser.add_argument(
        "--nf_decoder_type",
        type=str,
        default="continuous",
        help="NeuralFlow: Decoder type",
    )
    parser.add_argument(
        "--nf_rnn", type=str, default="gru", help="NeuralFlow: RNN type"
    )
    parser.add_argument(
        "--nf_marks", type=int, default=0, help="NeuralFlow: Marks flag (0 or 1)"
    )
    parser.add_argument(
        "--nf_density_model",
        type=str,
        default="independent",
        help="NeuralFlow: Density model type",
    )
    parser.add_argument(
        "--nf_extrap",
        type=int,
        default=0,
        help="NeuralFlow: Extrapolation flag (0 or 1)",
    )

    # ── CRU Specific Hyperparameters ─────────────────────────────────────────────
    parser.add_argument(
        "--cru_lsd",
        type=int,
        default=None,
        help="CRU: Latent state dimension (defaults to hid_dim if None)",
    )
    parser.add_argument(
        "--cru_hidden_units",
        type=int,
        default=None,
        help="CRU: Hidden units for internal MLPs (defaults to hid_dim if None)",
    )
    parser.add_argument(
        "--cru_enc_num_layers",
        type=int,
        default=1,
        help="CRU: Number of encoder layers",
    )
    parser.add_argument(
        "--cru_dec_num_layers",
        type=int,
        default=1,
        help="CRU: Number of decoder layers",
    )
    parser.add_argument(
        "--cru_num_layers", type=int, default=1, help="CRU: Number of CRU layers"
    )
    parser.add_argument(
        "--cru_dropout_type",
        type=str,
        default="None",
        choices=["None", "Zoneout", "Variational"],
        help="CRU: Dropout type",
    )
    parser.add_argument(
        "--cru_dropout_rate", type=float, default=0.0, help="CRU: Dropout rate"
    )
    parser.add_argument(
        "--cru_use_gate_hidden_states",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="CRU: Use gate hidden states (True/False)",
    )
    parser.add_argument(
        "--cru_use_ode_for_gru",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="CRU: Use ODE for GRU (True/False)",
    )
    parser.add_argument(
        "--cru_use_decay_gravity_gate",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="CRU: Use decay gravity gate (True/False)",
    )
    parser.add_argument(
        "--cru_use_gravity_gate",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="CRU: Use gravity gate (True/False)",
    )
    parser.add_argument(
        "--cru_use_decay_input_gate",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="CRU: Use decay input gate (True/False)",
    )
    parser.add_argument(
        "--cru_use_input_gate",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="CRU: Use input gate (True/False)",
    )
    parser.add_argument(
        "--cru_use_skip_connection",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="CRU: Use skip connection (True/False)",
    )
    parser.add_argument(
        "--cru_solver",
        type=str,
        default="euler",
        choices=["euler", "rk4"],
        help="CRU: Solver for ODEs if used",
    )
    parser.add_argument(
        "--ts",
        type=float,
        default=0.3,
        help="Scaling factor of timestamps for numerical stability.",
    )
    parser.add_argument(
        "--grad_clip", action="store_true", help="If to use gradient clipping."
    )

    # ── LatentODE Specific Hyperparameters ───────────────────────────────────────
    parser.add_argument(
        "--ode_latents", type=int, default=20, help="LatentODE: Latent dimension"
    )
    parser.add_argument(
        "--ode_units",
        type=int,
        default=32,
        help="LatentODE: Units in ODE function network",
    )
    parser.add_argument(
        "--ode_gen_layers",
        type=int,
        default=1,
        help="LatentODE: Layers in ODE function generator",
    )
    parser.add_argument(
        "--ode_rec_dims",
        type=int,
        default=32,
        help="LatentODE: Recognition RNN hidden dimensions",
    )
    parser.add_argument(
        "--ode_rec_layers",
        type=int,
        default=1,
        help="LatentODE: Layers in recognition RNN",
    )
    parser.add_argument(
        "--ode_gru_units",
        type=int,
        default=32,
        help="LatentODE: GRU units in recognition RNN (defaults to hid_dim if None)",
    )
    parser.add_argument(
        "--ode_poisson",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="LatentODE: Use Poisson process for observations (True/False)",
    )
    parser.add_argument(
        "--ode_classif",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="LatentODE: Perform classification task (True/False)",
    )
    parser.add_argument(
        "--ode_linear_classif",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="LatentODE: Use linear classifier (True/False)",
    )
    parser.add_argument(
        "--ode_z0_encoder",
        type=str,
        default="odernn",
        choices=["odernn", "rnn"],
        help="LatentODE: Type of encoder for z0",
    )
    parser.add_argument(
        "--ode_obsrv_std",
        type=float,
        default=0.01,
        help="LatentODE: Observation standard deviation",
    )
    parser.add_argument(
        "--ode_n_traj_samples",
        type=int,
        default=1,
        help="LatentODE: Number of trajectory samples for reconstruction",
    )

    # ── Fusion Modules ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--TTF_module",
        type=str,
        default="TTF_T2V_XAttn",
        choices=["TTF_RecAvg", "TTF_T2V_XAttn"],
        help="Timestamp-to-Time Fusion module",
    )
    parser.add_argument(
        "--MMF_module",
        type=str,
        default="MMF_XAttn_Add",
        choices=["MMF_GR_Add", "MMF_XAttn_Add"],
        help="Multimodal Fusion module",
    )
    parser.add_argument(
        "--llm_model_fusion",
        type=str,
        default="GPT2",
        help="LLM model (for Fusion), LLAMA, GPT2, BERT are supported",
    )
    parser.add_argument(
        "--llm_layers_fusion",
        type=int,
        default=6,
        help="number of layers in llm fusion",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max tokens per note (for TTF)",
    )
    parser.add_argument(
        "--d_txt",
        type=int,
        default=768,
        help="Text embedding dimension (for TTF)",
    )
    parser.add_argument(
        "--recency_sigma",
        type=float,
        default=1.0,
        help="Recency sigma for TTF_RecAvg module",
    )
    parser.add_argument(
        "--n_heads_fusion",
        type=int,
        default=1,
        help="Number of attention heads for Fusion modules",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.5,
        help="Weight for the text‐time fusion module in MMF_XAttn_Add",
    )

    # ── Training Hyperparameters ─────────────────────────────────────────────────
    parser.add_argument("--epoch", type=int, default=1000, help="Max training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        # default=10,
        help="Early‐stopping patience",
    )
    parser.add_argument(
        "--early_stop_delta",
        type=float,
        default=1e-4,
        help="Minimum change in the monitored metric to qualify as improvement",
    )

    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--w_decay",
        type=float,
        # default=0.0,
        # default=0.001,
        default=0.01,
        help="Weight‐decay (L2 regularization)",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        # default=True,
        help="Enable Automatic Mixed Precision (AMP) training",
    )

    # ── Logging & Checkpointing ──────────────────────────────────────────────────
    parser.add_argument(
        "--logmode", type=str, default="a", help='File mode for logging (e.g. "w", "a")'
    )
    parser.add_argument(
        "--save",
        type=str,
        default="experiments/",
        help="Directory in which to save model checkpoints",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Experiment ID to load for evaluation (if any)",
    )

    args = parser.parse_args()

    # Default nf_gru_units and nf_hidden_dim to args.hid_dim if not provided
    if args.nf_gru_units is None:
        args.nf_gru_units = args.hid_dim
    if args.nf_hidden_dim is None:
        args.nf_hidden_dim = args.hid_dim

    # Default cru_lsd and cru_hidden_units to args.hid_dim if not provided
    if args.cru_lsd is None:
        args.cru_lsd = args.hid_dim
    if args.cru_hidden_units is None:
        args.cru_hidden_units = args.hid_dim

    # Note: ode_gru_units default handling is now inside the LatentODE model wrapper,
    # based on args.hid_dim if args.ode_gru_units is None when passed.

    args.npatch = (
        int(np.ceil((args.history - args.patch_size) / args.stride)) + 1
    )  # (window size for a patch)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    file_name = os.path.basename(__file__)[:-3]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")  # use cpu to debug
    args.PID = os.getpid()
    print("PID, device:", args.PID, args.device)

    return args


def update_args_from_fixed_params(
    args: argparse.Namespace, fixed_params: dict
) -> argparse.Namespace:
    # Update args
    for key, value in fixed_params.items():
        if not hasattr(args, key):
            print(f"AttributeError: {key} not found in args")
        print("### [Fixed] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args_from_tunable_params(
    args: argparse.Namespace, tunable_params: dict
) -> argparse.Namespace:
    # Update args
    for key, value in tunable_params.items():
        if not hasattr(args, key):
            print(f"AttributeError: {key} not found in args")
        print("### [Tunable] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args_for_dataset(args: argparse.Namespace) -> argparse.Namespace:
    # Update args based on the dataset
    if args.dataset == "GDELT":
        args.history = 14
        args.pred_window = 14
        args.stride = 14
        args.time_unit = "days"
    elif args.dataset == "RepoHealth":
        args.history = 31
        args.pred_window = 31
        args.stride = 31
        args.time_unit = "days"
    elif args.dataset == "MIMIC":
        args.history = 24
        args.pred_window = 24
        args.stride = 24
        args.time_unit = "hours"
    elif args.dataset == "FNSPID":
        args.history = 31
        args.pred_window = 31
        args.stride = 31
        args.time_unit = "days"
    elif args.dataset == "ClusterTrace":
        args.history = 12
        args.pred_window = 12
        args.stride = 12
        args.time_unit = "hours"
    elif args.dataset == "StudentLife":
        args.history = 31
        args.pred_window = 31
        args.stride = 31
        args.time_unit = "days"
    elif args.dataset == "ILINet":
        args.history = 36
        args.pred_window = 36
        args.stride = 4
        args.time_unit = "weeks"
    elif args.dataset == "CESNET":
        args.history = 7
        args.pred_window = 7
        args.stride = 7
        args.time_unit = "days"
    elif args.dataset == "EPA-Air":
        args.history = 7
        args.pred_window = 7
        args.stride = 7
        args.time_unit = "days"

    return args


def update_args_for_model(args: argparse.Namespace) -> argparse.Namespace:
    # Update args based on the model
    # ? MTS
    if args.model == "Informer":
        args.e_layers = 2
        args.d_layers = 1
        args.factor = 3
    elif args.model == "DLinear":
        pass
    elif args.model == "PatchTST":
        args.e_layers = 1
        args.d_layers = 1
        args.n_heads = 2
    elif args.model == "TimesNet":
        args.e_layers = 2
        args.d_layers = 1
        args.factor = 3
        args.d_model = 16
        args.d_ff = 32
        args.top_k = 5
    elif args.model == "TimeMixer":
        args.e_layers = 2
        args.d_model = 16
        args.d_ff = 32
        args.down_sampling_layers = 3
        args.down_sampling_method = "avg"
        args.down_sampling_window = 2
    # ? LMTS
    elif args.model == "TimeLLM":
        args.input_token_len = 16
        args.output_token_len = 96
        args.d_model = 32
        args.d_ff = 128
        args.llm_model_timellm = "GPT2"
        args.llm_layers_timellm = 6
    elif args.model == "TTM":
        args.input_token_len = 16
        args.output_token_len = 96
        args.d_model = 1024
        args.AP_levels = 3
        args.e_layers = 3
        args.d_layers = 2
        args.d_d_model = 64
        args.patch_size = args.history // 4

        # args.history = 96
        # args.pred_window = 96
        # args.stride = 96
        # args.time_unit = "days"
    # ? IMTS
    elif args.model == "CRU":
        args.cru_lsd = 32
        args.cru_hidden_units = 32
        args.ts = 0.3
        args.cru_enc_var_activation = "square"
        args.cru_dec_var_activation = "exp"
        args.grad_clip = True
    elif args.model == "LatentODE":
        args.ode_rec_dims = 32
        args.ode_units = 32
        args.ode_gru_units = 32
        args.ode_rec_layers = 1
        args.ode_gen_layers = 1
    elif args.model == "NeuralFlow":
        args.nf_extrap = 0
        args.nf_hidden_layers = 3
        args.nf_hidden_dim = 32
        args.nf_rec_dims = 40
        args.nf_latents = 20
        args.nf_gru_units = 32
        args.nf_flow_model = "coupling"
        args.nf_flow_layers = 2
        args.nf_time_net = "TimeLinear"
        args.nf_time_hidden_dim = 8
    elif args.model == "tPatchGNN":
        args.patch_size = 24
        args.n_heads = 1
        args.tf_layer = 1
        args.nlayer = 1
        args.te_dim = 10
        args.node_dim = 10
        args.hid_dim = 32
        args.outlayer = "Linear"

    return args


def update_args(
    args: argparse.Namespace,
    fixed_params: dict,
    tunable_params: dict,
) -> argparse.Namespace:
    # Check if there are duplicated keys
    duplicated_keys = set(fixed_params.keys()) & set(tunable_params.keys())
    assert not duplicated_keys, f"Duplicated keys found: {duplicated_keys}"

    # Update args from fixed_params, tunable_params, and dataset
    if args.overwrite_args:
        args = update_args_from_fixed_params(args, fixed_params)
        args = update_args_from_tunable_params(args, tunable_params)
        args = update_args_for_dataset(args)
        args = update_args_for_model(args)

    return args


def trainable(
    tunable_params: dict,
    fixed_params: dict,
    args: argparse.Namespace,
) -> dict:
    # Update args
    args = update_args(args, fixed_params, tunable_params)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + ".ckpt")

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2) :]
    input_command = " ".join(input_command)

    ##################################################################
    # Update max_length if needed
    if args.enable_text:
        args.max_length = 512 if args.llm_model_fusion == "BERT" else 1024
        # context_window_size = get_context_window_size(
        #     args.llm_model_fusion, args.device
        # )
        # if args.max_length > context_window_size:
        #     print(
        #         f"Overriding max_length from {args.max_length} to {context_window_size}"
        #         " to match the LLM model's context window size."
        #     )
        #     args.max_length = context_window_size

    # Pass model name to parse_datasets to select the correct collate_fn
    data_obj = parse_datasets(args)

    ### Model setting ###
    args.C = data_obj["input_dim"]
    args.enc_in = args.C
    args.c_out = args.C
    args.input_len, args.pred_len = get_input_and_pred_len(data_obj)
    model_class = globals()[args.model]
    model = model_class(args).to(args.device)
    fusion = FusionModel(args).to(args.device) if args.enable_text else None

    ##################################################################

    if args.n < 12000:
        args.state = "debug"
        log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
    else:
        log_path = "logs/{}_{}_{}_{}patch_{}stride_{}layer_{}lr.log".format(
            args.dataset,
            args.model,
            args.state,
            args.patch_size,
            args.stride,
            args.nlayer,
            args.lr,
        )

    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(
        logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode
    )
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)

    # Set trainable_parameters
    if not args.enable_text:
        trainable_parameters = list(model.parameters())
    else:
        assert fusion is not None
        trainable_parameters = list(model.parameters()) + list(fusion.parameters())

    optimizer = optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.w_decay)

    def _nan_hook(module, inputs, output, name):
        # output might be a Tensor or tuple of Tensors
        outs = output if isinstance(output, (list, tuple)) else (output,)
        for o in outs:
            if isinstance(o, torch.Tensor) and torch.isnan(o).any():
                raise RuntimeError(f"NaN in forward of {name}")

    def register_forward_nan_checks(model):
        for name, module in model.named_modules():
            module.register_forward_hook(
                lambda mod, inp, out, name=name: _nan_hook(mod, inp, out, name)
            )

    def _grad_hook(grad, name):
        if torch.isnan(grad).any():
            raise RuntimeError(f"NaN in grad for parameter {name}")
        return grad

    def register_grad_nan_checks(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: _grad_hook(grad, name))

    register_forward_nan_checks(model)
    register_grad_nan_checks(model)

    scaler = GradScaler() if args.use_amp else None

    best_val_mse = np.inf
    test_res = None
    no_improve_counter = 0
    for itr in range(args.epoch):
        st = time.time()

        ### Training ###
        model.train()
        if fusion is not None:
            fusion.train()
        iter_data = tqdm(data_obj["train_dataloader"], desc="Training")
        for step, batch_dict in enumerate(iter_data):
            optimizer.zero_grad()
            # with torch.autograd.set_detect_anomaly(True):
            #     train_res = compute_all_losses(
            #         model, fusion, batch_dict, args.enable_text
            #     )
            #     train_res["loss"].backward()
            #     torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            # optimizer.step()

            # # Update the progress bar description with current loss
            # current_loss = train_res["loss"].item()
            # iter_data.set_description(f"Epoch {itr}, Loss: {current_loss:.5f}")
            try:
                with torch.autograd.set_detect_anomaly(True):
                    if args.use_amp:
                        with autocast():
                            train_res = compute_all_losses(
                                model, fusion, batch_dict, args.enable_text
                            )
                            loss = train_res["loss"]
                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(
                            trainable_parameters, max_norm=1.0
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        train_res = compute_all_losses(
                            model, fusion, batch_dict, args.enable_text
                        )
                        loss = train_res["loss"]
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            trainable_parameters, max_norm=1.0
                        )
                        optimizer.step()

                    # Update the progress bar description with current loss
                    current_loss = loss.item()
                    iter_data.set_description(f"Epoch {itr}, Loss: {current_loss:.5f}")

            except (RuntimeError, AssertionError) as e:
                if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                    print(f"[OOM] Step {step}: Skipping batch due to out-of-memory.")
                    torch.cuda.empty_cache()
                elif isinstance(
                    e, AssertionError
                ) and "t must be strictly increasing or decreasing" in str(e):
                    print(e)
                    print(
                        f"[Bad Data] Step {step}: Skipping batch due to invalid timestamps."
                    )
                else:
                    raise e  # Re-raise unknown exceptions
                continue

        ### Validation ###
        model.eval()
        if fusion is not None:
            fusion.eval()
        with torch.no_grad():
            val_res = evaluation(
                model, fusion, data_obj["val_dataloader"], args.enable_text
            )

            # Compute improvement over best MSE
            improvement = best_val_mse - val_res["mse"]

            if improvement > args.early_stop_delta:
                best_val_mse = val_res["mse"]
                best_iter = itr
                no_improve_counter = 0  # Reset early stopping counter

                ### Testing ###
                test_res = evaluation(
                    model, fusion, data_obj["test_dataloader"], args.enable_text
                )
            else:
                no_improve_counter += 1

            logger.info("- Epoch {:03d}, ExpID {}".format(itr, experimentID))
            logger.info(
                "Train - Loss (one batch): {:.5f}".format(train_res["loss"].item())
            )
            logger.info(
                "Val - Loss, MSE, MAE: {:.5f}, {:.5f}, {:.5f}".format(
                    val_res["loss"],
                    val_res["mse"],
                    val_res["mae"],
                )
            )
            if test_res != None:
                logger.info(
                    "Test - Best epoch, Loss, MSE, MAE: {}, {:.5f}, {:.5f}, {:.5f}".format(
                        best_iter,
                        test_res["loss"],
                        test_res["mse"],
                        test_res["mae"],
                    )
                )
            logger.info("Time spent: {:.2f}s".format(time.time() - st))

        if no_improve_counter >= args.patience:
            print("Exp has been early stopped!")
            break

    assert (
        test_res is not None
    ), "No test results available. Please check the training loop."

    return test_res


#####################################################################################################

if __name__ == "__main__":
    """------------------------------------"""
    # data_name = "GDELT"  # type 1.1
    # data_name = "RepoHealth"  # type 1.2
    # data_name = "MIMIC"  # type 1.3 (not ready)
    # data_name = "FNSPID"  # type 2.1
    data_name = "ClusterTrace"  # type 2.2 (not ready)
    # data_name = "StudentLife"  # type 2.3
    # data_name = "ILINet"  # type 3.1
    # data_name = "CESNET"  # type 3.2
    # data_name = "EPA-Air"  # type 3.3

    # ? MTS
    # model_name = "Informer"
    # model_name = "DLinear"
    # model_name = "PatchTST"
    # model_name = "TimesNet"
    # model_name = "TimeMixer"
    # ? LMTS
    # model_name = "TimeLLM"
    # model_name = "TTM"
    # ? IMTS
    # model_name = "CRU"
    model_name = "LatentODE"
    # model_name = "NeuralFlow"
    # model_name = "tPatchGNN"

    enable_text = False
    # enable_text = True

    # use_text_embeddings = False
    use_text_embeddings = True

    TTF_module = "TTF_RecAvg"
    # TTF_module = "TTF_T2V_XAttn"
    MMF_module = "MMF_GR_Add"
    # MMF_module = "MMF_XAttn_Add"

    llm_model_fusion = "GPT2"
    # llm_model_fusion = "BERT"
    # llm_model_fusion = "Llama"
    # llm_model_fusion = "DeepSeek"

    llm_layers_fusion = None
    # llm_layers_fusion = 6

    split_method = "sample"
    # split_method = "instance"  # only for in-domain transfer learning

    tunable_params_path = None
    # tunable_params_path = Path(
    #     "exp_settings_and_results",
    #     "single_granularity",
    #     model_name,
    #     f"{data_name}.json",
    # )

    # batch_size = 1
    # batch_size = 2  # 8G
    batch_size = 8
    # batch_size = 16  # 24G
    # batch_size = 32
    # batch_size = 64
    # batch_size = 256
    """------------------------------------"""
    # Setup args
    args = get_args_from_parser()

    # Set all random seeds (Python, NumPy, PyTorch)
    set_seed(args.seed)

    # Setup fixed params
    fixed_params = {
        "dataset": data_name,
        "model": model_name,
        "batch_size": batch_size,
        "enable_text": enable_text,
        "use_text_embeddings": use_text_embeddings,
        "split_method": split_method,
        "TTF_module": TTF_module,
        "MMF_module": MMF_module,
        "llm_model_fusion": llm_model_fusion,
        "llm_layers_fusion": llm_layers_fusion,
    }

    # Setup tunable params
    if tunable_params_path is None:
        tunable_params = {
            # "lr": 1e-2,
            "lr": 1e-3,
            # "lr": 1e-4,
            "patience": 3,
            # "kappa": 0.1,
            # "recency_sigma": 0.1,
            # "n_heads_fusion": 2,
        }

    # Run
    best_metrics = trainable(tunable_params, fixed_params, args)
    print_formatted_dict(best_metrics)
    print("### Done ###")
