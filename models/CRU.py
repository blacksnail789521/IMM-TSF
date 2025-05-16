import torch
import torch.nn as nn

# Absolute imports from project root (MERGE_manus)
from lib.cru_components.cru_models import Physionet_USHCN # Using this as the general CRU model structure

class CRU(nn.Module):
    def __init__(self, configs):
        super(CRU, self).__init__()
        self.input_len = configs.input_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in # Number of features (C)
        self.device = configs.device

        # Internal configuration for CRU model
        # This creates a namespace object similar to what argparse would produce for CRU.
        class CRU_Args_Internal:
            def __init__(self):
                # Core CRU parameters (refer to CRU.py and cru_models.py in cru_baseline_code)
                self.latent_state_dim = getattr(configs, "cru_lsd", 32)
                self.hidden_units = getattr(configs, "cru_hidden_units", 32)
                self.enc_num_layers = getattr(configs, "cru_enc_num_layers", 1)
                self.dec_num_layers = getattr(configs, "cru_dec_num_layers", 1)
                self.num_cru_layers = getattr(configs, "cru_num_layers", 1)
                
                self.dropout_type = getattr(configs, "cru_dropout_type", "None") # e.g., "None", "Zoneout", "Variational"
                self.dropout_rate = getattr(configs, "cru_dropout_rate", 0.0)
                
                self.use_gate_hidden_states = getattr(configs, "cru_use_gate_hidden_states", True)
                self.use_ode_for_gru = getattr(configs, "cru_use_ode_for_gru", False)
                self.use_decay_gravity_gate = getattr(configs, "cru_use_decay_gravity_gate", True)
                self.use_gravity_gate = getattr(configs, "cru_use_gravity_gate", True)
                self.use_decay_input_gate = getattr(configs, "cru_use_decay_input_gate", True)
                self.use_input_gate = getattr(configs, "cru_use_input_gate", True)
                self.use_skip_connection = getattr(configs, "cru_use_skip_connection", True)
                
                self.solver = getattr(configs, "cru_solver", "euler") # "euler" or "rk4"
                self.extrapolation = True # Essential for forecasting
                self.device = configs.device
                self.batch_size = configs.batch_size # CRU model might use this internally
                
                self.lr = getattr(configs, "lr", 1e-3) # Learning rate for CRU model
                self.rkn = getattr(configs, "cru_rkn", False) # RKN flag
                self.f_cru = getattr(configs, "cru_f_cru", False) # f_cru flag
                self.bandwidth = getattr(configs, "cru_bandwidth", 3) # Bandwidth for CRU model
                self.num_basis = getattr(configs, "cru_num_basis", 15) # Number of basis functions
                self.trans_net_hidden_units = getattr(configs, "cru_trans_net_hidden_units", [])
                self.trans_net_hidden_activation = getattr(configs, "cru_trans_net_hidden_activation", "elup1")
                self.t_sensitive_trans_net = getattr(configs, "cru_t_sensitive_trans_net", False)
                self.trans_var_activation = getattr(configs, "cru_trans_var_activation", "elup1")
                self.trans_covar = getattr(configs, "cru_trans_covar", 0.1)
                self.enc_var_activation = getattr(configs, "cru_enc_var_activation", "square")
                self.dec_var_activation = getattr(configs, "cru_dec_var_activation", "exp")

                # Parameters for Physionet_USHCN (which inherits CRU)
                # These are mostly related to encoder/decoder MLP structures if not overridden by CRU itself
                # For simplicity, we assume Physionet_USHCN will use the hidden_units for its MLPs.

        cru_args_internal = CRU_Args_Internal()

        # Instantiate the core CRU model
        # We use Physionet_USHCN as it seems to be a more general instantiation from the CRU baseline code.
        # It expects target_dim, lsd, and args.
        self.cru_model_core = Physionet_USHCN(
            target_dim=self.enc_in,
            lsd=cru_args_internal.latent_state_dim,
            args=cru_args_internal, # Pass the internal args namespace
            use_cuda_if_available=(self.device.type == "cuda")
        ).to(self.device)

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        # tp_to_predict: (B, Lp_future)
        # observed_data: (B, L_hist, C)
        # observed_tp: (B, L_hist) - chunk-relative time steps
        # observed_mask: (B, L_hist, C) - 1 if valid, 0 if not

        B, L_hist, C = observed_data.shape
        _, Lp_future = tp_to_predict.shape

        all_tp = torch.cat((observed_tp, tp_to_predict), dim=1) 

        future_data_padding = torch.zeros(B, Lp_future, C, device=self.device, dtype=observed_data.dtype)
        all_data_input = torch.cat((observed_data, future_data_padding), dim=1) 

        hist_obs_valid = observed_mask.any(dim=-1).bool() 
        future_obs_valid = torch.zeros(B, Lp_future, device=self.device, dtype=torch.bool) 
        all_obs_valid = torch.cat((hist_obs_valid, future_obs_valid), dim=1) 
        
        output_mean_all, _, _ = self.cru_model_core.forward(
            obs_batch=all_data_input.float(), 
            time_points=all_tp.float(), 
            obs_valid=all_obs_valid
        )

        predictions_mean = output_mean_all[:, L_hist:, :] 

        return predictions_mean


