import torch
import torch.nn as nn

# Absolute imports from project root (MERGE_manus)
# Adjust path based on final location of create_LatentODE_model.py
# Assuming create_latent_ode_model.py is in lib/neural_flow_components/latent_ode_lib/
from lib.neural_flow_components.latent_ode_lib.create_latent_ode_model import create_LatentODE_model

class NeuralFlow(nn.Module):
    def __init__(self, configs):
        super(NeuralFlow, self).__init__()
        self.input_len = configs.input_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in # Number of features (C)
        self.device = configs.device

        # Internal configuration for NeuralFlow's core model
        # This creates a namespace object similar to what argparse would produce,
        # tailored for the needs of create_LatentODE_model.
        class NF_Args_Internal:
            def __init__(self):
                self.model = 'flow' # Focusing on latent_ode with flow model
                self.experiment = 'latent_ode'
                self.latents = getattr(configs, 'nf_latents', 20)
                self.rec_dims = getattr(configs, 'nf_rec_dims', 40)
                self.gru_units = getattr(configs, 'nf_gru_units', 32)
                self.hidden_layers = getattr(configs, 'nf_hidden_layers', 3)
                self.hidden_dim = getattr(configs, 'nf_hidden_dim', 32)
                self.flow_model = getattr(configs, 'nf_flow_model', 'coupling')
                self.flow_layers = getattr(configs, 'nf_flow_layers', 2)
                self.time_net = getattr(configs, 'nf_time_net', 'TimeLinear')
                self.time_hidden_dim = getattr(configs, 'nf_time_hidden_dim', 8)
                self.solver = getattr(configs, 'nf_solver', 'dopri5')
                self.solver_step = getattr(configs, 'nf_solver_step', 0.05) # Default, not in activity.sh
                self.atol = getattr(configs, 'nf_atol', 1e-4) # Default, not in activity.sh
                self.rtol = getattr(configs, 'nf_rtol', 1e-3) # Default, not in activity.sh
                self.odenet = getattr(configs, 'nf_odenet', 'concat') # Default, not in activity.sh
                self.activation = getattr(configs, 'nf_activation', 'Tanh') # Default, not in activity.sh
                self.final_activation = getattr(configs, 'nf_final_activation', 'Identity') # Default, not in activity.sh
                self.input_dim = configs.enc_in # For create_LatentODE_model, should be num features
                self.classify = 0 # As per plan, no classification for forecasting
                self.extrap = getattr(configs, 'nf_extrap', 0) # Default, not in activity.sh
                self.device = configs.device
                # Add any other args required by create_LatentODE_model or its sub-modules
                # based on activity.sh and nfe/train.py defaults if not overridden by user's configs
                self.weight_decay = getattr(configs, 'nf_weight_decay', 0.0001)
                self.quantization = getattr(configs, 'nf_quantization', 0.0)
                self.max_t = getattr(configs, 'nf_max_t', 5.)
                self.mixing = getattr(configs, 'nf_mixing', 0.0001)
                self.gob_prep_hidden = getattr(configs, 'nf_gob_prep_hidden', 10)
                self.gob_cov_hidden = getattr(configs, 'nf_gob_cov_hidden', 50)
                self.gob_p_hidden = getattr(configs, 'nf_gob_p_hidden', 25)
                self.invertible = getattr(configs, 'nf_invertible', 1)
                self.components = getattr(configs, 'nf_components', 8)
                self.decoder = getattr(configs, 'nf_decoder_type', 'continuous') # 'decoder' is a class name, use 'nf_decoder_type'
                self.rnn = getattr(configs, 'nf_rnn', 'gru')
                self.marks = getattr(configs, 'nf_marks', 0)
                self.density_model = getattr(configs, 'nf_density_model', 'independent')

        nf_args_internal = NF_Args_Internal()

        self.z0_prior = torch.distributions.Normal(
            torch.Tensor([0.0]).to(self.device),
            torch.Tensor([1.0]).to(self.device)
        )
        # obsrv_std might be dataset-specific or a small value for regression
        self.obsrv_std = torch.Tensor([getattr(configs, 'nf_obsrv_std', 0.01)]).to(self.device)
        self.n_classes = 0 # No classification output needed for the forecasting task

        # This is the core NeuralFlow model (LatentODEmodel instance)
        self.nf_model_core = create_LatentODE_model(nf_args_internal, 
                                               input_dim=self.enc_in, # input_dim for create_LatentODE_model
                                               z0_prior=self.z0_prior, 
                                               obsrv_std=self.obsrv_std, 
                                               device=self.device, 
                                               n_labels=self.n_classes)

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        # tp_to_predict: (B, Lp_future) - Timestamps for future predictions from your collate_fn
        # observed_data: (B, L_hist, C) - Observed values from your collate_fn
        # observed_tp: (B, L_hist) - Timestamps for observed_data from your collate_fn
        # observed_mask: (B, L_hist, C) - Mask for observed_data from your collate_fn

        # The core NeuralFlow model (self.nf_model_core, which is a LatentODEmodel instance)
        # has a `get_reconstruction` method that can be used for forecasting.
        # It expects inputs: time_steps_to_predict, truth (observed_data), 
        # truth_time_steps (observed_tp), and mask.

        # Ensure data types and devices are correct before passing to nf_model_core.
        # Your collate_fn should already handle device placement based on `configs.device`.

        # Call the core model's prediction method.
        # The mask format for `get_reconstruction` needs to be compatible.
        # If nf_model_core.get_reconstruction expects mask of shape (B,L) and yours is (B,L,C),
        # an adaptation might be needed, e.g., nf_mask = observed_mask.any(dim=-1).
        # However, LatentODEmodel's get_reconstruction seems to handle (B,L,C) mask if truth is (B,L,C).
        predictions, _ = self.nf_model_core.get_reconstruction(
            time_steps_to_predict=tp_to_predict, 
            truth=observed_data, 
            truth_time_steps=observed_tp, 
            mask=observed_mask, 
            n_traj_samples=1 # For deterministic forecasting
        )
        
        # Output of get_reconstruction is typically (n_traj_samples, B, Lp, C) 
        # or (B, Lp, C) if n_traj_samples=1 and squeezed internally by the model.
        # We need to ensure the final output is (B, Lp, C).
        if predictions.dim() == 4 and predictions.size(0) == 1:
            predictions = predictions.squeeze(0)

        return predictions # Expected shape: (B, Lp, C)

