import torch
import torch.nn as nn
from torch.distributions.normal import Normal

# Absolute import from the project root (MERGE_manus)
from lib.latent_ode_components.create_latent_ode_model import create_LatentODE_model

class LatentODE(nn.Module):
    def __init__(self, args):
        super(LatentODE, self).__init__()
        self.args = args
        self.device = args.device
        
        # if not hasattr(args, 'input_dim'):
        #     raise ValueError("args.input_dim must be set before initializing LatentODE model.")
        # self.input_dim = args.input_dim
        self.input_dim = args.C

        # Prior for z0
        self.z0_prior = Normal(torch.Tensor([0.0]).to(self.device), torch.Tensor([1.0]).to(self.device))
        
        # Observation noise standard deviation
        # Use prefixed arg: args.ode_obsrv_std
        self.obsrv_std_val = args.ode_obsrv_std if hasattr(args, 'ode_obsrv_std') else 0.01
        self.obsrv_std = torch.Tensor([self.obsrv_std_val]).to(self.device)

        # LatentODE specific arguments. These should be added to argparse in run_models.py
        # Using prefixed names for clarity and to avoid conflicts
        default_ode_args = {
            'ode_latents': 20, 
            'ode_units': getattr(args, 'ode_units', 32),
            'ode_gen_layers': getattr(args, 'ode_gen_layers', 1),
            'ode_rec_dims': getattr(args, 'ode_rec_dims', 32),
            'ode_rec_layers': getattr(args, 'ode_rec_layers', 1),
            'ode_gru_units': getattr(args, 'ode_gru_units', 32),
            'ode_poisson': False, 
            'ode_classif': False,
            'ode_linear_classif': False, 
            'ode_z0_encoder': 'odernn', 
            'dataset': 'custom_dataset' # Placeholder, actual dataset name comes from args.dataset
        }

        # Create a temporary args_for_ode object to pass to create_LatentODE_model
        # This avoids polluting the main args namespace with unprefixed versions if not intended
        class ArgsForODE:
            pass
        
        self.args_for_ode = ArgsForODE()
        self.args_for_ode.device = self.device # Pass device
        self.args_for_ode.dataset = self.args.dataset # Pass dataset name

        for key_prefixed, default_value in default_ode_args.items():
            # key_unprefixed is what create_LatentODE_model expects (e.g. 'latents', 'units')
            key_unprefixed = key_prefixed.replace('ode_', '', 1) 
            
            if hasattr(self.args, key_prefixed):
                setattr(self.args_for_ode, key_unprefixed, getattr(self.args, key_prefixed))
            else:
                # print(f"Warning: LatentODE argument acility_'{key_prefixed}acility_' not found in args. Using default value: {default_value}")
                setattr(self.args_for_ode, key_unprefixed, default_value)
        
        # Special handling for ode_gru_units, if None, it should use hid_dim from main args
        if self.args_for_ode.gru_units is None and hasattr(self.args, 'hid_dim'):
             self.args_for_ode.gru_units = self.args.hid_dim
        elif self.args_for_ode.gru_units is None: # if hid_dim also not there, use the default_ode_args value
            self.args_for_ode.gru_units = default_ode_args['ode_gru_units']


        self.latent_ode_model_core = create_LatentODE_model(
            args=self.args_for_ode, # Pass the specially prepared args_for_ode
            input_dim=self.input_dim,
            z0_prior=self.z0_prior,
            obsrv_std=self.obsrv_std,
            device=self.device,
            classif_per_tp=False, # Assuming not used for forecasting task
            n_labels=1            # Assuming not used for forecasting task
        )

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        observed_data = observed_data.to(self.device).float()
        observed_tp = observed_tp.to(self.device).float()
        observed_mask = observed_mask.to(self.device).float()
        tp_to_predict = tp_to_predict.to(self.device).float()

        # Check whether tp_to_predict is strictly increasing
        if not torch.all(torch.diff(tp_to_predict) > 0):
            raise ValueError(f"tp_to_predict must be strictly increasing. Found: {tp_to_predict}")

        # print(tp_to_predict.shape, observed_data.shape, observed_tp.shape, observed_mask.shape)
        # raise Exception(tp_to_predict)
        # raise Exception(tp_to_predict.shape, observed_data.shape, observed_tp.shape, observed_mask.shape)

        # Use prefixed arg: args.ode_n_traj_samples
        n_traj_samples = self.args.ode_n_traj_samples if hasattr(self.args, 'ode_n_traj_samples') else 1

        predictions_raw, _ = self.latent_ode_model_core.get_reconstruction(
            time_steps_to_predict=tp_to_predict,
            truth=observed_data,
            truth_time_steps=observed_tp,
            mask=observed_mask, 
            n_traj_samples=n_traj_samples,
            run_backwards=True 
        )
        
        if n_traj_samples == 1:
            predictions = predictions_raw.squeeze(0) 
        else:
            predictions = predictions_raw.mean(dim=0) 
            
        return predictions

