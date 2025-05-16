import torch
import torch.nn as nn

# Pre-import your fusion modules here
from fusions.TTF_RecAvg import TTF_RecAvg

from fusions.TTF_T2V_XAttn import TTF_T2V_XAttn
# from fusions.TTF_T2V_XAttn_old import TTF_T2V_XAttn
# from fusions.TTF_T2V_XAttn_new import TTF_T2V_XAttn
from fusions.MMF_GR_Add import MMF_GR_Add
from fusions.MMF_XAttn_Add import MMF_XAttn_Add

# Maps for string-based lookup
_TTF_CLASSES = {
    "TTF_RecAvg": TTF_RecAvg,
    "TTF_T2V_XAttn": TTF_T2V_XAttn,
}
_MMF_CLASSES = {
    "MMF_GR_Add": MMF_GR_Add,
    "MMF_XAttn_Add": MMF_XAttn_Add,
}


class FusionModel(nn.Module):
    """
    Composite fusion module combining a TTF (text-time fusion) and an MMF (multimodal fusion).
    Expects an argparse.Namespace (or similar) with:
      --TTF_module: str key or class for TTF (e.g., 'TTF_RecAvg')
      --MMF_module: str key or class for MMF (e.g., 'MMF_XAttn_Add')
      --llm_model_fusion: str alias for the LLM
      --llm_layers_fusion: int number of LLM layers
      --recency_sigma: float (for TTF_RecAvg)
      --dropout: float dropout probability
      --device: str 'cpu' or 'cuda'
      --n_heads_fusion: int number of attention heads
      --C: int number of channels in time series
      --kappa: float (for MMF_XAttn_Add)
    """

    def __init__(self, args):
        super().__init__()
        # Resolve module references
        TTF_ref = args.TTF_module
        MMF_ref = args.MMF_module
        TTF_cls = (
            _TTF_CLASSES.get(TTF_ref, TTF_ref) if isinstance(TTF_ref, str) else TTF_ref
        )
        MMF_cls = (
            _MMF_CLASSES.get(MMF_ref, MMF_ref) if isinstance(MMF_ref, str) else MMF_ref
        )

        print(f"Using TTF module: {args.TTF_module}")
        print(f"Using MMF module: {args.MMF_module}")

        # Instantiate TTF
        if TTF_cls is TTF_RecAvg:
            self.ttf = TTF_cls(
                args.llm_model_fusion,
                args.llm_layers_fusion,
                max_length=args.max_length,
                device=args.device,
                use_text_embeddings=args.use_text_embeddings,
                recency_sigma=args.recency_sigma,
                dropout=args.dropout,
                d_txt=args.d_txt,
            )
        else:
            self.ttf = TTF_cls(
                args.llm_model_fusion,
                args.llm_layers_fusion,
                max_length=args.max_length,
                device=args.device,
                use_text_embeddings=args.use_text_embeddings,
                n_heads_fusion=args.n_heads_fusion,
                dropout=args.dropout,
                d_txt=args.d_txt,
            )

        # Instantiate MMF with injected d_txt
        d_txt = self.ttf.d_txt
        if MMF_cls is MMF_GR_Add:
            self.mmf = MMF_cls(
                d_txt=d_txt,
                C=args.C,
                hidden_dim=args.C,
                dropout=args.dropout,
            )
        else:
            self.mmf = MMF_cls(
                d_txt=d_txt,
                C=args.C,
                d_attn=d_txt,
                n_heads_fusion=args.n_heads_fusion,
                dropout=args.dropout,
                kappa=args.kappa,
            )

    def forward(self, notes_input, tau, t_hat, Y_ts):
        """
        Forward through TTF then MMF.
        Returns fused output.
        """
        if torch.isnan(Y_ts).any():
            print(f"Y_ts: {Y_ts}")
            raise ValueError("Y_ts contains NaN values.")
        E_txt, M_txt = self.ttf(notes_input, tau, t_hat)
        if torch.isnan(E_txt).any():
            raise ValueError("E_txt contains NaN values.")
        # print(f"M_txt: {M_txt}")
        Y_out = self.mmf(Y_ts, E_txt, M_txt)
        if torch.isnan(Y_out).any():
            raise ValueError("Y_out contains NaN values.")
        return Y_out
