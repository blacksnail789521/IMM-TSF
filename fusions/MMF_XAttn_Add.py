import torch
import torch.nn as nn
import torch.nn.functional as F

from fusions.TTF_RecAvg import TTF_RecAvg
from fusions.TTF_T2V_XAttn import TTF_T2V_XAttn


# === 4. MMF_XAttn_Add with LayerNorm & Dropout ===
class MMF_XAttn_Add(nn.Module):
    """
    Cross-Attention Add fusion with a fixed hyperparameter `kappa`
    to control the relative weight of the text-derived residual.

    Fuse as:
        Y_fused = (Y_ts + kappa * Δ) / (1 + kappa)
    so that the weights on the two branches always sum to 1,
    and kappa < 1 downweights text, kappa > 1 upweights it.
    """

    def __init__(
        self,
        d_txt: int,
        C: int,
        d_attn: int,
        n_heads_fusion: int = 1,
        dropout: float = 0.1,
        kappa: float = 1.0,
    ):
        super().__init__()
        self.C = C
        self.d_attn = d_attn
        # kappa for text residual (hyperparameter)
        self.kappa = kappa

        # projections to attention dim
        self.proj_q = nn.Linear(C, d_attn, bias=False)
        self.proj_k = nn.Linear(d_txt, d_attn, bias=False)
        self.proj_v = nn.Linear(d_txt, d_attn, bias=False)

        # multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_attn,
            num_heads=n_heads_fusion,
            dropout=dropout,
            batch_first=True,
        )

        # map attn_out → ΔY
        self.residual_head = nn.Linear(d_attn, C)

        # shared normalization & dropout on residual
        self.layer_norm = nn.LayerNorm(C)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Y_ts, E_txt, M_txt):
        """
        Args:
            Y_ts:(B, T, C) base TS forecast
            E_txt:(B, T, d_txt) text embeddings
            M_txt:(B,)    text-presence mask
        Returns:
            Y_fused:(B, T, C) fused forecast
        """
        B, T, C = Y_ts.shape

        # 1) Project to Q/K/V
        Q = self.proj_q(Y_ts)  # (B, T, d_attn)
        K = self.proj_k(E_txt)  # (B, T, d_attn)
        V = self.proj_v(E_txt)  # (B, T, d_attn)

        # 2) Build key_padding_mask for MHA
        key_pad = (~M_txt).view(B, 1).expand(-1, T)

        # 3) Multi-head attention
        attn_out, _ = self.attn(Q, K, V, key_padding_mask=key_pad)

        # ——— NEW: nuke any NaNs for no-text samples ———
        mask_attn = M_txt.view(B, 1, 1).expand(-1, T, self.d_attn)
        attn_out = torch.where(mask_attn, attn_out, torch.zeros_like(attn_out))

        # 4) Map to ΔY
        delta_y = self.residual_head(attn_out)  # (B, T, C)
        if torch.isnan(delta_y).any():
            print(f"M_txt: {M_txt}")
            print(f"attn_out: {attn_out}")
            print(f"Q: {Q}")
            print(f"K: {K}")
            print(f"V: {V}")
            print(f"Y_ts: {Y_ts}")
            raise ValueError("delta_y contains NaN values.")

        # 5) LayerNorm + Dropout on residual
        delta_norm = self.layer_norm(delta_y)
        delta_drop = self.dropout(delta_norm)

        # 6) Zero out ΔY if no text
        mask = M_txt.view(B, 1, 1).expand(-1, T, C)
        delta_drop = torch.where(mask, delta_drop, torch.zeros_like(delta_drop))

        # 7) Fuse with fixed kappa (convex blend)
        Y_fused = (Y_ts + self.kappa * delta_drop) / (1.0 + self.kappa)
        return Y_fused


# Example usage
if __name__ == "__main__":
    """----------------------------------------"""
    llm_model_fusion = "GPT2"
    llm_layers_fusion = 6
    notes_text = [
        [
            "This is my first note.",
            "I love machine learning.",
            "Let's test tokenization.",
        ],
        [],  # second sample has no notes
        [
            "Here is another note.",
            "Data science is fun.",
            "PyTorch and transformers integration.",
            "Testing pad behavior.",
            "Final note in this batch.",
        ],
    ]
    max_length = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """----------------------------------------"""
    T_f = 10  # number of future time steps

    B = len(notes_text)
    lengths = [len(x) for x in notes_text]
    N_max = max(lengths) if lengths else 0

    # Dummy tau and t_hat
    tau = torch.zeros(B, N_max, device=device)
    for i, L_i in enumerate(lengths):
        tau[i, :L_i] = torch.linspace(0, L_i - 1, steps=L_i, device=device)
    t_hat = torch.linspace(
        max(lengths) - 0.5, max(lengths) + 0.5, steps=T_f, device=device
    )
    t_hat = t_hat.unsqueeze(0).repeat(B, 1)

    # instantiate and forward
    # fusion = TTF_RecAvg(
    #     llm_model_fusion,
    #     llm_layers_fusion,
    #     recency_sigma=1.0,
    #     dropout=0.1,
    #     max_length=max_length,
    #     device=device,
    #     use_text_embeddings=False,
    # ).to(device)
    fusion = TTF_T2V_XAttn(
        llm_model_fusion,
        llm_layers_fusion,
        n_heads_fusion=2,
        dropout=0.1,
        max_length=max_length,
        use_text_embeddings=False,
    ).to(device)
    E_txt, M_txt = fusion(notes_text, tau, t_hat)
    print("E_txt shape:", E_txt.shape)  # [B, T_f, d_txt]
    print("M_txt:", M_txt)

    if torch.isnan(E_txt).any():
        print(E_txt)
        # Find which samples are all nan
        for i in range(E_txt.shape[0]):
            if torch.isnan(E_txt[i]).all():
                print(f"Sample {i} has all NaN values in E_txt.")
        raise ValueError("E_txt contains NaN values.")

    """----------------------------------------"""
    C = 4

    d_txt = E_txt.shape[-1]

    # Dummy Y_ts
    Y_ts = torch.randn(B, T_f, C, device=device)  # [B, T_f, C]

    # Instantiate module and forward
    fusion = MMF_XAttn_Add(
        d_txt=d_txt, C=C, d_attn=d_txt, n_heads_fusion=1, dropout=0.1, kappa=1.0
    ).to(device)
    Y_out = fusion(Y_ts, E_txt, M_txt)

    print("Output shape:", Y_out.shape)

    if torch.isnan(Y_out).any():
        print(Y_out)
        raise ValueError("Y_out contains NaN values.")