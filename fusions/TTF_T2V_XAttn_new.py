import torch
import torch.nn as nn

from fusions.load_llm import load_llm, embed_notes, get_d_txt


class Time2Vec(nn.Module):
    """
    Time2Vec encoding: transforms a scalar time delta into a d_tau-dimensional vector.
    """

    def __init__(self, d_tau: int):
        super().__init__()
        assert d_tau > 1, "d_tau must be > 1"
        # Linear component for trend
        self.linear = nn.Linear(1, 1)
        # Periodic components for seasonality
        self.periodic = nn.Linear(1, d_tau - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 1)
        lin = self.linear(x)  # (..., 1)
        per = torch.sin(self.periodic(x))  # (..., d_tau-1)
        return torch.cat([lin, per], dim=-1)  # (..., d_tau)


class TTF_T2V_XAttn(nn.Module):
    """
    Multi-head T2V-augmented cross-attention module.

    Given past textual notes with timestamps, produces time-aligned text embeddings
    at desired future query times using cross-attention with Time2Vec.
    Args:
        llm_model_fusion (str): alias or path for tokenizer+model
        llm_layers_fusion (int): number of encoder layers to keep
        n_heads_fusion (int): number of attention heads
        dropout (float): dropout probability for attention and output
    """

    def __init__(
        self,
        llm_model_fusion: str,
        llm_layers_fusion: int,
        max_length: int = 1024,
        device: str = "cpu",
        use_text_embeddings: bool = True,
        n_heads_fusion: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_text_embeddings = use_text_embeddings
        if self.use_text_embeddings == False:
            # load tokenizer & LLM once
            self.tokenizer, self.llm_model = load_llm(
                llm_model_fusion, llm_layers_fusion, device
            )
        self.d_txt = get_d_txt(llm_model_fusion)
        # half for time dimension
        self.d_tau = self.d_txt // 2
        self.max_length = max_length

        # Modules
        self.time2vec = Time2Vec(self.d_tau)
        # Project concatenated [text; time] to model dimension
        self.KV_proj = nn.Linear(self.d_txt + self.d_tau, self.d_txt)
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_txt,
            num_heads=n_heads_fusion,
            dropout=dropout,
            batch_first=True,
        )
        # Residual norm and dropout
        self.layer_norm = nn.LayerNorm(self.d_txt)
        self.dropout = nn.Dropout(dropout)
        # Final projection back to text embedding dimension
        self.proj_out = nn.Linear(self.d_txt, self.d_txt)
        # Learnable fixed query vector of size d_txt
        self.Q_param = nn.Parameter(torch.randn(1, 1, self.d_txt))

    def forward(
        self,
        notes_input,  # either List[List[str]] or Tensor[B, N_max, d_txt]
        tau: torch.Tensor,  # (B, N_max)
        t_hat: torch.Tensor,  # (B, T_f)
    ):
        """
        notes_input: raw notes as List[List[str]] or precomputed embeddings Tensor[B, N_max, d_txt]
        tau:         timestamps for notes, shape (B, N_max)
        t_hat:       future query times, shape (B, T_f)
        """
        # 1) Embed or reuse
        if self.use_text_embeddings == True:
            V = notes_input
            note_mask = (V.abs().sum(dim=2) > 0).to(torch.bool)
        else:
            V, note_mask = embed_notes(
                notes_input,
                self.tokenizer,
                self.llm_model,
                max_length=self.max_length,
            )

        if torch.isnan(V).any():
            raise ValueError("Input embeddings V contain NaN values.")

        # 2) Presence mask
        M_txt = note_mask.any(dim=1, keepdim=True)  # [B,1]

        # Ensure t_hat is (B, T_f)
        B = V.shape[0]
        if t_hat.dim() == 1:
            t_hat = t_hat.unsqueeze(0).repeat(B, 1)  # [B, T_f]
        elif t_hat.shape[0] != B:
            raise ValueError(
                f"Expected t_hat shape (B, T_f) or (T_f,), got {t_hat.shape}"
            )

        # 3) Time2Vec: compute ONCE per note
        tau_feat = self.time2vec(tau.unsqueeze(-1))  # [B, N_max, d_tau]

        # 4) Fuse V and time embeddings: [B, N_max, d_txt + d_tau]
        V_fused = torch.cat([V, tau_feat], dim=-1)
        KV_proj = self.KV_proj(V_fused)  # [B, N_max, d_txt]

        # 5) Prepare queries from t_hat: [B, T_f, d_txt]
        Q = self.Q_param.expand(B, t_hat.shape[1], self.d_txt)  # [B, T_f, d_txt]

        # 6) Attention mask: [B, N_max]
        mask_pad = ~note_mask  # [B, N_max]

        # 7) Multi-head cross-attention
        # Flatten B dim for attention
        Q_flat = Q.reshape(B * t_hat.shape[1], 1, self.d_txt)
        KV_proj_flat = KV_proj.unsqueeze(1).expand(-1, t_hat.shape[1], -1, -1)
        KV_proj_flat = KV_proj_flat.reshape(B * t_hat.shape[1], KV_proj.shape[1], self.d_txt)
        mask_pad_flat = mask_pad.unsqueeze(1).expand(-1, t_hat.shape[1], -1).reshape(B * t_hat.shape[1], KV_proj.shape[1])

        attn_out, _ = self.attn(
            Q_flat,
            KV_proj_flat,
            KV_proj_flat,
            key_padding_mask=mask_pad_flat,
        )
        E_attn = attn_out.reshape(B, t_hat.shape[1], self.d_txt)

        # ----- NEW: override no-note samples -----
        # M_txt: [B, 1] → expand to [B, T_f, d_txt]
        mask = M_txt.view(B, 1, 1).expand(B, t_hat.shape[1], self.d_txt)
        # for samples with no notes, set E_attn to zero (or to Q2, etc.)
        E_attn = torch.where(mask, E_attn, torch.zeros_like(E_attn))
        # -----------------------------------------

        # 8) Residual + Norm + Dropout
        Q2 = self.Q_param.expand(B, t_hat.shape[1], self.d_txt)
        E_resid = self.layer_norm(E_attn + Q2)
        E_drop = self.dropout(E_resid)

        # 9) Final projection: [B, T_f, d_txt]
        E_txt = self.proj_out(E_drop)

        return E_txt, M_txt


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
