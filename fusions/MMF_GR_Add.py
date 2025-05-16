import torch
import torch.nn as nn
import torch.nn.functional as F

from fusions.TTF_RecAvg import TTF_RecAvg


# === 3. MMF_GR_Add with LayerNorm & Dropout ===
class MMF_GR_Add(nn.Module):
    """
    GRU‐Gated Residual Add fusion:
    computes ΔY via GRU, normalizes & drops it, then gates per‐step/channel.
    """

    def __init__(self, d_txt: int, C: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.C = C
        self.d_txt = d_txt
        # GRU over concat(Y_ts, E_txt)
        self.gru = nn.GRU(
            input_size=C + d_txt, hidden_size=hidden_dim, batch_first=True
        )
        # map hidden → residual ΔY
        self.residual_head = nn.Linear(hidden_dim, C)
        # gate network on concat features
        self.gate_net = nn.Linear(C + d_txt, C)
        # shared normalization & dropout on residual
        self.layer_norm = nn.LayerNorm(C)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Y_ts, E_txt, M_txt):
        """
        Args:
            Y_ts: (B, T, C) base TS forecast
            E_txt:(B, T, d_txt) text embeddings
            M_txt:(B,)    text‐presence mask
        Returns:
            Y_fused:(B, T, C) fused forecast
        """
        B, T, C = Y_ts.shape

        # 1) Concat TS + text
        x = torch.cat([Y_ts, E_txt], dim=-1)  # (B, T, C+d_txt)

        # 2) GRU → raw residual
        h, _ = self.gru(x)  # (B, T, hidden_dim)
        delta_y = self.residual_head(h)  # (B, T, C)

        # 3) LayerNorm + Dropout on residual
        delta_norm = self.layer_norm(delta_y)  # (B, T, C)
        delta_drop = self.dropout(delta_norm)  # (B, T, C)

        # 4) Compute per‐step/channel gate
        gate_logits = self.gate_net(x)  # (B, T, C)
        g = torch.sigmoid(gate_logits)  # (B, T, C)
        mask = M_txt.view(B, 1, 1).expand(-1, T, C)  # (B, T, C)
        g = torch.where(mask, g, torch.ones_like(g))  # force g=1 if no text

        # 5) Fuse: blend base & corrected forecasts
        Y_fused = g * Y_ts + (1 - g) * (Y_ts + delta_drop)
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
    fusion = TTF_RecAvg(
        llm_model_fusion,
        llm_layers_fusion,
        recency_sigma=1.0,
        dropout=0.1,
        max_length=max_length,
        device=device,
    ).to(device)
    E_txt, M_txt = fusion(notes_text, tau, t_hat)
    print("E_txt shape:", E_txt.shape)  # [B, T_f, d_txt]
    print("M_txt:", M_txt)

    """----------------------------------------"""
    C = 4

    d_txt = E_txt.shape[-1]

    # Dummy Y_ts
    Y_ts = torch.randn(B, T_f, C, device=device)  # [B, T_f, C]

    # Instantiate module and forward
    fusion = MMF_GR_Add(d_txt=d_txt, C=C, hidden_dim=C).to(device)
    Y_out = fusion(Y_ts, E_txt, M_txt)

    print("Output shape:", Y_out.shape)
