import torch
import torch.nn as nn

from fusions.load_llm import load_llm, embed_notes, get_d_model


# === 1. TTF_RecAvg with LayerNorm & Dropout ===
class TTF_RecAvg(nn.Module):
    """
    Recency-Weighted Averaging fusion with LayerNorm & Dropout.
    Produces time-aligned text embeddings E_txt and a mask M_txt.
    """

    def __init__(
        self,
        llm_model_fusion: str,
        llm_layers_fusion: int,
        max_length: int = 1024,
        device: str = "cpu",
        use_text_embeddings: bool = True,
        recency_sigma: float = 1.0,
        dropout: float = 0.1,
        d_txt: int | None = 768,
    ):
        super().__init__()
        self.use_text_embeddings = use_text_embeddings
        if self.use_text_embeddings == False:
            # load tokenizer & LLM once
            self.tokenizer, self.llm_model = load_llm(
                llm_model_fusion, llm_layers_fusion, device
            )

        # original embedding dim
        d_model = get_d_model(llm_model_fusion)
        # optional projection to fixed dimension
        if d_txt is not None:
            self.input_proj = nn.Linear(d_model, d_txt)
            self.d_txt = d_txt
        else:
            self.input_proj = None
            self.d_txt = d_model
        
        self.max_length = max_length

        assert recency_sigma > 0, "recency_sigma must be > 0"
        # learnable log-sigma for Gaussian recency weighting
        self.log_recency_sigma = nn.Parameter(torch.log(torch.tensor(recency_sigma)))
        # final projection of normalized & dropped embeddings
        self.proj = nn.Linear(self.d_txt, self.d_txt)
        # shared normalization & dropout
        self.layer_norm = nn.LayerNorm(self.d_txt)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        notes_input,  # either List[List[str]] or Tensor[B,N_max,d_model]
        tau: torch.Tensor,
        t_hat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        notes_input: if raw text → List[List[str]]; if embeddings → Tensor[B, N_max, d_model]
        tau:       (B, N_max) timestamps of each note
        t_hat:     (B, T_f) future query times
        """
        # 1) Decide whether to embed or use precomputed
        if self.use_text_embeddings == True:
            # already [B, N_max, d_model]
            V = notes_input
            note_mask = (V.abs().sum(dim=2) > 0).to(torch.bool)
        else:
            V, note_mask = embed_notes(
                notes_input, self.tokenizer, self.llm_model, max_length=self.max_length
            )

        if torch.isnan(V).any():
            raise ValueError("Input embeddings V contain NaN values.")
        
        # optional projection to fixed d_txt
        if self.input_proj is not None:
            V = self.input_proj(V)  # [B, N_max, d_txt]

        # 2) Compute Gaussian recency weights
        B, N_max, _ = V.shape

        # Ensure t_hat is (B, T_f)
        if t_hat.dim() == 1:
            t_hat = t_hat.unsqueeze(0).repeat(B, 1)  # [B, T_f]
        elif t_hat.shape[0] != B:
            raise ValueError(
                f"Expected t_hat shape (B, T_f) or (T_f,), got {t_hat.shape}"
            )

        _, T_f = t_hat.shape
        delta = (t_hat[:, None] - tau[:, :, None]).clamp_min(0)  # [B, N_max, T_f]
        sigma = self.log_recency_sigma.exp()
        w = torch.exp(-((delta / sigma) ** 2))  # [B, N_max, T_f]
        w = w * note_mask.to(w.dtype)[:, :, None]  # zero out missing notes

        # 3) Weighted sum + normalization
        E_wsum = torch.einsum("bnt,bnd->btd", w, V)  # [B, T_f, d_txt]
        denom = w.sum(dim=1).clamp_min(1e-6)  # [B, T_f]
        E_raw = E_wsum / denom.unsqueeze(-1)  # [B, T_f, d_txt]

        # 4) LayerNorm + Dropout
        E_norm = self.layer_norm(E_raw)  # [B, T_f, d_txt]
        E_drop = self.dropout(E_norm)  # [B, T_f, d_txt]

        # 5) Final projection
        E_txt = self.proj(E_drop)  # [B, T_f, d_txt]
        M_txt = note_mask.any(dim=1, keepdim=True)  # [B, 1]

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

    # instantiate and forward
    fusion = TTF_RecAvg(
        llm_model_fusion,
        llm_layers_fusion,
        recency_sigma=1.0,
        dropout=0.1,
        max_length=max_length,
        device=device,
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
