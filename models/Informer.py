import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    ConvLayer,
)
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Informer(nn.Module):
    """
    Informer adapted for irregular time‐series forecasting.
    """

    def __init__(self, configs):
        super(Informer, self).__init__()
        # History & forecast lengths
        self.input_len = configs.input_len
        self.seq_len = configs.input_len
        self.pred_len = configs.pred_len
        self.C = configs.C

        # Embeddings now take (value, mask, timestamp) ⇒ 2*C + 1 input channels
        in_channels = 2 * self.C + 1
        self.enc_embedding = DataEmbedding(
            in_channels, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            in_channels, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Encoder (unchanged)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            (
                [ConvLayer(configs.d_model) for _ in range(configs.e_layers - 1)]
                if configs.distil
                else None
            ),
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Decoder (unchanged)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        ProbAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

        # Padding buffer for irregular sequences
        self.zeros_pad = torch.zeros(
            configs.batch_size,
            max(self.input_len, self.pred_len),
            self.C,
            device=configs.device,
        )

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        """
        Irregular forecasting:
          tp_to_predict: [B, Lp]
          observed_data: [B, L, C]
          observed_tp:   [B, L]
          observed_mask:[B, L, C]
        Returns:
          [B, Lp, C]
        """
        B, L, C = observed_data.shape

        # 1) Pad history if too short
        if L < self.input_len:
            pad = self.input_len - L
            observed_data = torch.cat(
                [observed_data, self.zeros_pad[:B, :pad, :]], dim=1
            )
            observed_mask = torch.cat(
                [observed_mask, self.zeros_pad[:B, :pad, :]], dim=1
            )
            observed_tp = torch.cat([observed_tp, self.zeros_pad[:B, :pad, 0]], dim=1)

        # 2) Pad future time‐points if too short
        Lp = tp_to_predict.size(1)
        if Lp < self.pred_len:
            pad = self.pred_len - Lp
            tp_to_predict = torch.cat(
                [tp_to_predict, tp_to_predict.new_zeros(B, pad)], dim=1
            )

        # 3) Normalize observed values
        x = observed_data * observed_mask
        means = x.sum(1, keepdim=True) / observed_mask.sum(1, keepdim=True).clamp(min=1)
        x = x - means
        var = ((x * observed_mask) ** 2).sum(1, keepdim=True) / observed_mask.sum(
            1, keepdim=True
        ).clamp(min=1)
        stdev = torch.sqrt(var + 1e-5)
        x = x / stdev

        # 4) Build encoder & decoder inputs by stacking (value, mask, time)
        #    Now timestamp is a single channel, not repeated per variable:
        enc_in = torch.cat(
            [
                x,  # [B, L, C]
                observed_mask,  # [B, L, C]
                observed_tp.unsqueeze(-1),  # [B, L, 1]
            ],
            dim=-1,
        )  # → [B, L, 2*C+1]

        dec_in = torch.cat(
            [
                observed_data.new_zeros(
                    B, self.pred_len, C
                ),  # future-values placeholder
                observed_data.new_zeros(B, self.pred_len, C),  # future-mask placeholder
                tp_to_predict.unsqueeze(-1),  # [B, Lp, 1]
            ],
            dim=-1,
        )  # → [B, Lp, 2*C+1]

        # 5) Embed
        enc_out = self.enc_embedding(enc_in, None)  # [B, L, d_model]
        dec_out = self.dec_embedding(dec_in, None)  # [B, Lp, d_model]

        # 6) Encode & decode
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None
        )  # [B, Lp, d_model]

        # 7) Project to [B, Lp, C] via decoder's final linear, then de-normalize
        out = dec_out * stdev + means  # broadcast along time
        return out[:, :Lp, :]
