import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x, tp_to_predict):  # x: [bs x nvars x d_model x patch_num]
        B, K, _, _ = x.shape
        x = self.flatten(x)
        x = torch.cat([x, tp_to_predict.unsqueeze(dim=1).repeat(1, K, 1)], dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=6*3, stride=3*3):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.input_len = configs.input_len
        self.seq_len = 3 * configs.input_len
        self.pred_len = configs.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = configs.d_model * int((self.seq_len - patch_len) / stride + 2)

        self.head = FlattenHead(configs.enc_in, self.head_nf + configs.pred_len, configs.pred_len,
                                head_dropout=configs.dropout)

        self.zeros_pad = torch.zeros(configs.batch_size, max(configs.input_len, configs.pred_len), configs.enc_in).to(configs.device)

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        '''
        tp_to_predict [B, Lp]
        observed_data [B, L, K]
        observed_tp [B, L] 
        observed_mask: [B, L, K]
        '''

        ### padding input to the unified length ###
        B, L, K = observed_data.shape
        if(L < self.input_len):
            pad_len = self.input_len - L
            observed_data = torch.cat([observed_data, self.zeros_pad[:B, :pad_len, :]], dim=1)
            observed_mask = torch.cat([observed_mask, self.zeros_pad[:B, :pad_len, :]], dim=1)
            observed_tp = torch.cat([observed_tp, self.zeros_pad[:B, :pad_len, 0]], dim=1)
        
        _, Lp = tp_to_predict.shape
        if(Lp < self.pred_len):
            pad_len = self.pred_len - Lp
            tp_to_predict = torch.cat([tp_to_predict, self.zeros_pad[:B, :pad_len, 0]], dim=1)
        
        # Normalization from Non-stationary Transformer
        B, L, K = observed_data.shape
        x_enc = observed_data # [B, L, K]
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # concat data with masking and time
        x_enc = torch.stack([x_enc, observed_mask, observed_tp.unsqueeze(dim=-1).repeat(1,1,K)], dim=-1) 
        x_enc = x_enc.permute(0, 1, 3, 2).reshape(B, -1, K)
        # print(x_enc.shape, x_enc[0,:,10])
        
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder (transformer)
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        # print("enc_out:", enc_out.shape)

        # Decoder
        dec_out = self.head(enc_out, tp_to_predict)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)
        # print("dec_out:", dec_out.shape)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out[:, -self.pred_len:, :]
        
        return dec_out[:, :Lp, :]