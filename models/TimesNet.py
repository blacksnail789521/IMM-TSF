import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.configs = configs
        self.input_len = configs.input_len
        self.seq_len = configs.input_len
        self.pred_len = configs.pred_len
        print("seq len:", self.seq_len, self.pred_len)
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(2*configs.enc_in+1, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(
            self.seq_len + self.pred_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)

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
        x_enc = observed_data
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # concat data with masking and time
        x_enc = torch.cat([x_enc, observed_mask, observed_tp.unsqueeze(dim=-1)], dim=-1) 

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [B, L, d_model]
        tp_to_predict = tp_to_predict.unsqueeze(dim=-1).repeat(1, 1, enc_out.size(-1))
        enc_out = torch.cat([enc_out, tp_to_predict], dim=1)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # porject back
        # enc_out [B, L+Lp, d_model]
        # dec_out [B, L+Lp, K]
        dec_out = self.projection(enc_out)
        # print(enc_out.shape, dec_out.shape)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out[:, -self.pred_len:, :]
        
        return dec_out[:, :Lp, :]