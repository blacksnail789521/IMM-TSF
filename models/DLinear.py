import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class DLinear(nn.Module):
    """
    Irregular‐time‐series‐adapted DLinear, now using (value, mask, timestamp).
    """

    def __init__(self, configs, individual=False):
        super(DLinear, self).__init__()
        self.input_len = configs.input_len
        self.seq_len = configs.input_len
        self.pred_len = configs.pred_len
        self.individual = individual
        self.C = configs.enc_in

        # decomposition
        self.decomposition = series_decomp(configs.moving_avg)

        # seasonal & trend linears
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.C)]
            )
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.C)]
            )
            # time‐projection linears
            self.Linear_Time = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.C)]
            )
            # init all to uniform 1/seq_len
            for lin in (
                list(self.Linear_Seasonal)
                + list(self.Linear_Trend)
                + list(self.Linear_Time)
            ):
                lin.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones_like(lin.weight)
                )
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Time = nn.Linear(self.seq_len, self.pred_len)
            for lin in (self.Linear_Seasonal, self.Linear_Trend, self.Linear_Time):
                lin.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones_like(lin.weight)
                )

        # pad buffer
        self.zeros_pad = torch.zeros(
            configs.batch_size,
            max(self.seq_len, self.pred_len),
            self.C,
            device=configs.device,
        )

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        B, L, C = observed_data.shape
        assert C == self.C

        # 1) pad history if too short
        if L < self.input_len:
            pad = self.input_len - L
            observed_data = torch.cat(
                [observed_data, self.zeros_pad[:B, :pad, :]], dim=1
            )
            observed_mask = torch.cat(
                [observed_mask, self.zeros_pad[:B, :pad, :]], dim=1
            )
            observed_tp = torch.cat([observed_tp, self.zeros_pad[:B, :pad, 0]], dim=1)

        # 2) pad future tp
        Lp = tp_to_predict.size(1)
        if Lp < self.pred_len:
            tp_to_predict = torch.cat(
                [tp_to_predict, tp_to_predict.new_zeros(B, self.pred_len - Lp)], dim=1
            )

        # 3) mask & normalize data
        x = observed_data * observed_mask
        sums = observed_mask.sum(1, keepdim=True).clamp(min=1)
        means = x.sum(1, keepdim=True) / sums
        x = x - means
        var = ((x * observed_mask) ** 2).sum(1, keepdim=True) / sums
        stdev = torch.sqrt(var + 1e-5)
        x = x / stdev

        # 4) decompose
        seasonal_init, trend_init = self.decomposition(x)  # both [B, L, C]
        # bring to [B, C, L]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # 5) build time‐channel [B, C, L]
        time_init = observed_tp.unsqueeze(1).repeat(1, C, 1)

        # 6) project each component
        if self.individual:
            seasonal_out = torch.stack(
                [self.Linear_Seasonal[i](seasonal_init[:, i, :]) for i in range(C)],
                dim=1,
            )  # [B, C, pred_len]
            trend_out = torch.stack(
                [self.Linear_Trend[i](trend_init[:, i, :]) for i in range(C)], dim=1
            )
            time_out = torch.stack(
                [self.Linear_Time[i](time_init[:, i, :]) for i in range(C)], dim=1
            )
        else:
            # flatten B*C → apply single Linear, then reshape
            bc = B * C
            s_flat = seasonal_init.reshape(bc, self.seq_len)
            t_flat = trend_init.reshape(bc, self.seq_len)
            # print(time_init.shape, self.seq_len, bc)
            tp_flat = time_init.reshape(bc, self.seq_len)

            seasonal_out = self.Linear_Seasonal(s_flat).reshape(B, C, self.pred_len)
            trend_out = self.Linear_Trend(t_flat).reshape(B, C, self.pred_len)
            time_out = self.Linear_Time(tp_flat).reshape(B, C, self.pred_len)

        # 7) sum them, back to [B, Lp, C]
        dec = (seasonal_out + trend_out + time_out).permute(0, 2, 1)

        # 8) de‐normalize on horizon
        dec = dec * stdev.expand(-1, self.pred_len, -1) + means.expand(
            -1, self.pred_len, -1
        )

        # 9) slice to actual Lp
        return dec[:, :Lp, :]
