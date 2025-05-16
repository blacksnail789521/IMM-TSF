import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding
from layers.StandardNorm import Normalize


class DFT_series_decomp(nn.Module):
    """FFT‐based decomposition into seasonality & trend."""

    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        # x: [B, T, C]
        xf = torch.fft.rfft(x, dim=1)
        freq = xf.abs()
        freq[:, 0, :] = 0
        # find cutoff
        topk_vals, _ = torch.topk(freq, self.top_k, dim=1)
        cutoff = topk_vals.min(dim=1, keepdim=True)[0]
        xf[freq <= cutoff] = 0
        season = torch.fft.irfft(xf, n=x.size(1), dim=1)
        trend = x - season
        return season, trend


class MultiScaleSeasonMixing(nn.Module):
    """Bottom‐up mixing of multiscale seasonal components."""

    def __init__(self, configs):
        super().__init__()
        seq_len = configs.input_len
        down_w = configs.down_sampling_window
        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(seq_len // (down_w**i), seq_len // (down_w ** (i + 1))),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_w ** (i + 1)), seq_len // (down_w ** (i + 1))
                    ),
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        # season_list: each [B, d_model, T_i]
        out_high = season_list[0]
        out_seasons = [out_high.permute(0, 2, 1)]  # [B, T0, d_model]
        out_low = season_list[1]

        for i, layer in enumerate(self.down_sampling_layers):
            res = layer(out_high)  # maps last dim T_i -> T_{i+1}
            out_low = out_low + res  # [B, d_model, T_{i+1}]
            out_high = out_low
            if i + 2 < len(season_list):
                out_low = season_list[i + 2]
            out_seasons.append(out_high.permute(0, 2, 1))
        return out_seasons


class MultiScaleTrendMixing(nn.Module):
    """Top‐down mixing of multiscale trend components."""

    def __init__(self, configs):
        super().__init__()
        seq_len = configs.input_len
        down_w = configs.down_sampling_window
        layers = []
        for i in reversed(range(configs.down_sampling_layers)):
            layers.append(
                nn.Sequential(
                    nn.Linear(seq_len // (down_w ** (i + 1)), seq_len // (down_w**i)),
                    nn.GELU(),
                    nn.Linear(seq_len // (down_w**i), seq_len // (down_w**i)),
                )
            )
        self.up_sampling_layers = nn.ModuleList(layers)

    def forward(self, trend_list):
        rev = list(reversed(trend_list))  # each [B, d_model, T_i]
        out_low = rev[0]
        out_trends = [out_low.permute(0, 2, 1)]  # [B,T_last,d_model]
        out_high = rev[1]

        for i, layer in enumerate(self.up_sampling_layers):
            res = layer(out_low)  # upsample
            out_high = out_high + res
            out_low = out_high
            if i + 2 < len(rev):
                out_high = rev[i + 2]
            out_trends.append(out_low.permute(0, 2, 1))
        return list(reversed(out_trends))


class PastDecomposableMixing(nn.Module):
    """The multiscale decomposable‐mixing block."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.pred_len
        self.down_w = configs.down_sampling_window

        # choose decomposition
        if configs.decomp_method == "moving_avg":
            self.decomposition = series_decomp(configs.moving_avg)
        else:
            self.decomposition = DFT_series_decomp(configs.top_k)

        # optional cross‐channel mixing
        if not configs.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU(),
                nn.Linear(configs.d_ff, configs.d_model),
            )

        # mixers
        self.mix_season = MultiScaleSeasonMixing(configs)
        self.mix_trend = MultiScaleTrendMixing(configs)

        # final cross layer
        self.out_layer = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model),
        )

    def forward(self, x_list):
        # x_list: list of [B, T_i, d_model]
        lengths = [x.shape[1] for x in x_list]
        # decompose each
        seasons = []
        trends = []
        for x in x_list:
            s, t = self.decomposition(x)  # each [B, T_i, d_model]
            if hasattr(self, "cross_layer"):
                s = self.cross_layer(s)
                t = self.cross_layer(t)
            # prepare for mixing
            seasons.append(s.permute(0, 2, 1))  # [B, d_model, T_i]
            trends.append(t.permute(0, 2, 1))

        # mix at all scales
        out_seasons = self.mix_season(seasons)  # list of [B, T_i, d_model]
        out_trends = self.mix_trend(trends)

        # combine and add residual
        out_list = []
        for orig, os, ot, L in zip(x_list, out_seasons, out_trends, lengths):
            # os, ot: [B, T_i, d_model]
            combined = os + ot  # [B, T_i, d_model]
            if hasattr(self, "out_layer"):
                combined = orig + self.out_layer(combined)
            out_list.append(combined[:, :L, :])
        return out_list


class TimeMixer(nn.Module):
    """Irregular‐forecasting TimeMixer."""

    def __init__(self, configs):
        super().__init__()
        self.input_len = configs.input_len
        self.pred_len = configs.pred_len
        self.C = configs.enc_in
        self.layers = configs.e_layers

        # ────────────────────────────────────────────────────────────
        # Compute how many down‐sampling layers actually fit:
        max_layers = 0
        cur_len = configs.input_len
        while (
            max_layers < configs.down_sampling_layers
            and cur_len >= configs.down_sampling_window
        ):
            cur_len //= configs.down_sampling_window
            max_layers += 1
        configs.down_sampling_layers = max_layers
        self.down_layers = configs.down_sampling_layers
        self.down_w = configs.down_sampling_window
        self.configs = configs
        # ────────────────────────────────────────────────────────────

        # padding buffer
        self.zeros_pad = torch.zeros(
            configs.batch_size,
            max(self.input_len, self.pred_len),
            self.C,
            device=configs.device,
        )

        # embed (value, mask, timestamp) ⇒ 2*C+1 channels
        in_ch = 2 * self.C + 1
        self.enc_embedding = DataEmbedding(
            in_ch, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # normalize at each scale
        self.normalize_layers = nn.ModuleList(
            [
                Normalize(self.C, affine=True, non_norm=False)
                for _ in range(self.down_layers + 1)
            ]
        )

        # predictor at each down‐sampling scale
        self.predict_layers = nn.ModuleList(
            [
                nn.Linear(configs.input_len // (self.down_w**i), configs.pred_len)
                for i in range(self.down_layers + 1)
            ]
        )

        # final projection back to C channels
        self.projection = nn.Linear(configs.d_model, self.C, bias=True)

        # PDM blocks
        self.pdm_blocks = nn.ModuleList(
            [PastDecomposableMixing(configs) for _ in range(self.layers)]
        )

    def __multi_scale_process_inputs(self, x_enc, x_mask):
        """Down‐sample x_enc:[B,T,2C+1] & x_mask:[B,T,C]."""
        method = self.configs.down_sampling_method
        w = self.down_w
        if method == "max":
            pool = nn.MaxPool1d(w)
        elif method == "avg":
            pool = nn.AvgPool1d(w)
        elif method == "conv":
            pool = nn.Conv1d(
                in_channels=x_enc.size(-1),
                out_channels=x_enc.size(-1),
                kernel_size=3,
                padding=1,
                stride=w,
                padding_mode="circular",
                bias=False,
            )
        else:
            return [x_enc], [x_mask]

        x_list, m_list = [], []
        x_cur = x_enc.permute(0, 2, 1)  # [B, 2C+1, T]
        m_cur = x_mask  # [B, T, C]

        x_list.append(x_cur.permute(0, 2, 1))
        m_list.append(m_cur)

        for _ in range(self.down_layers):
            x_next = pool(x_cur)
            # if the time‐axis is too short to pool further, stop
            if x_next.size(-1) == 0:
                break
            m_next = m_cur[:, ::w, :]
            x_list.append(x_next.permute(0, 2, 1))
            m_list.append(m_next)
            x_cur, m_cur = x_next, m_next

        return x_list, m_list

    def forecasting(
        self,
        tp_to_predict: torch.Tensor,
        observed_data: torch.Tensor,
        observed_tp: torch.Tensor,
        observed_mask: torch.Tensor,
    ):
        B, L, C = observed_data.shape

        # 1) pad history
        if L < self.input_len:
            pad = self.input_len - L
            observed_data = torch.cat(
                [observed_data, self.zeros_pad[:B, :pad, :]], dim=1
            )
            observed_mask = torch.cat(
                [observed_mask, self.zeros_pad[:B, :pad, :]], dim=1
            )
            observed_tp = torch.cat([observed_tp, self.zeros_pad[:B, :pad, 0]], dim=1)

        # 2) pad future times
        Lp = tp_to_predict.size(1)
        if Lp < self.pred_len:
            pad = self.pred_len - Lp
            tp_to_predict = torch.cat(
                [tp_to_predict, tp_to_predict.new_zeros(B, pad)], dim=1
            )

        # 3) mask & normalize
        x = observed_data * observed_mask
        sums = observed_mask.sum(1, keepdim=True).clamp(min=1)
        means = x.sum(1, keepdim=True) / sums
        x = x - means
        var = ((x * observed_mask) ** 2).sum(1, keepdim=True) / sums
        stdev = torch.sqrt(var + 1e-5)
        x = x / stdev

        # 4) build input = [value, mask, timestamp]
        enc_in = torch.cat([x, observed_mask, observed_tp.unsqueeze(-1)], dim=-1)

        # 5) multiscale down‐sampling
        x_list, m_list = self.__multi_scale_process_inputs(enc_in, observed_mask)

        # 6) embed each scale (feed through both value+mask+time together)
        enc_out_list = [self.enc_embedding(xi, None) for xi in x_list]

        # 7) Past Decomposable Mixing blocks
        for block in self.pdm_blocks:
            enc_out_list = block(enc_out_list)

        # 8) predict at coarsest scale
        dec = self.predict_layers[-1](enc_out_list[-1].permute(0, 2, 1)).permute(
            0, 2, 1
        )
        dec = self.projection(dec)

        # 9) de‐normalize & slice
        dec = dec * stdev + means
        return dec[:, :Lp, :]
