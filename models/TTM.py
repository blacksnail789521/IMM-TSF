import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.MLP import TTMLayer

###############################################################################
# ——— the original TTM building-blocks and head, verbatim ——————————————————
###############################################################################


class TTMAPBlock(nn.Module):
    def __init__(
        self, e_layers, d_model, num_patches, n_vars, mode, adapt_patch_level, dropout
    ):
        super().__init__()
        self.adapt_patch_level = adapt_patch_level
        adaptive_patch_factor = 2**adapt_patch_level
        self.adaptive_patch_factor = adaptive_patch_factor
        self.mixer_layers = nn.ModuleList(
            [
                TTMLayer(
                    d_model=d_model // adaptive_patch_factor,
                    num_patches=num_patches * adaptive_patch_factor,
                    n_vars=n_vars,
                    mode=mode,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )

    def forward(self, x):
        B, M, N, D = x.shape
        # reshape channels→patch dims
        x = x.reshape(
            B, M, N * self.adaptive_patch_factor, D // self.adaptive_patch_factor
        )
        for m in self.mixer_layers:
            x = m(x)
        # restore
        B, M, n2, d2 = x.shape
        return x.reshape(
            B, M, n2 // self.adaptive_patch_factor, d2 * self.adaptive_patch_factor
        )


class TTMBlock(nn.Module):
    def __init__(
        self, e_layers, AP_levels, d_model, num_patches, n_vars, mode, dropout
    ):
        super().__init__()
        self.AP_levels = AP_levels
        if AP_levels > 0:
            self.mixers = nn.ModuleList(
                [
                    TTMAPBlock(
                        e_layers=e_layers,
                        d_model=d_model,
                        num_patches=num_patches,
                        n_vars=n_vars,
                        mode=mode,
                        adapt_patch_level=i,
                        dropout=dropout,
                    )
                    for i in reversed(range(AP_levels))
                ]
            )
        else:
            self.mixers = nn.ModuleList(
                [
                    TTMLayer(
                        d_model=d_model,
                        num_patches=num_patches,
                        n_vars=n_vars,
                        mode=mode,
                        dropout=dropout,
                    )
                    for _ in range(e_layers)
                ]
            )

    def forward(self, x):
        for m in self.mixers:
            x = m(x)
        return x


class TTMPredicationHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.dropout_layer = nn.Dropout(configs.dropout)
        head_d_model = configs.d_d_model if configs.use_decoder else configs.d_model
        self.base_forecast_block = nn.Linear(
            configs.num_patches * head_d_model, configs.pred_len
        )
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x):
        # x: [B, M, N, D]
        x = self.flatten(x)  # → [B, M, N*D]
        x = self.dropout_layer(x)  # → [B, M, N*D]
        out = self.base_forecast_block(x)  # → [B, M, pred_len]
        return out.transpose(-1, -2).contiguous()  # → [B, pred_len, M]


class TTMBackbone(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.encoder = TTMBlock(
            e_layers=configs.e_layers,
            AP_levels=configs.AP_levels,
            d_model=configs.d_model,
            num_patches=configs.num_patches,
            n_vars=configs.n_vars,
            mode=configs.mode,
            dropout=configs.dropout,
        )
        self.patcher = nn.Linear(configs.patch_size, configs.d_model)
        self.patch_size = configs.patch_size
        self.stride = configs.stride

    def forward(self, x):
        # x: [B, L, M] → [B, M, L]
        x = x.permute(0, 2, 1)
        # unfold → [B, M, N, P]
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # P→D
        x = self.patcher(x)  # [B, M, N, D]
        return self.encoder(x)  # → [B, M, N, D]


class Model(nn.Module):
    """
    The original TTM interface:
        forward(x, x_mark, _, y_mark) → [B, pred_len, M]
    """

    def __init__(self, configs):
        super().__init__()
        # recompute num_patches from input_len
        configs.num_patches = (
            max(configs.input_len, configs.patch_size) - configs.patch_size
        ) // configs.stride + 1

        self.configs = configs
        self.pred_len = configs.pred_len
        self.n_vars = configs.n_vars
        self.backbone = TTMBackbone(configs)
        self.use_decoder = configs.use_decoder
        self.use_norm = configs.use_norm

        if self.use_decoder:
            self.decoder_adapter = nn.Linear(configs.d_model, configs.d_d_model)
            self.decoder = TTMBlock(
                e_layers=configs.d_layers,
                AP_levels=0,
                d_model=configs.d_d_model,
                num_patches=configs.num_patches,
                n_vars=configs.n_vars,
                mode=configs.mode,
                dropout=configs.dropout,
            )

        self.head = TTMPredicationHead(configs)

    def forward(self, x, x_mark, _, y_mark):
        # x: [B, L, M]
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - means) / stdev

        dec_in = self.backbone(x)  # [B, M, N, D]

        if self.use_decoder:
            dec_in = self.decoder_adapter(dec_in)
            dec_out = self.decoder(dec_in)
        else:
            dec_out = dec_in

        y_hat = self.head(dec_out)  # [B, pred_len, M]

        if self.use_norm:
            y_hat = y_hat * stdev + means

        return y_hat


###############################################################################
# ——— Irregular-TS adapter around the original TTM ——————————————————————
###############################################################################


class TTM(Model):
    """
    Irregular‐TS adapter around the original TTM backbone,
    using the 2C+1 trick (data, mask, timestamp).
    """

    def __init__(self, configs):
        self.C = configs.enc_in  # number of channels

        # 0) figure out how many real variables you have
        original_vars = configs.enc_in

        # 1) bump n_vars so backbone builds with new channel count
        configs.n_vars = original_vars * 2 + 1

        # 2) now call original TTM __init__ (it will see configs.n_vars)
        super().__init__(configs)

        # 3) stash for de-normalization & slicing
        self.orig_vars = original_vars
        self.input_len = configs.input_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        # 4) pad‐buffer for short sequences
        self.zeros_pad = torch.zeros(
            configs.batch_size,
            max(self.input_len, self.pred_len),
            self.C,
            device=configs.device,
        )

    def forecasting(
        self,
        tp_to_predict: torch.Tensor,  # [B, Lp]
        observed_data: torch.Tensor,  # [B, L, C]
        observed_tp: torch.Tensor,  # [B, L]
        observed_mask: torch.Tensor,  # [B, L, C]
    ) -> torch.Tensor:  # → [B, Lp, C]
        B, L, C = observed_data.shape
        assert C == self.orig_vars, f"expected {self.orig_vars} channels, got {C}"

        # # 1) pad history if too short
        # if L < self.input_len:
        #     p = self.input_len - L
        #     pad_all = self.zeros_pad[:B, :p, :]  # [B, p, 2C+1]
        #     # split pad into data/mask/tp parts
        #     observed_data = torch.cat([observed_data, pad_all[..., :C]], dim=1)
        #     observed_mask = torch.cat([observed_mask, pad_all[..., C : 2 * C]], dim=1)
        #     observed_tp = torch.cat(
        #         [observed_tp, torch.zeros_like(observed_tp[:, :p])], dim=1
        #     )

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

        # 2) pad future tp if too short
        Lp = tp_to_predict.size(1)
        if Lp < self.pred_len:
            tp_to_predict = F.pad(tp_to_predict, (0, self.pred_len - Lp))

        # 3) build 2C+1 channels: (value*mask), mask, timestamp
        vals = observed_data * observed_mask  # zero out missing
        mk = observed_mask
        tp_ch = observed_tp.unsqueeze(-1)  # [B, L, 1]

        enc_in = torch.cat([vals, mk, tp_ch], dim=-1)  # [B, L, 2C+1]

        # 4) optional normalization (non‐stationary transformer style)
        if self.use_norm:
            sums = observed_mask.sum(1).clamp(min=1)  # [B, C]
            means = vals.sum(1) / sums  # [B, C]
            centered = vals - means.unsqueeze(1)
            var = ((centered * observed_mask) ** 2).sum(1) / sums  # [B, C]
            stdev = torch.sqrt(var + 1e-5)  # [B, C]
            vals_n = centered / stdev.unsqueeze(1)

            # center mask around 0.5, leave its scale
            mk_n = mk - 0.5

            # standardize timestamp
            tp_mean = tp_ch.mean(1, keepdim=True)
            tp_std = tp_ch.std(1, keepdim=True) + 1e-5
            tp_n = (tp_ch - tp_mean) / tp_std

            enc_in = torch.cat([vals_n, mk_n, tp_n], dim=-1)

        # 5) call original forward (it will run through your patched backbone)
        #    we pass `None` for the extra args; the Model.forward ignores them.
        y_full = super().forward(enc_in, None, None, None)  # [B, pred_len, 2C+1]

        # 6) extract only the data‐channel forecasts
        y_data = y_full[..., :C]  # [B, pred_len, C]

        # 7) de-normalize if needed
        if self.use_norm:
            y_data = y_data * stdev.unsqueeze(1) + means.unsqueeze(1)

        # 8) slice to actual Lp
        return y_data[:, :Lp, :]
