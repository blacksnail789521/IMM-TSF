import os
import pandas as pd
import glob
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from prettytable import PrettyTable
import math
import argparse

import lib.utils as utils


class ChunkedTimeSeriesDataset(Dataset):
    """
    Dataset for irregular time series that optionally includes precomputed text embeddings.

    Each record lives under `root/processed/<record_id>/` and must contain
      - `time_series.csv`
      - optional `text_embeddings_{llm_model_fusion}_{llm_layers_fusion or 'full'}.pt`

    Yields per‐chunk items:
      (chunk_id, tt, vals, mask, texts)
    where `texts` is a list of (t, payload), with payload either a str
    (raw note) or a Tensor(d_txt) (precomputed embedding), filtered to the
    history window only.
    """

    UNIT_SECONDS = {
        "seconds": 1.0,
        "minutes": 60.0,
        "hours": 3600.0,
        "days": 86400.0,
        "weeks": 604800.0,
    }

    def __init__(
        self,
        root: str,
        history: int,
        pred_window: int,
        stride: int,
        device: torch.device = torch.device("cpu"),
        time_unit: str = "days",
        unit_scale: float | None = None,
        normalize: bool = True,
        enable_text: bool = False,
        use_text_embeddings: bool = False,
        llm_model_fusion: str | None = None,
        llm_layers_fusion: int | None = None,
        max_length: int = 1024,
        args: argparse.Namespace | None = None,
    ):
        super().__init__()
        self.history = history
        self.pred_window = pred_window
        self.stride = stride
        self.device = device
        self.normalize = normalize
        self.enable_text = enable_text
        self.use_text_embeddings = use_text_embeddings
        self.llm_model_fusion = llm_model_fusion
        self.llm_layers_fusion = llm_layers_fusion

        # determine time‐unit scale
        if time_unit == "custom":
            if unit_scale is None:
                raise ValueError("Must set unit_scale when time_unit='custom'")
            self._sec_per_unit = float(unit_scale)
        else:
            try:
                self._sec_per_unit = self.UNIT_SECONDS[time_unit]
            except KeyError:
                raise ValueError(f"Unknown time_unit '{time_unit}'")

        proc_dir = os.path.join(root, "processed")
        rec_ids = sorted(
            d for d in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, d))
        )

        # TODO: If rec_ids is in args, we use that instead of all rec_ids
        if isinstance(args, argparse.Namespace) and getattr(args, "rec_ids", None) is not None:
            rec_ids = args.rec_ids
            # raise Exception(rec_ids)

        raw_data = []  # List of tuples: (rec, tt, vals, mask, record_texts)
        for rec in rec_ids:
            ts_path = os.path.join(proc_dir, rec, "time_series.csv")
            if not os.path.isfile(ts_path):
                continue

            # load and normalize time series
            df = pd.read_csv(ts_path)
            df["_ts_raw"] = pd.to_datetime(df["date_time"])
            df = df.sort_values("_ts_raw")

            feat_cols = [
                c for c in df.columns if c not in ("date_time", "record_id", "_ts_raw")
            ]
            if normalize:
                df[feat_cols] = df[feat_cols].apply(
                    lambda col: (
                        ((col - col.mean()) / col.std())
                        if col.std()
                        else (col - col.mean())
                    ),
                    axis=0,
                )

            # Time → float Tensor
            secs = (df["_ts_raw"] - df["_ts_raw"].min()).dt.total_seconds()
            units = secs / self._sec_per_unit
            tt = torch.tensor(units.values, dtype=torch.float32, device=device)

            

            # Values & mask
            vals_np = df[feat_cols].values.astype("float32")
            mask_np = ~pd.isna(vals_np)
            vals = torch.nan_to_num(torch.tensor(vals_np, device=device))
            mask = torch.tensor(mask_np.astype("float32"), device=device)

            # Check if mask are all 0
            if mask.sum() == 0:
                raise ValueError(f"Mask for {rec} is all zeros")

            # load notes or embeddings (even when enable_text=False)
            texts: list[tuple[float, object]] = []
            if use_text_embeddings and llm_model_fusion and enable_text:
                # find the .pt file
                fname = (
                    f"text_embeddings_model={llm_model_fusion}"
                    f"_layers={llm_layers_fusion or 'full'}"
                    f"_maxlen={max_length}.pt"
                )
                path = os.path.join(proc_dir, rec, fname)
                if os.path.isfile(path):
                    data = torch.load(path, map_location=device)
                    emb = data["embeddings"]  # [N_notes, d_txt]
                    if torch.isnan(emb).any():
                        raise ValueError("text embeddings contains NaN values.")
                    rel = data["rel_times"].to(device)  # [N_notes]
                    for i, t in enumerate(rel):
                        texts.append((t.item(), emb[i]))
                else:
                    raise FileNotFoundError(f"Missing text embeddings file: {path}")
            else:
                # raw text fallback
                text_path = os.path.join(proc_dir, rec, "text.csv")
                if os.path.isfile(text_path):
                    tdf = pd.read_csv(text_path, parse_dates=["date_time"])
                    tdf = tdf.sort_values("date_time")  # Ensure chronological order
                    cols = [
                        c for c in tdf.columns if c not in ("date_time", "record_id")
                    ]
                    if len(cols) != 1:
                        raise ValueError(f"{rec}: expected 1 text column, got {cols}")
                    text_col = cols[0]
                    base = df["_ts_raw"].min()
                    for _, row in tdf.iterrows():
                        txt = row[text_col]
                        if pd.isna(txt):
                            continue
                        t_rel = (
                            row["date_time"] - base
                        ).total_seconds() / self._sec_per_unit
                        texts.append((t_rel, txt))

            raw_data.append((rec, tt, vals, mask, texts))

        # chunking
        total = history + pred_window
        chunks: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, list]] = []
        for rec, tt, vals, mask, record_texts in raw_data:
            t_max = tt.max().item()
            st = tt.min().item()
            cnt = 0
            drop_count = 0
            while st + total <= t_max:
                idx = (
                    ((tt >= st) & (tt < st + total)).nonzero(as_tuple=False).squeeze(1)
                )
                if idx.numel() >= 2:
                    sub_tt = tt[idx] - st
                    sub_vals = vals[idx]
                    sub_mask = mask[idx]

                    # # Check whether sub_tt is strictly increasing
                    # if not torch.all(torch.diff(sub_tt) > 0):
                    #     raise ValueError(f"Sub tt for {rec} is not strictly increasing")

                    # Split into history and prediction portions
                    hist_mask = sub_mask[sub_tt < history]
                    pred_mask = sub_mask[sub_tt >= history]

                    # Ensure both windows contain at least one valid value
                    if hist_mask.sum() == 0 or pred_mask.sum() == 0:
                        st += stride
                        continue

                    if sub_mask.sum() == 0:
                        raise ValueError(f"Sub mask for {rec} is all zeros")

                    # filter texts in history window only
                    hist_end = st + history
                    selected = [
                        (t - st, payload)
                        for (t, payload) in record_texts
                        if st <= t < hist_end
                    ]
                    chunk_id = f"{rec}_chunk{cnt}"
                    cnt += 1

                    # We drop the samples with no text (even when enable_text=False)
                    if len(selected) == 0:
                        drop_count += 1
                        st += self.stride
                        continue

                    if enable_text:
                        chunks.append((chunk_id, sub_tt, sub_vals, sub_mask, selected))
                    else:
                        chunks.append((chunk_id, sub_tt, sub_vals, sub_mask, []))
                st += stride

            # Show total count and drop count
            drop_ratio = drop_count / (cnt + drop_count)
            print(
                f"Record {rec}: {cnt} chunks created, {drop_count} dropped ({drop_ratio:.2%})"
            )

        if not chunks:
            raise RuntimeError("No chunks created; check history/pred_window/stride")
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        # returns: chunk_id, tt, vals, mask, texts
        return self.chunks[idx]


#####################################################################################################
# Collate Functions
#####################################################################################################


def variable_time_collate_fn(batch, args, time_max=None):
    observed_tp, observed_data, observed_mask = [], [], []
    predicted_tp, predicted_data, predicted_mask = [], [], []
    chunk_time_max = (
        time_max
        if time_max is not None
        else torch.tensor(args.history + args.pred_window).to(args.device)
    )
    for _, tt, vals, mask in batch:
        hist_idx = torch.where(tt < args.history)[0]
        pred_idx = torch.where(tt >= args.history)[0]
        # print(tt, args.history)
        if mask[pred_idx].sum() == 0:
            raise Exception(tt, args.history)
            raise ValueError(
                f"Mask for batch is all zeros in collate_fn, predicted index: {pred_idx}"
            )
        observed_tp.append(tt[hist_idx])
        observed_data.append(vals[hist_idx])
        observed_mask.append(mask[hist_idx])
        predicted_tp.append(tt[pred_idx])
        predicted_data.append(vals[pred_idx])
        predicted_mask.append(mask[pred_idx])
    observed_tp = pad_sequence(observed_tp, batch_first=True, padding_value=0.0)
    observed_data = pad_sequence(observed_data, batch_first=True, padding_value=0.0)
    observed_mask = pad_sequence(observed_mask, batch_first=True, padding_value=0.0)
    predicted_tp = pad_sequence(predicted_tp, batch_first=True, padding_value=0.0)
    predicted_data = pad_sequence(predicted_data, batch_first=True, padding_value=0.0)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True, padding_value=0.0)

    observed_tp = utils.normalize_masked_tp(
        observed_tp, att_min=0.0, att_max=chunk_time_max
    )
    predicted_tp = utils.normalize_masked_tp(
        predicted_tp, att_min=0.0, att_max=chunk_time_max
    )
    return {
        "observed_data": observed_data,
        "observed_tp": observed_tp,
        "observed_mask": observed_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask,
    }


def patch_variable_time_collate_fn(batch, args, time_max=None):
    if not batch:
        return None
    D = batch[0][2].shape[1]
    chunk_time_max = (
        time_max
        if time_max is not None
        else torch.tensor(args.history + args.pred_window).to(args.device)
    )
    obs_tps, obs_vals, obs_masks = [], [], []
    pred_tps, pred_vals, pred_masks = [], [], []
    for _, tt, vals, mask in batch:
        hidx = torch.where(tt < args.history)[0]
        pidx = torch.where(tt >= args.history)[0]
        obs_tps.append(tt[hidx])
        obs_vals.append(vals[hidx])
        obs_masks.append(mask[hidx])
        pred_tps.append(tt[pidx])
        pred_vals.append(vals[pidx])
        pred_masks.append(mask[pidx])
    ptp = pad_sequence(pred_tps, batch_first=True, padding_value=0.0)
    pval = pad_sequence(pred_vals, batch_first=True, padding_value=0.0)
    pmask = pad_sequence(pred_masks, batch_first=True, padding_value=0.0)
    non_empty = [t for t in obs_tps if len(t) > 0]
    if non_empty:
        combined_tt, inv = torch.unique(
            torch.cat(non_empty), sorted=True, return_inverse=True
        )
        n_pts = len(combined_tt)
    else:
        combined_tt = torch.tensor([], device=args.device)
        inv = torch.tensor([], dtype=torch.long, device=args.device)
        n_pts = 0
    B = len(batch)
    combined_vals = torch.zeros(B, n_pts, D, device=args.device)
    combined_mask = torch.zeros(B, n_pts, D, device=args.device)
    offset = 0
    for i in range(B):
        tpi = obs_tps[i]
        if len(tpi) > 0:
            idxs = inv[offset : offset + len(tpi)]
            combined_vals[i, idxs] = obs_vals[i]
            combined_mask[i, idxs] = obs_masks[i]
            offset += len(tpi)
    norm_combined_tt = utils.normalize_masked_tp(
        combined_tt, att_min=0.0, att_max=chunk_time_max
    )
    norm_ptp = utils.normalize_masked_tp(ptp, att_min=0.0, att_max=chunk_time_max)
    unnorm_tt = combined_tt
    patch_indices = []
    patch_size = args.patch_size
    patch_stride = args.patch_stride
    for i in range(args.npatch):
        st = i * patch_stride
        ed = st + patch_size
        if i == args.npatch - 1:
            mask_idx = (unnorm_tt >= st) & (unnorm_tt < args.history)
        else:
            mask_idx = (unnorm_tt >= st) & (unnorm_tt < ed)
        patch_indices.append(torch.where(mask_idx)[0])
    data_dict = {
        "data": combined_vals,
        "time_steps": norm_combined_tt,
        "mask": combined_mask,
        "data_to_predict": pval,
        "tp_to_predict": norm_ptp,
        "mask_predicted_data": pmask,
    }
    return utils.split_and_patch_batch(data_dict, args, n_pts, patch_indices)


def variable_time_collate_fn_CRU(batch, args, time_max=None):
    """
    Collate function for CRU model.
    Differs from standard variable_time_collate_fn by not normalizing time points (observed_tp, tp_to_predict),
    as CRU might expect chunk-relative time steps directly.
    """
    observed_tp, observed_data, observed_mask = [], [], []
    predicted_tp, predicted_data, predicted_mask = [], [], []

    for _, tt, vals, mask in batch:
        hist_idx = torch.where(tt < args.history)[0]
        pred_idx = torch.where(tt >= args.history)[0]

        observed_tp.append(tt[hist_idx])
        observed_data.append(vals[hist_idx])
        observed_mask.append(mask[hist_idx])

        predicted_tp.append(tt[pred_idx])
        predicted_data.append(vals[pred_idx])
        predicted_mask.append(mask[pred_idx])

    observed_tp = pad_sequence(observed_tp, batch_first=True, padding_value=0.0)
    observed_data = pad_sequence(observed_data, batch_first=True, padding_value=0.0)
    observed_mask = pad_sequence(observed_mask, batch_first=True, padding_value=0.0)

    predicted_tp = pad_sequence(predicted_tp, batch_first=True, padding_value=0.0)
    predicted_data = pad_sequence(predicted_data, batch_first=True, padding_value=0.0)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True, padding_value=0.0)

    # Time point normalization is SKIPPED for CRU.
    # observed_tp and predicted_tp remain as chunk-relative time steps.

    return {
        "observed_data": observed_data,
        "observed_tp": observed_tp,
        "observed_mask": observed_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask,
    }


def variable_time_collate_fn_ODE(batch, args, time_max=None):
    """
    Collate for Latent ODE: returns per-batch data & masks plus a single 1D time vector
    that’s then split into observed vs. to-predict—just like the old version, but with
    batch-size timestamps.
    """
    # 1) concatenate all raw timestamps
    all_tt = torch.cat([tt for (_, tt, _, _) in batch])
    # 2) get unique sorted time axis & inverse indices
    combined_tt_raw, inverse_indices = torch.unique(
        all_tt, sorted=True, return_inverse=True
    )
    # 3) compute how many are in the “history” window
    n_obs = torch.lt(combined_tt_raw, args.history).sum().item()

    B = len(batch)
    D = batch[0][2].size(-1)
    T = combined_tt_raw.size(0)

    # 4) scatter values & masks into a (B, T, D) grid
    combined_vals = torch.zeros(B, T, D, device=args.device)
    combined_mask = torch.zeros(B, T, D, device=args.device)
    offset = 0
    for b, (_, tt, vals, mask) in enumerate(batch):
        L = tt.size(0)
        idx = inverse_indices[offset : offset + L]
        combined_vals[b, idx] = vals.to(args.device)
        combined_mask[b, idx] = mask.to(args.device)
        offset += L

    # 6) compute the normalization cap for time
    cap = (
        time_max
        if time_max is not None
        else torch.tensor(args.history + args.pred_window, device=args.device)
    )
    # 7) normalize the *1D* time vector
    combined_tt = utils.normalize_masked_tp(combined_tt_raw, att_min=0.0, att_max=cap)

    # 7.5) enforce strict monotonicity by adding a tiny per‐step jitter
    #     this shifts any dupes or zero‐/negative‐diffs by ~eps*cap
    eps = torch.finfo(combined_tt.dtype).eps * cap
    idxs = torch.arange(combined_tt.size(0), device=combined_tt.device)
    combined_tt = combined_tt + idxs * eps

    # 8) split back into observed vs. prediction
    observed_tp = combined_tt[:n_obs]  # (n_obs,)
    predicted_tp = combined_tt[n_obs:]  # (T - n_obs,)
    observed_data = combined_vals[:, :n_obs, :]  # (B, n_obs, D)
    predicted_data = combined_vals[:, n_obs:, :]  # (B, T-n_obs, D)
    observed_mask = combined_mask[:, :n_obs, :]
    predicted_mask = combined_mask[:, n_obs:, :]

    return {
        "observed_data": observed_data,
        "observed_tp": observed_tp,
        "observed_mask": observed_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask,
    }


#####################################################################################################
# Main Data Parsing Function
#####################################################################################################


def get_input_and_pred_len(data_obj):
    """
    Scans one full epoch of train/val/test data to find:
      - max_input_len  = largest observed_data.shape[1]
      - max_pred_len   = largest data_to_predict.shape[1]
    Returns (max_input_len, max_pred_len).
    """

    max_input_len = 0
    max_pred_len = 0

    # We'll iterate through exactly one epoch per split.
    splits = [
        ("train", data_obj["train_dataloader"]),
        ("val", data_obj["val_dataloader"]),
    ]
    # test_dataloader may be None or have its own count
    if data_obj.get("test_dataloader") is not None:
        splits.append(("test", data_obj["test_dataloader"]))

    for name, dataloader in splits:
        print(f"Scanning {name} split ({len(dataloader)} batches)...")
        for batch in dataloader:
            # observed_data: (B, T_obs, D)
            # data_to_predict: (B, T_pred, D)
            T_obs = batch["observed_data"].shape[1]
            T_pred = batch["data_to_predict"].shape[1]

            if T_obs > max_input_len:
                max_input_len = T_obs
            if T_pred > max_pred_len:
                max_pred_len = T_pred

    return max_input_len, max_pred_len


def show_ds_summary(args):
    # 1) Gather all paths to raw time_series.csv files
    data_glob = os.path.join(
        args.data_root, args.dataset, "processed", "*", "time_series.csv"
    )
    paths = glob.glob(data_glob)

    # 2) num_entities & num_features (from the first file)
    num_entities = len(paths)
    first_df = pd.read_csv(paths[0], parse_dates=["date_time"])
    feature_cols = [c for c in first_df.columns if c not in ["date_time", "record_id"]]
    num_features = len(feature_cols)

    # Prepare accumulators
    total_obs = 0
    feat_counts = np.zeros(num_features, dtype=float)
    all_times = []
    all_dts = []
    all_text_times = []  # NEW: to compute text temporal entropy
    total_text = 0

    # 3) Loop files to accumulate
    for p in paths:
        df = pd.read_csv(p, parse_dates=["date_time"])
        # a) observations per feature
        mask = df[feature_cols].notna().to_numpy(dtype=int)
        total_obs += mask.sum()
        feat_counts += mask.sum(axis=0)
        # b) timestamps
        times = df["date_time"].sort_values().to_numpy()
        all_times.append(times)
        # c) inter-obs intervals
        dts = np.diff(times).astype("timedelta64[s]").astype(float)
        all_dts.append(dts)

        # d) text
        text_path = p.replace("time_series.csv", "text.csv")
        if os.path.isfile(text_path):
            tdf = pd.read_csv(text_path, parse_dates=["date_time"])
            text_cols = [c for c in tdf.columns if c not in ("date_time", "record_id")]
            if len(text_cols) == 1:
                total_text += tdf[text_cols[0]].notna().sum()
                all_text_times.append(tdf["date_time"].dropna().to_numpy())

    # 4) Consolidate arrays
    all_times = np.concatenate(all_times)
    all_dts = np.concatenate(all_dts)
    num_unique_timestamps = len(np.unique(all_times))

    # 5) Feature Observability Entropy
    p_feat = feat_counts / total_obs
    feat_obs_entropy = -(p_feat * np.log(p_feat + 1e-12)).sum()
    H_feat_max = math.log(num_features)
    feat_obs_entropy_norm = feat_obs_entropy / H_feat_max

    # 6) Temporal Observation Entropy (numeric)
    K = 10
    t_min = all_times.min().astype("datetime64[s]").astype(float)
    t_max = all_times.max().astype("datetime64[s]").astype(float)
    bins = np.linspace(t_min, t_max, K + 1)
    counts, _ = np.histogram(all_times.astype("datetime64[s]").astype(float), bins=bins)
    p_time = counts / counts.sum()
    temp_obs_entropy = -(p_time * np.log(p_time + 1e-12)).sum()
    H_temp_max = math.log(K)
    temp_obs_entropy_norm = temp_obs_entropy / H_temp_max

    # 7) Temporal Observation Entropy (text)
    if total_text > 0 and len(all_text_times) > 0:
        all_text_times = np.concatenate(all_text_times)
        t_text_min = all_text_times.min().astype("datetime64[s]").astype(float)
        t_text_max = all_text_times.max().astype("datetime64[s]").astype(float)
        bins_text = np.linspace(t_text_min, t_text_max, K + 1)
        counts_text, _ = np.histogram(
            all_text_times.astype("datetime64[s]").astype(float), bins=bins_text
        )
        p_text_time = counts_text / counts_text.sum()
        temp_text_entropy = -(p_text_time * np.log(p_text_time + 1e-12)).sum()
        temp_text_entropy_norm = temp_text_entropy / H_temp_max
    else:
        temp_text_entropy_norm = None

    # 8) Mean IOI
    SEC_PER_UNIT = {
        "seconds": 1,
        "minutes": 60,
        "hours": 3600,
        "days": 86400,
        "weeks": 604800,
    }
    sec_per_unit = SEC_PER_UNIT[args.time_unit]
    mean_ioi = (all_dts / sec_per_unit).mean()

    # 9) Timespan
    start_str = pd.to_datetime(t_min, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    end_str = pd.to_datetime(t_max, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    timespan = f"{start_str}~{end_str}"

    # 10) Pretty print
    summary = {
        "num_entities": num_entities,
        "num_features": num_features,
        "num_unique_timestamps": num_unique_timestamps,
        "num_observations": int(total_obs),
        "Feat observability entropy (norm)": round(feat_obs_entropy_norm, 4),
        "Temporal observation entropy (norm)": round(temp_obs_entropy_norm, 4),
        "Mean IOI": f"{round(mean_ioi, 4)} {args.time_unit}",
        "timespan": timespan,
        "num_text": int(total_text),
        "Text temporal entropy (norm)": (
            round(temp_text_entropy_norm, 4)
            if temp_text_entropy_norm is not None
            else "N/A"
        ),
    }

    table = PrettyTable(["Metric", "Value"])
    for metric, value in summary.items():
        table.add_row([metric, value])
    print(table)


def parse_datasets(args, show_summary=True):
    """
    Load and split time-series data either by instance or by sliding-window samples.

    Parameters
    ----------
    args : argparse.Namespace
        Must have attributes:
          - data_root, dataset, device,
          - history, pred_window, stride, batch_size
          - (if patch_ts=True) patch_size, npatch, patch_stride
          - model, split_method, enable_text
    show_summary : bool
        If True, prints a summary of the dataset.
    """
    # — resolve and print data path —
    base = (
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.data_root))
        if not os.path.isabs(args.data_root)
        else args.data_root
    )
    dataset_path = os.path.join(base, args.dataset)
    print(f"Using dataset path: {dataset_path}")

    # — load & chunk everything in one go —
    ds = ChunkedTimeSeriesDataset(
        root=dataset_path,
        history=args.history,
        pred_window=args.pred_window,
        stride=args.stride,
        device=args.device,
        time_unit=args.time_unit,
        unit_scale=getattr(args, "unit_scale", None),
        normalize=True,  # always normalize
        enable_text=args.enable_text,
        use_text_embeddings=args.use_text_embeddings,
        llm_model_fusion=args.llm_model_fusion,
        llm_layers_fusion=args.llm_layers_fusion,
        max_length=args.max_length,
        args=args,
    )
    if show_summary:
        show_ds_summary(args)

    all_chunks = ds.chunks  # each is (chunk_id, tt, vals, mask, texts)
    if not all_chunks:
        raise ValueError("No chunks available! Check history/pred_window/stride.")

    # infer input_dim
    _, _, first_vals, _, _ = all_chunks[0]
    input_dim = first_vals.size(-1)

    # — split into train/val/test by index —
    if args.split_method == "instance":
        # keep all chunks of each record together
        rec_ids = sorted({cid.rsplit("_chunk", 1)[0] for cid, *_ in all_chunks})
        train_recs, test_recs = train_test_split(
            rec_ids, train_size=0.8, random_state=42, shuffle=True
        )
        train_recs, val_recs = train_test_split(
            train_recs, train_size=0.75, random_state=42, shuffle=False
        )

        train_idx = [
            i
            for i, (cid, *_) in enumerate(all_chunks)
            if cid.rsplit("_chunk", 1)[0] in train_recs
        ]
        val_idx = [
            i
            for i, (cid, *_) in enumerate(all_chunks)
            if cid.rsplit("_chunk", 1)[0] in val_recs
        ]
        test_idx = [
            i
            for i, (cid, *_) in enumerate(all_chunks)
            if cid.rsplit("_chunk", 1)[0] in test_recs
        ]

    elif args.split_method == "sample":
        # split within each record by temporal order
        grouped = defaultdict(list)
        for i, (cid, *_) in enumerate(all_chunks):
            rec_id, idx_str = cid.rsplit("_chunk", 1)
            grouped[rec_id].append((int(idx_str), i))

        train_idx, val_idx, test_idx = [], [], []
        for rec_id, lst in grouped.items():
            lst.sort(key=lambda x: x[0])
            N = len(lst)
            t_end = int(N * 0.6)
            v_end = int(N * 0.8)
            train_idx += [i for _, i in lst[:t_end]]
            val_idx += [i for _, i in lst[t_end:v_end]]
            test_idx += [i for _, i in lst[v_end:]]
    else:
        raise ValueError(f"Unknown split_method: {args.split_method!r}")

    print(
        f"After chunking & splitting ({args.split_method}): "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    # — choose collate fn and defaults for patching —
    if args.model == "tPatchGNN":
        collate_fn_to_use = patch_variable_time_collate_fn
        args.patch_size = args.patch_size or args.history // 5
        args.npatch = args.npatch or 5
        args.patch_stride = args.patch_stride or args.patch_size
        print(
            f"Using Patch Collate Fn: patch_size={args.patch_size}, "
            f"npatch={args.npatch}, patch_stride={args.patch_stride}"
        )
    elif args.model == "CRU":
        collate_fn_to_use = variable_time_collate_fn_CRU
        print("Using CRU Specific Collate Fn")
    elif args.model == "LatentODE":
        collate_fn_to_use = variable_time_collate_fn_ODE
        print("Using ODE Specific Collate Fn")
    else:
        collate_fn_to_use = variable_time_collate_fn
        print("Using Standard Collate Fn")

    tm = torch.tensor(
        args.history + args.pred_window, dtype=torch.float32, device=args.device
    )

    # — multimodal collate wrapper, now padding tau to a Tensor —
    def make_multimodal_collate_fn(base_collate, args, time_max):
        """
        Wraps an existing collate function to add either raw text or precomputed embeddings.

        Returns:
        - always adds 'tau': Tensor[B, N_max]
        - if args.enable_text and not args.use_text_embeddings:
            adds 'notes_text': List[List[str]]
        - if args.enable_text and args.use_text_embeddings:
            adds 'notes_embeddings': Tensor[B, N_max, d_txt]
        """
        from torch.nn.utils.rnn import pad_sequence

        def multimodal_collate(batch):
            # 1) get the base numeric batch
            numeric_batch = [item[:4] for item in batch]
            out = base_collate(numeric_batch, args, time_max)

            # 2) extract the fifth element: List[List[(t, payload)]]
            raws = [item[4] for item in batch]

            # 3) build tau (times) for every sample
            time_seqs = [
                torch.tensor(
                    [t for (t, _) in seq], dtype=torch.float32, device=args.device
                )
                for seq in raws
            ]
            tau = pad_sequence(time_seqs, batch_first=True, padding_value=0.0)
            out["tau"] = tau

            # 4) raw‐text branch
            if args.enable_text and not args.use_text_embeddings:
                out["notes_text"] = [[txt for (_, txt) in seq] for seq in raws]

            # 5) embeddings branch
            if args.enable_text and args.use_text_embeddings:
                # find d_txt from any non-empty seq
                d_txt = None
                for seq in raws:
                    if seq:
                        d_txt = seq[0][1].size(-1)
                        break
                if d_txt is None:
                    # no embeddings anywhere → zero‐size
                    emb_padded = torch.zeros((len(batch), 0, 0), device=args.device)
                else:
                    emb_seqs = []
                    for seq in raws:
                        if seq:
                            emb_seqs.append(torch.stack([e for (_, e) in seq], dim=0))
                        else:
                            emb_seqs.append(torch.zeros((0, d_txt), device=args.device))
                    emb_padded = pad_sequence(
                        emb_seqs, batch_first=True, padding_value=0.0
                    )
                out["notes_embeddings"] = emb_padded

            return out

        return multimodal_collate

    collate_fn = make_multimodal_collate_fn(collate_fn_to_use, args, tm)

    # — build DataLoaders via Subset —
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx) if test_idx else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = (
        DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )
        if test_ds
        else None
    )

    return {
        "train_dataloader": train_loader,
        "val_dataloader": val_loader,
        "test_dataloader": test_loader,
        "input_dim": input_dim,
        "time_max": tm,
        "ds": ds,
    }
