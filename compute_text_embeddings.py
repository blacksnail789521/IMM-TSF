import os
import torch
import pandas as pd
from tqdm import tqdm
from fusions.load_llm import load_llm, embed_notes, get_context_window_size


def compute_text_embeddings(
    data_name: str,
    llm_model_fusion: str,
    llm_layers_fusion: int | None,
    max_length: int = 1024,
    device: str = "cpu",
) -> None:
    """
    Loop over all records in base_dir, read each text.csv, embed notes one at a time,
    and save text_embeddings_{llm_model_fusion}_{llm_layers_fusion or 'full'}.pt

    Args:
      data_name: name of the dataset (e.g. 'ILINet', 'FNSPID')
      llm_model_fusion: key or model ID (e.g. 'GPT2')
      llm_layers_fusion: number of layers to keep, or None for all
      max_length: maximum length of input tokens
      device: 'cpu' or 'cuda'
    """
    base_dir = f"data/{data_name}/processed"
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Load LLM once
    tokenizer, llm_model = load_llm(
        llm_model_fusion,
        llm_layers_fusion,
        device,
        # use_device_map=False,
        use_device_map=True,
    )

    # Discover all record subfolders
    record_ids = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )
    if not record_ids:
        raise RuntimeError(f"No record subfolders under {base_dir}")

    # Iterate records with progress bar
    for idx, rec in enumerate(record_ids):
        print(f"[{idx + 1}/{len(record_ids)}] Processing record: {rec}")
        rec_dir = os.path.join(base_dir, rec)
        text_csv = os.path.join(rec_dir, "text.csv")
        if not os.path.isfile(text_csv):
            tqdm.write(f"[SKIP] no text.csv in {rec_dir}")
            continue

        # Prepare output path
        out_name = (
            f"text_embeddings_model={llm_model_fusion}"
            f"_layers={llm_layers_fusion or 'full'}"
            f"_maxlen={max_length}.pt"
        )
        out_path = os.path.join(rec_dir, out_name)

        # Skip if output already exists
        if os.path.isfile(out_path):
            tqdm.write(f"[SKIP] Embeddings already exist for '{rec}', skipping.")
            continue

        tqdm.write(f"Embedding notes for record '{rec}'...")
        df = pd.read_csv(text_csv, parse_dates=["date_time"])
        base_ts = df["date_time"].min()
        rel_times = ((df["date_time"] - base_ts).dt.total_seconds() / 86400.0).tolist()
        text_cols = [c for c in df.columns if c not in ("date_time", "record_id")]
        if len(text_cols) != 1:
            raise ValueError(f"{rec_dir}: expected 1 text col, got {text_cols}")
        notes = df[text_cols[0]].astype(str).tolist()

        # Embed each note one by one to save memory
        embeddings = []
        for note in tqdm(notes, desc=f"Notes/{rec}", leave=False, unit="note"):
            emb, _ = embed_notes([[note]], tokenizer, llm_model, max_length=max_length)
            embeddings.append(emb.squeeze(0).squeeze(0).cpu())
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        # Stack into Tensor [N_notes, d_txt]
        if embeddings:
            emb_tensor = torch.stack(embeddings, dim=0)
        else:
            emb_tensor = torch.empty((0, 0), dtype=torch.float32)

        # Save embeddings and rel_times
        torch.save(
            {
                "embeddings": emb_tensor,
                "rel_times": torch.tensor(rel_times, dtype=torch.float32),
            },
            out_path,
        )
        tqdm.write(f"Wrote embeddings to {out_path}")


if __name__ == "__main__":
    # # * GPU settings
    # gpu_id = 0
    # # gpu_id = 1
    # # gpu_id = 2
    # # gpu_id = 3
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # * Parameters
    data_name_list = [
        "GDELT",  # type 1.1
        "RepoHealth",  # type 1.2
        "MIMIC",  # type 1.3
        "FNSPID",  # type 2.1
        # "ClusterTrace",  # type 2.2
        "StudentLife",  # type 2.3
        "ILINet",  # type 3.1
        "CESNET",  # type 3.2
        "EPA-Air",  # type 3.3
    ]

    llm_model_fusion = "GPT2"
    # llm_model_fusion = "GPT2XL"
    # llm_model_fusion = "BERT"
    # llm_model_fusion = "Llama"
    # llm_model_fusion = "DeepSeek"
    # llm_layers_fusion = 6
    llm_layers_fusion = None
    max_length = 512 if llm_model_fusion == "BERT" else 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"### LLM model: {llm_model_fusion} ###")

    # * Update max_length if needed
    # context_window_size = get_context_window_size(llm_model_fusion, device)
    # if max_length > context_window_size:
    #     print(
    #         f"Overriding max_length from {max_length} to {context_window_size}"
    #         " to match the LLM model's context window size."
    #     )
    #     max_length = context_window_size

    for data_name in data_name_list:
        print(f"### Processing dataset: {data_name} ###")
        compute_text_embeddings(
            data_name, llm_model_fusion, llm_layers_fusion, max_length, device
        )
