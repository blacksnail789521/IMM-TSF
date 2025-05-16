import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig

_ALIAS = {
    "GPT2": "openai-community/gpt2",  # context window size 1024, d_model 768
    "GPT2M": "openai-community/gpt2-medium",  # context window size 1024, d_model 1024
    "GPT2L": "openai-community/gpt2-large",  # context window size 1024, d_model 1280
    "GPT2XL": "openai-community/gpt2-xl",  # context window size 1024, d_model 1600
    "BERT": "google-bert/bert-base-uncased",  # context window size 512, d_model 768
    "Llama": "meta-llama/Llama-3.1-8B",  # context window size 128K, d_model 4096
    "DeepSeek": "deepseek-ai/deepseek-llm-7b-base",  # context window size 4096, d_model 4096
}


def get_d_model(
    llm_model_fusion: str,
):
    """
    Return the hidden size of the given LLM.

    Args:
      llm_model_fusion: key or model ID
    Returns:
      d_model (int)
    """
    # Load the model config
    model_id = _ALIAS.get(llm_model_fusion, llm_model_fusion)
    cfg = AutoConfig.from_pretrained(model_id)

    # Fetch hidden size from config
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    else:
        raise AttributeError("Cannot determine hidden size from model/config.")


def get_context_window_size(
    llm_model_fusion: str,
    device: str | torch.device = "cpu",
) -> int:
    """
    Return the maximum positional context window (number of tokens) for the given LLM.

    This will load the model (via load_llm), read its config, then free the model from memory.
    Args:
      llm_model_fusion: key or model ID
      device: device string
    Returns:
      context window size (int)
    """
    # Normalize device to torch.device
    if isinstance(device, str):
        device = torch.device(device)

    # Load the tokenizer and full model (no layer truncation)
    tokenizer, llm_model = load_llm(llm_model_fusion, None, device)
    cfg = llm_model.config

    # Fetch window size from config or embeddings
    if hasattr(cfg, "n_positions"):
        size = cfg.n_positions
    elif hasattr(cfg, "max_position_embeddings"):
        size = cfg.max_position_embeddings
    elif hasattr(llm_model, "wpe"):
        size = llm_model.wpe.weight.size(0)
    else:
        raise AttributeError("Cannot determine context window size from model/config.")

    # Clean up to free memory
    del llm_model
    del tokenizer
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return size


def load_llm(
    llm_model_fusion: str,
    llm_layers_fusion: int | None = None,
    device: str | torch.device = "cpu",
    use_device_map: bool = False,
):
    """
    Load a pretrained LLM and tokenizer, optionally truncating encoder layers.

    Args:
      llm_model_fusion: key or model ID for HuggingFace AutoModel
      llm_layers_fusion: number of encoder layers to keep, or None to keep all
      device: 'cpu' or 'cuda'

    Returns:
      tokenizer, llm_model
    """
    model_id = _ALIAS.get(llm_model_fusion, llm_model_fusion)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if use_device_map:
        print(f"Detected {torch.cuda.device_count()} GPUs. Using model parallelism.")
        llm_model = AutoModel.from_pretrained(model_id, device_map="auto")
    else:
        print(f"Loading model on {device}.")
        llm_model = AutoModel.from_pretrained(model_id).to(device)

    # Truncate encoder layers only if requested
    if llm_layers_fusion is not None:
        if hasattr(llm_model, "encoder") and hasattr(llm_model.encoder, "layer"):
            llm_model.encoder.layer = nn.ModuleList(
                llm_model.encoder.layer[:llm_layers_fusion]
            )

    # Freeze all parameters
    for param in llm_model.parameters():
        param.requires_grad = False

    # Extend token embeddings to include pad_token_id
    llm_model.resize_token_embeddings(len(tokenizer))

    print(
        f"Loaded {llm_model_fusion} with "
        f"{'full' if llm_layers_fusion is None else llm_layers_fusion} layers for fusion."
    )
    return tokenizer, llm_model


def embed_notes(
    notes_text: list[list[str]],
    tokenizer,
    llm_model,
    max_length: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Turn a batch of variable-length lists of text notes into embeddings.

    Args:
      notes_text: List of B samples, each a list of N_i strings.
      tokenizer : a HuggingFace AutoTokenizer (with pad_token set).
      llm_model : a HuggingFace AutoModel (outputs last_hidden_state).
      max_length: number of tokens per note.

    Returns:
      note_embeddings: Tensor [B, N_max, hidden_size]
      note_mask      : BoolTensor [B, N_max], True for real notes.
    """
    B = len(notes_text)
    lengths = [len(x) for x in notes_text]
    N_max = max(lengths) if lengths else 0

    # 1) pad each sample's note-list up to N_max with empty strings
    padded = [sample + [""] * (N_max - len(sample)) for sample in notes_text]

    # 2) flatten into B * N_max strings
    flat_notes = [note for sample in padded for note in sample]

    # 3) tokenize all notes in one batch
    enc = tokenizer(
        flat_notes,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    # reshape into [B, N_max, L_tok]
    notes_input_ids = enc.input_ids.view(B, N_max, max_length)
    attention_mask = enc.attention_mask.view(B, N_max, max_length)

    # 4) build note_mask [B, N_max]
    note_mask = torch.tensor(
        [[True] * l + [False] * (N_max - l) for l in lengths],
        dtype=torch.bool,
        device=next(llm_model.parameters()).device,
    )

    # 5) flatten again to feed the model
    flat_ids = notes_input_ids.view(B * N_max, max_length)
    flat_attn = attention_mask.view(B * N_max, max_length)

    # Move inputs to the same device as the model
    device = next(llm_model.parameters()).device
    flat_ids = flat_ids.to(device)
    flat_attn = flat_attn.to(device)

    # 6) forward pass
    outputs = llm_model(input_ids=flat_ids, attention_mask=flat_attn)
    hidden_states = outputs.last_hidden_state  # [B*N_max, L_tok, hidden_size]

    # 7) masked mean-pooling over tokens
    mask_expanded = flat_attn.unsqueeze(-1)  # [B*N_max, L_tok, 1]
    sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
    lengths_tok = mask_expanded.sum(dim=1).clamp(min=1)
    pooled = sum_hidden / lengths_tok  # [B*N_max, hidden_size]

    # 8) un-flatten to [B, N_max, hidden_size]
    hidden_size = pooled.size(-1)
    note_embeddings = pooled.view(B, N_max, hidden_size)

    return note_embeddings, note_mask


if __name__ == "__main__":
    # Example usage
    llm_model_fusion = "GPT2"
    llm_layers_fusion = 6
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, llm_model = load_llm(llm_model_fusion, llm_layers_fusion, device)

    # Example real-world notes:
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

    embs, mask = embed_notes(notes_text, tokenizer, llm_model, max_length=1024)
    print(embs.shape)  # -> [3, 5, d_txt]
    print(mask)  # -> tensor([[ True,  True,  True,  True,  True],
    #                               [False, False, False, False, False],
    #                               [ True,  True,  True,  True,  True]])

    print(embs.device)  # -> cuda:0
    print(mask.device)  # -> cuda:0

    context_window_size = get_context_window_size(llm_model_fusion, device)
    print(f"Context window size: {context_window_size}")  # -> 1024
