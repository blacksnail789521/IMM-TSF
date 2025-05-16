import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from transformers import (
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
)
from layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class ReprogrammingLayer(nn.Module):
    def __init__(
        self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1
    ):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, Q, K_src, V_src):
        Bm, Lq, _ = Q.shape
        Vs, _ = K_src.shape
        H = self.n_heads
        Ek = K_src.size(-1) // H

        q = self.query_projection(Q).view(Bm, Lq, H, -1)
        k = self.key_projection(K_src).view(Vs, H, -1)
        v = self.value_projection(V_src).view(Vs, H, -1)

        scale = 1.0 / sqrt(Ek)
        scores = torch.einsum("blhe,she->bhls", q, k)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,she->blhe", A, v)

        out = out.reshape(Bm, Lq, -1)
        return self.out_projection(out)


class TimeLLM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # lengths & dims
        self.input_len = configs.input_len
        self.seq_len = configs.input_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.d_ff = configs.d_ff
        self.num_tokens = configs.ts_vocab_size
        self.patch_len = configs.input_token_len
        self.stride = configs.stride
        self.domain_des = configs.domain_des
        self.top_k = configs.top_k
        self.C = configs.C

        # pick LLM hidden size
        if configs.llm_model_timellm == "LLAMA":
            self.d_llm = 4096
        elif configs.llm_model_timellm in ("GPT2", "BERT"):
            self.d_llm = 768
        else:
            raise ValueError("Unknown llm_model for TimeLLM")

        # compute patch count, clamp ≥1
        raw_p = (self.seq_len - self.patch_len) // self.stride + 2
        self.patch_nums = max(1, raw_p)
        self.head_nf = self.d_ff * self.patch_nums

        # load & freeze LLM
        self._get_model_and_tokenizer(
            configs.llm_model_timellm, configs.llm_layers_timellm
        )
        self._get_llm_pad_token()
        for p in self.llm_model.parameters():
            p.requires_grad = False

        self.dropout = nn.Dropout(configs.dropout)

        # the same patch‐embedder for values & time
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, self.stride, configs.dropout
        )

        # mapping & reprogramming
        self.word_embeddings = self.llm_model.get_input_embeddings().weight  # [V,H]
        self.mapping_layer = nn.Linear(self.word_embeddings.size(0), self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(
            configs.d_model, configs.n_heads, d_llm=self.d_llm
        )

        # final head
        self.output_projection = FlattenHead(
            self.head_nf, self.pred_len, head_dropout=configs.dropout
        )

        # Padding buffer for irregular sequences
        self.zeros_pad = torch.zeros(
            configs.batch_size,
            max(self.input_len, self.pred_len),
            self.C,
            device=configs.device,
        )

    def _get_model_and_tokenizer(self, model_name, layers):
        if model_name == "LLAMA":
            cfg = LlamaConfig.from_pretrained("huggyllama/llama-7b")
            cfg.num_hidden_layers = layers
            cfg.output_hidden_states = True
            cfg.output_attentions = True
            self.llm_model = LlamaModel.from_pretrained(
                "huggyllama/llama-7b", config=cfg
            )
            self.tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
        elif model_name == "GPT2":
            cfg = GPT2Config.from_pretrained("openai-community/gpt2")
            cfg.num_hidden_layers = layers
            cfg.output_hidden_states = True
            cfg.output_attentions = True
            self.llm_model = GPT2Model.from_pretrained(
                "openai-community/gpt2", config=cfg
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        elif model_name == "BERT":
            cfg = BertConfig.from_pretrained("google-bert/bert-base-uncased")
            cfg.num_hidden_layers = layers
            cfg.output_hidden_states = True
            cfg.output_attentions = True
            self.llm_model = BertModel.from_pretrained(
                "google-bert/bert-base-uncased", config=cfg
            )
            self.tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-uncased"
            )
        else:
            raise ValueError("Unsupported LLM")

    def _get_llm_pad_token(self):
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token = "[PAD]"

    def _get_prompt(self, x_enc: torch.Tensor) -> list[str]:
        B, L, N = x_enc.shape
        mins = x_enc.min(dim=1)[0]
        maxs = x_enc.max(dim=1)[0]
        meds = x_enc.median(dim=1).values
        trend = x_enc.diff(dim=1).sum(dim=1).mean(dim=1)
        FFT = torch.fft.rfft(x_enc.permute(0, 2, 1), dim=-1)
        corr = torch.fft.irfft(FFT * FFT.conj(), n=L, dim=-1).mean(dim=1)
        _, lags = corr.topk(min(self.top_k, L), dim=-1)
        if lags.size(1) < self.top_k:
            pad = lags[:, -1, None].repeat(1, self.top_k - lags.size(1))
            lags = torch.cat([lags, pad], dim=-1)

        prompts = []
        for b in range(B):
            tr = "upward" if trend[b].item() > 0 else "downward"
            prompts.append(
                f"<|start_prompt|>"
                f"Dataset: {self.domain_des}. "
                f"Forecast next {self.pred_len} from past {self.input_len}. "
                f"Min {mins[b].tolist()}, "
                f"Max {maxs[b].tolist()}, "
                f"Median {meds[b].tolist()}, "
                f"Trend {tr}, "
                f"Top lags {lags[b].tolist()}."
                f"<|end_prompt|>"
            )
        return prompts

    def forecasting(
        self,
        tp_to_predict: torch.Tensor,
        observed_data: torch.Tensor,
        observed_tp: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L, N = observed_data.shape

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

        # 2) pad future times just to get Lp
        Lp = tp_to_predict.size(1)
        if Lp < self.pred_len:
            tp_to_predict = F.pad(tp_to_predict, (0, self.pred_len - Lp))

        # 3) normalize
        x = observed_data * observed_mask
        sums = observed_mask.sum(1).clamp(min=1)
        means = x.sum(1) / sums
        x = x - means.unsqueeze(1)
        var = ((x * observed_mask) ** 2).sum(1) / sums
        stdev = torch.sqrt(var + 1e-5)
        x = x / stdev.unsqueeze(1)

        # 4) prompt
        prompts = self._get_prompt(x)
        tokens = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).input_ids.to(x.device)
        prompt_embeds = self.llm_model.get_input_embeddings()(tokens)  # [B,P,d_llm]

        # 5) patch‐embed the **values**
        x_ts = x.permute(0, 2, 1)  # [B,N,L]
        if x_ts.size(-1) < self.patch_len:
            pad = self.patch_len - x_ts.size(-1)
            x_ts = F.pad(x_ts, (0, pad))
        ts_out, n_vars = self.patch_embedding(x_ts)  # [B*N, Pn, d_ff]

        # 6) patch‐embed the **timestamps**  ← keep observed_tp
        x_tp = observed_tp.unsqueeze(1).repeat(1, N, 1)  # [B,N,L]
        if x_tp.size(-1) < self.patch_len:
            pad = self.patch_len - x_tp.size(-1)
            x_tp = F.pad(x_tp, (0, pad))
        tp_out, _ = self.patch_embedding(x_tp)  # [B*N, Pn, d_ff]

        # 7) combine them
        rep_in = ts_out + tp_out  # [B*N, Pn, d_ff]

        # 8) reprogram
        src_emb = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        rep_out = self.reprogramming_layer(rep_in, src_emb, src_emb)  # [B*N, Pn, d_llm]

        # 9) run through LLM
        rep_out = rep_out.view(B, N, self.patch_nums, self.d_llm)
        rep_out = rep_out.permute(0, 2, 1, 3).reshape(B, -1, self.d_llm)
        llama_inp = torch.cat([prompt_embeds, rep_out], dim=1)
        llama_out = self.llm_model(inputs_embeds=llama_inp).last_hidden_state

        # 10) extract & project
        total_ts = self.patch_nums * n_vars
        dec = llama_out[:, -total_ts:, : self.d_ff]
        dec = dec.view(B, self.patch_nums, n_vars, self.d_ff)
        dec = dec.permute(0, 2, 3, 1).reshape(B * n_vars, self.d_ff, self.patch_nums)
        out_seq = self.output_projection(dec)  # [B*N, pred_len]

        # 11) reshape back & de‐normalize
        out_seq = out_seq.view(B, n_vars, self.pred_len).permute(0, 2, 1)
        if self.use_norm:
            out_seq = out_seq * stdev.unsqueeze(1) + means.unsqueeze(1)

        # 12) slice to true Lp
        return out_seq[:, :Lp, :]
