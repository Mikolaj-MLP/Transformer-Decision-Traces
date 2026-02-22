# Transformer Decision Traces - Usage

This repository is for extracting and analyzing **internal decision traces** from Transformer models.

Main outputs are saved per run in `traces/<RUN_ID>/`:
- `meta.json` (run metadata)
- `tokens.parquet` (text + tokenization + predictions/labels)
- one or more Zarr arrays (`*.zarr`) for attention, Q/K/V, hidden states, and residual streams

Default output is always inside this repo under `traces/`.

## What Works Here

### 1) Decoder traces on CSQA prompts (GPT-2 style)
Script: `src/cli/extract_trace_csqa_gpt.py`

Captures decoder self-attention/QKV/hidden/residual traces for CSQA-formatted prompts.

### 2) Decoder next-token traces (label-free LM analysis)
Script: `src/cli/extract_trace_nexttok_dec.py`

Datasets supported:
- `ud_ewt`
- `go_emotions`
- `csqa`
- `wikitext2`
- `wikitext103`

Adds next-token diagnostics to `tokens.parquet`:
- `next_pos`, `next_true_id`, `next_pred_id`, `next_correct`
- `next_true_prob`, `next_pred_prob`, `next_entropy`
- `next_topk_ids`, `next_topk_tokens`, `next_topk_probs`

### 3) Encoder multiple-choice traces on CSQA
Script: `src/cli/extract_trace_csqa_enc.py`

Captures per-choice encoder traces for multiple-choice models (`AutoModelForMultipleChoice`), with:
- `label_idx`, `pred_idx`, `pred_label`, `is_correct`
- `choice_logits`, `choice_probs`

### 4) Fine-tune encoder model on CSQA
Script: `src/cli/finetune_csqa_enc.py`

Trains an encoder MCQ model and saves a checkpoint for trace extraction.

## Typical Workflows

### A) Decoder next-token workflow
1. Extract traces:

```powershell
python -m src.cli.extract_trace_nexttok_dec `
  --model gpt2 `
  --dataset wikitext2 `
  --split validation `
  --limit 500 `
  --max_seq_len 256 `
  --capture hidden resid `
  --batch_size 8 `
  --out_dir traces\wt2_val500_gpt2
```

2. Validate run:

```powershell
python -m src.cli.index_traces traces\wt2_val500_gpt2 --validate
```

3. Analyze in notebook:
- `nexttok_trace_analysis.ipynb`

### B) Decoder CSQA prompt workflow

```powershell
python -m src.cli.extract_trace_csqa_gpt `
  --split validation `
  --limit 500 `
  --batch_size 8 `
  --max_seq_len auto `
  --capture attn qkv hidden resid `
  --out_dir traces\csqa_dec500
```

### C) Encoder CSQA classification workflow
1. Fine-tune:

```powershell
python -m src.cli.finetune_csqa_enc `
  --model roberta-base `
  --max_seq_len 128 `
  --batch_size 8 `
  --lr 2e-5 `
  --epochs 2 `
  --gradient_checkpointing
```

2. Extract traces from trained checkpoint:

```powershell
python -m src.cli.extract_trace_csqa_enc `
  --model roberta-base `
  --model_path checkpoints\YOUR_RUN_DIR `
  --split validation `
  --limit 250 `
  --batch_size 4 `
  --max_seq_len 128 `
  --capture hidden resid `
  --out_dir traces\csqa_enc_val250_ft
```

3. Validate run:

```powershell
python -m src.cli.index_traces traces\csqa_enc_val250_ft --validate
```

## Output Shapes (Quick Reference)

### Decoder traces
- `dec_self_attn.zarr/attn`: `(N, L, H, T, T)`
- `dec_self_qkv.zarr/{q,k,v}`: `(N, L, H, T, d_head)`
- `dec_hidden.zarr/h`: `(N, L+1, T, D)`
- `dec_res_embed.zarr/x`: `(N, T, D)`
- `dec_res_pre_attn.zarr/x`: `(N, L, T, D)`
- `dec_res_post_attn.zarr/x`: `(N, L, T, D)`
- `dec_res_post_mlp.zarr/x`: `(N, L, T, D)`

### Encoder MCQ traces
- `enc_mc_attn.zarr/attn`: `(N, C, L, H, T, T)`
- `enc_mc_qkv.zarr/{q,k,v}`: `(N, C, L, H, T, d_head)`
- `enc_mc_hidden.zarr/h`: `(N, C, L+1, T, D)`
- `enc_mc_res_embed.zarr/x`: `(N, C, T, D)`
- `enc_mc_res_pre_attn.zarr/x`: `(N, C, L, T, D)`
- `enc_mc_res_post_attn.zarr/x`: `(N, C, L, T, D)`
- `enc_mc_res_post_mlp.zarr/x`: `(N, C, L, T, D)`

## Python API (TraceStore)

```python
from pathlib import Path
from src.traces_utils.store import TraceStore

run_dir = str(sorted(Path("traces").iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[0])
st = TraceStore(run_dir)

print(st.meta)
print(st.arrays())
print(st.tokens.head())

eid = st.tokens.iloc[0]["example_id"]
```

### Decoder accessors

```python
A = st.attn(eid, side="dec", kind="self", layer=0, head=0)       # (T, T)
Q = st.qkv(eid, which="q", side="dec", kind="self", layer=0, head=0)  # (T, d)
H = st.hidden(eid, side="dec")                                    # (L+1, T, D)
R = st.resid(eid, stage="post_mlp", side="dec", layer=0)          # (T, D)
```

### Encoder MCQ accessors

```python
choice_idx = int(st.tokens.iloc[0]["pred_idx"])

A_mc = st.mcq_attn(eid, choice_idx=choice_idx, layer=0, head=0)          # (T, T)
Q_mc = st.mcq_qkv(eid, choice_idx=choice_idx, which="q", layer=0, head=0) # (T, d)
H_mc = st.mcq_hidden(eid, choice_idx=choice_idx)                           # (L+1, T, D)
R_mc = st.mcq_resid(eid, choice_idx=choice_idx, stage="post_mlp", layer=0) # (T, D)
```