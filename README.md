# Transformer Decision Traces

Structured extraction of transformer **decision traces** (attention + Q/K/V + optional MLP) with clean, per-example access for downstream analysis.

## Why this exists
- Make internal signals **queryable**: given an input, quickly slice heads/layers/tokens.
- Keep artifacts **portable**: Zarr (chunked, compressed) + Parquet metadata.
- Be **reproducible**: every run has `meta.json` and stable `example_id`s.

---

## Install

```bash
# create and activate your venv (example)
python -m venv .venv
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt

## GET THE DATA

# UD English–EWT + GoEmotions
python -m src.data.prepare_datasets --which all
# or one at a time:
# python -m src.data.prepare_datasets --which ud
# python -m src.data.prepare_datasets --which go
## OUTPUT : 
data/processed/
  ├─ ud_ewt.parquet
  ├─ go_emotions.parquet
  └─ go_emotions.labels.json

# EXTRACT TRACES : 

# Example: RoBERTa-base, UD validation, 32 examples, 128 tokens
python -m src.cli.extract_traces --model roberta-base --dataset ud_ewt --split validation --limit 32 --max_seq_len 128 --capture attn qkv

# out : 
traces/<RUN_ID>/
  attn.zarr/attn          # (N,L,H,T,T) float16
  qkv.zarr/q,k,v          # (N,L,H,T,d_head) float16
  tokens.parquet          # text + tokenizer encodings
  meta.json               # run config & shapes

#Programmatic access (TraceStore)

from src.traces.store import TraceStore

run_dir = r"traces/<RUN_ID>"
store = TraceStore(run_dir)

# Inspect
print(store.meta)         # model/dataset/shapes
print(store.arrays())     # which arrays exist and shapes

# Pick an example
eid = store.tokens.iloc[0]["example_id"]
enc = store.encodings(eid)             # input_ids, attention_mask, offset_mapping, tokens
A = store.attn(eid, layer=0, head=0)   # (T, T)
Q = store.qkv(eid, "q", layer=0, head=0)   # (T, d_head)
K = store.qkv(eid, "k", layer=0, head=0)
V = store.qkv(eid, "v", layer=0, head=0)
