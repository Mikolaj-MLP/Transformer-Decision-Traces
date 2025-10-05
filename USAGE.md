# Transformer-Decision-Traces — Usage Guide

This repo extracts **structured decision traces** from Transformer models on text inputs and lets you inspect them offline.

It can capture:
- **Attention** matrices (per layer/head)
- **Q/K/V** vectors (per layer/head)
- **Hidden states** (optional)

Supports:
- **Encoders** (e.g., `roberta-base`)
- **Decoders** (e.g., `gpt2`) — causal self-attn
- **Encoder–decoders** (e.g., `t5-small`) — encoder self, decoder self, and cross-attn (when targets provided)

Outputs are chunked **Zarr** arrays with a **tokens.parquet** and **meta.json** per run. A small `TraceStore` API gives random access to any example.

---

## Quick start

```powershell
# (Optional) activate your venv
# .\venvs\Masters_env\Scripts\activate

# 1) Prepare datasets (UD English-EWT, GoEmotions)
python -m src.data.prepare_datasets

# 2) Run an encoder (RoBERTa) trace extraction
python -m src.cli.extract_traces --model roberta-base --dataset ud_ewt --split validation --limit 24 --max_seq_len 128 --capture attn qkv hidden --layers-enc "0,6,11" --heads-enc "0,3,7"

# 3) Inspect latest run in a notebook
# (See "Python API" below)


  Data preparation
Two public English datasets are supported:

# UD English-EWT
python -m src.data.prepare_datasets --which ud

# GoEmotions
python -m src.data.prepare_datasets --which go

# Both
python -m src.data.prepare_datasets


Artifacts land in data/processed/:

ud_ewt.parquet

go_emotions.parquet


Extract traces (CLI)

python -m src.cli.extract_traces `
  --model <hf_model_name> `
  --dataset {ud_ewt|go_emotions} --split {train|validation|test} `
  --limit <N> --max_seq_len <T_enc> `
  --capture <one-or-more of: attn qkv hidden> `
  [--layers-enc "i,j,k"] [--heads-enc "i,j,k"] `
  [--layers-dec "i,j,k"] [--heads-dec "i,j,k"] `
  [--dec_max_len <T_dec>] [--targets_file <path>] `
  [--out_dir <traces/custom_name>]


Encoder-only (RoBERTa)

python -m src.cli.extract_traces --model roberta-base `
  --dataset ud_ewt --split validation `
  --limit 24 --max_seq_len 128 `
  --capture attn qkv hidden `
  --layers-enc "0,6,11" --heads-enc "0,3,7"

Decoder-only (GPT-2)

python -m src.cli.extract_traces --model gpt2 `
  --dataset ud_ewt --split validation `
  --limit 16 --max_seq_len 128 `
  --capture attn qkv `
  --layers-dec "0,5,11" --heads-dec "0,1,7"

Notes:

set a pad token for GPT-2 automatically (uses eos).

Expect causal (lower-triangular) attention.

Encoder–decoder (T5)
Encoder side only (no targets)

python -m src.cli.extract_traces --model t5-small `
  --dataset ud_ewt --split validation `
  --limit 12 --max_seq_len 128 `
  --capture attn qkv hidden `
  --layers-enc "0,3,5" --heads-enc "0,2,6"


If you omit --targets_file, we inject a 1-token dummy decoder_input_ids so the encoder runs; only encoder arrays are saved.

Encoder + Decoder Self + Cross (with targets)

python -c "import pandas as pd; df=pd.read_parquet('data/processed/ud_ewt.parquet'); val=df[df.split=='validation'].head(12); open('data/targets_dummy.txt','w',encoding='utf-8').write('\n'.join(val.text.tolist()))"

python -m src.cli.extract_traces --model t5-small `
  --dataset ud_ewt --split validation `
  --limit 12 --max_seq_len 128 --dec_max_len 64 `
  --targets_file data/targets_dummy.txt `
  --capture attn qkv hidden `
  --layers-enc '0,3,5' --heads-enc '0,2,6' `
  --layers-dec '0,3,5' --heads-dec '0,2,6'


Index validation: If you pass out-of-range layers/heads (e.g., 11 for t5-small which has 6 layers), the CLI drops them and prints:
[warn] dropping out-of-range layers-enc [11]; valid range = 0..5


Output layout
Each run writes to traces/<RUN_ID>/ (atomic: _tmp_<RUN_ID> → <RUN_ID>):
traces/<RUN_ID>/
  meta.json
  tokens.parquet
  # encoder-only
  attn.zarr/attn
  qkv.zarr/{q,k,v}
  hidden.zarr/h

  # decoder-only
  dec_self_attn.zarr/attn
  dec_self_qkv.zarr/{q,k,v}
  dec_hidden.zarr/h

  # encoder–decoder
  enc_self_attn.zarr/attn
  enc_self_qkv.zarr/{q,k,v}
  enc_hidden.zarr/h
  # (when targets provided)
  dec_self_attn.zarr/attn
  dec_self_qkv.zarr/{q,k,v}
  dec_cross_attn.zarr/attn
  dec_cross_qkv.zarr/{q,k,v}
  dec_hidden.zarr/h

attn chunks: (1,1,1,T,T)

q/k/v chunks: (1,1,1,T,d_head)

Default dtype: fp16 (use --dtype float32 to change)


Python API (TraceStore)

from pathlib import Path
from src.traces.store import TraceStore
import numpy as np

# Load latest run
run_dir = str(sorted(Path("traces").iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[0])
st = TraceStore(run_dir)

print(st.arrays())      # dict: name -> shape
print(st.meta)          # run metadata
print(st.tokens.head()) # example_id, text, ids, masks, tokens

eid = st.tokens.iloc[0]["example_id"]

# Encoder
A_enc = st.attn(eid, side="enc", layer=0, head=0)           # (T_enc, T_enc)
Q_enc = st.qkv(eid, "q", side="enc", layer=0, head=0)       # (T_enc, d_head)
H_enc = st.hidden(eid, side="enc")                          # (L+1, T_enc, D) if captured

# Decoder (GPT-2)
A_dec = st.attn(eid, side="dec", layer=0, head=0)           # (T_dec, T_dec)
K_dec = st.qkv(eid, "k", side="dec", layer=0, head=0)       # (T_dec, d_head)

# Encoder–decoder (T5)
A_es  = st.attn(eid, side="enc", kind="self",  layer=0, head=0)  # (T_enc, T_enc)
A_ds  = st.attn(eid, side="dec", kind="self",  layer=0, head=0)  # (T_dec, T_dec) if targets
A_x   = st.attn(eid, side="dec", kind="cross", layer=0, head=0)  # (T_dec, T_enc) if targets
Q_x   = st.qkv(eid, "q", side="dec", kind="cross", layer=0, head=0)
K_x   = st.qkv(eid, "k", side="dec", kind="cross", layer=0, head=0)

# Trim pads & sanity
enc = st.encodings(eid)
mask = np.array(enc["attention_mask"], bool)
L = int(mask.sum())
print("row-sum≈1 (enc, non-pad):", float(A_enc[:L,:L].sum(-1).mean()))



Common commands (copy/paste)

# RoBERTa, 32 samples, all layers/heads
python -m src.cli.extract_traces --model roberta-base --dataset ud_ewt --split validation --limit 32 --max_seq_len 128 --capture attn qkv

# GPT-2, 16 samples, selected layers/heads
python -m src.cli.extract_traces --model gpt2 --dataset ud_ewt --split validation --limit 16 --max_seq_len 128 --capture attn qkv --layers-dec "0,5,11" --heads-dec "0,1,7"

# T5 encoder-only, 12 samples (invalid indices auto-dropped)
python -m src.cli.extract_traces --model t5-small --dataset ud_ewt --split validation --limit 12 --max_seq_len 128 --capture attn qkv hidden --layers-enc "0,5,11" --heads-enc "0,2,6"

# T5 with decoder & cross (needs targets)
python -c "import pandas as pd; df=pd.read_parquet('data/processed/ud_ewt.parquet'); val=df[df.split=='validation'].head(12); open('data/targets_dummy.txt','w',encoding='utf-8').write('\n'.join(val.text.tolist()))"
python -m src.cli.extract_traces --model t5-small --dataset ud_ewt --split validation --limit 12 --max_seq_len 128 --dec_max_len 64 --targets_file data/targets_dummy.txt --capture attn qkv hidden --layers-enc '0,3,5' --heads-enc '0,2,6' --layers-dec '0,3,5' --heads-dec '0,2,6'


Troubleshooting
Windows symlink warning (HF hub): safe to ignore; uses copies if symlinks disabled.

T5 index out of bounds: we now drop invalid layer/head indices and warn.

T5 requires decoder_input_ids: handled (dummy 1-token when no --targets_file).

GPT-2 pad token missing: handled (we set pad_token = eos_token).




Roadmap

--capture mlp persistence (hooks exist; on-disk writing will be added on demand).

MNLI baseline + eval CLI to tag examples correct/incorrect for error-trace analyses.


::contentReference[oaicite:0]{index=0}


