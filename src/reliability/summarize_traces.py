# src/reliability/summarize_traces.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.traces.store import TraceStore

# --- helpers (arch-aware) ---

def attn_entropy(A, eps=1e-12):
    rs = A.sum(-1, keepdims=True)
    P = np.divide(A, rs, out=np.zeros_like(A, dtype=np.float64), where=rs>0)
    with np.errstate(divide='ignore', invalid='ignore'):
        logP = np.zeros_like(P, dtype=np.float64)
        m = P > 0
        logP[m] = np.log(P[m])
    return -(P*logP).sum(-1)  # (T,)

def diag_mass(A, w=1):
    M = np.zeros_like(A, dtype=bool)
    for i in range(A.shape[0]):
        j0, j1 = max(0, i-w), min(A.shape[1], i+w+1)
        M[i, j0:j1] = True
    return A[M].sum() / (A.sum() + 1e-12)

def _pick(enc, keys):
    for k in keys:
        if k in enc and enc[k] is not None:
            return enc[k]
    return None

def seq_mask(enc, arch, side="enc"):
    """Return boolean mask of non-pad tokens for the requested stream."""
    if arch == "enc":
        m = _pick(enc, ["attention_mask"])
    elif arch == "dec":
        m = _pick(enc, ["dec_attention_mask", "attention_mask"])
    elif arch == "encdec":
        if side == "enc":
            m = _pick(enc, ["attention_mask"])
        else:
            m = _pick(enc, ["dec_attention_mask", "attention_mask"])
    else:
        m = _pick(enc, ["attention_mask"])  # safe default
    return np.asarray(m, dtype=bool)

def per_example_features(st: TraceStore, eid: str, take_pca=8):
    """Compute compact, per-example features from attn/QKV. Arch-aware."""
    enc = st.encodings(eid)
    arch = st.meta.get("arch", "enc")  # 'enc'|'dec'|'encdec'
    if arch == "enc":
        side_for_self = "enc"
    elif arch == "dec":
        side_for_self = "dec"
    else:  # encdec
        side_for_self = "enc"  # use encoder self-attn for features in first pass

    # indices of non-pad tokens for the chosen stream
    mask = seq_mask(enc, arch, side=side_for_self)
    idx = np.flatnonzero(mask)

    # Effective L,H from meta (falls back to shapes if needed)
    L = st.meta.get("num_layers")
    H = st.meta.get("num_heads")
    if L is None or H is None:
        arrs = st.arrays()
        key = "attn"
        if arch == "dec":
            key = "dec_self_attn" if "dec_self_attn" in arrs else key
        elif arch == "enc":
            key = "enc_self_attn" if "enc_self_attn" in arrs else key
        if key in arrs:
            _, L, H, _, _ = arrs[key]

    # Attention stats per layer (avg over heads)
    attn_mean_entropy, attn_mean_diag = [], []
    for Lloc in range(L):
        ent_h, dm_h = [], []
        for Hloc in range(H):
            A = st.attn(eid, side=side_for_self, kind="self",
                        layer=Lloc, head=Hloc)
            A = A[np.ix_(idx, idx)]
            ent_h.append(attn_entropy(A).mean())
            dm_h.append(diag_mass(A))
        attn_mean_entropy.append(np.mean(ent_h))
        attn_mean_diag.append(np.mean(dm_h))

    # Q/K/V norms per layer averaged over heads & tokens
    q_means, k_means, v_means = [], [], []
    Q_stack, K_stack, V_stack = [], [], []
    for Lloc in range(L):
        qh, kh, vh = [], [], []
        for Hloc in range(H):
            Q = st.qkv(eid, "q", side=side_for_self, kind="self", layer=Lloc, head=Hloc)[idx]
            K = st.qkv(eid, "k", side=side_for_self, kind="self", layer=Lloc, head=Hloc)[idx]
            V = st.qkv(eid, "v", side=side_for_self, kind="self", layer=Lloc, head=Hloc)[idx]
            qh.append(np.linalg.norm(Q, axis=-1).mean())
            kh.append(np.linalg.norm(K, axis=-1).mean())
            vh.append(np.linalg.norm(V, axis=-1).mean())
            if take_pca and take_pca > 0:
                Q_stack.append(Q.mean(0))
                K_stack.append(K.mean(0))
                V_stack.append(V.mean(0))
        q_means.append(np.mean(qh)); k_means.append(np.mean(kh)); v_means.append(np.mean(vh))

    feats = {
        "attn_entropy_mean": float(np.mean(attn_mean_entropy)),
        "attn_entropy_std":  float(np.std(attn_mean_entropy)),
        "attn_diag_mean":    float(np.mean(attn_mean_diag)),
        "q_norm_mean": float(np.mean(q_means)),
        "k_norm_mean": float(np.mean(k_means)),
        "v_norm_mean": float(np.mean(v_means)),
    }

    # Optional: small PCA summary of Q/K/V (global per example)
    if take_pca and take_pca > 0 and len(Q_stack) > 0:
        Qm = np.stack(Q_stack); Km = np.stack(K_stack); Vm = np.stack(V_stack)
        pcaQ = PCA(n_components=min(take_pca, Qm.shape[1])).fit(Qm)
        pcaK = PCA(n_components=min(take_pca, Km.shape[1])).fit(Km)
        pcaV = PCA(n_components=min(take_pca, Vm.shape[1])).fit(Vm)
        Qc = pcaQ.transform(Qm).mean(0)
        Kc = pcaK.transform(Km).mean(0)
        Vc = pcaV.transform(Vm).mean(0)
        for i,v in enumerate(Qc): feats[f"Q_pca{i}"]=float(v)
        for i,v in enumerate(Kc): feats[f"K_pca{i}"]=float(v)
        for i,v in enumerate(Vc): feats[f"V_pca{i}"]=float(v)

    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="path to traces/<RUN_ID>")
    ap.add_argument("--pred_csv",  required=True, help="classifier val_predictions.csv with example_id & error")
    ap.add_argument("--out_parquet", default="features/reliability_features.parquet")
    ap.add_argument("--take_pca", type=int, default=8)
    args = ap.parse_args()

    st = TraceStore(args.run_dir)
    pred = pd.read_csv(args.pred_csv)[["example_id","y_true","y_pred","p_pos","error"]]
    pred = pred.set_index("example_id")

    rows = []
    # optional: fit PCA on a subset globally (simple approach: fit per example on its stacks; adequate for first pass)
    for eid in st.tokens["example_id"]:
        feats = per_example_features(st, eid, take_pca=args.take_pca)
        feats["example_id"] = eid
        # attach targets if present
        if eid in pred.index:
            feats["error"] = int(pred.loc[eid,"error"])
            feats["p_pos"] = float(pred.loc[eid,"p_pos"])
            feats["y_true"] = int(pred.loc[eid,"y_true"])
            feats["y_pred"] = int(pred.loc[eid,"y_pred"])
        rows.append(feats)

    df = pd.DataFrame(rows)
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print("[done] wrote", args.out_parquet, "rows:", len(df))

if __name__ == "__main__":
    main()
