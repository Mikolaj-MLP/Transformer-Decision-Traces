# src/reliability/eval_reliability.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch, joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from .autoencoder import DualHeadAutoencoder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features/rel_feats.parquet")
    ap.add_argument("--run_dir",   default="runs/reliability")
    args = ap.parse_args()

    meta = torch.load(Path(args.run_dir)/"scaler.pt")
    mu, sd, feat_cols = meta["mu"], meta["sd"], meta["feat_cols"]

    df = pd.read_parquet(args.features).dropna(subset=["error"])
    X = df[feat_cols].astype(float).values
    y = df["error"].astype(int).values
    X = (X - mu)/sd

    # load model + calibration
    ckpt = torch.load(Path(args.run_dir)/"ae.pt", map_location="cpu")
    model = DualHeadAutoencoder(in_dim=X.shape[1])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = []
        for i in range(0, len(X), 1024):
            xb = torch.tensor(X[i:i+1024], dtype=torch.float32)
            _, logit, _ = model(xb)
            logits.append(logit.numpy())
        logits = np.concatenate(logits, 0)

    lr = joblib.load(Path(args.run_dir)/"platt.joblib")
    p_err = lr.predict_proba(logits.reshape(-1,1))[:,1]

    thr = json.loads((Path(args.run_dir)/"operating_point.json").read_text())["threshold"]
    reliable = (p_err < thr).astype(int)  # lower p(error) → more reliable
    cov = reliable.mean()

    print("AUROC:", roc_auc_score(y, p_err))
    print("AUPRC:", average_precision_score(y, p_err))
    print("Brier:", brier_score_loss(y, p_err))
    print("Coverage @threshold:", cov)

if __name__ == "__main__":
    main()
