# src/reliability/train_reliability.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib

from .autoencoder import DualHeadVAE

def standardize(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd

def kld_gaussian(mu, logvar):
    # KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + logσ² - μ² - σ²)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features/rel_feats.parquet")
    ap.add_argument("--out_dir", default="runs/reliability", type=str)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lambda_recon", type=float, default=0.5, help="weight for recon MSE")
    ap.add_argument("--beta_kl", type=float, default=1e-3, help="weight for KL term")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.features).dropna(subset=["error"])
    y = df["error"].astype(int).values
    feat_cols = [c for c in df.columns if c not in {"example_id","error","p_pos","y_true","y_pred"}]
    X = df[feat_cols].astype(float).values

    X, mu, sd = standardize(X)
    torch.save({"mu":mu, "sd":sd, "feat_cols":feat_cols}, out_dir/"scaler.pt")

    # simple split (last 20% for calibration)
    n = len(X); n_cal = max(64, int(0.2*n))
    X_tr, X_cal = X[:-n_cal], X[-n_cal:]
    y_tr, y_cal = y[:-n_cal], y[-n_cal:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadVAE(in_dim=X.shape[1], latent=args.latent, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    mse = nn.MSELoss()
    pos_weight = (y_tr==1).sum() / max(1, (y_tr==0).sum())
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight+1e-6)]).to(device))

    ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                       torch.tensor(y_tr, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    for ep in range(1, args.epochs+1):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            recon, logits, z, mu_z, logvar_z = model(xb)
            loss_recon = mse(recon, xb)
            loss_bce   = bce(logits, yb)
            loss_kl    = kld_gaussian(mu_z, logvar_z).mean()
            loss = args.lambda_recon * loss_recon + (1 - args.lambda_recon) * loss_bce + args.beta_kl * loss_kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        print(f"[epoch {ep}] loss={tot/len(ds):.4f}  (recon={loss_recon.item():.4f}, bce={loss_bce.item():.4f}, kl={loss_kl.item():.4f})")
        
    # Calibration set logits
    model.eval()
    with torch.no_grad():
        logits_cal = []
        for i in range(0, len(X_cal), 1024):
            xb = torch.tensor(X_cal[i:i+1024], dtype=torch.float32, device=device)
            _, logit, *_ = model(xb)
            logits_cal.append(logit.detach().cpu().numpy())
        logits_cal = np.concatenate(logits_cal, 0)

    # Platt scaling → calibrated p(error)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(logits_cal.reshape(-1,1), y_cal)
    torch.save({"state_dict": model.state_dict(),
                "config": {"in_dim": X.shape[1], "latent": args.latent, "hidden": args.hidden}},
               out_dir/"vae.pt")
    joblib.dump(lr, out_dir/"platt.joblib")
    print("[done] saved VAE+calibration to", out_dir)

    # Quick eval
    p_err = lr.predict_proba(logits_cal.reshape(-1,1))[:,1]
    print("AUROC:", roc_auc_score(y_cal, p_err))
    print("AUPRC:", average_precision_score(y_cal, p_err))
    print("Brier:", brier_score_loss(y_cal, p_err))

    # Choose operating threshold (Youden J)
    from sklearn.metrics import roc_curve
    fpr,tpr,thr = roc_curve(y_cal, p_err)
    j = tpr - fpr; best = int(np.argmax(j))
    thr_star = float(thr[best])
    (out_dir/"operating_point.json").write_text(
        __import__("json").dumps({"threshold": thr_star}, indent=2)
    )
    print("Chosen threshold:", thr_star)

if __name__ == "__main__":
    main()
