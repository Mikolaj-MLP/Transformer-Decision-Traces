# src/reliability/autoencoder.py
import torch
import torch.nn as nn

class DualHeadVAE(nn.Module):
    """
    Variational AE with:
      - Encoder: x -> (mu, logvar) -> z (reparam)
      - Decoder (reconstruction): z -> x_hat
      - Reliability head: z -> logit(error)
    """
    def __init__(self, in_dim: int, latent: int = 128, hidden: int = 256, p_dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.latent = latent

        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(p_dropout),
        )
        self.enc_mu = nn.Linear(hidden, latent)
        self.enc_logvar = nn.Linear(hidden, latent)

        self.dec_recon = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden, in_dim)
        )
        self.dec_rel = nn.Sequential(
            nn.Linear(latent, hidden // 2), nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden // 2, 1)
        )

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        z = self.reparam(mu, logvar)
        recon = self.dec_recon(z)
        logits = self.dec_rel(z).squeeze(-1)
        return recon, logits, z, mu, logvar
