"""
models/domain_adaptation.py
=============================
Domain-adversarial training component (Gradient Reversal Layer).

The domain discriminator is trained to distinguish MIMIC-IV (source)
from eICU-CRD (target) patients, while the gradient reversal layer
makes the TCN/fusion encoder produce domain-invariant representations.

Reference: Ganin et al., "Domain-Adversarial Training of Neural Networks",
JMLR 2016.

The full HybXAI-Sepsis v3 model integrates all components:
  ┌────────────────────────────────────────┐
  │  x_ts (B,T,24) → TCNEncoder → h_tcn   │
  │  x_st (B,43)   → XGBEmbedder → h_xgb  │
  │  [h_tcn, h_xgb] → AttentionFusion → p │  ← sepsis prediction
  │  GRL(h_tcn)    → DomainDisc → d        │  ← domain label (0=MIMIC, 1=eICU)
  └────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Gradient Reversal Layer ───────────────────────────────────

class GradientReversal(torch.autograd.Function):
    """
    Identity in forward pass; multiplies gradient by −λ in backward pass.
    λ is annealed from 0 → 1 over training following Ganin et al. schedule.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lam)


def grl_lambda(epoch: int, max_epochs: int,
               gamma: float = 10.0) -> float:
    """
    Annealing schedule for λ: 0 → 1 over training.
    p = epoch / max_epochs; λ = 2/(1 + exp(−γ·p)) − 1
    """
    p = epoch / max(max_epochs, 1)
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))) - 1.0)


# ── Domain Discriminator ──────────────────────────────────────

class DomainDiscriminator(nn.Module):
    """
    Binary classifier that predicts dataset origin.
    Applied to the GRL-transformed TCN embedding.

    Architecture: 128 → 64 → 1 (sigmoid)

    Parameters
    ----------
    in_dim  : int   — input dimension (TCN embedding dim = 128)
    dropout : float — dropout probability (default 0.3)
    """

    def __init__(self, in_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor, lam: float = 0.3) -> torch.Tensor:
        """
        Parameters
        ----------
        h   : (B, in_dim)  — TCN embedding
        lam : float        — GRL coefficient (annealed by caller)

        Returns
        -------
        d   : (B,)  — domain probability (1 = eICU / target domain)
        """
        h_rev = grad_reverse(h, lam)
        return self.net(h_rev).squeeze(-1)


# ── Full HybXAI-Sepsis v3 model ───────────────────────────────

class HybXAISepsisV3(nn.Module):
    """
    Complete HybXAI-Sepsis v3 model integrating all four components.

    Parameters
    ----------
    tcn_in_channels : int   — TCN input channels (24 after augmentation)
    xgb_embed_dim   : int   — XGBoost leaf embedding dimension (128)
    fusion_dropout  : float — dropout in fusion MLP (0.3)
    domain_dropout  : float — dropout in domain discriminator (0.3)
    """

    def __init__(self, tcn_in_channels: int = 24,
                 xgb_embed_dim: int = 128,
                 fusion_dropout: float = 0.3,
                 domain_dropout: float = 0.3):
        super().__init__()

        from models.tcn_branch import TCNEncoder
        from models.attention_fusion import AttentionFusionClassifier

        self.tcn     = TCNEncoder(in_channels=tcn_in_channels)
        self.fusion  = AttentionFusionClassifier(
            tcn_dim=self.tcn.out_dim, xgb_dim=xgb_embed_dim,
            dropout=fusion_dropout
        )
        self.domain  = DomainDiscriminator(
            in_dim=self.tcn.out_dim, dropout=domain_dropout
        )

    def forward(self, x_ts: torch.Tensor,
                h_xgb: torch.Tensor,
                lam: float = 0.3) -> tuple:
        """
        Parameters
        ----------
        x_ts  : (B, T, 24)  — normalised vital-sign tensor
        h_xgb : (B, 128)    — XGBoost leaf embedding
        lam   : float       — GRL coefficient

        Returns
        -------
        p_sep    : (B,)  — sepsis probability
        d_domain : (B,)  — domain probability (for adversarial loss)
        """
        h_tcn    = self.tcn(x_ts)
        p_sep    = self.fusion(h_tcn, h_xgb)
        d_domain = self.domain(h_tcn, lam)
        return p_sep, d_domain

    def predict(self, x_ts: torch.Tensor,
                h_xgb: torch.Tensor) -> torch.Tensor:
        """Inference-only forward (no domain head, no GRL)."""
        with torch.no_grad():
            h_tcn = self.tcn(x_ts)
            return self.fusion(h_tcn, h_xgb)


# ── Training step helper ──────────────────────────────────────

def domain_adversarial_step(model: HybXAISepsisV3,
                             x_ts_src: torch.Tensor, h_xgb_src: torch.Tensor, y_src: torch.Tensor,
                             x_ts_tgt: torch.Tensor, h_xgb_tgt: torch.Tensor,
                             focal_loss_fn, optimizer: torch.optim.Optimizer,
                             lam: float = 0.3,
                             domain_weight: float = 0.1) -> dict:
    """
    One training step combining:
      L_total = L_focal(sepsis, source) + domain_weight × L_bce(domain)

    Parameters
    ----------
    x_ts_src, h_xgb_src, y_src : source (MIMIC-IV) batch
    x_ts_tgt, h_xgb_tgt        : target (eICU) batch (labels unused)
    focal_loss_fn               : FocalLoss instance
    domain_weight               : weight for domain adversarial loss (λ_d = 0.1)
    """
    model.train()
    optimizer.zero_grad()

    # Source: sepsis prediction + domain classification (label=0)
    p_src, d_src = model(x_ts_src, h_xgb_src, lam)
    p_src = torch.clamp(torch.nan_to_num(p_src, nan=0.5), 1e-7, 1 - 1e-7)
    loss_sep = focal_loss_fn(p_src, (y_src >= 0.5).float())

    # Target: domain classification only (label=1, no sepsis labels)
    _, d_tgt = model(x_ts_tgt, h_xgb_tgt, lam)

    # Domain loss: source=0, target=1
    d_labels = torch.cat([
        torch.zeros(len(d_src), device=d_src.device),
        torch.ones(len(d_tgt),  device=d_tgt.device)
    ])
    d_preds = torch.cat([d_src, d_tgt])
    loss_dom = F.binary_cross_entropy(
        torch.clamp(d_preds, 1e-7, 1 - 1e-7), d_labels
    )

    loss_total = loss_sep + domain_weight * loss_dom

    if not torch.isnan(loss_total):
        loss_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        'loss_total':  loss_total.item(),
        'loss_sepsis': loss_sep.item(),
        'loss_domain': loss_dom.item(),
    }


# ── Quick sanity check ────────────────────────────────────────
if __name__ == '__main__':
    B, T, C = 8, 12, 24
    model = HybXAISepsisV3()
    x_ts  = torch.randn(B, T, C)
    h_xgb = torch.randn(B, 128)
    p, d  = model(x_ts, h_xgb, lam=0.3)
    print(f"Sepsis probs: {p.shape}  Domain probs: {d.shape}")
    print(f"p range: [{p.min():.3f}, {p.max():.3f}]")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    print("Domain adaptation OK")
