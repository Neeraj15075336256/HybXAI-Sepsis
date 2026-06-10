"""
models/attention_fusion.py
===========================
Attention-weighted fusion gate and classification head.

Architecture:
  concat([h_tcn(128), h_xgb(128)]) → 256-dim
  gate  = sigmoid(Linear(256, 256))      ← learned stream weighting
  fused = gate ⊙ concat                  ← element-wise gating
  MLP:  256 → 128 → 64 → 1 (sigmoid)    ← classification head

The gate implicitly allocates contribution between the TCN stream
(temporal patterns) and the XGBoost stream (static/physics knowledge)
on a per-sample, per-dimension basis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced binary classification.
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    FL(p) = -α(1-p)^γ log(p)

    Parameters
    ----------
    gamma : float  — focusing parameter (default 2.0)
    alpha : float  — class-weight for positive class (default 0.25)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target.float(), reduction='none')
        pt  = torch.exp(-bce)
        w   = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (w * (1 - pt) ** self.gamma * bce).mean()


class AttentionFusionClassifier(nn.Module):
    """
    Fusion gate + MLP classifier head.

    Parameters
    ----------
    tcn_dim    : int  — TCN embedding dimension (default 128)
    xgb_dim    : int  — XGBoost embedding dimension (default 128)
    hidden_dim : int  — first hidden layer (default 128)
    dropout    : float — dropout probability (default 0.3)
    """

    def __init__(self, tcn_dim: int = 128, xgb_dim: int = 128,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        fused_dim = tcn_dim + xgb_dim   # 256

        # Attention gate — one scalar weight per fused dimension
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid()
        )

        # Classification MLP: 256 → 128 → 64 → 1
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, h_tcn: torch.Tensor,
                h_xgb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h_tcn : (B, tcn_dim)
        h_xgb : (B, xgb_dim)

        Returns
        -------
        prob  : (B,)   — predicted sepsis probability
        """
        h_cat  = torch.cat([h_tcn, h_xgb], dim=-1)   # (B, 256)
        gate   = self.gate(h_cat)                      # (B, 256)
        fused  = gate * h_cat                          # (B, 256)
        fused  = torch.nan_to_num(fused, nan=0.0)
        return self.classifier(fused).squeeze(-1)      # (B,)

    def gate_weights(self, h_tcn: torch.Tensor,
                     h_xgb: torch.Tensor) -> dict:
        """
        Return mean gate weight for each stream (diagnostic / interpretability).
        """
        with torch.no_grad():
            h_cat = torch.cat([h_tcn, h_xgb], dim=-1)
            g     = self.gate(h_cat)
            d     = h_tcn.shape[-1]
            return {
                'tcn_weight': g[:, :d].mean().item(),
                'xgb_weight': g[:, d:].mean().item(),
            }


# ── Quick sanity check ────────────────────────────────────────
if __name__ == '__main__':
    B = 16
    h_t = torch.randn(B, 128)
    h_x = torch.randn(B, 128)
    model = AttentionFusionClassifier()
    prob  = model(h_t, h_x)
    print(f"Probabilities shape: {prob.shape} | range: [{prob.min():.3f}, {prob.max():.3f}]")
    print("Gate weights:", model.gate_weights(h_t, h_x))
    loss_fn = FocalLoss()
    y = torch.randint(0, 2, (B,))
    loss = loss_fn(prob, y)
    print(f"Focal loss: {loss.item():.4f}")
    print("Attention fusion OK")
