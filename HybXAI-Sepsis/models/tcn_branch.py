"""
models/tcn_branch.py
=====================
Temporal Convolutional Network (TCN) encoder for the time-series branch.

Architecture:
  4 × TemporalBlock with dilation schedule {1, 2, 4, 8}
  Kernel size = 3, channels 24→64→64→128→128
  Global average pooling → 128-dim embedding

Reference: Bai et al., "An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modelling", arXiv:1803.01271
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """
    One residual dilated-causal-conv block.
    Causal masking: output at time t depends only on t' ≤ t.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        # Causal padding = (kernel - 1) * dilation on the left only
        self.causal_pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels,  out_channels, kernel_size,
                               padding=self.causal_pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=self.causal_pad, dilation=dilation)

        self.bn1   = nn.BatchNorm1d(out_channels)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.drop  = nn.Dropout(dropout)
        self.relu  = nn.ReLU()

        # 1×1 projection for residual when channel dims differ
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels
                         else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # Remove causal padding from the right to keep output length = T
        h = self.relu(self.bn1(self.drop(self.conv1(x)[:, :, :-self.causal_pad])))
        h = self.relu(self.bn2(self.drop(self.conv2(h)[:, :, :-self.causal_pad])))
        res = self.residual(x)
        return self.relu(torch.nan_to_num(h) + torch.nan_to_num(res))


class TCNEncoder(nn.Module):
    """
    Stack of TemporalBlocks → global average pool → fixed-dim embedding.

    Parameters
    ----------
    in_channels : int
        Number of input channels (24 after augmentation: original 8 +
        delta 8 + CV 8).
    channels : tuple of int
        Output channels per block. Default: (64, 64, 128, 128).
    kernel_size : int
        Convolutional kernel width. Default: 3.
    dropout : float
        Dropout probability within each block. Default: 0.2.
    """

    def __init__(self, in_channels: int = 24,
                 channels: tuple = (64, 64, 128, 128),
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        ch = in_channels
        for i, out_ch in enumerate(channels):
            layers.append(
                TemporalBlock(ch, out_ch, kernel_size, dilation=2**i, dropout=dropout)
            )
            ch = out_ch
        self.network   = nn.Sequential(*layers)
        self.pool      = nn.AdaptiveAvgPool1d(1)
        self.out_dim   = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, C)  — batch × time × channels

        Returns
        -------
        h : (B, out_dim)
        """
        h = self.network(torch.nan_to_num(x.permute(0, 2, 1)))  # (B, C_out, T)
        return self.pool(torch.nan_to_num(h)).squeeze(-1)         # (B, out_dim)


# ── Quick sanity check ────────────────────────────────────────
if __name__ == '__main__':
    B, T, C = 8, 12, 24
    model = TCNEncoder(in_channels=C)
    x = torch.randn(B, T, C)
    out = model(x)
    print(f"Input: {x.shape}  →  TCN output: {out.shape}")
    assert out.shape == (B, model.out_dim), "Shape mismatch"
    print("TCN encoder OK")
