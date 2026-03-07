"""
dl_models.py — Hybrid 1D-CNN + Transformer model for raw CIR classification.

This module defines the end-to-end deep learning architecture that operates
directly on the raw 1016-sample CIR waveform, bypassing manual feature engineering.

Architecture overview:
  Raw CIR [batch, 1016]
    → 1D-CNN Encoder (3 conv blocks: 1→32→64→128 channels with MaxPool)
    → Sequence of 127 temporal embeddings [batch, 127, 128]
    → Sinusoidal Positional Encoding
    → Transformer Encoder (2 layers, 4 attention heads)
    → Global Average Pooling → [batch, 128]
    → Concatenate with 11 scalar features → [batch, 139]
    → MLP Head (139→64→1) → Binary logit for LOS/NLOS

Design rationale:
  - The CNN encoder captures LOCAL patterns (peak shapes, rise/decay profiles,
    multipath signatures) through hierarchical convolution filters.
  - The Transformer encoder captures LONG-RANGE dependencies across the entire
    CIR (relationships between distant multipath components, overall energy
    distribution patterns) via self-attention.
  - Scalar features (RANGE, FP_AMP1-3, noise metrics) are fused at the
    classification head to provide hardware-level context.

Libraries: torch (PyTorch), math (positional encoding computation)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer input.

    Since transformers have no inherent notion of sequence order, positional
    encoding injects position information using sin/cos functions at different
    frequencies. This allows the transformer to distinguish between CIR
    temporal positions (e.g., early-arriving vs late-arriving signal components).

    Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
             PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=512):
        """
        Parameters:
            d_model (int): Dimension of the model (must match CNN output channels).
            max_len (int): Maximum sequence length to pre-compute encodings for.
        """
        super().__init__()
        # Pre-compute positional encodings as a non-trainable buffer
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions: cos
        self.register_buffer('pe', pe.unsqueeze(0))    # Shape: [1, max_len, d_model]

    def forward(self, x):
        """Add positional encoding to input embeddings. x: [batch, seq_len, d_model]."""
        return x + self.pe[:, :x.size(1)]


class MultiScaleConv(nn.Module):
    """
    Multi-scale first convolutional layer with parallel branches at different kernel sizes.

    Captures both fine-grained (kernel=3) and broad (kernel=7, 15) patterns in the
    raw CIR waveform simultaneously, then concatenates outputs to 32 channels.
    """

    def __init__(self, out_channels=32):
        super().__init__()
        # Split output channels across 3 kernel sizes
        c1 = out_channels // 3           # ~10 channels for kernel=3
        c2 = out_channels // 3           # ~10 channels for kernel=7
        c3 = out_channels - c1 - c2      # remaining for kernel=15
        self.branch3 = nn.Conv1d(1, c1, kernel_size=3, padding=1)
        self.branch7 = nn.Conv1d(1, c2, kernel_size=7, padding=3)
        self.branch15 = nn.Conv1d(1, c3, kernel_size=15, padding=7)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: [batch, 1, 1016]
        b3 = self.branch3(x)
        b7 = self.branch7(x)
        b15 = self.branch15(x)
        out = torch.cat([b3, b7, b15], dim=1)  # [batch, 32, 1016]
        return torch.relu(self.bn(out))


class ResidualCNNBlock(nn.Module):
    """
    Residual 1D-CNN block: Conv+BN+ReLU → Conv+BN → add residual → ReLU → MaxPool.

    Uses a 1x1 convolution shortcut when input/output channels differ.
    Residual connections improve gradient flow and enable deeper networks.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(2)

        # 1x1 conv shortcut for channel mismatch
        self.shortcut = (nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch
                         else nn.Identity())

    def forward(self, x):
        residual = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.relu(out + residual)
        return self.pool(out)


class CIRTransformerClassifier(nn.Module):
    """
    Hybrid multi-scale CNN with residual blocks + Transformer for LOS/NLOS classification.

    Architecture:
      Raw CIR [batch, 1016]
        → Multi-scale Conv (kernels 3/7/15) → 32 channels + MaxPool (508)
        → ResidualCNNBlock 32→64 + MaxPool (254)
        → ResidualCNNBlock 64→128 + MaxPool (127)
        → Positional Encoding → Transformer Encoder (2 layers, 4 heads)
        → Global Average Pooling → [batch, 128]
        → Concat scalars → MLP Head → binary logit
    """

    def __init__(self, n_scalar=11, cnn_channels=128, n_heads=4,
                 n_transformer_layers=2, mlp_hidden=64, dropout=0.1):
        super().__init__()

        # ── Multi-scale first layer + residual CNN blocks ─────────────
        self.multi_scale = MultiScaleConv(out_channels=32)
        self.pool1 = nn.MaxPool1d(2)  # 1016 → 508
        self.res_block2 = ResidualCNNBlock(32, 64, kernel_size=5)   # 508 → 254
        self.res_block3 = ResidualCNNBlock(64, cnn_channels, kernel_size=3)  # 254 → 127

        # ── Transformer Encoder ──────────────────────────────────────
        self.pos_enc = PositionalEncoding(cnn_channels, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_channels, nhead=n_heads,
            dim_feedforward=cnn_channels * 2, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # ── MLP Classification Head ──────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(cnn_channels + n_scalar, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

        self.last_attention_weights = None
        self._register_attention_hook()

    def _register_attention_hook(self):
        """Hook into the first transformer layer's self-attention to capture weights."""
        def hook_fn(module, input, output):
            if not self.training:
                src = input[0] if isinstance(input, tuple) else input
                with torch.no_grad():
                    _, self.last_attention_weights = module.self_attn(
                        src, src, src, need_weights=True, average_attn_weights=True
                    )
        self.transformer.layers[0].register_forward_hook(hook_fn)

    def forward(self, cir, scalars):
        """Forward pass: CIR [batch, 1016] + scalars → logit [batch, 1]."""
        x = cir.unsqueeze(1)  # [batch, 1, 1016]

        # Multi-scale CNN + residual blocks → [batch, 128, 127]
        x = self.multi_scale(x)
        x = self.pool1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # Transformer: [batch, 127, 128]
        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)
        x = self.transformer(x)

        # Global average pooling + classification
        x = x.mean(dim=1)
        combined = torch.cat([x, scalars], dim=1)
        return self.head(combined)
