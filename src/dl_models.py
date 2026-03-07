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


class CIRTransformerClassifier(nn.Module):
    """
    Hybrid 1D-CNN + Transformer for LOS/NLOS classification from raw CIR.

    Input:
        cir: [batch, 1016] — raw CIR waveform (1016 time-domain samples)
        scalars: [batch, n_scalar] — scalar features from Decawave hardware

    Output:
        logits: [batch, 1] — binary logit (use BCEWithLogitsLoss for training)
                Positive logit → NLOS, Negative logit → LOS
    """

    def __init__(self, n_scalar=11, cnn_channels=128, n_heads=4,
                 n_transformer_layers=2, mlp_hidden=64, dropout=0.1):
        """
        Parameters:
            n_scalar (int): Number of scalar features to concatenate (default 11).
            cnn_channels (int): Output channels of the CNN encoder (default 128).
            n_heads (int): Number of attention heads in the transformer (default 4).
            n_transformer_layers (int): Number of transformer encoder layers (default 2).
            mlp_hidden (int): Hidden dimension of the classification MLP head (default 64).
            dropout (float): Dropout rate for regularisation (default 0.1).
        """
        super().__init__()

        # ── 1D-CNN Encoder: 3 convolutional blocks ───────────────────
        # Progressively increases channels (1→32→64→128) while halving
        # temporal resolution via MaxPool (1016→508→254→127).
        # BatchNorm stabilises training; ReLU provides non-linearity.
        self.cnn = nn.Sequential(
            # Block 1: Raw CIR (1 channel) → 32 feature maps
            nn.Conv1d(1, 32, kernel_size=7, padding=3),   # Large kernel for initial patterns
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 1016 → 508

            # Block 2: 32 → 64 feature maps
            nn.Conv1d(32, 64, kernel_size=5, padding=2),  # Medium kernel
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 508 → 254

            # Block 3: 64 → 128 feature maps (final embedding dimension)
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),  # Small kernel for fine detail
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 254 → 127
        )

        # ── Transformer Encoder ──────────────────────────────────────
        # Processes the 127-step sequence of 128-dim CNN embeddings.
        # Self-attention allows each temporal position to attend to all others,
        # capturing long-range dependencies across the CIR waveform.
        self.pos_enc = PositionalEncoding(cnn_channels, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_channels, nhead=n_heads,
            dim_feedforward=cnn_channels * 2, dropout=dropout,
            batch_first=True,  # Input shape: [batch, seq_len, d_model]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # ── MLP Classification Head ──────────────────────────────────
        # Takes the pooled CIR representation (128-dim) concatenated with
        # scalar features (11-dim) → 139-dim input → 64 hidden → 1 logit.
        self.head = nn.Sequential(
            nn.Linear(cnn_channels + n_scalar, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),  # Single logit for binary classification
        )

        # ── Attention Weight Storage ─────────────────────────────────
        # Store attention weights from the first transformer layer for
        # interpretability/visualization (only captured during eval mode).
        self.last_attention_weights = None
        self._register_attention_hook()

    def _register_attention_hook(self):
        """
        Hook into the first transformer layer's self-attention to capture weights.

        This enables visualization of which CIR regions the model attends to
        when making LOS/NLOS predictions. The hook only runs during evaluation
        (not training) to avoid impacting training performance.
        """
        def hook_fn(module, input, output):
            # Re-run attention to get weights (the standard forward doesn't return them)
            if not self.training:
                src = input[0] if isinstance(input, tuple) else input
                with torch.no_grad():
                    _, self.last_attention_weights = module.self_attn(
                        src, src, src, need_weights=True, average_attn_weights=True
                    )

        # Register on the first transformer encoder layer
        self.transformer.layers[0].register_forward_hook(hook_fn)

    def forward(self, cir, scalars):
        """
        Forward pass: CIR waveform + scalar features → LOS/NLOS logit.

        Parameters:
            cir (torch.Tensor): Raw CIR waveform, shape [batch, 1016].
            scalars (torch.Tensor): Scalar features, shape [batch, n_scalar].

        Returns:
            torch.Tensor: Binary logit, shape [batch, 1].
        """
        # Reshape CIR for 1D convolution: [batch, 1016] → [batch, 1, 1016]
        x = cir.unsqueeze(1)

        # CNN encoder: extract local features → [batch, 128, 127]
        x = self.cnn(x)

        # Reshape for transformer: [batch, 128, 127] → [batch, 127, 128]
        # (sequence of 127 temporal positions, each with 128-dim embedding)
        x = x.permute(0, 2, 1)

        # Add positional encoding and process through transformer
        x = self.pos_enc(x)
        x = self.transformer(x)  # Self-attention across all 127 positions

        # Global average pooling: aggregate all temporal positions → [batch, 128]
        x = x.mean(dim=1)

        # Concatenate scalar features and classify → [batch, 139] → [batch, 1]
        combined = torch.cat([x, scalars], dim=1)
        return self.head(combined)
