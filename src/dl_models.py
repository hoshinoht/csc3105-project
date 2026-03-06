"""Hybrid 1D-CNN + Transformer model for raw CIR classification."""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class CIRTransformerClassifier(nn.Module):
    """
    Hybrid 1D-CNN + Transformer for LOS/NLOS classification from raw CIR.

    Input:
        cir: [batch, 1016] raw CIR waveform
        scalars: [batch, n_scalar] scalar features

    Output:
        logits: [batch, 1] (use BCEWithLogitsLoss)
    """

    def __init__(self, n_scalar=11, cnn_channels=128, n_heads=4,
                 n_transformer_layers=2, mlp_hidden=64, dropout=0.1):
        super().__init__()

        # 1D-CNN encoder: 3 conv blocks
        self.cnn = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 1016 -> 508

            # Block 2: 32 -> 64 channels
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 508 -> 254

            # Block 3: 64 -> cnn_channels
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 254 -> 127
        )

        # Transformer encoder
        self.pos_enc = PositionalEncoding(cnn_channels, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_channels, nhead=n_heads,
            dim_feedforward=cnn_channels * 2, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # MLP classification head
        self.head = nn.Sequential(
            nn.Linear(cnn_channels + n_scalar, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

        # Store attention weights for visualization
        self.last_attention_weights = None
        self._register_attention_hook()

    def _register_attention_hook(self):
        """Hook into the first transformer layer's self-attention to capture weights."""
        def hook_fn(module, input, output):
            # Re-run attention to get weights (forward doesn't return them)
            # We only do this during eval for visualization
            if not self.training:
                src = input[0] if isinstance(input, tuple) else input
                with torch.no_grad():
                    _, self.last_attention_weights = module.self_attn(
                        src, src, src, need_weights=True, average_attn_weights=True
                    )

        self.transformer.layers[0].register_forward_hook(hook_fn)

    def forward(self, cir, scalars):
        # cir: [batch, 1016] -> [batch, 1, 1016]
        x = cir.unsqueeze(1)

        # CNN encoder -> [batch, cnn_channels, 127]
        x = self.cnn(x)

        # Reshape for transformer: [batch, 127, cnn_channels]
        x = x.permute(0, 2, 1)

        # Positional encoding + transformer
        x = self.pos_enc(x)
        x = self.transformer(x)

        # Global average pooling -> [batch, cnn_channels]
        x = x.mean(dim=1)

        # Concatenate scalar features and classify
        combined = torch.cat([x, scalars], dim=1)
        return self.head(combined)
