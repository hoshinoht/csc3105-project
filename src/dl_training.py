"""
dl_training.py — Training loop and assessment for the CNN+Transformer CIR classifier.

This module handles the end-to-end training and assessment of the deep learning model
defined in dl_models.py. It implements:
  - A PyTorch Dataset class for batching CIR + scalar + label data.
  - Automatic device selection (CUDA GPU → Apple MPS → CPU fallback).
  - Training loop with Adam optimiser, BCEWithLogitsLoss, and ReduceLROnPlateau
    learning rate scheduler.
  - Early stopping (patience=5) to prevent overfitting.
  - Full assessment on the test set with metrics compatible with the ML pipeline
    (accuracy, AUC, confusion matrix, ROC curve).

Libraries: torch (PyTorch), sklearn.metrics (assessment), numpy
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc,
)
from src.dl_models import CIRTransformerClassifier


class CIRDataset(Dataset):
    """
    PyTorch Dataset for CIR waveforms + scalar features + binary labels.

    Wraps numpy arrays as torch tensors for efficient batching via DataLoader.
    Each sample provides:
      - cir: 1016-dim CIR waveform (float32)
      - scalars: 11-dim scalar feature vector (float32)
      - label: binary NLOS label (float32, for BCEWithLogitsLoss)
    """

    def __init__(self, cir, scalars, labels):
        """
        Parameters:
            cir (np.ndarray): CIR waveforms, shape (N, 1016).
            scalars (np.ndarray): Scalar features, shape (N, 11).
            labels (np.ndarray): Binary labels, shape (N,).
        """
        self.cir = torch.tensor(cir, dtype=torch.float32)
        self.scalars = torch.tensor(scalars, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cir[idx], self.scalars[idx], self.labels[idx]


def _get_device():
    """
    Select the best available compute device.

    Priority: CUDA GPU → Apple MPS (Metal Performance Shaders) → CPU.
    CUDA provides the fastest training on NVIDIA GPUs. MPS accelerates
    training on Apple Silicon (M1/M2/M3) Macs. CPU is the fallback.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_dl_classifier(X_cir_train, X_scalar_train, y_train,
                         X_cir_test, X_scalar_test, y_test,
                         lr=1e-3, batch_size=256, max_epochs=50, patience=5):
    """
    Train the CNN+Transformer model and return results dict compatible
    with the existing classification pipeline.

    Training procedure:
      1. Initialise model, optimiser (Adam), and loss function (BCEWithLogitsLoss).
      2. For each epoch: train on batches, assess on test set.
      3. ReduceLROnPlateau halves the learning rate if validation loss plateaus
         for 2 consecutive epochs.
      4. Early stopping: if validation loss doesn't improve for `patience` epochs,
         stop training and restore the best model weights.

    Parameters:
        X_cir_train (np.ndarray): Training CIR waveforms, shape (N_train, 1016).
        X_scalar_train (np.ndarray): Training scalar features, shape (N_train, 11).
        y_train (np.ndarray): Training labels, shape (N_train,).
        X_cir_test (np.ndarray): Test CIR waveforms, shape (N_test, 1016).
        X_scalar_test (np.ndarray): Test scalar features, shape (N_test, 11).
        y_test (np.ndarray): Test labels, shape (N_test,).
        lr (float): Initial learning rate for Adam (default 1e-3).
        batch_size (int): Mini-batch size for training (default 256).
        max_epochs (int): Maximum training epochs (default 50).
        patience (int): Early stopping patience in epochs (default 5).

    Returns:
        dict: {model, y_pred, y_prob, accuracy, auc, confusion_matrix,
               fpr, tpr, report} — compatible with ML pipeline results.
    """
    device = _get_device()
    print(f"\n--- CNN+Transformer (device: {device}) ---")

    # ── Initialise model ─────────────────────────────────────────────
    n_scalar = X_scalar_train.shape[1]  # Number of scalar features (11)
    model = CIRTransformerClassifier(n_scalar=n_scalar).to(device)

    # ── Create DataLoaders for batched training/assessment ───────────
    train_ds = CIRDataset(X_cir_train, X_scalar_train, y_train)
    test_ds = CIRDataset(X_cir_test, X_scalar_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # ── Optimiser and loss function ──────────────────────────────────
    # Adam: adaptive learning rate optimiser (good default for DL)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # BCEWithLogitsLoss: combines sigmoid + binary cross-entropy in one numerically
    # stable operation. Input is raw logit, not probability.
    criterion = nn.BCEWithLogitsLoss()
    # ReduceLROnPlateau: halve LR when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,
    )

    # ── Training loop with early stopping ────────────────────────────
    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # --- Training phase ---
        model.train()  # Enable dropout, BatchNorm in training mode
        train_loss = 0.0
        for cir_batch, scalar_batch, label_batch in train_loader:
            # Move batch to compute device (GPU/MPS/CPU)
            cir_batch = cir_batch.to(device)
            scalar_batch = scalar_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()                              # Reset gradients
            logits = model(cir_batch, scalar_batch).squeeze(-1)  # Forward pass → logits
            loss = criterion(logits, label_batch)              # Compute loss
            loss.backward()                                    # Backpropagate gradients
            optimizer.step()                                   # Update weights
            train_loss += loss.item() * len(label_batch)       # Accumulate weighted loss

        train_loss /= len(train_ds)  # Average loss per sample

        # --- Validation phase ---
        model.eval()  # Disable dropout, use running stats for BatchNorm
        val_loss = 0.0
        with torch.no_grad():  # No gradient computation for validation
            for cir_batch, scalar_batch, label_batch in test_loader:
                cir_batch = cir_batch.to(device)
                scalar_batch = scalar_batch.to(device)
                label_batch = label_batch.to(device)
                logits = model(cir_batch, scalar_batch).squeeze(-1)
                val_loss += criterion(logits, label_batch).item() * len(label_batch)
        val_loss /= len(test_ds)

        # Adjust learning rate based on validation loss plateau
        scheduler.step(val_loss)

        # Print progress every 5 epochs (and always the first epoch)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # --- Early stopping check ---
        if val_loss < best_loss:
            best_loss = val_loss
            # Save best model weights (cloned to CPU to avoid GPU memory issues)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # ── Load best model and run assessment ───────────────────────────
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Collect all test predictions
    all_logits = []
    with torch.no_grad():
        for cir_batch, scalar_batch, _ in test_loader:
            logits = model(cir_batch.to(device), scalar_batch.to(device)).squeeze(-1)
            all_logits.append(logits.cpu())

    # Convert logits to probabilities and binary predictions
    all_logits = torch.cat(all_logits).numpy()
    y_prob = _sigmoid(all_logits)          # Apply sigmoid to get P(NLOS)
    y_pred = (y_prob >= 0.5).astype(int)   # Threshold at 0.5 for binary decision

    # Compute assessment metrics (same as ML pipeline for fair comparison)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['LOS', 'NLOS'])
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"  Accuracy: {acc:.4f}, AUC: {roc_auc:.4f}")
    print(report)

    return {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': acc,
        'auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'report': report,
    }


def _sigmoid(x):
    """
    Numerically stable sigmoid function.

    Uses conditional computation to avoid overflow:
      - For x >= 0: sigma(x) = 1 / (1 + exp(-x))  (standard formula)
      - For x < 0: sigma(x) = exp(x) / (1 + exp(x))  (avoids exp(large positive))

    Parameters:
        x (np.ndarray): Input logits.

    Returns:
        np.ndarray: Sigmoid probabilities in [0, 1].
    """
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
