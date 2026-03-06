"""Training loop and evaluation for the DL CIR classifier."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc,
)
from src.dl_models import CIRTransformerClassifier


class CIRDataset(Dataset):
    """PyTorch Dataset for CIR waveforms + scalar features."""

    def __init__(self, cir, scalars, labels):
        self.cir = torch.tensor(cir, dtype=torch.float32)
        self.scalars = torch.tensor(scalars, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cir[idx], self.scalars[idx], self.labels[idx]


def _get_device():
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
    """
    device = _get_device()
    print(f"\n--- CNN+Transformer (device: {device}) ---")

    n_scalar = X_scalar_train.shape[1]
    model = CIRTransformerClassifier(n_scalar=n_scalar).to(device)

    train_ds = CIRDataset(X_cir_train, X_scalar_train, y_train)
    test_ds = CIRDataset(X_cir_test, X_scalar_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,
    )

    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for cir_batch, scalar_batch, label_batch in train_loader:
            cir_batch = cir_batch.to(device)
            scalar_batch = scalar_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits = model(cir_batch, scalar_batch).squeeze(-1)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(label_batch)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cir_batch, scalar_batch, label_batch in test_loader:
                cir_batch = cir_batch.to(device)
                scalar_batch = scalar_batch.to(device)
                label_batch = label_batch.to(device)
                logits = model(cir_batch, scalar_batch).squeeze(-1)
                val_loss += criterion(logits, label_batch).item() * len(label_batch)
        val_loss /= len(test_ds)

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Evaluate
    all_logits = []
    with torch.no_grad():
        for cir_batch, scalar_batch, _ in test_loader:
            logits = model(cir_batch.to(device), scalar_batch.to(device)).squeeze(-1)
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits).numpy()
    y_prob = _sigmoid(all_logits)
    y_pred = (y_prob >= 0.5).astype(int)

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
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
