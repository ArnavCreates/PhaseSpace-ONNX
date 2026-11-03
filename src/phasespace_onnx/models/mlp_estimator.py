"""
Multi-Layer Perceptron neural likelihood ratio classifier with embedded
input normalization buffers for zero-overhead ONNX deployment.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Any, Iterable, Iterator, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class PhaseSpaceMLP(nn.Module):
    """
    Neural likelihood ratio discriminator with self-contained feature standardization.
    Bakes mean and std into model buffers so downstream ONNX runtime inference
    accepts raw unstandardized kinematic observables without separate preprocessors.
    """

    def __init__(
        self,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]
        self.hidden_dims = list(hidden_dims)
        self.dropout_probability = float(dropout)

        # Convert and register standardization buffers
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).to(torch.float32)
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std).to(torch.float32)

        self.register_buffer("mean", mean.view(1, -1))
        self.register_buffer("std", std.view(1, -1))

        layers: List[nn.Module] = []
        in_dim = mean.numel()

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Final projection to single logit
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def fit_streaming(
        self,
        chunk_stream_fn: Callable[[], Iterable[Tuple[np.ndarray, np.ndarray]]],
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
        seed: int = 42,
    ) -> "PhaseSpaceMLP":
        """Fit the model from repeatable chunks and update scaling with Welford statistics."""
        mlp_cfg = config.get("mlp", {})
        learning_rate = float(mlp_cfg.get("learning_rate", 0.0015))
        batch_size = int(mlp_cfg.get("batch_size", 1024))
        epochs = int(mlp_cfg.get("epochs", 20))
        weight_decay = float(mlp_cfg.get("weight_decay", 1e-4))
        torch.manual_seed(seed)

        count = 0
        running_mean = np.zeros(self.mean.shape[1], dtype=np.float64)
        running_m2 = np.zeros_like(running_mean)
        for features, _ in chunk_stream_fn():
            values = np.asarray(features, dtype=np.float64)
            if values.size == 0:
                continue
            chunk_count = int(values.shape[0])
            chunk_mean = np.mean(values, axis=0)
            centered = values - chunk_mean
            chunk_m2 = np.sum(centered * centered, axis=0)
            delta = chunk_mean - running_mean
            combined_count = count + chunk_count
            running_mean += delta * (chunk_count / combined_count)
            running_m2 += chunk_m2 + delta * delta * (count * chunk_count / combined_count)
            count = combined_count
        if count == 0:
            raise ValueError("The training stream produced no events")

        running_std = np.sqrt(running_m2 / count)
        running_std = np.where(running_std < 1e-6, 1.0, running_std)
        with torch.no_grad():
            self.mean.copy_(torch.as_tensor(running_mean, dtype=self.mean.dtype).view_as(self.mean))
            self.std.copy_(torch.as_tensor(running_std, dtype=self.std.dtype).view_as(self.std))

        val_features = torch.as_tensor(np.asarray(X_val), dtype=torch.float32)
        val_labels = torch.as_tensor(np.asarray(y_val), dtype=torch.float32).reshape(-1, 1)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        best_val_loss = float("inf")
        best_state: Dict[str, torch.Tensor] = {}

        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0.0
            trained_events = 0
            for features, labels in chunk_stream_fn():
                if len(labels) == 0:
                    continue
                order = torch.randperm(len(labels))
                features_tensor = torch.as_tensor(np.asarray(features), dtype=torch.float32)
                labels_tensor = torch.as_tensor(np.asarray(labels), dtype=torch.float32).reshape(-1, 1)
                for start in range(0, len(labels), batch_size):
                    indices = order[start : start + batch_size]
                    batch_features = features_tensor[indices]
                    batch_labels = labels_tensor[indices]
                    optimizer.zero_grad(set_to_none=True)
                    predictions = self(batch_features)
                    loss = criterion(predictions, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * len(indices)
                    trained_events += len(indices)

            self.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for start in range(0, len(val_labels), batch_size):
                    stop = min(start + batch_size, len(val_labels))
                    predictions = self(val_features[start:stop])
                    loss = criterion(predictions, val_labels[start:stop])
                    val_loss_sum += float(loss.item()) * (stop - start)
            val_loss = val_loss_sum / max(len(val_labels), 1)
            train_loss = total_loss / max(trained_events, 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {name: value.detach().clone() for name, value in self.state_dict().items()}
            if epoch % 5 == 0 or epoch == epochs:
                print(
                    f"[PhaseSpaceMLP Epoch {epoch:02d}/{epochs:02d}] "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )

        if best_state:
            self.load_state_dict(best_state)
        self.eval()
        return self

    def predict_proba_streaming(
        self,
        chunk_stream_fn: Callable[[], Iterable[Tuple[np.ndarray, np.ndarray]]],
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield model scores and labels while consuming one feature chunk at a time."""
        self.eval()
        with torch.no_grad():
            for features, labels in chunk_stream_fn():
                values = torch.as_tensor(np.asarray(features), dtype=torch.float32)
                scores = self(values).cpu().numpy().reshape(-1)
                yield scores, np.asarray(labels)

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save network parameters, scaler buffers, and reconstruction metadata."""
        payload = {
            "state_dict": {name: value.detach().cpu() for name, value in self.state_dict().items()},
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout_probability,
        }
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        torch.save(payload, checkpoint_path)

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str) -> "PhaseSpaceMLP":
        """Restore a model checkpoint written by ``save_checkpoint``."""
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        state = payload["state_dict"]
        model = cls(
            mean=state["mean"].reshape(-1),
            std=state["std"].reshape(-1),
            hidden_dims=list(payload["hidden_dims"]),
            dropout=float(payload.get("dropout", 0.0)),
        )
        model.load_state_dict(state)
        model.eval()
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward evaluation mapping raw kinematic features directly to calibrated score domain.
        Args:
            x: Raw physical kinematics tensor of shape (batch_size, 10).
        Returns:
            Probability score in (0, 1) of shape (batch_size, 1).
        """
        # In-graph standardization: zero external preprocessing needed
        z = (x - self.mean) / (self.std + 1e-7)
        logits = self.net(z)
        return torch.sigmoid(logits)


def train_mlp_estimator(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
) -> PhaseSpaceMLP:
    """
    Train PhaseSpaceMLP discriminator on CPU using AdamW and Binary Cross-Entropy loss.

    Returns:
        Trained PhaseSpaceMLP module in eval mode.
    """
    mlp_cfg = config.get("mlp", {})
    hidden_dims = mlp_cfg.get("hidden_dims", [128, 64, 32])
    lr = float(mlp_cfg.get("learning_rate", 0.0015))
    batch_size = int(mlp_cfg.get("batch_size", 1024))
    epochs = int(mlp_cfg.get("epochs", 20))
    weight_decay = float(mlp_cfg.get("weight_decay", 1e-4))
    dropout = float(mlp_cfg.get("dropout", 0.0))

    # Compute training set empirical moments
    mean = np.mean(X_train, axis=0, keepdims=False).astype(np.float32)
    std = np.std(X_train, axis=0, keepdims=False).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    model = PhaseSpaceMLP(mean=mean, std=std, hidden_dims=hidden_dims, dropout=dropout)
    device = torch.device("cpu")
    model.to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).to(torch.float32),
        torch.from_numpy(y_train).to(torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).to(torch.float32),
        torch.from_numpy(y_val).to(torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    model.train()
    best_val_loss = float("inf")
    best_state: Dict[str, torch.Tensor] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(x_b)
            n_samples += len(x_b)

        train_loss = running_loss / max(n_samples, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                v_preds = model(x_v)
                v_loss = criterion(v_preds, y_v)
                val_loss += v_loss.item() * len(x_v)
                val_samples += len(x_v)

        val_loss = val_loss / max(val_samples, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"[PhaseSpaceMLP Epoch {epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    return model
