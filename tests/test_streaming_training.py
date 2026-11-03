from __future__ import annotations

import os
import tempfile
from typing import Iterator, Tuple

import numpy as np
import torch

from phasespace_onnx.models.mlp_estimator import PhaseSpaceMLP


def test_streaming_fit_scaler_checkpoint_and_prediction() -> None:
    rng = np.random.default_rng(17)
    X_train = rng.normal(size=(160, 10)).astype(np.float32)
    y_train = (X_train[:, 0] + 0.2 * X_train[:, 1] > 0).astype(np.int32)
    X_val = rng.normal(size=(40, 10)).astype(np.float32)
    y_val = (X_val[:, 0] + 0.2 * X_val[:, 1] > 0).astype(np.int32)

    def train_chunks() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for start in range(0, len(X_train), 23):
            stop = min(start + 23, len(X_train))
            yield X_train[start:stop], y_train[start:stop]

    model = PhaseSpaceMLP(
        mean=np.zeros(10, dtype=np.float32),
        std=np.ones(10, dtype=np.float32),
        hidden_dims=[16, 8],
    )
    model.fit_streaming(
        train_chunks,
        X_val,
        y_val,
        {"mlp": {"epochs": 2, "batch_size": 16, "learning_rate": 0.001}},
        seed=17,
    )

    np.testing.assert_allclose(model.mean.cpu().numpy().reshape(-1), X_train.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(model.std.cpu().numpy().reshape(-1), X_train.std(axis=0), atol=1e-6)

    def validation_chunks() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for start in range(0, len(X_val), 13):
            stop = min(start + 13, len(X_val))
            yield X_val[start:stop], y_val[start:stop]

    predictions = list(model.predict_proba_streaming(validation_chunks))
    assert sum(len(labels) for _, labels in predictions) == len(X_val)
    assert all(np.all((scores >= 0.0) & (scores <= 1.0)) for scores, _ in predictions)

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = os.path.join(directory, "phasespace_mlp.pt")
        model.save_checkpoint(checkpoint)
        restored = PhaseSpaceMLP.load_checkpoint(checkpoint)
        np.testing.assert_allclose(
            model(torch.as_tensor(X_val)).detach().numpy(),
            restored(torch.as_tensor(X_val)).detach().numpy(),
            atol=0.0,
        )
