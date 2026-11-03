"""
LightGBM Gradient Boosted Decision Tree likelihood ratio baseline estimator.
Provides non-parametric tree ensemble density ratio approximation for comparison.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
import lightgbm as lgb


class LGBMReweighter:
    """
    LightGBM gradient boosted decision tree classifier for likelihood ratio estimation.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("lightgbm", {})
        n_estimators = int(cfg.get("n_estimators", 200))
        learning_rate = float(cfg.get("learning_rate", 0.05))
        num_leaves = int(cfg.get("num_leaves", 31))
        max_depth = int(cfg.get("max_depth", 6))
        subsample = float(cfg.get("subsample", 0.8))
        colsample_bytree = float(cfg.get("colsample_bytree", 0.8))
        random_state = int(cfg.get("random_state", 42))

        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LGBMReweighter":
        """Fit LightGBM model on training features and hypothesis labels."""
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability score s(x) = P(y=1 | x).
        Returns 1D array of shape (N,) containing class 1 probabilities.
        """
        probas = self.model.predict_proba(X)
        return probas[:, 1].astype(np.float32)
