"""
Monotonic histogram probability calibration for neural likelihood ratio estimation.
Converts raw discriminator scores into calibrated probabilities and likelihood ratios
r(x) = s_cal(x) / (1 - s_cal(x)), exporting a fixed-grid lookup table (LUT)
for C++20 sub-microsecond linear interpolation.
"""

from __future__ import annotations

import os
import struct
import json
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression


class HistogramCalibrator:
    """
    Binned histogram calibrator with isotonic monotonicity smoothing.
    Given raw scores s in [0, 1] and labels y in {0, 1}, maps s -> s_cal in [eps, 1-eps]
    and computes the likelihood ratio r(x) = s_cal / (1 - s_cal).
    """

    def __init__(
        self,
        num_bins: int = 50,
        eps_clip: float = 1.0e-5,
        weight_clip_max: float = 20.0,
        lut_grid_size: int = 1001,
    ) -> None:
        self.num_bins = num_bins
        self.eps_clip = eps_clip
        self.weight_clip_max = weight_clip_max
        self.lut_grid_size = lut_grid_size

        self.bin_edges: np.ndarray | None = None
        self.bin_centers: np.ndarray | None = None
        self.grid_x = np.linspace(0.0, 1.0, lut_grid_size, dtype=np.float32)
        self.grid_y = np.linspace(0.0, 1.0, lut_grid_size, dtype=np.float32)
        self.is_fitted = False

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> "HistogramCalibrator":
        """
        Fit monotonic calibration mapping from validation scores and labels.
        """
        scores = np.asarray(raw_scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()

        scores = np.clip(scores, 0.0, 1.0)
        self.bin_edges = np.linspace(0.0, 1.0, self.num_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        # Digitize scores into bins 1..num_bins
        bin_indices = np.digitize(scores, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        raw_cal_p = np.zeros(self.num_bins, dtype=np.float64)
        bin_weights = np.zeros(self.num_bins, dtype=np.float64)

        for k in range(self.num_bins):
            mask = (bin_indices == k)
            n_total = np.sum(mask)
            n_one = np.sum(y[mask])
            n_zero = n_total - n_one

            # Laplace smoothing: (n1 + 1) / (n0 + n1 + 2)
            raw_cal_p[k] = (n_one + 1.0) / (n_zero + n_one + 2.0)
            bin_weights[k] = n_total + 2.0

        # Enforce strict monotonicity using weighted isotonic regression
        iso = IsotonicRegression(
            y_min=self.eps_clip,
            y_max=1.0 - self.eps_clip,
            increasing=True,
            out_of_bounds="clip",
        )
        iso.fit(self.bin_centers, raw_cal_p, sample_weight=bin_weights)

        # Evaluate isotonic mapping over the dense uniform LUT grid
        smooth_y = iso.predict(self.grid_x)
        # Ensure strict boundary conditions
        smooth_y = np.clip(smooth_y, self.eps_clip, 1.0 - self.eps_clip).astype(np.float32)
        # Force monotonic non-decreasing
        self.grid_y = np.maximum.accumulate(smooth_y)
        self.is_fitted = True
        return self

    def calibrate_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Map uncalibrated classifier scores to calibrated probabilities via piecewise-linear LUT.
        """
        if not self.is_fitted:
            raise RuntimeError("HistogramCalibrator is not fitted yet.")
        scores = np.asarray(raw_scores, dtype=np.float32).ravel()
        scores_clipped = np.clip(scores, 0.0, 1.0)
        calibrated = np.interp(scores_clipped, self.grid_x, self.grid_y)
        return np.clip(calibrated, self.eps_clip, 1.0 - self.eps_clip).astype(np.float32)

    def compute_likelihood_ratio(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Compute density ratio r(x) = p1(x) / p0(x) = s_cal(x) / (1 - s_cal(x)).
        """
        s_cal = self.calibrate_scores(raw_scores)
        denom = 1.0 - s_cal
        denom = np.maximum(denom, self.eps_clip)
        weights = s_cal / denom
        weights = np.clip(weights, 0.0, self.weight_clip_max).astype(np.float32)
        return weights

    def export_lut_json(self, json_path: str) -> None:
        """Export calibration LUT parameters and grid to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        payload = {
            "num_points": int(self.lut_grid_size),
            "eps_clip": float(self.eps_clip),
            "weight_clip_max": float(self.weight_clip_max),
            "grid_x": self.grid_x.tolist(),
            "grid_y": self.grid_y.tolist(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def export_lut_binary(self, bin_path: str) -> None:
        """
        Export calibration LUT to compact binary for fast C++20 direct deserialization.
        Layout:
          uint32: num_points (1001)
          float32: eps_clip
          float32: weight_clip_max
          float32[num_points]: grid_y
        """
        os.makedirs(os.path.dirname(os.path.abspath(bin_path)), exist_ok=True)
        num_points = len(self.grid_y)
        header = struct.pack("=Iff", num_points, float(self.eps_clip), float(self.weight_clip_max))
        data = self.grid_y.astype(np.float32).tobytes()
        with open(bin_path, "wb") as f:
            f.write(header)
            f.write(data)
