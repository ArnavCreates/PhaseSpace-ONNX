"""
Unit tests for histogram calibration mathematics, isotonic smoothing monotonicity,
likelihood ratio computation bounds, and binary LUT serialization roundtrip.
"""

from __future__ import annotations

import os
import struct
import tempfile
import numpy as np
import pytest

from phasespace_onnx.models.calibrator import HistogramCalibrator


def test_monotonic_calibration_and_lut_roundtrip() -> None:
    """Verify monotonically non-decreasing probability mapping and binary serialization."""
    rng = np.random.default_rng(42)
    n_samples = 10000

    # Simulate realistic uncalibrated sigmoid scores and labels
    scores_nom = rng.beta(2.0, 5.0, size=n_samples // 2)
    scores_var = rng.beta(5.0, 2.0, size=n_samples // 2)

    raw_scores = np.concatenate([scores_nom, scores_var])
    labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    calibrator = HistogramCalibrator(num_bins=40, eps_clip=1e-5, weight_clip_max=25.0, lut_grid_size=1001)
    calibrator.fit(raw_scores, labels)

    # 1. Monotonicity verification
    diffs = np.diff(calibrator.grid_y)
    assert np.all(diffs >= -1e-7), "Calibrated probabilities must be monotonically non-decreasing"

    # 2. Probability bounds verification
    assert np.all(calibrator.grid_y >= 1e-5), "Calibrated probabilities below eps_clip"
    assert np.all(calibrator.grid_y <= 1.0 - 1e-5), "Calibrated probabilities above 1 - eps_clip"

    # 3. Likelihood ratio computation & clipping
    test_scores = np.array([0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0], dtype=np.float32)
    weights = calibrator.compute_likelihood_ratio(test_scores)

    assert len(weights) == len(test_scores)
    assert np.all(weights >= 0.0), "Weights must be non-negative"
    assert np.all(weights <= 25.0), "Weights exceeded weight_clip_max"
    # Monotonicity of density ratio
    assert np.all(np.diff(weights) >= -1e-6), "Density ratio r(x) must increase with classifier score"

    # 4. Binary LUT roundtrip test
    with tempfile.TemporaryDirectory() as tmp_dir:
        bin_path = os.path.join(tmp_dir, "test_lut.bin")
        calibrator.export_lut_binary(bin_path)
        assert os.path.exists(bin_path)

        with open(bin_path, "rb") as f:
            header = f.read(12)
            num_points, eps, clip_max = struct.unpack("=Iff", header)
            raw_floats = f.read()

        read_grid_y = np.frombuffer(raw_floats, dtype=np.float32)
        assert num_points == 1001
        assert np.isclose(eps, 1e-5)
        assert np.isclose(clip_max, 25.0)
        assert np.allclose(read_grid_y, calibrator.grid_y, atol=1e-6)
