"""
Unit and integration tests for columnar Awkward/Uproot scientific data ingestion.
Verifies zero-copy Awkward v2 jagged array slicing, generator streaming, and dual ROOT/HDF5 persistence.
"""

from __future__ import annotations

import os
import tempfile
import h5py
import numpy as np
import pytest

from phasespace_onnx.data.generator import generate_synthetic_phasespace, OBSERVABLE_NAMES
from phasespace_onnx.data.columnar_loader import ColumnarPhaseSpaceLoader


def test_synthetic_phasespace_generation_and_loading() -> None:
    """Verify synthetic dataset generation and roundtrip columnar loading."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = os.path.join(tmp_dir, "test_events.root")
        h5_path = os.path.join(tmp_dir, "test_events.h5")
        n_per_class = 2000

        X_gen, y_gen = generate_synthetic_phasespace(
            root_path=root_path,
            h5_path=h5_path,
            n_per_class=n_per_class,
            seed=123,
        )

        assert os.path.exists(root_path), "ROOT file was not created"
        assert os.path.exists(h5_path), "HDF5 file was not created"
        assert X_gen.shape == (2 * n_per_class, len(OBSERVABLE_NAMES))
        assert y_gen.shape == (2 * n_per_class,)

        # Test ROOT columnar ingestion with Awkward v2 slicing
        loader = ColumnarPhaseSpaceLoader(observable_names=OBSERVABLE_NAMES)
        X_root, y_root = loader.load_from_root(root_path, step_size="10 MB")

        assert X_root.shape == (2 * n_per_class, 10)
        assert y_root.shape == (2 * n_per_class,)
        assert np.all(np.isfinite(X_root)), "Found NaN or Inf in ROOT features"
        assert np.array_equal(y_gen, y_root), "Labels mismatch between generator and ROOT loader"

        # Check lead jet and sublead jet logic: lead_jet_pt >= sublead_jet_pt
        lead_jet = X_root[:, 5]
        sublead_jet = X_root[:, 6]
        assert np.all(lead_jet >= sublead_jet - 1e-5), "Lead jet pT must be >= sublead jet pT"

        # Test streaming generator interface
        streamed_x = []
        streamed_y = []
        for x_chunk, y_chunk in loader.stream_root_chunks(root_path, step_size="5 MB"):
            streamed_x.append(x_chunk)
            streamed_y.append(y_chunk)

        X_streamed = np.concatenate(streamed_x, axis=0)
        y_streamed = np.concatenate(streamed_y, axis=0)
        assert np.allclose(X_root, X_streamed, atol=1e-5)
        assert np.array_equal(y_root, y_streamed)

        # Test HDF5 bounded loader
        X_h5, y_h5 = loader.load_from_hdf5(h5_path, chunk_size=1000)
        assert X_h5.shape == (2 * n_per_class, 10)
        assert np.allclose(X_root, X_h5, atol=1e-5), "Discrepancy between ROOT and HDF5 ingested features"
        assert np.array_equal(y_root, y_h5)

        with h5py.File(h5_path, "r") as h5_file:
            group = h5_file["kinematics"]
            flat = group["jet_pt_flat"]
            offsets = group["jet_pt_offsets"]
            assert flat.ndim == 1
            assert flat.dtype == np.dtype(np.float32)
            assert offsets.ndim == 1
            assert offsets.dtype == np.dtype(np.int64)
            assert offsets.shape[0] == 2 * n_per_class + 1
            assert offsets[0] == 0
            assert offsets[-1] == flat.shape[0]
