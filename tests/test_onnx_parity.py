"""
Unit tests for PyTorch PhaseSpaceMLP forward pass and ONNX export parity.
Verifies that the PyTorch in-graph standardization matches ONNX Runtime numerical outputs.
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import torch
import onnxruntime as ort
import pytest

from phasespace_onnx.models.mlp_estimator import PhaseSpaceMLP
from phasespace_onnx.export.onnx_exporter import export_to_onnx


def test_pytorch_to_onnx_numerical_parity() -> None:
    """Assert PyTorch forward pass matches ONNX Runtime within 1e-5 tolerance."""
    rng = np.random.default_rng(99)
    mean = rng.normal(50.0, 10.0, size=10).astype(np.float32)
    std = rng.uniform(5.0, 20.0, size=10).astype(np.float32)

    model = PhaseSpaceMLP(mean=mean, std=std, hidden_dims=[64, 32], dropout=0.0)
    model.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = os.path.join(tmp_dir, "test_model.onnx")
        export_to_onnx(model, onnx_file, opset_version=17)
        assert os.path.exists(onnx_file)

        # Generate diverse test evaluation inputs
        test_inputs = rng.normal(50.0, 25.0, size=(500, 10)).astype(np.float32)

        # 1. PyTorch forward evaluation
        with torch.no_grad():
            torch_out = model(torch.from_numpy(test_inputs)).numpy()

        # 2. ONNX Runtime evaluation
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(onnx_file, sess_opts, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        ort_out = sess.run([out_name], {in_name: test_inputs})[0]

        max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
        assert max_abs_diff < 1.0e-5, f"Discrepancy between PyTorch and ONNX Runtime: {max_abs_diff:.4e}"
