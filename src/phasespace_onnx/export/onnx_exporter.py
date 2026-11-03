"""
ONNX export and integrity verification for PhaseSpaceMLP models.
Exports the complete PyTorch neural graph including embedded standardization buffers.
"""

from __future__ import annotations

import os
import warnings
import torch
import onnx
from ..models.mlp_estimator import PhaseSpaceMLP


def export_to_onnx(
    model: PhaseSpaceMLP,
    onnx_path: str,
    opset_version: int = 17,
) -> str:
    """
    Export trained PhaseSpaceMLP to ONNX format with dynamic batch axis.
    Validates model graph integrity using the ONNX model checker.

    Returns:
        Absolute path to the verified ONNX model file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    model.eval()
    model.to("cpu")

    dummy_x = torch.randn(1, 10, dtype=torch.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.filterwarnings(
            "ignore",
            message=r"You are using the legacy TorchScript-based ONNX exporter.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The feature will be removed\..*",
            category=DeprecationWarning,
        )
        torch.onnx.export(
            model,
            dummy_x,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["kinematics"],
            output_names=["raw_score"],
            dynamic_axes={
                "kinematics": {0: "batch_size"},
                "raw_score": {0: "batch_size"},
            },
            dynamo=False,
        )

    # Validate exported model structure
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    return os.path.abspath(onnx_path)
