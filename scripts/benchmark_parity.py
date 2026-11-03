"""Measure trained-model inference paths and verify Python/C++ numerical parity."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Tuple

import h5py
import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from phasespace_onnx.evaluation.plotting import plot_latency_comparison
from phasespace_onnx.models.calibrator import HistogramCalibrator
from phasespace_onnx.models.mlp_estimator import PhaseSpaceMLP

WARMUP_PASSES = 5
TIMED_PASSES = 50


def _timed_configuration(
    operation: Callable[[], np.ndarray],
    n_events: int,
    batch_size: int,
    thread_count: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    for _ in range(WARMUP_PASSES):
        operation()
    timings_ms = []
    output = np.empty(n_events, dtype=np.float32)
    for _ in range(TIMED_PASSES):
        started = time.perf_counter()
        output = operation()
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(timings_ms, dtype=np.float64)
    mean_ms = float(np.mean(values))
    std_ms = float(np.std(values, ddof=1))
    per_event_us = mean_ms * 1000.0 / n_events
    return output, {
        "events_per_pass": int(n_events),
        "batch_size": int(batch_size),
        "thread_count": int(thread_count),
        "warmup_passes": WARMUP_PASSES,
        "timed_passes": TIMED_PASSES,
        "mean_batch_ms": mean_ms,
        "std_batch_ms": std_ms,
        "per_event_us": per_event_us,
        "std_per_event_us": std_ms * 1000.0 / n_events,
        "throughput_eps": n_events / (mean_ms / 1000.0),
    }


def _error_summary(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    absolute = np.abs(reference - candidate)
    relative = absolute / np.maximum(np.abs(reference), 1e-7)
    return {
        "max_abs_error": float(np.max(absolute)),
        "mean_abs_error": float(np.mean(absolute)),
        "max_rel_error": float(np.max(relative)),
        "mean_rel_error": float(np.mean(relative)),
    }


def _run_cpp(
    executable: str,
    model_path: str,
    lut_path: str,
    input_h5: str,
    output_h5: str,
    threads: int,
) -> Dict[str, float]:
    command = [
        executable,
        "--model", model_path,
        "--lut", lut_path,
        "--input-h5", input_h5,
        "--output-h5", output_h5,
        "--threads", str(threads),
        "--iterations", str(TIMED_PASSES),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    print(completed.stdout, end="")
    values: Dict[str, float] = {}
    for line in completed.stdout.splitlines():
        if line.startswith("CPP_") and "=" in line:
            key, value = line.split("=", 1)
            values[key.removeprefix("CPP_").lower()] = float(value)
    required = ("total_events", "mean_batch_ms", "std_batch_ms", "per_event_us", "throughput_events_per_sec")
    if any(name not in values for name in required):
        raise RuntimeError("C++ benchmark output did not contain all timing fields")
    return values


def main() -> None:
    print("=" * 75)
    print("PHASESPACE-ONNX: TRAINED MODEL PARITY AND LATENCY MEASUREMENTS")
    print("=" * 75)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    data_dir = os.path.join(PROJECT_ROOT, "data")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    assets_dir = os.path.join(PROJECT_ROOT, "assets")
    build_dir = os.path.join(PROJECT_ROOT, "build")
    test_h5 = os.path.join(data_dir, "test_kinematics.h5")
    checkpoint_path = os.path.join(models_dir, "phasespace_mlp.pt")
    onnx_path = os.path.join(models_dir, "phasespace_mlp.onnx")
    lut_bin_path = os.path.join(models_dir, "calibration_lut.bin")
    lut_json_path = os.path.join(models_dir, "calibration_lut.json")
    output_h5 = os.path.join(data_dir, "cpp_weights_out.h5")

    executable_candidates = (
        os.path.join(build_dir, "phasespace_infer_bench.exe"),
        os.path.join(build_dir, "Release", "phasespace_infer_bench.exe"),
        os.path.join(build_dir, "phasespace_infer_bench"),
    )
    executable = next((path for path in executable_candidates if os.path.isfile(path)), None)
    if executable is None:
        raise FileNotFoundError("Build phasespace_infer_bench before running this script")
    for path in (test_h5, checkpoint_path, onnx_path, lut_bin_path, lut_json_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required pipeline output is missing: {path}")

    with h5py.File(test_h5, "r") as sample:
        X_test = np.asarray(sample["kinematics/features"], dtype=np.float32)
        y_test = np.asarray(sample["kinematics/labels"], dtype=np.int32)
    n_events, n_features = X_test.shape
    if n_events != 20000:
        raise ValueError(f"Expected 20,000 held-out events, found {n_events:,}")

    model = PhaseSpaceMLP.load_checkpoint(checkpoint_path)
    with open(lut_json_path, "r", encoding="utf-8") as lut_file:
        lut_data = json.load(lut_file)
    calibrator = HistogramCalibrator(
        eps_clip=float(lut_data["eps_clip"]),
        weight_clip_max=float(lut_data["weight_clip_max"]),
        lut_grid_size=int(lut_data["num_points"]),
    )
    calibrator.grid_x = np.asarray(lut_data["grid_x"], dtype=np.float32)
    calibrator.grid_y = np.asarray(lut_data["grid_y"], dtype=np.float32)
    calibrator.is_fitted = True

    X_tensor = torch.as_tensor(X_test, dtype=torch.float32)

    def score_online() -> np.ndarray:
        weights = np.empty(n_events, dtype=np.float32)
        with torch.no_grad():
            for index in range(n_events):
                raw_score = float(model(X_tensor[index : index + 1]).item())
                weights[index] = calibrator.compute_likelihood_ratio(
                    np.asarray([raw_score], dtype=np.float32)
                )[0]
        return weights

    def score_microbatch() -> np.ndarray:
        weights = np.empty(n_events, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n_events, 256):
                stop = min(start + 256, n_events)
                raw_scores = model(X_tensor[start:stop]).cpu().numpy().reshape(-1)
                weights[start:stop] = calibrator.compute_likelihood_ratio(raw_scores)
        return weights

    def score_torch_vectorized() -> np.ndarray:
        with torch.no_grad():
            raw_scores = model(X_tensor).cpu().numpy().reshape(-1)
        return calibrator.compute_likelihood_ratio(raw_scores)

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, session_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def score_onnx_vectorized() -> np.ndarray:
        raw_scores = session.run([output_name], {input_name: X_test})[0].reshape(-1)
        return calibrator.compute_likelihood_ratio(raw_scores)

    print(f"Loaded trained checkpoint and {n_events:,} held-out events ({n_features} features).")
    print("Running five warmups and 50 timed passes per Python configuration...")
    weights_torch, torch_online = _timed_configuration(score_online, n_events, 1, 1)
    _, torch_microbatch = _timed_configuration(score_microbatch, n_events, 256, 1)
    weights_torch_batch, torch_batch = _timed_configuration(score_torch_vectorized, n_events, n_events, 1)
    weights_onnx, onnx_batch = _timed_configuration(score_onnx_vectorized, n_events, n_events, 1)

    print("Running the C++20 OpenMP driver over the same held-out sample...")
    cpp_values = _run_cpp(
        executable,
        onnx_path,
        lut_bin_path,
        test_h5,
        output_h5,
        threads=min(12, max(1, os.cpu_count() or 1)),
    )
    with h5py.File(output_h5, "r") as result_file:
        weights_cpp = np.asarray(result_file["weights"], dtype=np.float32)

    parity = {
        "pytorch_vs_onnx": _error_summary(weights_torch_batch, weights_onnx),
        "pytorch_vs_cpp": _error_summary(weights_torch_batch, weights_cpp),
        "onnx_vs_cpp": _error_summary(weights_onnx, weights_cpp),
    }
    max_abs_error = max(item["max_abs_error"] for item in parity.values())
    print("\nNumerical parity across trained-model execution paths:")
    for name, errors in parity.items():
        print(
            f"  {name:<18} max abs {errors['max_abs_error']:.4e} | "
            f"mean abs {errors['mean_abs_error']:.4e} | "
            f"max rel {errors['max_rel_error']:.4e} | "
            f"mean rel {errors['mean_rel_error']:.4e}"
        )
    if max_abs_error >= 1e-5:
        raise AssertionError(f"Numerical parity threshold failed: {max_abs_error:.4e} >= 1e-5")

    latency = {
        "python_online_event_loop": torch_online,
        "python_streaming_microbatch": torch_microbatch,
        "python_vectorized_batch": torch_batch,
        "python_onnxruntime_vectorized_batch": onnx_batch,
        "cpp_onnxruntime_openmp": {
            "events_per_pass": int(cpp_values["total_events"]),
            "batch_size": n_events,
            "thread_count": min(12, max(1, os.cpu_count() or 1)),
            "ort_intra_op_threads": 1,
            "ort_inter_op_threads": 1,
            "warmup_passes": WARMUP_PASSES,
            "timed_passes": TIMED_PASSES,
            "mean_batch_ms": cpp_values["mean_batch_ms"],
            "std_batch_ms": cpp_values["std_batch_ms"],
            "per_event_us": cpp_values["per_event_us"],
            "std_per_event_us": cpp_values["std_batch_ms"] * 1000.0 / n_events,
            "throughput_eps": cpp_values["throughput_events_per_sec"],
        },
    }
    cpp_latency = float(latency["cpp_onnxruntime_openmp"]["per_event_us"])
    speedups = {
        name: float(values["per_event_us"]) / max(cpp_latency, 1e-12)
        for name, values in latency.items()
        if name != "cpp_onnxruntime_openmp"
    }
    print("\nMean per-event latency and C++ speedup:")
    for name, values in latency.items():
        suffix = "" if name == "cpp_onnxruntime_openmp" else f" | C++ speedup {speedups[name]:.2f}x"
        print(
            f"  {name:<37} {values['per_event_us']:.4f} ± "
            f"{values['std_per_event_us']:.4f} µs/event{suffix}"
        )

    results = {
        "test_events": int(n_events),
        "test_features": int(n_features),
        "checkpoint": "models/phasespace_mlp.pt",
        "parity_threshold": 1e-5,
        "parity_verified": True,
        "parity": parity,
        "max_abs_error": max_abs_error,
        "mean_abs_error": parity["pytorch_vs_cpp"]["mean_abs_error"],
        "max_rel_error": max(errors["max_rel_error"] for errors in parity.values()),
        "mean_rel_error": parity["pytorch_vs_cpp"]["mean_rel_error"],
        "benchmark_protocol": {
            "warmup_passes": WARMUP_PASSES,
            "timed_passes": TIMED_PASSES,
            "python_torch_intra_op_threads": torch.get_num_threads(),
            "python_torch_inter_op_threads": torch.get_num_interop_threads(),
            "python_onnx_intra_op_threads": session_options.intra_op_num_threads,
            "python_onnx_inter_op_threads": session_options.inter_op_num_threads,
            "sample": "20,000 held-out events; scoring and LUT evaluation only",
        },
        "latency_configurations": latency,
        "speedup_cpp_over_python": speedups,
        "target_speedup_over_microbatch": speedups["python_streaming_microbatch"] >= 4.2,
    }
    result_path = os.path.join(assets_dir, "benchmark_results.json")
    with open(result_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    figure_path = os.path.join(assets_dir, "cpp_vs_python_latency.png")
    plot_latency_comparison(results, figure_path)
    print(f"Saved benchmark results: {result_path}")
    print(f"Saved latency figure: {figure_path}")

    if not results["target_speedup_over_microbatch"]:
        raise AssertionError(
            f"C++ speedup over 256-event Python chunks is {speedups['python_streaming_microbatch']:.2f}x; 4.2x is required"
        )


if __name__ == "__main__":
    main()
