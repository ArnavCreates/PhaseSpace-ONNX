"""Generate a synthetic sample, train estimators, and evaluate reweighting closure."""

from __future__ import annotations

import json
import os
import sys
from typing import Callable, Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from phasespace_onnx.data.generator import OBSERVABLE_NAMES, generate_synthetic_phasespace
from phasespace_onnx.data.columnar_loader import ColumnarPhaseSpaceLoader
from phasespace_onnx.evaluation.metrics import compute_reweighting_metrics
from phasespace_onnx.evaluation.plotting import plot_reweighting_closure
from phasespace_onnx.export.onnx_exporter import export_to_onnx
from phasespace_onnx.models.calibrator import HistogramCalibrator
from phasespace_onnx.models.lgbm_estimator import LGBMReweighter
from phasespace_onnx.models.mlp_estimator import PhaseSpaceMLP

ChunkStream = Callable[[], Iterable[Tuple[np.ndarray, np.ndarray]]]


def _partition_stream(
    loader: ColumnarPhaseSpaceLoader,
    root_path: str,
    tree_name: str,
    step_size: str,
    partition: str,
    seed: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Assign events to fixed, class-balanced train, calibration, and test splits."""
    class_counts = [0, 0]
    for features, labels in loader.stream_root_chunks(root_path, tree_name, step_size):
        selected = np.zeros(len(labels), dtype=bool)
        for index, label_value in enumerate(labels):
            label = int(label_value)
            if label not in (0, 1):
                raise ValueError(f"Expected binary labels, received {label}")
            event_number = class_counts[label]
            class_counts[label] += 1
            fold = (event_number + seed + label * 2) % 5
            if partition == "calibration":
                selected[index] = fold == 0
            elif partition == "test":
                selected[index] = fold == 1
            elif partition == "train":
                selected[index] = fold >= 2
            else:
                raise ValueError(f"Unknown partition: {partition}")
        if np.any(selected):
            yield features[selected], labels[selected]


def _collect_partition(stream_fn: ChunkStream) -> Tuple[np.ndarray, np.ndarray]:
    feature_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    for features, labels in stream_fn():
        feature_chunks.append(features)
        label_chunks.append(labels)
    if not feature_chunks:
        return np.empty((0, len(OBSERVABLE_NAMES)), dtype=np.float32), np.empty(0, dtype=np.int32)
    return (
        np.ascontiguousarray(np.concatenate(feature_chunks, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(label_chunks, axis=0), dtype=np.int32),
    )


def main() -> None:
    print("=" * 75)
    print("PHASESPACE-ONNX: NEURAL LIKELIHOOD RATIO REWEIGHTING PIPELINE")
    print("=" * 75)

    config_path = os.path.join(PROJECT_ROOT, "configs", "reweight_config.yaml")
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    data_cfg = config["dataset"]
    n_per_class = int(data_cfg["num_events_per_hypothesis"])
    seed = int(data_cfg["random_seed"])
    data_dir = os.path.join(PROJECT_ROOT, "data")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    assets_dir = os.path.join(PROJECT_ROOT, "assets")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    root_path = os.path.join(data_dir, "mc_phasespace.root")
    h5_path = os.path.join(data_dir, "mc_phasespace.h5")
    print(f"\n[1/6] Generating {2 * n_per_class:,} synthetic Monte Carlo events...")
    generate_synthetic_phasespace(root_path=root_path, h5_path=h5_path, n_per_class=n_per_class, seed=seed)
    print(f"      ROOT sample: {root_path}")
    print(f"      HDF5 sample: {h5_path}")

    loader = ColumnarPhaseSpaceLoader(observable_names=OBSERVABLE_NAMES)
    tree_name = str(data_cfg["tree_name"])
    step_size = str(data_cfg["step_size"])
    split_seed = seed
    stream_fns: Dict[str, ChunkStream] = {
        name: (
            lambda name=name: _partition_stream(
                loader, root_path, tree_name, step_size, name, split_seed
            )
        )
        for name in ("train", "calibration", "test")
    }

    print("\n[2/6] Preparing calibration and held-out samples from ROOT chunks...")
    X_val, y_val = _collect_partition(stream_fns["calibration"])
    X_test, y_test = _collect_partition(stream_fns["test"])
    if X_val.shape[1] != len(OBSERVABLE_NAMES) or X_test.shape[1] != len(OBSERVABLE_NAMES):
        raise ValueError("The input feature matrix has an unexpected shape")
    if not np.all(np.isfinite(X_val)) or not np.all(np.isfinite(X_test)):
        raise ValueError("Non-finite values were found in the input features")
    print(
        f"      Calibration: {len(y_val):,} events | "
        f"Test: {len(y_test):,} events | Features: {len(OBSERVABLE_NAMES)}"
    )

    test_h5_path = os.path.join(data_dir, "test_kinematics.h5")
    with h5py.File(test_h5_path, "w") as output_file:
        group = output_file.create_group("kinematics")
        group.create_dataset("features", data=X_test, compression="gzip", compression_opts=4)
        group.create_dataset("labels", data=y_test, compression="gzip", compression_opts=4)
        group.attrs["num_events"] = int(len(X_test))
        group.attrs["num_features"] = int(X_test.shape[1])
    print(f"      Held-out sample: {test_h5_path}")

    print("\n[3/6] Training PhaseSpaceMLP from bounded ROOT chunks...")
    mlp_cfg = config.get("mlp", {})
    torch.manual_seed(seed)
    mlp_model = PhaseSpaceMLP(
        mean=np.zeros(len(OBSERVABLE_NAMES), dtype=np.float32),
        std=np.ones(len(OBSERVABLE_NAMES), dtype=np.float32),
        hidden_dims=list(mlp_cfg.get("hidden_dims", [128, 64, 32])),
        dropout=float(mlp_cfg.get("dropout", 0.0)),
    )
    mlp_model.fit_streaming(stream_fns["train"], X_val, y_val, config, seed=seed)
    checkpoint_path = os.path.join(models_dir, "phasespace_mlp.pt")
    mlp_model.save_checkpoint(checkpoint_path)
    print(f"      Checkpoint: {checkpoint_path}")

    print("\n[4/6] Training the LightGBM comparison model...")
    X_train, y_train = _collect_partition(stream_fns["train"])
    lgbm_model = LGBMReweighter(config).fit(X_train, y_train)
    print(f"      Training events: {len(y_train):,}")

    print("\n[5/6] Calibrating both models and exporting the MLP deployment files...")
    with torch.no_grad():
        raw_val_scores_mlp = mlp_model(torch.as_tensor(X_val, dtype=torch.float32)).numpy().reshape(-1)
    raw_val_scores_lgb = lgbm_model.predict_proba(X_val)
    calib_cfg = config.get("calibration", {})
    calibrator_mlp = HistogramCalibrator(
        num_bins=int(calib_cfg.get("num_bins", 50)),
        eps_clip=float(calib_cfg.get("eps_clip", 1e-5)),
        weight_clip_max=float(calib_cfg.get("weight_clip_max", 20.0)),
        lut_grid_size=int(calib_cfg.get("lut_grid_size", 1001)),
    ).fit(raw_val_scores_mlp, y_val)
    calibrator_lgb = HistogramCalibrator(
        num_bins=int(calib_cfg.get("num_bins", 50)),
        eps_clip=float(calib_cfg.get("eps_clip", 1e-5)),
        weight_clip_max=float(calib_cfg.get("weight_clip_max", 20.0)),
        lut_grid_size=int(calib_cfg.get("lut_grid_size", 1001)),
    ).fit(raw_val_scores_lgb, y_val)

    lut_bin_path = os.path.join(models_dir, "calibration_lut.bin")
    lut_json_path = os.path.join(models_dir, "calibration_lut.json")
    calibrator_mlp.export_lut_binary(lut_bin_path)
    calibrator_mlp.export_lut_json(lut_json_path)
    onnx_path = os.path.join(models_dir, "phasespace_mlp.onnx")
    export_to_onnx(
        mlp_model,
        onnx_path,
        opset_version=int(config.get("export", {}).get("onnx_opset", 17)),
    )
    print(f"      ONNX model: {onnx_path}")
    print(f"      Calibration table: {lut_bin_path}")

    print("\n[6/6] Evaluating held-out reweighting closure...")
    with torch.no_grad():
        raw_test_scores_mlp = mlp_model(torch.as_tensor(X_test, dtype=torch.float32)).numpy().reshape(-1)
    raw_test_scores_lgb = lgbm_model.predict_proba(X_test)
    weights_mlp = calibrator_mlp.compute_likelihood_ratio(raw_test_scores_mlp)
    weights_lgb = calibrator_lgb.compute_likelihood_ratio(raw_test_scores_lgb)
    mask_nom = y_test == 0
    mask_var = y_test == 1
    X_test_nom = X_test[mask_nom]
    X_test_var = X_test[mask_var]

    metrics_mlp = compute_reweighting_metrics(
        X_test_nom,
        X_test_var,
        weights_mlp[mask_nom],
        raw_test_scores_mlp[mask_nom],
        raw_test_scores_mlp[mask_var],
        OBSERVABLE_NAMES,
    )
    metrics_lgb = compute_reweighting_metrics(
        X_test_nom,
        X_test_var,
        weights_lgb[mask_nom],
        raw_test_scores_lgb[mask_nom],
        raw_test_scores_lgb[mask_var],
        OBSERVABLE_NAMES,
    )

    print("\nMetric                                | PhaseSpaceMLP | LightGBM")
    print("-------------------------------------|---------------|---------")
    for label, key, precision in (
        ("Initial separation ROC AUC", "initial_separation_auc", ".4f"),
        ("Held-out closure ROC AUC", "post_reweighting_closure_auc", ".4f"),
        ("Kish effective sample size", "effective_sample_size", ".1f"),
        ("Mean KS after reweighting", "mean_ks_reweighted", ".4f"),
        ("Mean chi-squared / ndof after reweighting", "mean_chi2_reweighted", ".4f"),
    ):
        print(f"{label:<37}| {format(metrics_mlp[key], precision):>13} | {format(metrics_lgb[key], precision):>8}")

    comp_results = {
        "phasespace_mlp": metrics_mlp,
        "lightgbm": metrics_lgb,
        "architectural_insights": {
            "mlp_cxx_deployable": True,
            "lgbm_cxx_deployable": False,
            "density_ratio_continuity": "The MLP score is continuous; LUT calibration is monotone and piecewise linear.",
        },
    }
    comp_json = os.path.join(assets_dir, "model_comparison.json")
    with open(comp_json, "w", encoding="utf-8") as output_file:
        json.dump(comp_results, output_file, indent=2)
    closure_png = os.path.join(assets_dir, "kinematic_reweighting_closure.png")
    plot_reweighting_closure(
        X_nom=X_test_nom,
        X_var=X_test_var,
        weights_nom=weights_mlp[mask_nom],
        observable_names=OBSERVABLE_NAMES,
        output_png=closure_png,
    )
    print(f"Saved model metrics: {comp_json}")
    print(f"Saved closure figure: {closure_png}")
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
