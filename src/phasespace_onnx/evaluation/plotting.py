"""
High-resolution visualization generation for kinematic reweighting closure
and C++20 vs Python inference latency benchmarks.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_reweighting_closure(
    X_nom: np.ndarray,
    X_var: np.ndarray,
    weights_nom: np.ndarray,
    observable_names: List[str],
    output_png: str,
) -> None:
    """
    Generate publication-quality multi-panel kinematic reweighting closure plot
    with ratio subplots demonstrating convergence of reweighted nominal to variation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)

    # Select 6 primary observables for 3x2 grid
    plot_indices = [0, 3, 4, 5, 7, 8]  # lead_lep_pt, inv_mass_ll, met, lead_jet_pt, delta_r_ll, scalar_ht
    selected_indices = [i for i in plot_indices if i < len(observable_names)]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    colors = {
        "nom": "#2b5c8f",      # Dark Blue
        "var": "#d95f02",      # Vermillion / Orange
        "rew": "#1b9e77",      # Teal / Green
    }

    for plot_idx, feat_idx in enumerate(selected_indices):
        name = observable_names[feat_idx]
        ax = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])

        v_nom = X_nom[:, feat_idx]
        v_var = X_var[:, feat_idx]

        # Robust percentile binning
        q_low = float(min(np.percentile(v_nom, 0.5), np.percentile(v_var, 0.5)))
        q_high = float(max(np.percentile(v_nom, 99.5), np.percentile(v_var, 99.5)))
        bins = np.linspace(q_low, q_high, 40)

        # Compute histograms normalized to density
        h_nom, _ = np.histogram(v_nom, bins=bins, density=True)
        h_var, _ = np.histogram(v_var, bins=bins, density=True)
        h_rew, _ = np.histogram(v_nom, bins=bins, weights=weights_nom, density=True)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        ax.step(bin_centers, h_nom, where="mid", color=colors["nom"], label="Nominal (p0, Unweighted)", linestyle="--", linewidth=1.8)
        ax.step(bin_centers, h_var, where="mid", color=colors["var"], label="Target Variation (p1)", linewidth=2.0)
        ax.step(bin_centers, h_rew, where="mid", color=colors["rew"], label="Nominal Reweighted (p0 * r(x))", linewidth=2.2)

        ax.set_title(f"Observable: {name}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{name} [GeV / rad]", fontsize=10)
        ax.set_ylabel("Normalized Density", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        if plot_idx == 0:
            ax.legend(frameon=True, fontsize=8.5, loc="upper right")

    plt.suptitle("PhaseSpace-ONNX: Neural Likelihood-Ratio Reweighting Closure", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_latency_comparison(
    benchmarks: Dict[str, Any],
    output_png: str,
) -> None:
    """Plot per-event latency and throughput for the measured execution paths."""
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    configs = benchmarks.get("latency_configurations", {})
    order = (
        ("python_online_event_loop", "Python event loop\n(batch 1)"),
        ("python_streaming_microbatch", "Python chunks\n(batch 256)"),
        ("python_vectorized_batch", "PyTorch vectorized\n(batch 20k)"),
        ("python_onnxruntime_vectorized_batch", "Python ONNX Runtime\n(batch 20k)"),
        ("cpp_onnxruntime_openmp", "C++ ONNX Runtime\n(OpenMP)"),
    )
    selected = [(key, label, configs[key]) for key, label in order if key in configs]
    labels = [label for _, label, _ in selected]
    per_event_us = [float(item["per_event_us"]) for _, _, item in selected]
    throughput_eps = [float(item["throughput_eps"]) for _, _, item in selected]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    bar_colors = ["#e41a1c", "#984ea3", "#7570b3", "#386cb0", "#2ca02c"][:len(labels)]
    bars1 = ax1.bar(labels, per_event_us, color=bar_colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("Per-Event Latency (µs / event)", fontsize=11, fontweight="bold")
    ax1.set_title("Inference Latency (Lower is Better)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.7)

    for bar, val in zip(bars1, per_event_us):
        height = bar.get_height()
        ax1.annotate(f"{val:.2f} µs",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # Throughput Plot (higher is better)
    bars2 = ax2.bar(labels, throughput_eps, color=bar_colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax2.set_ylabel("Throughput (events / second)", fontsize=11, fontweight="bold")
    ax2.set_title("Throughput Scalability (Higher is Better)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.7)

    for bar, val in zip(bars2, throughput_eps):
        height = bar.get_height()
        ax2.annotate(f"{val:,.0f} ev/s",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    fig.suptitle("PhaseSpace-ONNX inference latency and throughput", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
