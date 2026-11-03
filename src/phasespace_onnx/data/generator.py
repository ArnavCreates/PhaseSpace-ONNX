"""
Synthetic Monte Carlo phase-space simulation for scientific ML workflows.
Generates nominal (p0) and systematic variation (p1) kinematic samples,
writing dual ROOT TTree and HDF5 columnar representations.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Tuple
import numpy as np
import awkward as ak
import uproot
import h5py

OBSERVABLE_NAMES = [
    "lead_lepton_pt",
    "sublead_lepton_pt",
    "lead_lepton_eta",
    "invariant_mass_ll",
    "missing_et",
    "lead_jet_pt",
    "sublead_jet_pt",
    "delta_r_ll",
    "scalar_ht",
    "pt_balance_ratio",
]


def _simulate_hypothesis_events(
    n_events: int,
    is_variation: bool,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], ak.Array, ak.Array]:
    """
    Simulate correlated phase-space kinematics for a given hypothesis.
    Nominal: Standard electroweak / multijet phase space.
    Variation: Coherent +8% hardness shift on pT/HT, +5% on MET, angular tilt on delta_r_ll.
    """
    scale_hardness = 1.08 if is_variation else 1.00
    scale_met = 1.05 if is_variation else 1.00

    # 1. Lepton kinematics (e.g. Z -> l+ l- candidate)
    # Lead lepton pT: base gamma distribution shifted by hardness
    lead_lep_pt = (25.0 + rng.gamma(shape=3.0, scale=12.0, size=n_events) * scale_hardness).astype(np.float32)
    # Subleading lepton pT: correlated fraction (0.4 to 0.95) of leading lepton pT
    frac_sub = rng.uniform(0.40, 0.95, size=n_events)
    sublead_lep_pt = np.clip(lead_lep_pt * frac_sub, 15.0, 400.0).astype(np.float32)

    # Lead lepton pseudorapidity
    lead_lep_eta = np.clip(rng.normal(0.0, 1.25, size=n_events), -2.49, 2.49).astype(np.float32)
    sublead_lep_eta = np.clip(rng.normal(0.0, 1.30, size=n_events), -2.49, 2.49).astype(np.float32)

    # Invariant mass: Breit-Wigner Z peak (91.187 GeV, width 2.495 GeV) + exponential continuum
    is_peak = rng.random(size=n_events) < 0.85
    cauchy_mass = 91.187 + (2.495 / 2.0) * np.tan(np.pi * (rng.uniform(0.05, 0.95, size=n_events) - 0.5))
    continuum_mass = 50.0 + rng.exponential(scale=40.0, size=n_events)
    inv_mass = np.where(is_peak, cauchy_mass, continuum_mass)
    if is_variation:
        # Slight line-shape deformation in systematic variation
        inv_mass = inv_mass * (1.0 + 0.02 * rng.normal(0.0, 1.0, size=n_events))
    inv_mass = np.clip(inv_mass, 40.0, 200.0).astype(np.float32)

    # Lepton separation delta_r_ll
    if is_variation:
        delta_r_ll = np.clip(rng.beta(2.5, 2.0, size=n_events) * 3.5 + 0.4, 0.4, 4.0).astype(np.float32)
    else:
        delta_r_ll = np.clip(rng.beta(2.0, 2.0, size=n_events) * 3.5 + 0.4, 0.4, 4.0).astype(np.float32)

    # Missing transverse energy (MET)
    met = (rng.rayleigh(scale=32.0, size=n_events) * scale_met).astype(np.float32)

    # 2. Jagged Jets simulation (1 to 6 jets per event)
    # Number of jets per event follows Poisson(mean=2.6) truncated to [1, 6]
    n_jets_per_event = np.clip(rng.poisson(lam=2.6, size=n_events), 1, 6)
    total_jets = int(np.sum(n_jets_per_event))

    all_jet_pt = (20.0 + rng.exponential(scale=35.0, size=total_jets) * scale_hardness).astype(np.float32)
    all_jet_eta = np.clip(rng.normal(0.0, 1.8, size=total_jets), -4.4, 4.4).astype(np.float32)

    # Construct Awkward Array with unflattened offsets
    offsets = np.empty(n_events + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(n_jets_per_event, out=offsets[1:])
    jet_pt_ak = ak.unflatten(ak.Array(all_jet_pt), n_jets_per_event)
    jet_eta_ak = ak.unflatten(ak.Array(all_jet_eta), n_jets_per_event)

    # Sort jets descending by pT within each event
    sort_idx = ak.argsort(jet_pt_ak, axis=1, ascending=False)
    jet_pt_sorted = jet_pt_ak[sort_idx]
    jet_eta_sorted = jet_eta_ak[sort_idx]

    # Extract lead and sublead jet pT for event-level representation
    padded_jets = ak.fill_none(ak.pad_none(jet_pt_sorted, 2, axis=1), 0.0)
    lead_jet_pt = ak.to_numpy(padded_jets[:, 0]).astype(np.float32)
    sublead_jet_pt = ak.to_numpy(padded_jets[:, 1]).astype(np.float32)

    # 3. Global event variables
    # Sum of all jet pTs in event
    sum_jet_pt = ak.to_numpy(ak.sum(jet_pt_sorted, axis=1)).astype(np.float32)
    scalar_ht = (lead_lep_pt + sublead_lep_pt + sum_jet_pt + met).astype(np.float32)

    # Transverse momentum balance ratio
    pt_balance_ratio = ((lead_lep_pt + sublead_lep_pt) / (scalar_ht + 1e-5)).astype(np.float32)

    event_features = {
        "lead_lepton_pt": lead_lep_pt,
        "sublead_lepton_pt": sublead_lep_pt,
        "lead_lepton_eta": lead_lep_eta,
        "invariant_mass_ll": inv_mass,
        "missing_et": met,
        "lead_jet_pt": lead_jet_pt,
        "sublead_jet_pt": sublead_jet_pt,
        "delta_r_ll": delta_r_ll,
        "scalar_ht": scalar_ht,
        "pt_balance_ratio": pt_balance_ratio,
    }

    return event_features, jet_pt_sorted, jet_eta_sorted


def generate_synthetic_phasespace(
    root_path: str,
    h5_path: str,
    n_per_class: int = 50000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic Monte Carlo samples for nominal (y=0) and systematic (y=1) hypotheses,
    persisting to both ROOT TTree and HDF5 formats.

    Returns:
        (X, y): Complete dataset matrix (2*n_per_class, 10) and binary labels.
    """
    rng = np.random.default_rng(seed)

    # 1. Simulate Nominal (y = 0) and Variation (y = 1)
    feat_nom, jet_pt_nom, jet_eta_nom = _simulate_hypothesis_events(n_per_class, is_variation=False, rng=rng)
    feat_var, jet_pt_var, jet_eta_var = _simulate_hypothesis_events(n_per_class, is_variation=True, rng=rng)

    # 2. Combine and shuffle
    labels_nom = np.zeros(n_per_class, dtype=np.int32)
    labels_var = np.ones(n_per_class, dtype=np.int32)

    combined_features: Dict[str, np.ndarray] = {}
    for name in OBSERVABLE_NAMES:
        combined_features[name] = np.concatenate([feat_nom[name], feat_var[name]], axis=0)

    combined_labels = np.concatenate([labels_nom, labels_var], axis=0)
    combined_jet_pt = ak.concatenate([jet_pt_nom, jet_pt_var], axis=0)
    combined_jet_eta = ak.concatenate([jet_eta_nom, jet_eta_var], axis=0)

    # Random permutation across nominal and variation
    perm = rng.permutation(2 * n_per_class)
    for name in OBSERVABLE_NAMES:
        combined_features[name] = combined_features[name][perm]
    combined_labels = combined_labels[perm]
    combined_jet_pt = combined_jet_pt[perm]
    combined_jet_eta = combined_jet_eta[perm]

    # Build 2D feature matrix (2 * n_per_class, 10)
    feature_matrix = np.column_stack([combined_features[name] for name in OBSERVABLE_NAMES]).astype(np.float32)

    # 3. Write to ROOT file using uproot
    os.makedirs(os.path.dirname(os.path.abspath(root_path)), exist_ok=True)
    with uproot.recreate(root_path) as root_file:
        tree_dict: Dict[str, Any] = {
            name: combined_features[name] for name in OBSERVABLE_NAMES
        }
        tree_dict["jet_pt"] = combined_jet_pt
        tree_dict["jet_eta"] = combined_jet_eta
        tree_dict["is_variation"] = combined_labels
        root_file["PhaseSpaceTree"] = tree_dict

    # 4. Write to HDF5 file
    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    with h5py.File(h5_path, "w") as h5_file:
        grp = h5_file.create_group("kinematics")
        grp.create_dataset("features", data=feature_matrix, compression="gzip", compression_opts=4)
        grp.create_dataset("labels", data=combined_labels, compression="gzip", compression_opts=4)
        jet_counts = np.asarray(ak.num(combined_jet_pt, axis=1), dtype=np.int64)
        jet_offsets = np.empty(len(jet_counts) + 1, dtype=np.int64)
        jet_offsets[0] = 0
        np.cumsum(jet_counts, out=jet_offsets[1:])
        jet_pt_flat = np.asarray(ak.to_numpy(ak.flatten(combined_jet_pt, axis=1)), dtype=np.float32)
        grp.create_dataset("jet_pt_flat", data=jet_pt_flat, compression="gzip", compression_opts=4)
        grp.create_dataset("jet_pt_offsets", data=jet_offsets, compression="gzip", compression_opts=4)
        grp.attrs["observable_names"] = OBSERVABLE_NAMES
        grp.attrs["num_events"] = int(2 * n_per_class)
        grp.attrs["num_observables"] = len(OBSERVABLE_NAMES)

    return feature_matrix, combined_labels
