"""
Statistical evaluation and closure metrics for likelihood ratio reweighting.
Computes weighted Kolmogorov-Smirnov statistics, binned chi2/ndof,
disentangled initial separation vs post-reweighting closure ROC AUC,
and Kish Effective Sample Size (ESS).
"""

from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def compute_weighted_cdf(values: np.ndarray, weights: np.ndarray, eval_points: np.ndarray) -> np.ndarray:
    """Compute empirical cumulative distribution function with sample weights."""
    sorter = np.argsort(values)
    v_sorted = values[sorter]
    w_sorted = weights[sorter]

    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1] if len(cum_w) > 0 and cum_w[-1] > 0 else 1.0
    norm_cum_w = cum_w / total_w

    idx = np.searchsorted(v_sorted, eval_points, side="right") - 1
    cdf = np.where(idx < 0, 0.0, norm_cum_w[np.clip(idx, 0, len(norm_cum_w) - 1)])
    return cdf


def compute_ks_test(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
) -> float:
    """
    Compute two-sample Kolmogorov-Smirnov statistic with optional weights.
    Returns maximum vertical distance between empirical CDFs in [0, 1].
    """
    if weights_a is None:
        weights_a = np.ones(len(sample_a), dtype=np.float64)
    if weights_b is None:
        weights_b = np.ones(len(sample_b), dtype=np.float64)

    all_vals = np.sort(np.unique(np.concatenate([sample_a, sample_b])))
    cdf_a = compute_weighted_cdf(sample_a, weights_a, all_vals)
    cdf_b = compute_weighted_cdf(sample_b, weights_b, all_vals)
    ks_stat = float(np.max(np.abs(cdf_a - cdf_b)))
    return ks_stat


def compute_chi2_ndof(
    sample_nom: np.ndarray,
    sample_var: np.ndarray,
    weights_nom: np.ndarray | None = None,
    n_bins: int = 30,
) -> float:
    """
    Compute binned chi-square / ndof between nominal (optionally reweighted) and variation.
    """
    min_v = min(np.min(sample_nom), np.min(sample_var))
    max_v = max(np.max(sample_nom), np.max(sample_var))
    bins = np.linspace(min_v, max_v, n_bins + 1)

    nominal_weights = np.ones(len(sample_nom), dtype=np.float64) if weights_nom is None else np.asarray(weights_nom, dtype=np.float64)
    variation_weights = np.ones(len(sample_var), dtype=np.float64)
    hist_nom, _ = np.histogram(sample_nom, bins=bins, weights=nominal_weights)
    hist_var, _ = np.histogram(sample_var, bins=bins, weights=variation_weights)
    sumw2_nom, _ = np.histogram(sample_nom, bins=bins, weights=nominal_weights ** 2)
    sumw2_var, _ = np.histogram(sample_var, bins=bins, weights=variation_weights ** 2)

    # Normalize both histograms to unit area
    sum_var = np.sum(hist_var)
    sum_nom = np.sum(hist_nom)
    if sum_var <= 0 or sum_nom <= 0:
        return 1.0

    p_var = hist_var / sum_var
    p_nom = hist_nom / sum_nom

    # Propagate each bin's sum of squared weights through unit-area normalization.
    variance = (sumw2_var / (sum_var ** 2)) + (sumw2_nom / (sum_nom ** 2))
    valid = variance > 1e-12

    chi2 = np.sum(((p_nom[valid] - p_var[valid]) ** 2) / variance[valid])
    ndof = max(int(np.sum(valid)) - 1, 1)
    return float(chi2 / ndof)


def compute_kish_effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute Kish Effective Sample Size (ESS) for weighted events:
    N_eff = (sum w)^2 / (sum w^2).
    """
    w = np.asarray(weights, dtype=np.float64)
    sum_w = np.sum(w)
    sum_w2 = np.sum(w ** 2)
    if sum_w2 <= 0:
        return 0.0
    return float((sum_w ** 2) / sum_w2)


def compute_closure_auc(
    X_nom: np.ndarray,
    X_var: np.ndarray,
    weights_nom: np.ndarray,
) -> float:
    """
    Compute post-reweighting adversarial closure AUC.
    Trains a secondary regularized linear classifier to distinguish reweighted nominal
    events (weight = r(x)) from unweighted systematic variation events (weight = 1.0).
    A successful closure reweighting drives this AUC to ~0.50 (statistically indistinguishable).
    """
    nom_fit_idx, nom_test_idx = train_test_split(
        np.arange(len(X_nom)), test_size=0.5, random_state=42, shuffle=True
    )
    var_fit_idx, var_test_idx = train_test_split(
        np.arange(len(X_var)), test_size=0.5, random_state=42, shuffle=True
    )
    weights_nom = np.asarray(weights_nom, dtype=np.float64)
    weights_fit = weights_nom[nom_fit_idx]
    weights_test = weights_nom[nom_test_idx]
    weights_fit *= len(nom_fit_idx) / max(np.sum(weights_fit), 1e-12)
    weights_test *= len(nom_test_idx) / max(np.sum(weights_test), 1e-12)

    X_fit = np.concatenate([X_nom[nom_fit_idx], X_var[var_fit_idx]], axis=0)
    y_fit = np.concatenate([
        np.zeros(len(nom_fit_idx), dtype=np.int32),
        np.ones(len(var_fit_idx), dtype=np.int32),
    ])
    sample_weight_fit = np.concatenate([weights_fit, np.ones(len(var_fit_idx), dtype=np.float64)])

    X_test = np.concatenate([X_nom[nom_test_idx], X_var[var_test_idx]], axis=0)
    y_test = np.concatenate([
        np.zeros(len(nom_test_idx), dtype=np.int32),
        np.ones(len(var_test_idx), dtype=np.int32),
    ])
    sample_weight_test = np.concatenate([weights_test, np.ones(len(var_test_idx), dtype=np.float64)])

    scaler = StandardScaler().fit(X_fit)
    X_fit_scaled = scaler.transform(X_fit)
    X_test_scaled = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=300, solver="lbfgs", random_state=42)
    clf.fit(X_fit_scaled, y_fit, sample_weight=sample_weight_fit)

    probs = clf.predict_proba(X_test_scaled)[:, 1]
    closure_auc = float(roc_auc_score(y_test, probs, sample_weight=sample_weight_test))
    return closure_auc


def compute_reweighting_metrics(
    X_nom: np.ndarray,
    X_var: np.ndarray,
    weights_nom: np.ndarray,
    scores_nom: np.ndarray,
    scores_var: np.ndarray,
    observable_names: List[str],
) -> Dict[str, Any]:
    """
    Compute comprehensive reweighting closure metrics across all observables.
    Disentangles initial separation ROC AUC from post-reweighting closure ROC AUC.
    """
    # 1. Initial Separation ROC AUC (baseline discriminator discrimination)
    y_true = np.concatenate([np.zeros(len(scores_nom)), np.ones(len(scores_var))])
    y_scores = np.concatenate([scores_nom, scores_var])
    initial_sep_auc = float(roc_auc_score(y_true, y_scores))

    # 2. Post-reweighting secondary closure AUC
    post_reweighting_closure_auc = compute_closure_auc(X_nom, X_var, weights_nom)

    # 3. Kish ESS
    n_nom = len(weights_nom)
    ess = compute_kish_effective_sample_size(weights_nom)
    ess_fraction = ess / max(n_nom, 1)

    ks_unweighted: Dict[str, float] = {}
    ks_reweighted: Dict[str, float] = {}
    chi2_unweighted: Dict[str, float] = {}
    chi2_reweighted: Dict[str, float] = {}

    for idx, name in enumerate(observable_names):
        col_nom = X_nom[:, idx]
        col_var = X_var[:, idx]

        ks_unw = compute_ks_test(col_nom, col_var)
        ks_rew = compute_ks_test(col_nom, col_var, weights_a=weights_nom)

        chi2_unw = compute_chi2_ndof(col_nom, col_var)
        chi2_rew = compute_chi2_ndof(col_nom, col_var, weights_nom=weights_nom)

        ks_unweighted[name] = ks_unw
        ks_reweighted[name] = ks_rew
        chi2_unweighted[name] = chi2_unw
        chi2_reweighted[name] = chi2_rew

    mean_ks_unw = float(np.mean(list(ks_unweighted.values())))
    mean_ks_rew = float(np.mean(list(ks_reweighted.values())))
    ks_reduction_pct = float(max(0.0, (1.0 - mean_ks_rew / max(mean_ks_unw, 1e-9)) * 100.0))

    mean_chi2_unw = float(np.mean(list(chi2_unweighted.values())))
    mean_chi2_rew = float(np.mean(list(chi2_reweighted.values())))

    return {
        "roc_auc": initial_sep_auc,
        "initial_separation_auc": initial_sep_auc,
        "post_reweighting_closure_auc": post_reweighting_closure_auc,
        "nominal_events": n_nom,
        "effective_sample_size": ess,
        "effective_sample_size_fraction": ess_fraction,
        "mean_ks_unweighted": mean_ks_unw,
        "mean_ks_reweighted": mean_ks_rew,
        "ks_reduction_percent": ks_reduction_pct,
        "mean_chi2_unweighted": mean_chi2_unw,
        "mean_chi2_reweighted": mean_chi2_rew,
        "per_observable_ks_unweighted": ks_unweighted,
        "per_observable_ks_reweighted": ks_reweighted,
        "per_observable_chi2_unweighted": chi2_unweighted,
        "per_observable_chi2_reweighted": chi2_reweighted,
    }
