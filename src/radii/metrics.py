"""
RADII Metrics: Characterizing the Extrapolation Frontier of Graph Generative Models

This module provides metrics for mapping where and how generative models fail
as output structure size departs from the training distribution.

Metrics are organized into three tiers:
1. GENERATION QUALITY - Per-radius error measures that serve as the "thermometer"
   tracked across scales (RMSD, size error, local bond fidelity)
2. FAILURE DECOMPOSITION - Diagnostics revealing *where* in the structure errors
   concentrate (surface vs interior, coordination, orientation stability)
3. FRONTIER CHARACTERIZATION - Metrics that quantify the location and severity
   of the extrapolation frontier (smoothness, degradation ratio, frontier radius)

All metrics are O(N) or O(N log N) and remain memory-efficient up to ~2×10^4 atoms.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# =============================================================================
# UTILITIES
# =============================================================================


def center(pos: np.ndarray) -> np.ndarray:
    """
    Center atomic positions at the origin (centroid).

    Prerequisite for rotation-invariant comparisons. Removes translational
    degrees of freedom so comparisons focus on shape and arrangement.

    Args:
        pos: Atomic positions, shape (N, 3)

    Returns:
        Centered positions with mean at origin, shape (N, 3)

    Complexity: O(N)
    """
    return pos - pos.mean(axis=0)


def kabsch_align(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimal rotation alignment using the Kabsch algorithm.

    Finds the rotation matrix R that minimizes RMSD between pred and target,
    enabling comparison independent of absolute orientation.

    Args:
        pred: Predicted positions, shape (N, 3)
        target: Target positions, shape (N, 3)

    Returns:
        Tuple of:
            - aligned_pred: Predicted positions after optimal rotation, shape (N, 3)
            - R: The rotation matrix applied, shape (3, 3)

    Complexity: O(N) for matrix operations, O(1) for SVD (3x3 matrix)

    Reference:
        Kabsch, W. (1976). Acta Crystallographica, A32, 922-923.
    """
    pred_c = center(pred)
    target_c = center(target)

    H = pred_c.T @ target_c

    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return pred_c, np.eye(3)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return pred_c @ R, R


# =============================================================================
# 1. GENERATION QUALITY MEASURES
#    Evaluated at each radius independently to produce per-radius error profiles
#    that form the basis for frontier identification.
# =============================================================================


def rmsd(pred: np.ndarray, gt: np.ndarray, align: bool = True) -> float:
    """
    Root Mean Square Deviation between predicted and ground truth positions.

    The primary metric tracked across radii. When plotted against radius,
    RMSD profiles directly reveal where each model's scaling ceiling lies.

    Args:
        pred: Predicted positions, shape (N, 3)
        gt: Ground truth positions, shape (N, 3)
        align: Whether to apply Kabsch alignment first (default: True)

    Returns:
        RMSD value in same units as input (typically Angstroms)

    Complexity: O(N)

    Interpretation:
        - RMSD < 0.5 Å: Excellent generation
        - RMSD 0.5-1.0 Å: Good generation
        - RMSD 1.0-2.0 Å: Moderate errors
        - RMSD > 2.0 Å: Significant structural deviation
    """
    if align and len(pred) == len(gt):
        pred_aligned, _ = kabsch_align(pred, gt)
        gt_c = center(gt)
        return float(np.sqrt(((pred_aligned - gt_c) ** 2).mean()))
    return float(np.sqrt(((pred - gt) ** 2).mean()))


def radius_of_gyration(pos: np.ndarray) -> float:
    """
    Radius of gyration (Rg) — a measure of structure compactness/size.

    Rg = sqrt(mean(|r_i - centroid|²))

    Cheap O(N) alternative to convex hull volume. Captures overall spatial
    extent; degrades when models fail to scale morphology with radius.

    Args:
        pos: Atomic positions, shape (N, 3)

    Returns:
        Radius of gyration (typically Angstroms)

    Complexity: O(N)
    """
    c = center(pos)
    return float(np.sqrt((c**2).sum(axis=1).mean()))


def rg_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Relative error in radius of gyration.

    Checks whether the model produces structures of the correct overall size,
    independent of local atomic arrangement. Errors here indicate failure to
    understand scale — a direct signal of the extrapolation frontier.

    Args:
        pred: Predicted positions, shape (N, 3)
        gt: Ground truth positions, shape (N, 3)

    Returns:
        Relative error: |Rg_pred - Rg_gt| / Rg_gt (0 = perfect)

    Complexity: O(N)
    """
    rg_pred = radius_of_gyration(pred)
    rg_gt = radius_of_gyration(gt)
    return float(abs(rg_pred - rg_gt) / (rg_gt + 1e-8))


def bond_length_mae(pred: np.ndarray, gt: np.ndarray, k: int = 6) -> float:
    """
    Mean Absolute Error in k-nearest neighbor distances.

    Evaluates local structural fidelity by comparing bond-length distributions.
    By tracking BondMAE alongside RMSD across radii, we can distinguish models
    that lose *local* chemical order from those that maintain short-range
    structure but produce incorrect *global* morphology — a key diagnostic
    for characterizing *how* each architecture fails at the frontier.

    Args:
        pred: Predicted positions, shape (N, 3)
        gt: Ground truth positions, shape (N, 3)
        k: Number of nearest neighbors (default: 6)

    Returns:
        Mean absolute error in neighbor distances (Angstroms)

    Complexity: O(N log N) for KDTree construction and queries
    """
    k_use = min(k + 1, len(pred), len(gt))

    tree_pred = cKDTree(pred)
    tree_gt = cKDTree(gt)

    dist_pred, _ = tree_pred.query(pred, k=k_use)
    dist_gt, _ = tree_gt.query(gt, k=k_use)

    dist_pred = np.sort(dist_pred[:, 1:].flatten())
    dist_gt = np.sort(dist_gt[:, 1:].flatten())

    min_len = min(len(dist_pred), len(dist_gt))
    return float(np.abs(dist_pred[:min_len] - dist_gt[:min_len]).mean())


# =============================================================================
# 2. FAILURE DECOMPOSITION DIAGNOSTICS
#    When generation quality degrades, these identify *where* in the structure
#    the breakdown originates — surface vs interior, local coordination vs
#    global arrangement.
# =============================================================================


def surface_interior_ratio(
    pred: np.ndarray, gt: np.ndarray, surface_fraction: float = 0.25
) -> Dict[str, float]:
    """
    Compare prediction error on surface atoms vs interior atoms.

    Diagnoses WHERE models fail geometrically. Tracking this ratio across
    radii reveals whether extrapolation failures originate at the boundary
    (surface atoms lacking full coordination) or propagate from the bulk
    interior. A ratio that increases with radius indicates boundary-driven
    collapse; a stable ratio suggests uniform degradation.

    Args:
        pred: Predicted positions, shape (N, 3)
        gt: Ground truth positions, shape (N, 3)
        surface_fraction: Fraction of atoms as surface/interior (default: 0.25)

    Returns:
        Dictionary with surface_rmsd, interior_rmsd, surface_interior_ratio

    Complexity: O(N log N) for sorting

    Interpretation:
        - ratio ≈ 1.0: Errors uniformly distributed
        - ratio > 1.5: Errors concentrated on surfaces
        - ratio < 0.8: Errors concentrated in interior (unusual)
    """
    gt_c = center(gt)
    dists_from_center = np.linalg.norm(gt_c, axis=1)

    n = len(gt)
    n_subset = max(3, int(n * surface_fraction))

    sorted_idx = np.argsort(dists_from_center)
    interior_idx = sorted_idx[:n_subset]
    surface_idx = sorted_idx[-n_subset:]

    pred_aligned, _ = kabsch_align(pred, gt)
    gt_c = center(gt)

    surface_rmsd = np.sqrt(
        ((pred_aligned[surface_idx] - gt_c[surface_idx]) ** 2).mean()
    )
    interior_rmsd = np.sqrt(
        ((pred_aligned[interior_idx] - gt_c[interior_idx]) ** 2).mean()
    )

    return {
        "surface_rmsd": float(surface_rmsd),
        "interior_rmsd": float(interior_rmsd),
        "surface_interior_ratio": float(surface_rmsd / (interior_rmsd + 1e-8)),
    }


def coordination_preservation(
    pred: np.ndarray, gt: np.ndarray, cutoff: float = 3.0
) -> float:
    """
    Correlation between predicted and ground truth coordination numbers.

    Measures whether local chemical environments are preserved as size
    increases. A sharp drop in CoordCorr at a specific radius signals
    that the model has exceeded the scale at which it can maintain local
    structural rules — identifying the frontier for local order.

    High CoordCorr + high RMSD = "right local pieces, wrong global assembly."
    Low CoordCorr = fundamental local structure failure.

    Args:
        pred: Predicted positions, shape (N, 3)
        gt: Ground truth positions, shape (N, 3)
        cutoff: Distance cutoff for neighbor counting (default: 3.0 Å)

    Returns:
        Pearson correlation coefficient in [-1, 1]

    Complexity: O(N log N) for KDTree operations
    """
    tree_pred = cKDTree(pred)
    tree_gt = cKDTree(gt)

    coord_pred = np.array(
        [len(tree_pred.query_ball_point(p, cutoff)) - 1 for p in pred]
    )
    coord_gt = np.array([len(tree_gt.query_ball_point(g, cutoff)) - 1 for g in gt])

    if coord_gt.std() < 1e-8 or coord_pred.std() < 1e-8:
        return 1.0 if np.allclose(coord_pred, coord_gt) else 0.0

    corr, _ = pearsonr(coord_pred, coord_gt)
    return float(corr) if not np.isnan(corr) else 0.0


def rotation_consistency(
    predictions: List[np.ndarray], ground_truths: List[np.ndarray]
) -> Dict[str, float]:
    """
    Measure consistency of predictions for the SAME structure under different
    input orientations (orientation stability).

    For a model that produces consistent outputs regardless of input
    orientation, all aligned predictions should be identical (Δ_ij ≈ 0).
    When tracked across radii, increasing RotMean identifies the scale
    at which orientation sensitivity emerges — a secondary extrapolation
    frontier distinct from the primary quality frontier.

    Args:
        predictions: List of predicted positions for same structure,
                    different input orientations. Each shape (N, 3).
        ground_truths: Corresponding ground truths (for reference)

    Returns:
        Dictionary with:
            - rot_consistency_error: Std of pairwise RMSDs
            - rot_mean_pairwise_rmsd: Mean pairwise RMSD
            - rot_max_pairwise_rmsd: Worst-case inconsistency

    Complexity: O(K² × N) where K = number of orientations
    """
    if len(predictions) < 2:
        return {
            "rot_consistency_error": np.nan,
            "rot_mean_pairwise_rmsd": np.nan,
            "rot_max_pairwise_rmsd": np.nan,
        }

    reference = center(predictions[0])
    aligned_preds = [reference]

    for pred in predictions[1:]:
        aligned, _ = kabsch_align(pred, predictions[0])
        aligned_preds.append(aligned)

    pairwise_rmsds = []
    n = len(aligned_preds)
    for i in range(n):
        for j in range(i + 1, n):
            r = np.sqrt(((aligned_preds[i] - aligned_preds[j]) ** 2).mean())
            pairwise_rmsds.append(r)

    pairwise_rmsds = np.array(pairwise_rmsds)

    return {
        "rot_consistency_error": float(pairwise_rmsds.std()),
        "rot_mean_pairwise_rmsd": float(pairwise_rmsds.mean()),
        "rot_max_pairwise_rmsd": float(pairwise_rmsds.max()),
    }


# =============================================================================
# 3. FRONTIER CHARACTERIZATION METRICS
#    Operate on *per-radius error profiles* rather than individual structures,
#    directly quantifying the location and severity of the extrapolation frontier.
# =============================================================================


def scale_smoothness(
    metric_by_radius: Dict[float, float],
) -> Dict[str, float]:
    """
    Measure how smoothly a metric varies across scales.

    High Smooth values indicate monotonic degradation with size (gradual
    frontier); large Jump values reveal abrupt quality collapse at a
    specific radius (sharp frontier). Together, these distinguish
    architectures that degrade gracefully from those that fail
    catastrophically.

    Args:
        metric_by_radius: Dict mapping radius → metric value

    Returns:
        Dictionary with:
            - smoothness: Spearman correlation between radius and metric
            - jump_ratio: max_delta / mean_delta (abrupt transition detector)

    Complexity: O(R log R) where R = number of radii

    Interpretation:
        - smoothness > 0.8: Gradual, predictable degradation
        - smoothness < 0.5: Erratic behavior across scales
        - jump_ratio > 3.0: Abrupt quality collapse at specific radius
        - jump_ratio < 2.0: Relatively uniform transitions
    """
    radii = sorted(metric_by_radius.keys())
    values = [metric_by_radius[r] for r in radii]

    valid_pairs = [(r, v) for r, v in zip(radii, values) if not np.isnan(v)]
    if len(valid_pairs) < 3:
        return {"smoothness": np.nan, "jump_ratio": np.nan}

    radii_clean, values_clean = zip(*valid_pairs)
    radii_clean = np.array(radii_clean)
    values_clean = np.array(values_clean)

    corr, _ = spearmanr(radii_clean, values_clean)

    deltas = np.abs(np.diff(values_clean))
    if len(deltas) == 0 or deltas.mean() < 1e-8:
        jump_ratio = 1.0
    else:
        jump_ratio = deltas.max() / deltas.mean()

    return {
        "smoothness": float(corr) if not np.isnan(corr) else 0.0,
        "jump_ratio": float(jump_ratio),
    }


def degradation_ratio(
    metric_by_radius: Dict[float, float], id_radii: List[float], ood_radii: List[float]
) -> Dict[str, float]:
    """
    Quantify performance degradation from ID to OOD sizes.

    This single scalar summarizes the severity of the extrapolation frontier:
    values near 1 indicate robust scaling; values >> 1 indicate that the
    model's effective generation range does not extend beyond its training
    distribution.

    Args:
        metric_by_radius: Dict mapping radius → metric value
        id_radii: List of in-distribution radii
        ood_radii: List of out-of-distribution radii

    Returns:
        Dictionary with degradation_ratio, id_mean, ood_mean

    Complexity: O(R)

    Interpretation:
        - ratio ≈ 1.0: Robust generalization beyond training radii
        - ratio 1.5-2.0: Moderate degradation at the frontier
        - ratio > 2.0: Significant extrapolation failure
        - ratio > 5.0: Catastrophic frontier collapse
    """
    id_values = [metric_by_radius.get(r) for r in id_radii]
    ood_values = [metric_by_radius.get(r) for r in ood_radii]

    id_values = [v for v in id_values if v is not None and not np.isnan(v)]
    ood_values = [v for v in ood_values if v is not None and not np.isnan(v)]

    if not id_values or not ood_values:
        return {"degradation_ratio": np.nan, "id_mean": np.nan, "ood_mean": np.nan}

    id_mean = np.mean(id_values)
    ood_mean = np.mean(ood_values)

    return {
        "degradation_ratio": float(ood_mean / (id_mean + 1e-8)),
        "id_mean": float(id_mean),
        "ood_mean": float(ood_mean),
    }


def frontier_radius(
    metric_by_radius: Dict[float, float],
    threshold: float,
    lower_is_better: bool = True,
) -> Dict[str, float]:
    """
    Identify the extrapolation frontier: the largest radius at which
    generation quality remains acceptable.

    For a given error metric m and quality threshold τ:
        r*(m, τ) = max{r : m(r) ≤ τ}

    Comparing r* across models, materials, and metrics provides a compact
    summary of each architecture's scaling ceiling. When r* falls within
    the training range, the model fails to generalize even to interpolated
    scales; when r* extends into OOD radii, the model demonstrates genuine
    size extrapolation.

    Args:
        metric_by_radius: Dict mapping radius → metric value
        threshold: Quality threshold τ
        lower_is_better: If True (default for error metrics), frontier is
                        max r where metric ≤ threshold. If False (e.g.,
                        CoordCorr), frontier is max r where metric ≥ threshold.

    Returns:
        Dictionary with:
            - frontier_radius: Largest acceptable radius (nan if none qualify)
            - frontier_fraction: Fraction of radii within the frontier
            - beyond_frontier_mean: Mean metric value beyond the frontier

    Complexity: O(R log R)
    """
    radii = sorted(metric_by_radius.keys())
    values = [metric_by_radius[r] for r in radii]

    valid = [(r, v) for r, v in zip(radii, values) if not np.isnan(v)]
    if not valid:
        return {
            "frontier_radius": np.nan,
            "frontier_fraction": 0.0,
            "beyond_frontier_mean": np.nan,
        }

    if lower_is_better:
        acceptable = [r for r, v in valid if v <= threshold]
        beyond = [v for r, v in valid if v > threshold]
    else:
        acceptable = [r for r, v in valid if v >= threshold]
        beyond = [v for r, v in valid if v < threshold]

    fr = max(acceptable) if acceptable else np.nan
    frac = len(acceptable) / len(valid)
    beyond_mean = float(np.mean(beyond)) if beyond else np.nan

    return {
        "frontier_radius": float(fr) if not np.isnan(fr) else np.nan,
        "frontier_fraction": float(frac),
        "beyond_frontier_mean": float(beyond_mean),
    }


def orientation_degradation_ratio(
    consistency_by_radius: Dict[float, float],
    id_radii: List[float],
    ood_radii: List[float],
) -> Dict[str, float]:
    """
    Measure whether orientation stability transfers from trained to unseen scales.

    This is the degradation ratio applied specifically to orientation consistency
    error. Values significantly > 1 indicate that consistency achieved on
    training sizes does not generalize — a secondary frontier for orientation
    sensitivity that may differ from the primary quality frontier.

    Args:
        consistency_by_radius: Dict mapping radius → rot_consistency_error
        id_radii: In-distribution radii
        ood_radii: Out-of-distribution radii

    Returns:
        Dictionary with transfer_score, id_consistency_mean, ood_consistency_mean

    Complexity: O(R)

    Interpretation:
        - score ≈ 1.0: Orientation stability holds across scales
        - score 1.5-2.0: Partial degradation at larger sizes
        - score > 2.0: Orientation sensitivity emerges beyond training scales
    """
    id_values = [consistency_by_radius.get(r) for r in id_radii]
    ood_values = [consistency_by_radius.get(r) for r in ood_radii]

    id_values = [v for v in id_values if v is not None and not np.isnan(v)]
    ood_values = [v for v in ood_values if v is not None and not np.isnan(v)]

    if not id_values or not ood_values:
        return {
            "transfer_score": np.nan,
            "id_consistency_mean": np.nan,
            "ood_consistency_mean": np.nan,
        }

    id_mean = np.mean(id_values)
    ood_mean = np.mean(ood_values)

    return {
        "transfer_score": float(ood_mean / (id_mean + 1e-8)),
        "id_consistency_mean": float(id_mean),
        "ood_consistency_mean": float(ood_mean),
    }


# =============================================================================
# MAIN COMPUTE FUNCTIONS
# =============================================================================


def compute_sample_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    radius: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all per-sample metrics for a single prediction–ground truth pair.

    Called for every (prediction, ground_truth) pair during evaluation.
    All metrics are O(N) or O(N log N), safe for structures up to 20k atoms.
    Returns nan values for numeric metrics if inputs contain NaN/Inf.

    Args:
        pred: Predicted atomic positions, shape (N, 3)
        gt: Ground truth atomic positions, shape (N, 3)
        radius: Nanoparticle radius (optional, for per-radius profiling)

    Returns:
        Dictionary containing all per-sample metrics
    """
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(gt)):
        nan = float("nan")
        return {
            "rmsd": nan,
            "rg_error": nan,
            "bond_mae": nan,
            "surface_rmsd": nan,
            "interior_rmsd": nan,
            "surface_interior_ratio": nan,
            "coord_preservation": nan,
            "n_atoms": len(pred),
            "rg_pred": nan,
            "rg_gt": radius_of_gyration(gt) if np.all(np.isfinite(gt)) else nan,
            **({"radius": float(radius)} if radius is not None else {}),
        }
    metrics = {}

    # === Generation quality ===
    metrics["rmsd"] = rmsd(pred, gt, align=True)
    metrics["rg_error"] = rg_error(pred, gt)
    metrics["bond_mae"] = bond_length_mae(pred, gt)

    # === Failure decomposition ===
    si = surface_interior_ratio(pred, gt)
    metrics["surface_rmsd"] = si["surface_rmsd"]
    metrics["interior_rmsd"] = si["interior_rmsd"]
    metrics["surface_interior_ratio"] = si["surface_interior_ratio"]

    metrics["coord_preservation"] = coordination_preservation(pred, gt)

    # === Metadata ===
    metrics["n_atoms"] = len(pred)
    metrics["rg_pred"] = radius_of_gyration(pred)
    metrics["rg_gt"] = radius_of_gyration(gt)
    if radius is not None:
        metrics["radius"] = float(radius)

    return metrics


def compute_rotation_group_metrics(
    predictions: Dict[int, np.ndarray],
    ground_truths: Dict[int, np.ndarray],
) -> Dict[str, float]:
    """
    Compute orientation stability metrics for a group of predictions
    representing the same structure under different input orientations.

    Called once per (material, radius) group, NOT per sample.

    Args:
        predictions: Dict mapping rot_idx → predicted positions
        ground_truths: Dict mapping rot_idx → ground truth positions

    Returns:
        Dictionary containing orientation stability metrics
    """
    if len(predictions) < 2:
        return {}

    rot_indices = sorted(predictions.keys())
    pred_list = [predictions[i] for i in rot_indices]
    gt_list = [ground_truths[i] for i in rot_indices]

    return rotation_consistency(pred_list, gt_list)


def compute_aggregate_metrics(
    all_sample_metrics: List[Dict[str, float]],
    rotation_group_metrics: Dict[Tuple[str, float], Dict[str, float]],
    id_radii: List[float],
    ood_radii: List[float],
    frontier_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute aggregate metrics across all samples and rotation groups.

    Called once at the end of evaluation to produce summary statistics
    including frontier characterization.

    Args:
        all_sample_metrics: List of per-sample metric dictionaries
        rotation_group_metrics: Dict mapping (material, radius) → rotation metrics
        id_radii: In-distribution radii
        ood_radii: Out-of-distribution radii
        frontier_thresholds: Optional dict of metric_name → threshold for
                            frontier radius computation. Defaults provided
                            if None.

    Returns:
        Dictionary containing all aggregate metrics for reporting
    """
    if frontier_thresholds is None:
        frontier_thresholds = {
            "rmsd": 2.0,  # 2 Å threshold for acceptable generation
            "bond_mae": 0.5,  # 0.5 Å threshold for local structure
            "rg_error": 0.15,  # 15% size error threshold
        }

    agg = {}

    # === Group by radius ===
    by_radius = defaultdict(list)
    for m in all_sample_metrics:
        r = m.get("radius")
        if r is not None:
            by_radius[r].append(m)

    # === Mean metrics per radius (for frontier profiling) ===
    rmsd_by_radius = {
        r: np.mean([m["rmsd"] for m in ms]) for r, ms in by_radius.items()
    }
    bond_mae_by_radius = {
        r: np.mean([m["bond_mae"] for m in ms]) for r, ms in by_radius.items()
    }
    rg_error_by_radius = {
        r: np.mean([m["rg_error"] for m in ms]) for r, ms in by_radius.items()
    }
    surface_ratio_by_radius = {
        r: np.mean([m["surface_interior_ratio"] for m in ms])
        for r, ms in by_radius.items()
    }
    coord_by_radius = {
        r: np.mean([m["coord_preservation"] for m in ms]) for r, ms in by_radius.items()
    }

    # === Overall summary ===
    agg["rmsd_mean"] = np.mean([m["rmsd"] for m in all_sample_metrics])
    agg["rmsd_std"] = np.std([m["rmsd"] for m in all_sample_metrics])
    agg["rg_error_mean"] = np.mean([m["rg_error"] for m in all_sample_metrics])
    agg["bond_mae_mean"] = np.mean([m["bond_mae"] for m in all_sample_metrics])
    agg["surface_interior_ratio_mean"] = np.mean(
        [m["surface_interior_ratio"] for m in all_sample_metrics]
    )
    agg["coord_preservation_mean"] = np.mean(
        [m["coord_preservation"] for m in all_sample_metrics]
    )

    # === Per-split (ID vs OOD) for generation quality ===
    id_samples = [m for m in all_sample_metrics if m.get("split") == "id_test"]
    ood_samples = [m for m in all_sample_metrics if m.get("split") == "ood_test"]
    if id_samples:
        agg["id_rmsd_std"] = float(np.std([m["rmsd"] for m in id_samples]))
        agg["id_rg_error_mean"] = float(np.mean([m["rg_error"] for m in id_samples]))
        agg["id_bond_mae_mean"] = float(np.mean([m["bond_mae"] for m in id_samples]))
    else:
        agg["id_rmsd_std"] = agg["id_rg_error_mean"] = agg["id_bond_mae_mean"] = np.nan
    if ood_samples:
        agg["ood_rmsd_std"] = float(np.std([m["rmsd"] for m in ood_samples]))
        agg["ood_rg_error_mean"] = float(np.mean([m["rg_error"] for m in ood_samples]))
        agg["ood_bond_mae_mean"] = float(np.mean([m["bond_mae"] for m in ood_samples]))
    else:
        agg["ood_rmsd_std"] = agg["ood_rg_error_mean"] = agg["ood_bond_mae_mean"] = (
            np.nan
        )

    # === Frontier characterization: scale smoothness & jump detection ===
    smooth = scale_smoothness(rmsd_by_radius)
    agg["scale_smoothness"] = smooth["smoothness"]
    agg["scale_jump_ratio"] = smooth["jump_ratio"]

    # === Frontier characterization: ID-OOD degradation ratio ===
    degrad = degradation_ratio(rmsd_by_radius, id_radii, ood_radii)
    agg["degradation_ratio"] = degrad["degradation_ratio"]
    agg["id_rmsd_mean"] = degrad["id_mean"]
    agg["ood_rmsd_mean"] = degrad["ood_mean"]

    # === Frontier characterization: frontier radius per metric ===
    for metric_name, by_rad, threshold, lower in [
        ("rmsd", rmsd_by_radius, frontier_thresholds.get("rmsd", 2.0), True),
        (
            "bond_mae",
            bond_mae_by_radius,
            frontier_thresholds.get("bond_mae", 0.5),
            True,
        ),
        (
            "rg_error",
            rg_error_by_radius,
            frontier_thresholds.get("rg_error", 0.15),
            True,
        ),
    ]:
        fr = frontier_radius(by_rad, threshold, lower_is_better=lower)
        agg[f"frontier_radius_{metric_name}"] = fr["frontier_radius"]
        agg[f"frontier_fraction_{metric_name}"] = fr["frontier_fraction"]
        agg[f"beyond_frontier_mean_{metric_name}"] = fr["beyond_frontier_mean"]

    # === Failure decomposition: surface/interior degradation ===
    surface_degrad = degradation_ratio(surface_ratio_by_radius, id_radii, ood_radii)
    agg["surface_ratio_degradation"] = surface_degrad["degradation_ratio"]
    agg["id_surface_ratio_mean"] = surface_degrad["id_mean"]
    agg["ood_surface_ratio_mean"] = surface_degrad["ood_mean"]

    # === Failure decomposition: coordination degradation ===
    coord_degrad = degradation_ratio(
        {r: 1.0 - v for r, v in coord_by_radius.items()},
        id_radii,
        ood_radii,
    )
    agg["coord_degradation"] = coord_degrad["degradation_ratio"]
    agg["id_coord_mean"] = (
        1.0 - coord_degrad["id_mean"] if coord_degrad["id_mean"] else np.nan
    )
    agg["ood_coord_mean"] = (
        1.0 - coord_degrad["ood_mean"] if coord_degrad["ood_mean"] else np.nan
    )

    # === Orientation stability aggregates ===
    if rotation_group_metrics:
        consistency_errors = [
            m.get("rot_consistency_error", np.nan)
            for m in rotation_group_metrics.values()
        ]
        consistency_errors = [v for v in consistency_errors if not np.isnan(v)]

        if consistency_errors:
            agg["rot_consistency_mean"] = np.mean(consistency_errors)
            agg["rot_consistency_std"] = np.std(consistency_errors)

            consistency_by_radius = defaultdict(list)
            for (material, radius), m in rotation_group_metrics.items():
                err = m.get("rot_consistency_error")
                if err is not None and not np.isnan(err):
                    consistency_by_radius[radius].append(err)

            consistency_by_radius = {
                r: np.mean(vs) for r, vs in consistency_by_radius.items()
            }

            transfer = orientation_degradation_ratio(
                consistency_by_radius, id_radii, ood_radii
            )
            agg["orientation_degradation_score"] = transfer["transfer_score"]
            agg["id_consistency_mean"] = transfer["id_consistency_mean"]
            agg["ood_consistency_mean"] = transfer["ood_consistency_mean"]

    return agg
