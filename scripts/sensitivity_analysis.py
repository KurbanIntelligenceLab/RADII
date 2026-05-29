"""
Sensitivity analysis for RADII metric design choices.

Addresses reviewer concerns about parameter sensitivity:
- Reviewer 4ixz W1/D1: CoordCorr cutoff (r_c) sweep and SurfIntRatio shell fraction sweep
- Reviewer XQdQ Point 4: Per-atom local environment comparison (new metric)

Reads existing NPZ predictions and GT cache — no model retraining needed.

Usage:
    conda run -n aclWork python -m scripts.sensitivity_analysis
    conda run -n aclWork python -m scripts.sensitivity_analysis --results-dir results --out-dir results/sensitivity --n-jobs 8
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from radii.metrics import coordination_preservation, surface_interior_ratio
from radii.train_config import TrainConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ID_RADII = set(TrainConfig.ID_RADII)
OOD_RADII = set(TrainConfig.OOD_RADII)
MODELS = ["adit", "cdvae", "diffcsp", "flowmm", "mattergen"]
SEEDS = TrainConfig.SEEDS

CUTOFF_VALUES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
FRACTION_VALUES = [0.15, 0.20, 0.25, 0.30]
DEFAULT_CUTOFF = 3.0
DEFAULT_FRACTION = 0.25


# ---------------------------------------------------------------------------
# Data loading (lightweight, no dependency on compute_metrics_from_predictions)
# ---------------------------------------------------------------------------
def load_npz_lookup(
    npz_path: str, pos_key: str = "pred_pos"
) -> Dict[Tuple[str, float, int], np.ndarray]:
    """Load an NPZ file into a (material, radius, rot_idx) -> positions dict."""
    data = np.load(npz_path, allow_pickle=True)
    ptr = data["ptr"]
    positions = data[pos_key]
    materials = data["materials"]
    radius = data["radius"]
    rot_idx = data["rot_idx"]
    lookup = {}
    for i in range(len(ptr) - 1):
        s, e = int(ptr[i]), int(ptr[i + 1])
        key = (str(materials[i]), float(radius[i]), int(rot_idx[i]))
        lookup[key] = positions[s:e].copy()
    return lookup


def load_gt_cache(data_root: str) -> Dict[str, Dict[Tuple[str, float, int], np.ndarray]]:
    """Load ground truth cache for id_test and ood_test splits."""
    gt = {}
    for split in ("id_test", "ood_test"):
        path = os.path.join(data_root, "gt_cache", f"{split}.npz")
        gt[split] = load_npz_lookup(path, pos_key="gt_pos")
        print(f"  GT {split}: {len(gt[split])} samples")
    return gt


def discover_runs(results_dir: str) -> List[Tuple[str, int, str]]:
    """Find all (model, seed, run_dir) tuples with eval_predictions."""
    runs = []
    task_dir = os.path.join(results_dir, "task_1")
    for model in MODELS:
        for seed in SEEDS:
            pred_dir = os.path.join(task_dir, model, str(seed), "eval_predictions")
            if os.path.isdir(pred_dir):
                runs.append((model, seed, pred_dir))
    return runs


def load_all_samples(
    runs: List[Tuple[str, int, str]],
    gt: Dict[str, Dict[Tuple[str, float, int], np.ndarray]],
) -> List[Dict]:
    """Load all (pred, gt) pairs across all runs, matched by key."""
    samples = []
    for model, seed, pred_dir in runs:
        for split in ("id_test", "ood_test"):
            npz_path = os.path.join(pred_dir, f"{split}.npz")
            if not os.path.isfile(npz_path):
                continue
            preds = load_npz_lookup(npz_path, pos_key="pred_pos")
            gt_split = gt[split]
            matched = 0
            for key, pred_pos in preds.items():
                gt_pos = gt_split.get(key)
                if gt_pos is None:
                    continue
                material, radius, rot_idx = key
                samples.append({
                    "model": model,
                    "seed": seed,
                    "split": split,
                    "material": material,
                    "radius": radius,
                    "rot_idx": rot_idx,
                    "pred": pred_pos,
                    "gt": gt_pos,
                })
                matched += 1
            print(f"  {model}/seed{seed}/{split}: {matched} matched samples")
    return samples


# ---------------------------------------------------------------------------
# Per-atom BondMAE (new metric — Reviewer XQdQ Point 4)
# ---------------------------------------------------------------------------
def per_atom_bond_mae(pred: np.ndarray, gt: np.ndarray, k: int = 6) -> float:
    """Per-atom kNN distance MAE — compares each atom's local environment individually.

    Unlike the global bond_length_mae which sorts all kNN distances before comparing
    (conflating distinct local environments), this compares each atom's k-nearest
    neighbor distances independently.
    """
    k_use = min(k + 1, len(pred), len(gt))
    tree_pred = cKDTree(pred)
    tree_gt = cKDTree(gt)
    dist_pred, _ = tree_pred.query(pred, k=k_use)
    dist_gt, _ = tree_gt.query(gt, k=k_use)
    d_pred = np.sort(dist_pred[:, 1:], axis=1)
    d_gt = np.sort(dist_gt[:, 1:], axis=1)
    per_atom = np.abs(d_pred - d_gt).mean(axis=1)
    return float(per_atom.mean())


# ---------------------------------------------------------------------------
# Worker functions for parallel execution
# ---------------------------------------------------------------------------
def _worker_coord(args):
    """Compute coordination_preservation at a given cutoff for one sample."""
    pred, gt, cutoff = args
    return coordination_preservation(pred, gt, cutoff=cutoff)


def _worker_surfint(args):
    """Compute surface_interior_ratio at a given fraction for one sample."""
    pred, gt, frac = args
    result = surface_interior_ratio(pred, gt, surface_fraction=frac)
    return result["surface_interior_ratio"]


def _worker_per_atom(args):
    """Compute per_atom_bond_mae for one sample."""
    pred, gt = args
    return per_atom_bond_mae(pred, gt)


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------
def run_coord_sweep(
    samples: List[Dict], n_jobs: int
) -> pd.DataFrame:
    """Sweep coordination_preservation across cutoff values."""
    records = []
    for cutoff in CUTOFF_VALUES:
        print(f"  CoordCorr cutoff={cutoff:.1f} A ...")
        tasks = [(s["pred"], s["gt"], cutoff) for s in samples]
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                values = list(pool.map(_worker_coord, tasks, chunksize=64))
        else:
            values = [_worker_coord(t) for t in tasks]
        for s, val in zip(samples, values):
            records.append({
                "cutoff": cutoff,
                "model": s["model"],
                "seed": s["seed"],
                "split": s["split"],
                "material": s["material"],
                "radius": s["radius"],
                "coord_preservation": val,
            })
    return pd.DataFrame(records)


def run_surfint_sweep(
    samples: List[Dict], n_jobs: int
) -> pd.DataFrame:
    """Sweep surface_interior_ratio across fraction values."""
    records = []
    for frac in FRACTION_VALUES:
        print(f"  SurfIntRatio fraction={frac:.2f} ...")
        tasks = [(s["pred"], s["gt"], frac) for s in samples]
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                values = list(pool.map(_worker_surfint, tasks, chunksize=64))
        else:
            values = [_worker_surfint(t) for t in tasks]
        for s, val in zip(samples, values):
            records.append({
                "fraction": frac,
                "model": s["model"],
                "seed": s["seed"],
                "split": s["split"],
                "material": s["material"],
                "radius": s["radius"],
                "surface_interior_ratio": val,
            })
    return pd.DataFrame(records)


def run_per_atom_bond_mae(
    samples: List[Dict], n_jobs: int
) -> pd.DataFrame:
    """Compute per-atom BondMAE for all samples."""
    print("  Per-atom BondMAE ...")
    tasks = [(s["pred"], s["gt"]) for s in samples]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            values = list(pool.map(_worker_per_atom, tasks, chunksize=64))
    else:
        values = [_worker_per_atom(t) for t in tasks]
    records = []
    for s, val in zip(samples, values):
        records.append({
            "model": s["model"],
            "seed": s["seed"],
            "split": s["split"],
            "material": s["material"],
            "radius": s["radius"],
            "per_atom_bond_mae": val,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregation and table building
# ---------------------------------------------------------------------------
def _split_label(split: str) -> str:
    return "ID" if split == "id_test" else "OOD"


def build_coord_tables(df: pd.DataFrame) -> str:
    """Build markdown tables for CoordCorr sensitivity."""
    lines = []

    # --- Table 1: Overall ---
    lines.append("## Table S1: CoordCorr Sensitivity Analysis (Overall)")
    lines.append("**Addresses:** Reviewer 4ixz W1/D1 — CoordCorr cutoff robustness\n")

    # Average across seeds: group by (cutoff, model, seed, split) first, then across seeds
    seed_means = df.groupby(["cutoff", "model", "seed", "split"])["coord_preservation"].mean().reset_index()
    model_agg = seed_means.groupby(["cutoff", "model", "split"])["coord_preservation"].agg(["mean", "std"]).reset_index()

    # Pivot for table
    header = "| Cutoff (A) |"
    sep = "|------------|"
    for model in MODELS:
        header += f" {model} ID | {model} OOD | {model} Degrad |"
        sep += "----------|----------|----------|"
    lines.append(header)
    lines.append(sep)

    for cutoff in CUTOFF_VALUES:
        row = f"| {cutoff:.1f} |"
        for model in MODELS:
            id_rows = model_agg[(model_agg["cutoff"] == cutoff) & (model_agg["model"] == model) & (model_agg["split"] == "id_test")]
            ood_rows = model_agg[(model_agg["cutoff"] == cutoff) & (model_agg["model"] == model) & (model_agg["split"] == "ood_test")]
            id_mean = id_rows["mean"].values[0] if len(id_rows) else float("nan")
            id_std = id_rows["std"].values[0] if len(id_rows) else float("nan")
            ood_mean = ood_rows["mean"].values[0] if len(ood_rows) else float("nan")
            ood_std = ood_rows["std"].values[0] if len(ood_rows) else float("nan")
            degrad = ood_mean / (id_mean + 1e-8) if id_mean != 0 else float("nan")
            row += f" {id_mean:.4f}+/-{id_std:.4f} | {ood_mean:.4f}+/-{ood_std:.4f} | {degrad:.3f} |"
        lines.append(row)
    lines.append("")

    # --- Table 2: Per-material ---
    lines.append("## Table S2: CoordCorr Sensitivity (Per-Material)")
    lines.append("**Addresses:** Reviewer 4ixz W1/D1 — CoordCorr cutoff robustness across materials\n")
    materials = sorted(df["material"].unique())

    for split in ("id_test", "ood_test"):
        lines.append(f"### {_split_label(split)} Split\n")
        header = "| Cutoff (A) |"
        sep = "|------------|"
        for mat in materials:
            header += f" {mat} |"
            sep += "--------|"
        lines.append(header)
        lines.append(sep)

        mat_agg = df[df["split"] == split].groupby(["cutoff", "material"])["coord_preservation"].mean().reset_index()
        for cutoff in CUTOFF_VALUES:
            row = f"| {cutoff:.1f} |"
            for mat in materials:
                val = mat_agg[(mat_agg["cutoff"] == cutoff) & (mat_agg["material"] == mat)]["coord_preservation"].values
                row += f" {val[0]:.4f} |" if len(val) else " N/A |"
            lines.append(row)
        lines.append("")

    # --- Table 3: Ranking stability ---
    lines.append("## Table S3: CoordCorr Ranking Stability")
    lines.append("**Addresses:** Reviewer 4ixz W1/D1 — Cross-architecture rankings stable across cutoffs\n")
    lines.append("Spearman rho of model rankings at each cutoff vs. default (3.0 A).\n")

    # Get model rankings at each cutoff (by mean coord_preservation across seeds and samples)
    overall = df.groupby(["cutoff", "model"])["coord_preservation"].mean().reset_index()
    default_ranking = overall[overall["cutoff"] == DEFAULT_CUTOFF].sort_values("coord_preservation", ascending=False)["model"].tolist()
    default_rank = {m: i for i, m in enumerate(default_ranking)}

    lines.append("| Cutoff (A) | Spearman rho | Model ranking |")
    lines.append("|------------|-------------|---------------|")
    for cutoff in CUTOFF_VALUES:
        sub = overall[overall["cutoff"] == cutoff].sort_values("coord_preservation", ascending=False)
        ranking = sub["model"].tolist()
        rank_vals = [default_rank.get(m, len(MODELS)) for m in ranking]
        ref_vals = list(range(len(MODELS)))
        if len(rank_vals) >= 2:
            rho, _ = spearmanr(ref_vals, rank_vals)
        else:
            rho = float("nan")
        lines.append(f"| {cutoff:.1f} | {rho:.4f} | {' > '.join(ranking)} |")
    lines.append("")

    return "\n".join(lines)


def build_surfint_tables(df: pd.DataFrame) -> str:
    """Build markdown tables for SurfIntRatio sensitivity."""
    lines = []

    # --- Table 4: Overall ---
    lines.append("## Table S4: SurfIntRatio Sensitivity Analysis (Overall)")
    lines.append("**Addresses:** Reviewer 4ixz W1/D1 — SurfIntRatio shell fraction robustness\n")

    seed_means = df.groupby(["fraction", "model", "seed", "split"])["surface_interior_ratio"].mean().reset_index()
    model_agg = seed_means.groupby(["fraction", "model", "split"])["surface_interior_ratio"].agg(["mean", "std"]).reset_index()

    header = "| Fraction |"
    sep = "|----------|"
    for model in MODELS:
        header += f" {model} ID | {model} OOD | {model} Delta |"
        sep += "----------|----------|----------|"
    lines.append(header)
    lines.append(sep)

    for frac in FRACTION_VALUES:
        row = f"| {frac:.2f} |"
        for model in MODELS:
            id_rows = model_agg[(model_agg["fraction"] == frac) & (model_agg["model"] == model) & (model_agg["split"] == "id_test")]
            ood_rows = model_agg[(model_agg["fraction"] == frac) & (model_agg["model"] == model) & (model_agg["split"] == "ood_test")]
            id_mean = id_rows["mean"].values[0] if len(id_rows) else float("nan")
            ood_mean = ood_rows["mean"].values[0] if len(ood_rows) else float("nan")
            delta = ood_mean - id_mean
            row += f" {id_mean:.4f} | {ood_mean:.4f} | {delta:+.4f} |"
        lines.append(row)
    lines.append("")

    # --- Table 5: Per-material ---
    lines.append("## Table S5: SurfIntRatio Sensitivity (Per-Material)")
    lines.append("**Addresses:** Reviewer 4ixz W1/D1 — SurfIntRatio robustness across materials\n")
    materials = sorted(df["material"].unique())

    for split in ("id_test", "ood_test"):
        lines.append(f"### {_split_label(split)} Split\n")
        header = "| Fraction |"
        sep = "|----------|"
        for mat in materials:
            header += f" {mat} |"
            sep += "--------|"
        lines.append(header)
        lines.append(sep)

        mat_agg = df[df["split"] == split].groupby(["fraction", "material"])["surface_interior_ratio"].mean().reset_index()
        for frac in FRACTION_VALUES:
            row = f"| {frac:.2f} |"
            for mat in materials:
                val = mat_agg[(mat_agg["fraction"] == frac) & (mat_agg["material"] == mat)]["surface_interior_ratio"].values
                row += f" {val[0]:.4f} |" if len(val) else " N/A |"
            lines.append(row)
        lines.append("")

    # --- Table 6: Shift stability ---
    lines.append("## Table S6: SurfIntRatio ID->OOD Shift Stability")
    lines.append("**Addresses:** Reviewer 4ixz W1/D1 — Key claim: |Delta| bounded by +/-0.003\n")
    lines.append("Delta = OOD_mean - ID_mean for surface_interior_ratio.\n")

    header = "| Fraction |"
    sep = "|----------|"
    for model in MODELS:
        header += f" {model} Delta |"
        sep += "------------|"
    lines.append(header)
    lines.append(sep)

    for frac in FRACTION_VALUES:
        row = f"| {frac:.2f} |"
        for model in MODELS:
            id_rows = model_agg[(model_agg["fraction"] == frac) & (model_agg["model"] == model) & (model_agg["split"] == "id_test")]
            ood_rows = model_agg[(model_agg["fraction"] == frac) & (model_agg["model"] == model) & (model_agg["split"] == "ood_test")]
            id_mean = id_rows["mean"].values[0] if len(id_rows) else float("nan")
            ood_mean = ood_rows["mean"].values[0] if len(ood_rows) else float("nan")
            delta = ood_mean - id_mean
            row += f" {delta:+.4f} |"
        lines.append(row)

    lines.append("")
    # Summary line
    all_deltas = []
    for frac in FRACTION_VALUES:
        for model in MODELS:
            id_rows = model_agg[(model_agg["fraction"] == frac) & (model_agg["model"] == model) & (model_agg["split"] == "id_test")]
            ood_rows = model_agg[(model_agg["fraction"] == frac) & (model_agg["model"] == model) & (model_agg["split"] == "ood_test")]
            if len(id_rows) and len(ood_rows):
                all_deltas.append(ood_rows["mean"].values[0] - id_rows["mean"].values[0])
    if all_deltas:
        max_abs = max(abs(d) for d in all_deltas)
        lines.append(f"**Max |Delta| across all fractions and models: {max_abs:.4f}**\n")

    return "\n".join(lines)


def build_per_atom_table(df_per_atom: pd.DataFrame, samples_csv_dir: str) -> str:
    """Build markdown table comparing per-atom vs global BondMAE."""
    lines = []
    lines.append("## Table S7: Per-Atom BondMAE vs Global BondMAE")
    lines.append("**Addresses:** Reviewer XQdQ Point 4 — Per-atom local environment comparison\n")
    lines.append("Per-atom BondMAE compares each atom's kNN distances individually,")
    lines.append("rather than globally sorting all kNN distances before comparison.\n")

    # Load existing global bond_mae from sample_metrics.csv
    global_records = []
    for model in MODELS:
        for seed in SEEDS:
            csv_path = os.path.join(samples_csv_dir, "task_1", model, str(seed), "sample_metrics.csv")
            if os.path.isfile(csv_path):
                sdf = pd.read_csv(csv_path, usecols=["bond_mae", "split", "model", "seed"])
                global_records.append(sdf)
    if global_records:
        global_df = pd.concat(global_records, ignore_index=True)
        global_agg = global_df.groupby(["model", "split"])["bond_mae"].mean().reset_index()
    else:
        global_agg = pd.DataFrame()

    # Per-atom aggregation
    pa_agg = df_per_atom.groupby(["model", "split"])["per_atom_bond_mae"].mean().reset_index()

    # --- Overall table ---
    header = "| Model | ID Per-Atom | OOD Per-Atom | ID Global | OOD Global |"
    sep = "|-------|-----------|------------|----------|----------|"
    lines.append(header)
    lines.append(sep)

    for model in MODELS:
        pa_id = pa_agg[(pa_agg["model"] == model) & (pa_agg["split"] == "id_test")]["per_atom_bond_mae"].values
        pa_ood = pa_agg[(pa_agg["model"] == model) & (pa_agg["split"] == "ood_test")]["per_atom_bond_mae"].values
        gl_id = global_agg[(global_agg["model"] == model) & (global_agg["split"] == "id_test")]["bond_mae"].values if len(global_agg) else []
        gl_ood = global_agg[(global_agg["model"] == model) & (global_agg["split"] == "ood_test")]["bond_mae"].values if len(global_agg) else []

        pa_id_v = f"{pa_id[0]:.4f}" if len(pa_id) else "N/A"
        pa_ood_v = f"{pa_ood[0]:.4f}" if len(pa_ood) else "N/A"
        gl_id_v = f"{gl_id[0]:.4f}" if len(gl_id) else "N/A"
        gl_ood_v = f"{gl_ood[0]:.4f}" if len(gl_ood) else "N/A"
        lines.append(f"| {model} | {pa_id_v} | {pa_ood_v} | {gl_id_v} | {gl_ood_v} |")
    lines.append("")

    # --- Per-material breakdown ---
    lines.append("### Per-Material Breakdown\n")
    materials = sorted(df_per_atom["material"].unique())

    for split in ("id_test", "ood_test"):
        lines.append(f"#### {_split_label(split)} Split\n")
        header = "| Model |"
        sep = "|-------|"
        for mat in materials:
            header += f" {mat} |"
            sep += "--------|"
        lines.append(header)
        lines.append(sep)

        mat_agg = df_per_atom[df_per_atom["split"] == split].groupby(["model", "material"])["per_atom_bond_mae"].mean().reset_index()
        for model in MODELS:
            row = f"| {model} |"
            for mat in materials:
                val = mat_agg[(mat_agg["model"] == model) & (mat_agg["material"] == mat)]["per_atom_bond_mae"].values
                row += f" {val[0]:.4f} |" if len(val) else " N/A |"
            lines.append(row)
        lines.append("")

    # --- Failure sequence comparison ---
    lines.append("### Failure Sequence Comparison\n")
    lines.append("Does per-atom BondMAE alter the model failure ranking vs global BondMAE?\n")

    pa_overall = df_per_atom.groupby("model")["per_atom_bond_mae"].mean().sort_values()
    lines.append(f"**Per-atom BondMAE ranking (best to worst):** {' < '.join(pa_overall.index.tolist())}\n")
    if len(global_agg):
        gl_overall = global_agg.groupby("model")["bond_mae"].mean().sort_values()
        lines.append(f"**Global BondMAE ranking (best to worst):** {' < '.join(gl_overall.index.tolist())}\n")
        # Spearman correlation between rankings
        models_ordered = list(pa_overall.index)
        pa_ranks = list(range(len(models_ordered)))
        gl_ranks = [list(gl_overall.index).index(m) for m in models_ordered]
        if len(pa_ranks) >= 2:
            rho, _ = spearmanr(pa_ranks, gl_ranks)
            lines.append(f"**Spearman rho (per-atom vs global ranking):** {rho:.4f}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RADII metric sensitivity analysis")
    parser.add_argument("--results-dir", default="results", help="Results root directory")
    parser.add_argument("--data-root", default="radii", help="Dataset root directory")
    parser.add_argument("--out-dir", default="results/sensitivity", help="Output directory")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    print("=== Loading ground truth ===")
    gt = load_gt_cache(args.data_root)

    print("\n=== Discovering runs ===")
    runs = discover_runs(args.results_dir)
    print(f"  Found {len(runs)} runs")

    print("\n=== Loading predictions and matching to GT ===")
    samples = load_all_samples(runs, gt)
    print(f"\n  Total matched samples: {len(samples)}")

    # Run sweeps
    print("\n=== CoordCorr cutoff sweep ===")
    coord_df = run_coord_sweep(samples, args.n_jobs)

    print("\n=== SurfIntRatio fraction sweep ===")
    surfint_df = run_surfint_sweep(samples, args.n_jobs)

    print("\n=== Per-atom BondMAE ===")
    per_atom_df = run_per_atom_bond_mae(samples, args.n_jobs)

    # Build and save tables
    print("\n=== Building tables ===")

    coord_md = build_coord_tables(coord_df)
    coord_path = os.path.join(args.out_dir, "coordcorr_sensitivity.md")
    with open(coord_path, "w") as f:
        f.write(coord_md)
    print(f"  Saved: {coord_path}")

    surfint_md = build_surfint_tables(surfint_df)
    surfint_path = os.path.join(args.out_dir, "surfint_sensitivity.md")
    with open(surfint_path, "w") as f:
        f.write(surfint_md)
    print(f"  Saved: {surfint_path}")

    per_atom_md = build_per_atom_table(per_atom_df, args.results_dir)
    per_atom_path = os.path.join(args.out_dir, "per_atom_bond_mae.md")
    with open(per_atom_path, "w") as f:
        f.write(per_atom_md)
    print(f"  Saved: {per_atom_path}")

    # Also save raw CSVs for further analysis
    coord_df.to_csv(os.path.join(args.out_dir, "coordcorr_raw.csv"), index=False)
    surfint_df.to_csv(os.path.join(args.out_dir, "surfint_raw.csv"), index=False)
    per_atom_df.to_csv(os.path.join(args.out_dir, "per_atom_raw.csv"), index=False)
    print("  Saved raw CSVs")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
