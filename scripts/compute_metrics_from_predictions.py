"""
Compute metrics from saved predictions (eval_predictions/*.npz).

GT is matched from the dataset using (material, radius, rot_idx). Run after
training to compute all RADII metrics including rotation consistency analysis.

GT is loaded from data_root/gt_cache/ when present. If the cache is missing,
GT is built from the full dataset (loaded_frac=1.0) and saved there so later
runs and any training frac use the same cache.

Usage:
    python scripts/compute_metrics_from_predictions.py results/task_1/adit/42
    python scripts/compute_metrics_from_predictions.py results/task_1/adit/42 --data-root radii
    python scripts/compute_metrics_from_predictions.py results   # run for all dirs with eval_predictions under results/
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from radii.data import RADIIDataloader
from radii.metrics import (
    compute_sample_metrics,
    compute_rotation_group_metrics,
    compute_aggregate_metrics,
)
from radii.train_config import TrainConfig


ID_RADII = TrainConfig.ID_RADII
OOD_RADII = TrainConfig.OOD_RADII

# GT cache under data_root (one cache for all training fracs)
_GT_CACHE_DIR = "gt_cache"
_GT_META_FILE = "meta.json"


def _gt_cache_dir(data_root: str) -> str:
    return os.path.join(data_root, _GT_CACHE_DIR)


def _gt_lookup_to_arrays(lookup: Dict[Tuple[str, float, int], np.ndarray]):
    keys = sorted(lookup.keys())
    materials = np.array([k[0] for k in keys], dtype=object)
    radius = np.array([k[1] for k in keys], dtype=np.float64)
    rot_idx = np.array([k[2] for k in keys], dtype=np.int64)
    positions = [lookup[k] for k in keys]
    ptr = np.concatenate([[0], np.cumsum([len(p) for p in positions])]).astype(np.int64)
    gt_pos = np.concatenate(positions, axis=0).astype(np.float64)
    return materials, radius, rot_idx, ptr, gt_pos


def _arrays_to_gt_lookup(
    materials, radius, rot_idx, ptr, gt_pos
) -> Dict[Tuple[str, float, int], np.ndarray]:
    lookup = {}
    for i in range(len(ptr) - 1):
        s, e = int(ptr[i]), int(ptr[i + 1])
        lookup[(str(materials[i]), float(radius[i]), int(rot_idx[i]))] = gt_pos[
            s:e
        ].copy()
    return lookup


def _load_gt_cache(
    data_root: str, loaded_frac: float
) -> Optional[Dict[str, Dict[Tuple[str, float, int], np.ndarray]]]:
    cache_dir = _gt_cache_dir(data_root)
    meta_path = os.path.join(cache_dir, _GT_META_FILE)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        cached_frac = meta.get("loaded_frac")
        if cached_frac != loaded_frac and cached_frac != 1.0:
            return None
    except Exception:
        return None
    out = {}
    for split in ("id_test", "ood_test"):
        path = os.path.join(cache_dir, f"{split}.npz")
        if not os.path.isfile(path):
            return None
        try:
            data = np.load(path, allow_pickle=True)
            out[split] = _arrays_to_gt_lookup(
                data["materials"],
                data["radius"],
                data["rot_idx"],
                data["ptr"],
                data["gt_pos"],
            )
        except Exception:
            return None
    return out


def get_gt_cache(data_root: str, loaded_frac: float = 1.0):
    """Load GT lookup from cache if present (for use by analyze_results from NPZ)."""
    return _load_gt_cache(data_root, loaded_frac)


def ensure_gt_cache(data_root: str, loaded_frac: float = 1.0):
    """
    Return GT lookup for id_test and ood_test. Load from cache if present;
    otherwise build from full dataset, save cache, and return. Use this from
    analyze_results so NPZ-based metrics work even when cache was never built.
    """
    gt_lookup = _load_gt_cache(data_root, loaded_frac)
    if gt_lookup is not None:
        return gt_lookup

    def transform(data):
        data.y_pos = data.pos.clone()
        if not hasattr(data, "cell_ptr") or data.cell_ptr is None:
            data.cell_ptr = torch.tensor([0, data.cell_pos.size(0)], dtype=torch.long)
        data.num_atoms = torch.tensor([data.pos.size(0)], dtype=torch.long)
        return data

    ds = RADIIDataloader(root=data_root, transform=transform, loaded_frac=1.0)
    id_ds, ood_ds = ds.get_split("id_test"), ds.get_split("ood_test")
    id_ds.transform = ood_ds.transform = transform
    gt_lookup = {"id_test": build_gt_lookup(id_ds), "ood_test": build_gt_lookup(ood_ds)}
    _save_gt_cache(gt_lookup, data_root, 1.0)
    return gt_lookup


def _save_gt_cache(
    gt_lookup: Dict[str, Dict[Tuple[str, float, int], np.ndarray]],
    data_root: str,
    loaded_frac: float,
) -> None:
    cache_dir = _gt_cache_dir(data_root)
    os.makedirs(cache_dir, exist_ok=True)
    for split in ("id_test", "ood_test"):
        if split in gt_lookup:
            m, r, ri, ptr, pos = _gt_lookup_to_arrays(gt_lookup[split])
            np.savez_compressed(
                os.path.join(cache_dir, f"{split}.npz"),
                materials=m,
                radius=r,
                rot_idx=ri,
                ptr=ptr,
                gt_pos=pos,
            )
    with open(os.path.join(cache_dir, _GT_META_FILE), "w") as f:
        json.dump({"loaded_frac": loaded_frac}, f, indent=2)


def build_gt_lookup(dataset):
    """
    Build (material, radius, rot_idx) -> gt_pos lookup.

    Returns:
        Dict mapping (material, radius, rot_idx) -> positions array
    """
    lookup = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        mat = (
            sample.material
            if isinstance(sample.material, str)
            else str(sample.material)
        )
        rad = (
            float(sample.radius.item())
            if hasattr(sample.radius, "item")
            else float(sample.radius)
        )
        rot = (
            int(sample.rot_idx.item())
            if hasattr(sample.rot_idx, "item")
            else int(sample.rot_idx)
        )
        key = (mat, rad, rot)
        pos = (
            sample.pos.numpy()
            if torch.is_tensor(sample.pos)
            else np.asarray(sample.pos)
        )
        lookup[key] = pos
    return lookup


def _compute_one_sample(
    args: Tuple[Dict, np.ndarray, str],
) -> Tuple[Dict, Tuple[str, float], int, np.ndarray, np.ndarray]:
    """Worker: compute per-sample metrics. Used by ProcessPoolExecutor."""
    sample, gt, split = args
    pred = sample["pred"]
    rad = sample["radius"]
    mat = sample["material"]
    rot = sample["rot_idx"]
    metrics = compute_sample_metrics(pred, gt, radius=rad)
    metrics["material"] = mat
    metrics["radius"] = rad
    metrics["rot_idx"] = rot
    metrics["split"] = split
    group_key = (mat, rad)
    return (metrics, group_key, rot, pred.copy(), gt.copy())


def _compute_one_rotation_group(
    args: Tuple[Tuple[str, float], Dict[int, np.ndarray], Dict[int, np.ndarray]],
) -> Tuple[Tuple[str, float], Dict[str, float]]:
    """Worker: compute rotation group metrics. Used by ProcessPoolExecutor."""
    group_key, preds_dict, gt_dict = args
    rot_metrics = compute_rotation_group_metrics(preds_dict, gt_dict)
    return (group_key, rot_metrics)


def load_predictions(npz_path):
    """
    Load predictions from npz file.

    Returns:
        List of dicts with keys: pred, material, radius, rot_idx
    """
    data = np.load(npz_path, allow_pickle=True)
    pred_pos = data["pred_pos"]
    ptr = data["ptr"]
    materials = data["materials"]
    radius = data["radius"]
    rot_idx = data["rot_idx"]

    samples = []
    for i in range(len(ptr) - 1):
        s, e = int(ptr[i]), int(ptr[i + 1])
        samples.append(
            {
                "pred": pred_pos[s:e],
                "material": str(materials[i]),
                "radius": float(radius[i]),
                "rot_idx": int(rot_idx[i]),
            }
        )
    return samples


def find_results_dirs(root: str) -> List[str]:
    """
    Find all directories under root that contain eval_predictions/ with at least
    one of id_test.npz or ood_test.npz. Returns sorted list of result dir paths.
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return []
    found = []
    for dirpath, dirnames, _ in os.walk(root, topdown=True):
        if "eval_predictions" in dirnames:
            eval_dir = os.path.join(dirpath, "eval_predictions")
            if os.path.isfile(os.path.join(eval_dir, "id_test.npz")) or os.path.isfile(
                os.path.join(eval_dir, "ood_test.npz")
            ):
                found.append(os.path.normpath(dirpath))
        # Don't recurse into result dirs (they contain eval_predictions, not further runs)
        dirnames[:] = [d for d in dirnames if d != "eval_predictions"]
    return sorted(found)


def run_one(results_dir: str, args: argparse.Namespace) -> None:
    """Compute and save metrics for a single results directory."""
    eval_dir = os.path.join(results_dir, "eval_predictions")
    if not os.path.isdir(eval_dir):
        raise SystemExit(f"eval_predictions not found in {results_dir}")

    # Load config
    config_path = os.path.join(results_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        data_root = args.data_root or config.get("data_root", "radii")
        loaded_frac = config.get("loaded_frac", 1.0)
    else:
        data_root = args.data_root or "radii"
        loaded_frac = 1.0

    # Infer model/seed from path (per-dir when running multiple)
    parts = os.path.normpath(results_dir).replace("\\", "/").split("/")
    model_name = (
        args.model
        if args.model is not None
        else (parts[-2] if len(parts) >= 2 and parts[-1].isdigit() else "unknown")
    )
    seed = (
        args.seed
        if args.seed is not None
        else (int(parts[-1]) if len(parts) >= 1 and parts[-1].isdigit() else -1)
    )

    # Load GT from cache or build from full dataset and save (one cache for all fracs)
    gt_lookup = _load_gt_cache(data_root, loaded_frac)
    if gt_lookup is not None:
        print(
            f"Loaded GT from cache: id_test={len(gt_lookup['id_test'])}, ood_test={len(gt_lookup['ood_test'])}"
        )
    else:

        def transform(data):
            data.y_pos = data.pos.clone()
            if not hasattr(data, "cell_ptr") or data.cell_ptr is None:
                data.cell_ptr = torch.tensor(
                    [0, data.cell_pos.size(0)], dtype=torch.long
                )
            data.num_atoms = torch.tensor([data.pos.size(0)], dtype=torch.long)
            return data

        print("Loading dataset (no GT cache), building and saving cache...")
        ds = RADIIDataloader(root=data_root, transform=transform, loaded_frac=1.0)
        id_ds, ood_ds = ds.get_split("id_test"), ds.get_split("ood_test")
        id_ds.transform = ood_ds.transform = transform
        gt_lookup = {
            "id_test": build_gt_lookup(id_ds),
            "ood_test": build_gt_lookup(ood_ds),
        }
        _save_gt_cache(gt_lookup, data_root, 1.0)
        print(
            f"Saved GT cache: id_test={len(gt_lookup['id_test'])}, ood_test={len(gt_lookup['ood_test'])}"
        )

    # =========================================================================
    # PHASE 1: Compute per-sample metrics (parallel)
    # =========================================================================
    print("\n=== Phase 1: Per-sample metrics ===")

    all_sample_metrics: List[Dict] = []
    predictions_by_group: Dict[Tuple[str, float], Dict[int, np.ndarray]] = defaultdict(
        dict
    )
    gt_by_group: Dict[Tuple[str, float], Dict[int, np.ndarray]] = defaultdict(dict)

    n_jobs = args.jobs if args.jobs is not None else max(1, (os.cpu_count() or 4) - 1)
    phase1_tasks: List[Tuple[Dict, np.ndarray, str]] = []
    split_counts: Dict[str, Tuple[int, int]] = {}

    for split in ["id_test", "ood_test"]:
        npz_path = os.path.join(eval_dir, f"{split}.npz")
        if not os.path.exists(npz_path):
            print(f"Skipping {npz_path} (not found)")
            continue

        samples = load_predictions(npz_path)
        lookup = gt_lookup[split]
        matched, skipped = 0, 0
        for sample in samples:
            key = (sample["material"], sample["radius"], sample["rot_idx"])
            if key not in lookup:
                skipped += 1
                continue
            phase1_tasks.append((sample, lookup[key], split))
            matched += 1
        split_counts[split] = (matched, skipped)
        print(f"  {split}: matched={matched}, skipped={skipped}")

    if phase1_tasks:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for result in pool.map(
                _compute_one_sample,
                phase1_tasks,
                chunksize=max(1, len(phase1_tasks) // (n_jobs * 4)),
            ):
                metrics, group_key, rot, pred, gt = result
                all_sample_metrics.append(metrics)
                predictions_by_group[group_key][rot] = pred
                gt_by_group[group_key][rot] = gt
        print(f"  Computed {len(all_sample_metrics)} sample metrics ({n_jobs} workers)")

    # =========================================================================
    # PHASE 2: Compute rotation consistency metrics (per group, parallel)
    # =========================================================================
    print("\n=== Phase 2: Rotation consistency ===")

    rotation_group_metrics: Dict[Tuple[str, float], Dict[str, float]] = {}
    phase2_tasks: List[
        Tuple[Tuple[str, float], Dict[int, np.ndarray], Dict[int, np.ndarray]]
    ] = [
        (group_key, preds_dict, gt_by_group[group_key])
        for group_key, preds_dict in predictions_by_group.items()
        if len(preds_dict) >= 2
    ]

    if phase2_tasks:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for group_key, rot_metrics in pool.map(
                _compute_one_rotation_group,
                phase2_tasks,
                chunksize=max(1, len(phase2_tasks) // (n_jobs * 2)),
            ):
                if rot_metrics:
                    rotation_group_metrics[group_key] = rot_metrics

    print(f"  Computed rotation metrics for {len(rotation_group_metrics)} groups")

    # =========================================================================
    # PHASE 3: Compute aggregate metrics
    # =========================================================================
    print("\n=== Phase 3: Aggregate metrics ===")

    aggregate = compute_aggregate_metrics(
        all_sample_metrics=all_sample_metrics,
        rotation_group_metrics=rotation_group_metrics,
        id_radii=ID_RADII,
        ood_radii=OOD_RADII,
    )

    # Add model/seed
    aggregate["model"] = model_name
    aggregate["seed"] = seed

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================

    # 1. Per-sample metrics
    df_samples = pd.DataFrame(all_sample_metrics)
    df_samples["model"] = model_name
    df_samples["seed"] = seed
    samples_path = os.path.join(results_dir, "sample_metrics.csv")
    df_samples.to_csv(samples_path, index=False)
    print(f"\nSaved {len(df_samples)} sample metrics -> {samples_path}")

    # 2. Rotation group metrics
    rot_records = []
    for (mat, rad), metrics in rotation_group_metrics.items():
        record = {"material": mat, "radius": rad}
        record.update(metrics)
        rot_records.append(record)

    if rot_records:
        df_rotation = pd.DataFrame(rot_records)
        df_rotation["model"] = model_name
        df_rotation["seed"] = seed
        rotation_path = os.path.join(results_dir, "rotation_metrics.csv")
        df_rotation.to_csv(rotation_path, index=False)
        print(f"Saved {len(df_rotation)} rotation group metrics -> {rotation_path}")

    # 3. Aggregate metrics
    agg_path = os.path.join(results_dir, "aggregate_metrics.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Saved aggregate metrics -> {agg_path}")

    # Replace the print summary block with:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}, Seed: {seed}")
    print("\nGeneration Quality:")
    print(f"  RMSD (mean):        {aggregate.get('rmsd_mean', np.nan):.4f}")
    print(f"  Rg Error (mean):    {aggregate.get('rg_error_mean', np.nan):.4f}")
    print(f"  Bond MAE (mean):    {aggregate.get('bond_mae_mean', np.nan):.4f}")
    print("\nFailure Decomposition:")
    print(
        f"  Surface/Interior:   {aggregate.get('surface_interior_ratio_mean', np.nan):.4f}"
    )
    print(
        f"  Coord Preservation: {aggregate.get('coord_preservation_mean', np.nan):.4f}"
    )
    print(f"  Orient. Stability:  {aggregate.get('rot_consistency_mean', np.nan):.4f}")
    print("\nFrontier Characterization:")
    print(f"  Degradation Ratio:  {aggregate.get('degradation_ratio', np.nan):.4f}")
    print(f"  ID RMSD:            {aggregate.get('id_rmsd_mean', np.nan):.4f}")
    print(f"  OOD RMSD:           {aggregate.get('ood_rmsd_mean', np.nan):.4f}")
    print(
        f"  Orient. Degrad.:    {aggregate.get('orientation_degradation_score', np.nan):.4f}"
    )
    print(f"  Scale Smoothness:   {aggregate.get('scale_smoothness', np.nan):.4f}")
    print(f"  Scale Jump Ratio:   {aggregate.get('scale_jump_ratio', np.nan):.4f}")
    print(f"  Frontier (RMSD):    {aggregate.get('frontier_radius_rmsd', np.nan):.2f}")
    print(
        f"  Frontier (BondMAE): {aggregate.get('frontier_radius_bond_mae', np.nan):.2f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics from saved eval predictions"
    )
    parser.add_argument(
        "results_dir",
        help="Single run dir (e.g. results/task_1/adit/42) or root to run all (e.g. results)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to radii data (default: from config.json)",
    )
    parser.add_argument(
        "--model", default=None, help="Model name (inferred from path if omitted)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed (inferred from path if omitted)"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel jobs for metrics (default: CPU count - 1)",
    )
    args = parser.parse_args()

    path = os.path.abspath(args.results_dir)
    eval_dir_check = os.path.join(path, "eval_predictions")
    if os.path.isdir(eval_dir_check) and (
        os.path.isfile(os.path.join(eval_dir_check, "id_test.npz"))
        or os.path.isfile(os.path.join(eval_dir_check, "ood_test.npz"))
    ):
        dirs_to_run = [path]
    else:
        dirs_to_run = find_results_dirs(path)
        if not dirs_to_run:
            raise SystemExit(
                f"No result directories with eval_predictions found under {path}"
            )
        print(f"Found {len(dirs_to_run)} result directories under {path}\n")

    for i, results_dir in enumerate(dirs_to_run):
        if len(dirs_to_run) > 1:
            print(f"[{i + 1}/{len(dirs_to_run)}] {results_dir}")
        run_one(results_dir, args)


if __name__ == "__main__":
    main()
