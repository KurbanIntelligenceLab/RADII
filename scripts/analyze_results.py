"""
Load training/eval results for analysis. Real data only: aggregate and sample metrics
are computed from eval_predictions/*.npz (+ GT cache); epochs from epochs.csv.
Runs without valid NPZ are skipped.

Usage:
  python scripts/analyze_results.py
  python scripts/analyze_results.py --results-dir results --out-dir results/analysis
  python scripts/analyze_results.py --models adit cdvae --seeds 1 2 --quiet
  python scripts/analyze_results.py --recompute-metrics -j 4

From analysis code:
  from scripts.analyze_results import load_all_results
  aggregate_df, epochs_df, missing = load_all_results(results_root="results")
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from radii.train_config import TrainConfig

    DEFAULT_SEEDS = list(TrainConfig.SEEDS)
except Exception:
    DEFAULT_SEEDS = [1, 2]

DEFAULT_MODELS = ["adit", "cdvae", "diffcsp", "flowmm", "mattergen"]

# Keys produced by compute_aggregate_metrics
AGGREGATE_KEYS = [
    # Generation quality
    "rmsd_mean",
    "rmsd_std",
    "rg_error_mean",
    "bond_mae_mean",
    # Failure decomposition
    "surface_interior_ratio_mean",
    "coord_preservation_mean",
    # Frontier characterization
    "scale_smoothness",
    "scale_jump_ratio",
    "degradation_ratio",
    "id_rmsd_mean",
    "ood_rmsd_mean",
    "surface_ratio_degradation",
    "id_surface_ratio_mean",
    "ood_surface_ratio_mean",
    "coord_degradation",
    "id_coord_mean",
    "ood_coord_mean",
    "rot_consistency_mean",
    "rot_consistency_std",
    "orientation_degradation_score",
    "id_consistency_mean",
    "ood_consistency_mean",
    "frontier_radius_rmsd",
    "frontier_fraction_rmsd",
    "beyond_frontier_mean_rmsd",
    "frontier_radius_bond_mae",
    "frontier_fraction_bond_mae",
    "beyond_frontier_mean_bond_mae",
    "frontier_radius_rg_error",
    "frontier_fraction_rg_error",
    "beyond_frontier_mean_rg_error",
]

EPOCH_COLUMNS = ["epoch", "phase", "train_loss", "val_loss", "lr", "time"]
EPOCH_VAE_COLUMNS = ["recon_atom", "recon_pos", "kl"]

# Metric groups for table display: (display_name, id_col, ood_col)
# ood_col=None means single column (not split by ID/OOD)
METRIC_GROUPS = {
    "Generation Quality": [
        ("RMSD (mean)", "id_rmsd_mean", "ood_rmsd_mean"),
        ("RMSD (std)", "id_rmsd_std", "ood_rmsd_std"),
        ("Rg error (mean)", "id_rg_error_mean", "ood_rg_error_mean"),
        ("Bond MAE (mean)", "id_bond_mae_mean", "ood_bond_mae_mean"),
    ],
    "Failure Decomposition": [
        ("Surface/Interior", "id_surface_ratio_mean", "ood_surface_ratio_mean"),
        ("Coord preserv.", "id_coord_mean", "ood_coord_mean"),
        ("Orient. stability (mean)", "id_consistency_mean", "ood_consistency_mean"),
        ("Orient. stability (std)", "rot_consistency_std", None),
    ],
    "Frontier Characterization": [
        ("Degrad. ratio", "degradation_ratio", None),
        ("Scale smoothness", "scale_smoothness", None),
        ("Scale jump ratio", "scale_jump_ratio", None),
        ("Orient. degrad.", "orientation_degradation_score", None),
        ("Surface degrad.", "surface_ratio_degradation", None),
        ("Coord degrad.", "coord_degradation", None),
        ("Frontier r* (RMSD)", "frontier_radius_rmsd", None),
        ("Frontier r* (Bond)", "frontier_radius_bond_mae", None),
        ("Frontier r* (Rg)", "frontier_radius_rg_error", None),
        ("Frontier frac (RMSD)", "frontier_fraction_rmsd", None),
    ],
}

_ID_OOD_WIDTH = 14


def result_dir(results_root: str, model: str, seed: int) -> str:
    return os.path.join(results_root, "task_1", model, str(seed))


def discover_runs(results_root: str) -> List[Tuple[str, int]]:
    """
    Find all (model, seed) runs under results_root/task_1/ that have
    eval_predictions/id_test.npz and ood_test.npz. Sorted by model then seed.
    """
    task1 = os.path.join(results_root, "task_1")
    if not os.path.isdir(task1):
        return []
    runs = []
    for model in sorted(os.listdir(task1)):
        model_dir = os.path.join(task1, model)
        if not os.path.isdir(model_dir):
            continue
        for seed_name in sorted(os.listdir(model_dir)):
            if not seed_name.isdigit():
                continue
            seed_dir = os.path.join(model_dir, seed_name)
            if not os.path.isdir(seed_dir):
                continue
            id_npz = os.path.join(seed_dir, "eval_predictions", "id_test.npz")
            ood_npz = os.path.join(seed_dir, "eval_predictions", "ood_test.npz")
            if os.path.isfile(id_npz) and os.path.isfile(ood_npz):
                runs.append((model, int(seed_name)))
    return runs


def try_load_json(path: str) -> Tuple[Optional[Dict], bool]:
    if not os.path.isfile(path):
        return None, False
    try:
        with open(path) as f:
            return json.load(f), True
    except Exception:
        return None, False


def try_load_csv(path: str) -> Tuple[Optional[pd.DataFrame], bool]:
    if not os.path.isfile(path):
        return None, False
    try:
        return pd.read_csv(path), True
    except Exception:
        return None, False


def try_load_npz(path: str) -> Tuple[bool, bool]:
    exists = os.path.isfile(path)
    if not exists:
        return False, False
    try:
        np.load(path, allow_pickle=True)
        return True, True
    except Exception:
        return True, False


def load_or_fake_run(
    results_root: str,
    model: str,
    seed: int,
    recompute_metrics: bool = False,
    verbose: bool = True,
) -> Tuple[
    Optional[Dict[str, Any]], Optional[pd.DataFrame], List[str], Optional[pd.DataFrame]
]:
    """
    Load one run (model, seed). Only real data: aggregate and sample from NPZ;
    epochs from CSV.

    Returns:
        aggregate: dict from NPZ (or None if NPZ missing/failed)
        epochs_df: from epochs.csv (or None if missing)
        missing: list of missing artifacts
        sample_df: from NPZ (or None)
    """
    dir_path = result_dir(results_root, model, seed)
    missing: List[str] = []

    # --- config.json ---
    config_path = os.path.join(dir_path, "config.json")
    config, config_ok = try_load_json(config_path)
    if not config_ok:
        missing.append(f"{dir_path}: config.json")

    # --- epochs.csv ---
    epochs_path = os.path.join(dir_path, "epochs.csv")
    epochs_df, epochs_ok = try_load_csv(epochs_path)
    if not epochs_ok or epochs_df is None or epochs_df.empty:
        missing.append(f"{dir_path}: epochs.csv")
        epochs_df = None
    else:
        epochs_df = epochs_df.copy()
        epochs_df["model"] = model
        epochs_df["seed"] = seed

    # --- aggregate + sample: use saved if present (unless --recompute), else from NPZ ---
    if recompute_metrics:
        aggregate, sample_df = None, None
    else:
        aggregate, sample_df = _load_existing_metrics(dir_path, model, seed)
    if aggregate is None or sample_df is None or sample_df.empty:
        eval_dir = os.path.join(dir_path, "eval_predictions")
        id_npz = os.path.join(eval_dir, "id_test.npz")
        ood_npz = os.path.join(eval_dir, "ood_test.npz")
        for name, p in [
            ("eval_predictions/id_test.npz", id_npz),
            ("eval_predictions/ood_test.npz", ood_npz),
        ]:
            exists, loaded = try_load_npz(p)
            if not exists:
                missing.append(f"{dir_path}: {name}")
            elif not loaded:
                missing.append(f"{dir_path}: {name} (failed to load)")
        aggregate, sample_df = _metrics_from_npz(dir_path, model, seed, verbose=verbose)
    return aggregate, epochs_df, missing, sample_df


def _load_existing_metrics(
    dir_path: str, model: str, seed: int
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """Load aggregate and sample metrics from disk if already computed."""
    agg_path = os.path.join(dir_path, "aggregate_metrics.json")
    sample_path = os.path.join(dir_path, "sample_metrics.csv")
    if not os.path.isfile(agg_path) or not os.path.isfile(sample_path):
        return None, None
    try:
        with open(agg_path) as f:
            aggregate = json.load(f)
        sample_df = pd.read_csv(sample_path)
        if sample_df.empty:
            return None, None
        aggregate["model"] = aggregate.get("model", model)
        aggregate["seed"] = aggregate.get("seed", seed)
        if "model" not in sample_df.columns:
            sample_df["model"] = model
        if "seed" not in sample_df.columns:
            sample_df["seed"] = seed
        return aggregate, sample_df
    except Exception:
        return None, None


def _metrics_from_npz(
    dir_path: str, model: str, seed: int, verbose: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Compute aggregate + sample metrics from eval_predictions NPZ using GT cache.
    """
    eval_dir = os.path.join(dir_path, "eval_predictions")
    id_npz = os.path.join(eval_dir, "id_test.npz")
    ood_npz = os.path.join(eval_dir, "ood_test.npz")
    if not os.path.isfile(id_npz) or not os.path.isfile(ood_npz):
        return None, None

    try:
        from scripts.compute_metrics_from_predictions import (
            ensure_gt_cache,
            load_predictions,
        )
        from radii.metrics import (
            compute_sample_metrics,
            compute_rotation_group_metrics,
            compute_aggregate_metrics,
        )

        try:
            from radii.train_config import TrainConfig

            id_radii, ood_radii = (
                list(TrainConfig.ID_RADII),
                list(TrainConfig.OOD_RADII),
            )
        except Exception:
            id_radii, ood_radii = [11, 13, 15, 17, 19, 21], [6, 7, 29, 30]
    except Exception:
        return None, None

    config_path = os.path.join(dir_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            data_root = config.get("data_root", "radii")
            loaded_frac = config.get("loaded_frac", 1.0)
        except Exception:
            return None, None
    else:
        data_root = "radii"
        loaded_frac = 1.0

    try:
        gt_lookup = ensure_gt_cache(data_root, loaded_frac)
    except Exception:
        return None, None

    all_sample_metrics = []
    predictions_by_group = defaultdict(dict)
    gt_by_group = defaultdict(dict)

    for split in ("id_test", "ood_test"):
        npz_path = os.path.join(eval_dir, f"{split}.npz")
        try:
            samples = load_predictions(npz_path)
        except Exception:
            continue
        lookup = gt_lookup[split]
        iterator = (
            tqdm(samples, desc=f"{model}/{seed} {split}", leave=False, unit="sample")
            if verbose
            else samples
        )
        for s in iterator:
            key = (s["material"], s["radius"], s["rot_idx"])
            if key not in lookup:
                continue
            gt = lookup[key]
            m = compute_sample_metrics(s["pred"], gt, radius=s["radius"])
            m["material"] = s["material"]
            m["radius"] = s["radius"]
            m["rot_idx"] = s["rot_idx"]
            m["split"] = split
            all_sample_metrics.append(m)
            group_key = (s["material"], s["radius"])
            predictions_by_group[group_key][s["rot_idx"]] = s["pred"]
            gt_by_group[group_key][s["rot_idx"]] = gt

    if not all_sample_metrics:
        return None, None

    rotation_group_metrics = {}
    for group_key, preds_dict in predictions_by_group.items():
        gt_dict = gt_by_group[group_key]
        if len(preds_dict) >= 2:
            rot_metrics = compute_rotation_group_metrics(preds_dict, gt_dict)
            if rot_metrics:
                rotation_group_metrics[group_key] = rot_metrics

    aggregate = compute_aggregate_metrics(
        all_sample_metrics=all_sample_metrics,
        rotation_group_metrics=rotation_group_metrics,
        id_radii=id_radii,
        ood_radii=ood_radii,
    )
    aggregate["model"] = model
    aggregate["seed"] = seed

    sample_df = pd.DataFrame(all_sample_metrics)
    sample_df["model"] = model
    sample_df["seed"] = seed
    return aggregate, sample_df


def _data_root_from_first_run(
    results_root: str, run_list: List[Tuple[str, int]]
) -> str:
    if not run_list:
        return "radii"
    model, seed = run_list[0]
    config_path = os.path.join(result_dir(results_root, model, seed), "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                return json.load(f).get("data_root", "radii")
        except Exception:
            pass
    return "radii"


def load_all_results(
    results_root: str = "results",
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    recompute_metrics: bool = False,
    jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    """
    Load all runs; only real data (from NPZ and CSV). Skips runs with no NPZ.

    Args:
        jobs: If > 1, process (model, seed) runs in parallel with this many workers.

    Returns:
        aggregate_df: one row per (model, seed) with frontier metrics
        epochs_df: epoch logs from epochs.csv
        all_missing: list of missing artifacts
        sample_metrics_df: per-sample metrics from NPZ
    """
    if models is None and seeds is None:
        runs = discover_runs(results_root)
        run_list = runs
    else:
        models = models or DEFAULT_MODELS
        seeds = seeds or DEFAULT_SEEDS
        run_list = [(m, s) for m in models for s in seeds]

    all_aggregates = []
    all_epochs = []
    all_samples = []
    all_missing = []

    if jobs <= 1:
        for model, seed in tqdm(run_list, desc="Runs", unit="run"):
            agg, epochs, missing, sample_df = load_or_fake_run(
                results_root,
                model,
                seed,
                recompute_metrics=recompute_metrics,
                verbose=True,
            )
            all_missing.extend(missing)
            if agg is not None:
                all_aggregates.append(agg)
            if epochs is not None and not epochs.empty:
                all_epochs.append(epochs)
            if sample_df is not None and not sample_df.empty:
                all_samples.append(sample_df)
    else:
        # Parallel: one run per worker, no per-sample progress bars
        arg_list = [
            (results_root, model, seed, recompute_metrics, False)
            for model, seed in run_list
        ]
        with Pool(min(jobs, len(run_list))) as pool:
            results = list(
                tqdm(
                    pool.starmap(load_or_fake_run, arg_list),
                    total=len(arg_list),
                    desc="Runs",
                    unit="run",
                )
            )
        for agg, epochs, missing, sample_df in results:
            all_missing.extend(missing)
            if agg is not None:
                all_aggregates.append(agg)
            if epochs is not None and not epochs.empty:
                all_epochs.append(epochs)
            if sample_df is not None and not sample_df.empty:
                all_samples.append(sample_df)

    aggregate_df = pd.DataFrame(all_aggregates) if all_aggregates else pd.DataFrame()
    epochs_df = (
        pd.concat(all_epochs, ignore_index=True) if all_epochs else pd.DataFrame()
    )
    sample_metrics_df = (
        pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()
    )
    return aggregate_df, epochs_df, all_missing, sample_metrics_df


def _format_cell(mean_val, std_val):
    if pd.isna(mean_val):
        return "—"
    if pd.isna(std_val) or std_val == 0:
        return f"{mean_val:.3f}"
    return f"{mean_val:.3f} ± {std_val:.3f}"


def _table_block_id_ood(
    lines: List[str],
    title: str,
    row_names: List[str],
    means: pd.DataFrame,
    stds: pd.DataFrame,
    metrics: List[Tuple[str, Optional[str], Optional[str]]],
    col_width: int = _ID_OOD_WIDTH,
) -> None:
    """
    Render a table block with optional ID/OOD sub-columns.
    metrics: list of (display_name, id_col, ood_col). ood_col=None => single column.
    """
    sep = " "
    header1_parts = ["model".ljust(14), sep]
    header2_parts = ["".ljust(14), sep]
    cols = []
    block_w = 2 * col_width + 2 * len(sep)
    for display, id_col, ood_col in metrics:
        id_ok = id_col and id_col in means.columns
        ood_ok = ood_col and ood_col in means.columns
        if not id_ok and not ood_ok:
            continue
        if ood_ok:
            header1_parts.append(
                (
                    display[: block_w - len(sep)]
                    if len(display) > block_w - len(sep)
                    else display
                ).ljust(block_w)
                + sep
            )
            header2_parts.append(
                "ID".ljust(col_width) + sep + "OOD".ljust(col_width) + sep
            )
            cols.append((id_col, ood_col))
        else:
            header1_parts.append(
                (display[:col_width] if len(display) > col_width else display).ljust(
                    col_width
                )
                + sep
            )
            header2_parts.append("".ljust(col_width) + sep)
            cols.append((id_col, None))
    if not cols:
        return
    header1 = "".join(header1_parts).rstrip()
    header2 = "".join(header2_parts).rstrip()
    sep_len = max(len(header1), len(header2), 72)
    lines.append("")
    lines.append("=" * sep_len)
    lines.append(f"  {title}")
    lines.append("=" * sep_len)
    lines.append(header1)
    lines.append(header2)
    lines.append("-" * sep_len)
    for row_name in row_names:
        row_parts = [row_name.ljust(14), sep]
        for id_c, ood_c in cols:
            if ood_c is not None:
                m_id = means.loc[row_name, id_c] if row_name in means.index else np.nan
                s_id = (
                    stds.loc[row_name, id_c]
                    if row_name in stds.index and id_c in stds.columns
                    else np.nan
                )
                m_ood = (
                    means.loc[row_name, ood_c] if row_name in means.index else np.nan
                )
                s_ood = (
                    stds.loc[row_name, ood_c]
                    if row_name in stds.index and ood_c in stds.columns
                    else np.nan
                )
                row_parts.append(
                    _format_cell(m_id, s_id).ljust(col_width)
                    + sep
                    + _format_cell(m_ood, s_ood).ljust(col_width)
                    + sep
                )
            else:
                m = means.loc[row_name, id_c] if row_name in means.index else np.nan
                s = (
                    stds.loc[row_name, id_c]
                    if row_name in stds.index and id_c in stds.columns
                    else np.nan
                )
                row_parts.append(_format_cell(m, s).ljust(col_width) + sep)
        lines.append("".join(row_parts).rstrip())
    lines.append("")


def _top3_worst3_materials(
    sample_metrics_df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Rank materials by (ID mean RMSD + OOD mean RMSD)/2; return top 3 and worst 3."""
    if not all(c in sample_metrics_df.columns for c in ["material", "split", "rmsd"]):
        return [], []
    run_cols = ["model", "seed"]
    if not all(c in sample_metrics_df.columns for c in run_cols):
        return [], []
    per_run = (
        sample_metrics_df.groupby(run_cols + ["material", "split"])["rmsd"]
        .mean()
        .reset_index()
    )
    piv = per_run.pivot_table(
        index=run_cols + ["material"], columns="split", values="rmsd", aggfunc="mean"
    )
    if "id_test" not in piv.columns or "ood_test" not in piv.columns:
        return [], []
    combined = ((piv["id_test"] + piv["ood_test"]) / 2).dropna()
    if combined.empty:
        return [], []
    material_loss = combined.groupby("material").mean().sort_values()
    materials = material_loss.index.tolist()
    if len(materials) < 3:
        return materials, []
    return materials[:3], materials[-3:][::-1]


def _frontier_summary_block(
    lines: List[str],
    aggregate_df: pd.DataFrame,
    models: List[str],
) -> None:
    """
    Extra table summarizing frontier radius per model — the headline result.
    """
    frontier_cols = [
        c
        for c in [
            "frontier_radius_rmsd",
            "frontier_radius_bond_mae",
            "frontier_radius_rg_error",
            "frontier_fraction_rmsd",
        ]
        if c in aggregate_df.columns
    ]
    if not frontier_cols:
        return

    display_names = {
        "frontier_radius_rmsd": "r* (RMSD)",
        "frontier_radius_bond_mae": "r* (Bond)",
        "frontier_radius_rg_error": "r* (Rg)",
        "frontier_fraction_rmsd": "Frac (RMSD)",
    }
    col_w = 16
    sep = " "

    avail = [c for c in frontier_cols if c in aggregate_df.columns]
    if not avail:
        return

    means = aggregate_df.groupby("model")[avail].mean()
    stds = aggregate_df.groupby("model")[avail].std()

    # Overall row
    overall_mean = aggregate_df[avail].mean().to_frame().T
    overall_mean.index = ["Overall"]
    overall_std = aggregate_df[avail].std().to_frame().T
    overall_std.index = ["Overall"]
    means = pd.concat([means, overall_mean])
    stds = pd.concat([stds, overall_std])

    header = (
        "model".ljust(14)
        + sep
        + sep.join(display_names.get(c, c)[:col_w].ljust(col_w) for c in avail)
    )
    sep_len = max(len(header), 72)

    lines.append("")
    lines.append("=" * sep_len)
    lines.append("  Extrapolation Frontier Summary")
    lines.append("=" * sep_len)
    lines.append(header)
    lines.append("-" * sep_len)

    for row_name in models + ["Overall"]:
        if row_name not in means.index:
            continue
        parts = [row_name.ljust(14), sep]
        for c in avail:
            m = means.loc[row_name, c]
            s = stds.loc[row_name, c] if row_name in stds.index else np.nan
            parts.append(_format_cell(m, s).ljust(col_w) + sep)
        lines.append("".join(parts).rstrip())
    lines.append("")


def print_metric_tables(
    aggregate_df: pd.DataFrame,
    sample_metrics_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """
    Tables: Generation Quality, Failure Decomposition, Frontier Characterization
    (per model + Overall); Frontier Summary; ID vs OOD; Top3/Worst3 materials.
    """
    if aggregate_df.empty or "model" not in aggregate_df.columns:
        return []

    lines: List[str] = []
    models = sorted(aggregate_df["model"].unique().tolist())
    col_width = 16

    # ------ Group tables (Generation Quality, Failure Decomposition, Frontier) ------
    for group_name, metrics in METRIC_GROUPS.items():
        metrics_in_df = [
            (d, id_c, ood_c)
            for d, id_c, ood_c in metrics
            if (id_c and id_c in aggregate_df.columns)
            or (ood_c and ood_c in aggregate_df.columns)
        ]
        if not metrics_in_df:
            continue
        cols = []
        for _, id_c, ood_c in metrics_in_df:
            if id_c and id_c in aggregate_df.columns:
                cols.append(id_c)
            if ood_c and ood_c in aggregate_df.columns:
                cols.append(ood_c)
        cols = list(dict.fromkeys(cols))
        if not cols:
            continue
        means_by_model = aggregate_df.groupby("model")[cols].mean()
        stds_by_model = aggregate_df.groupby("model")[cols].std()
        overall_mean = aggregate_df[cols].mean().to_frame().T
        overall_mean.index = ["Overall (all)"]
        means = pd.concat([means_by_model, overall_mean])
        stds_over = aggregate_df[cols].std().to_frame().T
        stds_over.index = ["Overall (all)"]
        stds = pd.concat([stds_by_model, stds_over])
        _table_block_id_ood(
            lines,
            group_name,
            row_names=models + ["Overall (all)"],
            means=means,
            stds=stds,
            metrics=metrics_in_df,
            col_width=min(col_width, _ID_OOD_WIDTH),
        )

    # ------ Extrapolation Frontier Summary (headline table) ------
    _frontier_summary_block(lines, aggregate_df, models)

    # ------ Top 3 / Worst 3 materials ------
    if sample_metrics_df is None or sample_metrics_df.empty:
        return lines
    if "split" not in sample_metrics_df.columns:
        return lines
    top3, worst3 = _top3_worst3_materials(sample_metrics_df)
    run_cols = ["model", "seed"]
    sample_cols = [
        c
        for c in [
            "rmsd",
            "rg_error",
            "bond_mae",
            "surface_interior_ratio",
            "coord_preservation",
        ]
        if c in sample_metrics_df.columns
    ]
    if not sample_cols and "rmsd" in sample_metrics_df.columns:
        sample_cols = ["rmsd"]
    if not sample_cols or not (top3 or worst3):
        return lines
    display_s = ["RMSD", "Rg err", "Bond MAE", "Surf./inter.", "Coord preserv."][
        : len(sample_cols)
    ]
    sw = _ID_OOD_WIDTH
    sep = " "
    block_w = 2 * sw + 2 * len(sep)
    lines.append("")
    lines.append("=" * 72)
    lines.append("  By material performance (ID vs OOD)")
    lines.append("=" * 72)
    header1 = (
        "  "
        + "".join(
            (d[: block_w - len(sep)] if len(d) > block_w - len(sep) else d).ljust(
                block_w
            )
            + sep
            for d in display_s
        ).rstrip()
    )
    header2 = (
        "  "
        + "".join(
            "ID".ljust(sw) + sep + "OOD".ljust(sw) + sep for _ in sample_cols
        ).rstrip()
    )
    sep_len = max(len(header1), len(header2), 72)
    lines.append(header1)
    lines.append(header2)
    lines.append("  " + "-" * (sep_len - 2))
    for label, mat_list in [("Top 3 (best)", top3), ("Worst 3", worst3)]:
        if not mat_list:
            lines.append(f"  {label}: (none)")
            continue
        mask = sample_metrics_df["material"].isin(mat_list)
        sub = sample_metrics_df.loc[mask]
        run_split_means = (
            sub.groupby(run_cols + ["split"])[sample_cols].mean().reset_index()
        )
        id_means = run_split_means.loc[
            run_split_means["split"] == "id_test", sample_cols
        ]
        ood_means = run_split_means.loc[
            run_split_means["split"] == "ood_test", sample_cols
        ]
        id_overall_mean = (
            id_means.mean()
            if not id_means.empty
            else pd.Series({c: np.nan for c in sample_cols})
        )
        id_overall_std = (
            id_means.std()
            if not id_means.empty
            else pd.Series({c: np.nan for c in sample_cols})
        )
        ood_overall_mean = (
            ood_means.mean()
            if not ood_means.empty
            else pd.Series({c: np.nan for c in sample_cols})
        )
        ood_overall_std = (
            ood_means.std()
            if not ood_means.empty
            else pd.Series({c: np.nan for c in sample_cols})
        )
        row_parts = [f"  {label}".ljust(30)]
        for c in sample_cols:
            mid, sid = id_overall_mean.get(c, np.nan), id_overall_std.get(c, np.nan)
            moo, soo = ood_overall_mean.get(c, np.nan), ood_overall_std.get(c, np.nan)
            row_parts.append(
                _format_cell(mid, sid).ljust(sw)
                + sep
                + _format_cell(moo, soo).ljust(sw)
                + sep
            )
        lines.append("".join(row_parts).rstrip())
        lines.append("    Materials: " + ", ".join(mat_list))
    lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Load results for analysis (real data only: from id/ood NPZ and epochs CSV)."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root results directory (default: results)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Models to include (default: auto-discover)",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seeds to include (default: auto-discover)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="If set, write aggregate_df and epochs_df to CSV here",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print missing list (only summary count)",
    )
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Ignore saved metrics; recompute from NPZ",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel runs (default: 1). Use 4–8 for faster recompute.",
    )
    args = parser.parse_args()

    if args.recompute_metrics:
        print(
            "Recomputing metrics from NPZ (this can take a while: ~13k samples × 5 metrics per run).",
            flush=True,
        )
    if args.jobs > 1:
        print(f"Using {args.jobs} parallel workers.", flush=True)

    aggregate_df, epochs_df, missing, sample_metrics_df = load_all_results(
        results_root=args.results_dir,
        models=args.models,
        seeds=args.seeds,
        recompute_metrics=args.recompute_metrics,
        jobs=args.jobs,
    )

    # Report missing
    print("Missing or failed-to-load artifacts:")
    if not missing:
        print("  (none)")
    else:
        if not args.quiet:
            for m in missing:
                print(f"  {m}")
        print(f"  Total: {len(missing)} missing/failed.")

    # Summary
    print("\nLoaded data for analysis:")
    print(f"  Aggregate: {len(aggregate_df)} runs (rows)")
    if not aggregate_df.empty:
        print(f"  Models: {aggregate_df['model'].unique().tolist()}")
        print(f"  Seeds:  {aggregate_df['seed'].unique().tolist()}")
    print(f"  Epochs:  {len(epochs_df)} rows total")
    if aggregate_df.empty:
        print(
            "  (No runs with valid id/ood NPZ; run training and ensure eval_predictions exist.)"
        )
    else:
        # Warn when a model has no valid metrics (all "—" in tables)
        id_col = "id_rmsd_mean"
        ood_col = "ood_rmsd_mean"
        if id_col in aggregate_df.columns and ood_col in aggregate_df.columns:
            by_model = aggregate_df.groupby("model")[[id_col, ood_col]].mean()
            no_valid = [
                m
                for m in by_model.index
                if pd.isna(by_model.loc[m, id_col])
                and pd.isna(by_model.loc[m, ood_col])
            ]
            if no_valid:
                print(
                    f"  Note: {no_valid} have no valid metrics (—). Predictions may be NaN/Inf or atom counts may not match GT."
                )

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # Tables
    table_lines = print_metric_tables(aggregate_df, sample_metrics_df)
    if table_lines:
        print("\n" + "\n".join(table_lines))
        if args.out_dir:
            table_path = os.path.join(args.out_dir, "metric_tables.txt")
            with open(table_path, "w") as f:
                f.write("\n".join(table_lines))
            print(f"Wrote {table_path}")

    if args.out_dir:
        agg_path = os.path.join(args.out_dir, "aggregate_metrics.csv")
        ep_path = os.path.join(args.out_dir, "epochs.csv")
        aggregate_df.to_csv(agg_path, index=False)
        epochs_df.to_csv(ep_path, index=False)
        print(f"\nWrote {agg_path}, {ep_path}")

    return aggregate_df, epochs_df


if __name__ == "__main__":
    main()
