"""
Generate supplementary result tables from existing computed metrics.

Addresses reviewer concerns that require tabulated results (no recomputation
of metrics from raw predictions, except for frontier radii at multiple
thresholds and scaling law parameters):
- Reviewer 4ixz W2/D2: Quantitative result tables
- Reviewer XQdQ Point 2b: Baseline comparison table
- Reviewer XQdQ Point 5: Frontier radius table
- Reviewer sp5a Point 1: Atom count table (OOD gap evidence)
- Reviewer sp5a Point 3: Normalized OOD/ID degradation ratios
- Reviewer nm9J W2: Data summary

Usage:
    conda run -n aclWork python -m scripts.generate_tables
    conda run -n aclWork python -m scripts.generate_tables --results-dir results --out-dir results/tables
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import linregress

from radii.metrics import frontier_radius
from radii.train_config import TrainConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ID_RADII = TrainConfig.ID_RADII
OOD_RADII = TrainConfig.OOD_RADII
MODELS = ["adit", "cdvae", "diffcsp", "flowmm", "mattergen"]
SEEDS = TrainConfig.SEEDS

MATERIALS = [
    "Ag", "Au", "CH3NH3PbI3", "Fe2O3", "MoS2",
    "PbS", "SnO2", "SrTiO3", "TiO2", "ZnO",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_aggregate_metrics(results_dir: str) -> pd.DataFrame:
    """Load all aggregate_metrics.json files into a DataFrame."""
    records = []
    for model in MODELS:
        for seed in SEEDS:
            path = os.path.join(results_dir, "task_1", model, str(seed), "aggregate_metrics.json")
            if os.path.isfile(path):
                with open(path) as f:
                    records.append(json.load(f))
    return pd.DataFrame(records)


def load_all_sample_metrics(results_dir: str) -> pd.DataFrame:
    """Load all sample_metrics.csv files into a single DataFrame."""
    frames = []
    for model in MODELS:
        for seed in SEEDS:
            path = os.path.join(results_dir, "task_1", model, str(seed), "sample_metrics.csv")
            if os.path.isfile(path):
                frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_gt_atom_counts(data_root: str) -> pd.DataFrame:
    """Load atom counts per (material, radius) from GT cache."""
    records = []
    for split_name, split_label in [("id_test", "ID"), ("ood_test", "OOD")]:
        path = os.path.join(data_root, "gt_cache", f"{split_name}.npz")
        if not os.path.isfile(path):
            continue
        data = np.load(path, allow_pickle=True)
        ptr = data["ptr"]
        materials = data["materials"]
        radius = data["radius"]
        for i in range(len(ptr) - 1):
            n_atoms = int(ptr[i + 1]) - int(ptr[i])
            records.append({
                "material": str(materials[i]),
                "radius": float(radius[i]),
                "n_atoms": n_atoms,
                "split": split_label,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Table A+B: Metrics Summary + Degradation Ratios
# ---------------------------------------------------------------------------
def build_metrics_summary(agg_df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Table A: Full Metrics Summary")
    lines.append("**Addresses:** Reviewer 4ixz W2/D2, Reviewer XQdQ Point 2b — Quantitative results\n")
    lines.append("All values: mean +/- std across 3 seeds.\n")

    metrics = [
        ("RMSD", "id_rmsd_mean", "ood_rmsd_mean"),
        ("BondMAE", "id_bond_mae_mean", "ood_bond_mae_mean"),
        ("RgError", "id_rg_error_mean", "ood_rg_error_mean"),
        ("CoordCorr", "id_coord_mean", "ood_coord_mean"),
        ("SurfIntRatio", "id_surface_ratio_mean", "ood_surface_ratio_mean"),
    ]

    header = "| Model |"
    sep = "|-------|"
    for name, _, _ in metrics:
        header += f" {name} ID | {name} OOD |"
        sep += "---------|---------|"
    lines.append(header)
    lines.append(sep)

    for model in MODELS:
        mdf = agg_df[agg_df["model"] == model]
        row = f"| {model} |"
        for _, id_col, ood_col in metrics:
            id_mean = mdf[id_col].mean()
            id_std = mdf[id_col].std()
            ood_mean = mdf[ood_col].mean()
            ood_std = mdf[ood_col].std()
            row += f" {id_mean:.4f}+/-{id_std:.4f} | {ood_mean:.4f}+/-{ood_std:.4f} |"
        lines.append(row)
    lines.append("")

    # Table B: Degradation Ratios
    lines.append("## Table B: ID -> OOD Degradation Ratios")
    lines.append("**Addresses:** Reviewer 4ixz W2/D2 — Degradation ratios per metric\n")
    lines.append("Degradation = OOD_mean / ID_mean. Values > 1 indicate OOD is worse.\n")

    header = "| Model |"
    sep = "|-------|"
    for name, _, _ in metrics:
        header += f" {name} |"
        sep += "---------|"
    lines.append(header)
    lines.append(sep)

    for model in MODELS:
        mdf = agg_df[agg_df["model"] == model]
        row = f"| {model} |"
        for _, id_col, ood_col in metrics:
            id_mean = mdf[id_col].mean()
            ood_mean = mdf[ood_col].mean()
            ratio = ood_mean / (id_mean + 1e-8)
            row += f" {ratio:.4f} |"
        lines.append(row)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table C: Frontier Radii
# ---------------------------------------------------------------------------
def build_frontier_table(sample_df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Table C: Frontier Radii r*(m, tau)")
    lines.append("**Addresses:** Reviewer XQdQ Point 5 — Frontier radius table for main text\n")
    lines.append("r* = largest radius where metric <= threshold. NaN = no radius qualifies.\n")

    thresholds = [
        # Strict (likely NaN for 500K-param models — shows capacity limitation)
        ("RMSD", "rmsd", 1.0, True),
        ("RMSD", "rmsd", 2.0, True),
        # Practical (differentiate architectures)
        ("RMSD", "rmsd", 5.0, True),
        ("RMSD", "rmsd", 8.0, True),
        ("RMSD", "rmsd", 10.0, True),
        ("RMSD", "rmsd", 15.0, True),
        # Strict
        ("BondMAE", "bond_mae", 0.3, True),
        ("BondMAE", "bond_mae", 0.5, True),
        # Practical
        ("BondMAE", "bond_mae", 0.6, True),
        ("BondMAE", "bond_mae", 0.7, True),
        ("BondMAE", "bond_mae", 0.8, True),
        ("BondMAE", "bond_mae", 1.0, True),
    ]

    header = "| Model |"
    sep = "|-------|"
    for name, _, tau, _ in thresholds:
        header += f" r*({name}, {tau}) |"
        sep += "------------|"
    lines.append(header)
    lines.append(sep)

    for model in MODELS:
        mdf = sample_df[sample_df["model"] == model]
        row = f"| {model} |"
        for _, col, tau, lower_is_better in thresholds:
            # Average metric per radius across all seeds and rotations
            by_radius = mdf.groupby("radius")[col].mean().to_dict()
            result = frontier_radius(by_radius, tau, lower_is_better=lower_is_better)
            fr = result["frontier_radius"]
            cell = f"{fr:.0f}" if not np.isnan(fr) else "NaN"
            row += f" {cell} |"
        lines.append(row)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table D: Scaling Law Parameters
# ---------------------------------------------------------------------------
def build_scaling_law_table(sample_df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Table D: Scaling Law Parameters")
    lines.append("**Addresses:** Reviewer 4ixz W2/D2 — Power-law exponents, R^2, OOD residuals\n")
    lines.append("Fit: log10(RMSD) = alpha * log10(N_atoms) + intercept, on ID radii only.\n")

    header = "| Model | alpha (slope) | R^2 | Intercept | OOD Residual |"
    sep = "|-------|-------------|-----|-----------|-------------|"
    lines.append(header)
    lines.append(sep)

    id_set = set(ID_RADII)
    ood_set = set(OOD_RADII)

    for model in MODELS:
        mdf = sample_df[sample_df["model"] == model]
        # Group by radius, average across seeds and rotations
        by_radius = mdf.groupby("radius").agg({"n_atoms": "mean", "rmsd": "mean"}).reset_index()
        by_radius = by_radius[(by_radius["n_atoms"] > 0) & (by_radius["rmsd"] > 0)]

        log_n = np.log10(by_radius["n_atoms"].values)
        log_r = np.log10(by_radius["rmsd"].values)
        id_mask = by_radius["radius"].isin(id_set).values
        ood_mask = by_radius["radius"].isin(ood_set).values

        if id_mask.sum() >= 3:
            sl, ic, rv, _, se = linregress(log_n[id_mask], log_r[id_mask])
            r2 = rv ** 2
            if ood_mask.sum() > 0:
                pred_ood = sl * log_n[ood_mask] + ic
                actual_ood = log_r[ood_mask]
                ood_residual = np.mean(np.abs(actual_ood - pred_ood))
            else:
                ood_residual = float("nan")
            lines.append(f"| {model} | {sl:.4f} | {r2:.4f} | {ic:.4f} | {ood_residual:.4f} |")
        else:
            lines.append(f"| {model} | N/A | N/A | N/A | N/A |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table E: Atom Counts
# ---------------------------------------------------------------------------
def build_atom_count_table(gt_df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Table E: Atom Count per Radius and Material")
    lines.append("**Addresses:** Reviewer sp5a Point 1 — OOD gap is 20-40% by atom count, not just radius\n")
    lines.append("Train radii: 9-10, 12-26 (even). ID radii: 11, 13, 15, 17, 19, 21. OOD radii: 6, 7, 29, 30.\n")

    # Deduplicate: take one atom count per (material, radius)
    unique = gt_df.drop_duplicates(subset=["material", "radius"]).copy()
    pivot = unique.pivot_table(index="radius", columns="material", values="n_atoms", aggfunc="first")
    pivot = pivot.sort_index()

    materials = sorted(unique["material"].unique())
    header = "| Radius |"
    sep = "|--------|"
    for mat in materials:
        header += f" {mat} |"
        sep += "------|"
    header += " Split |"
    sep += "-------|"
    lines.append(header)
    lines.append(sep)

    id_set = set(ID_RADII)
    ood_set = set(OOD_RADII)

    for radius in sorted(pivot.index):
        row = f"| {radius:.0f} |"
        for mat in materials:
            val = pivot.loc[radius, mat] if mat in pivot.columns and not pd.isna(pivot.loc[radius].get(mat)) else None
            row += f" {int(val)} |" if val is not None else " - |"
        if radius in ood_set:
            row += " **OOD** |"
        elif radius in id_set:
            row += " ID |"
        else:
            row += " Train |"
        lines.append(row)
    lines.append("")

    # Summary statistics
    lines.append("### Atom Count Ranges by Split\n")
    for split in ["ID", "OOD"]:
        sdf = gt_df[gt_df["split"] == split]
        if len(sdf):
            lines.append(f"**{split}**: {sdf['n_atoms'].min()} - {sdf['n_atoms'].max()} atoms "
                         f"(mean: {sdf['n_atoms'].mean():.0f})")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table F: Normalized Degradation Ratios
# ---------------------------------------------------------------------------
def build_degradation_table(agg_df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Table F: Normalized OOD/ID Degradation Ratios")
    lines.append("**Addresses:** Reviewer sp5a Point 3 — Making the '13% degradation' claim explicit\n")
    lines.append("Values show OOD_mean / ID_mean per metric. >1 means OOD is worse.\n")

    metrics = [
        ("RMSD", "id_rmsd_mean", "ood_rmsd_mean"),
        ("BondMAE", "id_bond_mae_mean", "ood_bond_mae_mean"),
        ("RgError", "id_rg_error_mean", "ood_rg_error_mean"),
        ("CoordCorr", "id_coord_mean", "ood_coord_mean"),
    ]

    header = "| Model | Seed |"
    sep = "|-------|------|"
    for name, _, _ in metrics:
        header += f" {name} |"
        sep += "---------|"
    lines.append(header)
    lines.append(sep)

    for model in MODELS:
        for seed in SEEDS:
            mdf = agg_df[(agg_df["model"] == model) & (agg_df["seed"] == seed)]
            if len(mdf) == 0:
                continue
            row = f"| {model} | {seed} |"
            for _, id_col, ood_col in metrics:
                id_val = mdf[id_col].values[0]
                ood_val = mdf[ood_col].values[0]
                ratio = ood_val / (id_val + 1e-8)
                row += f" {ratio:.4f} |"
            lines.append(row)
    lines.append("")

    # Per-model average
    lines.append("### Per-Model Average (across seeds)\n")
    header = "| Model |"
    sep = "|-------|"
    for name, _, _ in metrics:
        header += f" {name} |"
        sep += "---------|"
    lines.append(header)
    lines.append(sep)

    for model in MODELS:
        mdf = agg_df[agg_df["model"] == model]
        row = f"| {model} |"
        for _, id_col, ood_col in metrics:
            ratios = mdf[ood_col].values / (mdf[id_col].values + 1e-8)
            row += f" {ratios.mean():.4f}+/-{ratios.std():.4f} |"
        lines.append(row)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table G: Data Summary
# ---------------------------------------------------------------------------
def build_data_summary(gt_df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Table G: Dataset Summary")
    lines.append("**Addresses:** Reviewer nm9J W2 — Description of data used\n")

    lines.append("### Materials\n")
    lines.append("| Material | Crystal System | Radii Available |")
    lines.append("|----------|---------------|----------------|")
    for mat in MATERIALS:
        mdf = gt_df[gt_df["material"] == mat]
        radii = sorted(mdf["radius"].unique())
        lines.append(f"| {mat} | - | {len(radii)} ({min(radii):.0f}-{max(radii):.0f} A) |")
    lines.append("")

    lines.append("### Split Statistics\n")
    lines.append("| Split | Radii | Structures | Atom Range | Mean Atoms |")
    lines.append("|-------|-------|-----------|-----------|-----------|")

    for split_label, split_name in [("ID", "ID"), ("OOD", "OOD")]:
        sdf = gt_df[gt_df["split"] == split_name]
        if len(sdf):
            radii = sorted(sdf["radius"].unique())
            radii_str = ", ".join(f"{r:.0f}" for r in radii)
            lines.append(
                f"| {split_label} | {radii_str} | {len(sdf)} | "
                f"{sdf['n_atoms'].min()}-{sdf['n_atoms'].max()} | "
                f"{sdf['n_atoms'].mean():.0f} |"
            )
    lines.append("")

    lines.append("### Atom Count Gap (OOD vs Training)\n")
    lines.append("This table shows that OOD structures differ by 20-40% in atom count,")
    lines.append("not just 5-10% in radius.\n")

    id_atoms = gt_df[gt_df["split"] == "ID"]
    ood_atoms = gt_df[gt_df["split"] == "OOD"]
    if len(id_atoms) and len(ood_atoms):
        id_min, id_max = id_atoms["n_atoms"].min(), id_atoms["n_atoms"].max()
        ood_min, ood_max = ood_atoms["n_atoms"].min(), ood_atoms["n_atoms"].max()
        lines.append(f"- **ID range**: {id_min} - {id_max} atoms")
        lines.append(f"- **OOD range**: {ood_min} - {ood_max} atoms")
        small_gap = (id_min - ood_min) / id_min * 100
        large_gap = (ood_max - id_max) / id_max * 100
        lines.append(f"- **Small-end gap**: OOD smallest ({ood_min}) is {small_gap:.0f}% smaller than ID smallest ({id_min})")
        lines.append(f"- **Large-end gap**: OOD largest ({ood_max}) is {large_gap:.0f}% larger than ID largest ({id_max})")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate supplementary result tables")
    parser.add_argument("--results-dir", default="results", help="Results root directory")
    parser.add_argument("--data-root", default="radii", help="Dataset root directory")
    parser.add_argument("--out-dir", default="results/tables", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Loading aggregate metrics ===")
    agg_df = load_aggregate_metrics(args.results_dir)
    print(f"  Loaded {len(agg_df)} runs")

    print("=== Loading sample metrics ===")
    sample_df = load_all_sample_metrics(args.results_dir)
    print(f"  Loaded {len(sample_df)} samples")

    print("=== Loading GT atom counts ===")
    gt_df = load_gt_atom_counts(args.data_root)
    print(f"  Loaded {len(gt_df)} entries")

    # Build and save tables
    tables = [
        ("metrics_summary.md", build_metrics_summary(agg_df)),
        ("frontier_radii.md", build_frontier_table(sample_df)),
        ("scaling_law.md", build_scaling_law_table(sample_df)),
        ("atom_counts.md", build_atom_count_table(gt_df)),
        ("degradation_ratios.md", build_degradation_table(agg_df)),
        ("data_summary.md", build_data_summary(gt_df)),
    ]

    for filename, content in tables:
        path = os.path.join(args.out_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        print(f"  Saved: {path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
