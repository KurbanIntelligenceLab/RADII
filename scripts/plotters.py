"""
RADII Paper Figures: Characterizing the Extrapolation Frontier

Generates three publication figures from analysis outputs:
  Figure 1 (2x2): Per-radius profiles — the hero figure
  Figure 2: Material × Model degradation heatmap
  Figure 3: Scaling law fits (log-log)

All error metrics are optionally normalized by per-model ID mean so that
ID baseline ≈ 1.0 and OOD values show relative degradation. This makes
cross-model comparison visually clean and numerically interpretable.

Usage:
  python scripts/plotters.py --results-dir results --out-dir figures
  python scripts/plotters.py --results-dir results --out-dir figures --no-normalize
  python scripts/plotters.py --results-dir results --out-dir figures --format pdf
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.lines import Line2D
from scipy.stats import linregress

try:
    from radii.train_config import TrainConfig
    ID_RADII = sorted(TrainConfig.ID_RADII)
    OOD_RADII = sorted(TrainConfig.OOD_RADII)
except Exception:
    ID_RADII = [11, 13, 15, 17, 19, 21]
    OOD_RADII = [6, 7, 29, 30]

ALL_RADII = sorted(set(ID_RADII + OOD_RADII))
ID_OOD_BOUNDARY = max(ID_RADII) + 0.5


# ── Style & Display ─────────────────────────────────────────────────────────

MODEL_COLORS = {
    "adit":      "#0072B2",
    "cdvae":     "#D55E00",
    "flowmm":    "#009E73",
    "diffcsp":   "#CC79A7",
    "mattergen": "#E69F00",
}
MODEL_MARKERS = {
    "adit": "o", "cdvae": "s", "flowmm": "^", "diffcsp": "D", "mattergen": "v",
}
MODEL_DISPLAY = {
    "adit": "ADiT", "cdvae": "CDVAE", "flowmm": "FlowMM",
    "diffcsp": "DiffCSP", "mattergen": "MatterGen",
}
MATERIAL_DISPLAY = {
    "Ag": "Ag", "Au": "Au", "CH3NH3PbI3": r"MAPbI$_3$",
    "Fe2O3": r"Fe$_2$O$_3$", "MoS2": r"MoS$_2$", "PbS": "PbS",
    "SnO2": r"SnO$_2$", "SrTiO3": r"SrTiO$_3$", "TiO2": r"TiO$_2$",
    "ZnO": "ZnO",
}


def _style_defaults():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    })


def _c(m): return MODEL_COLORS.get(m, "#555555")
def _mk(m): return MODEL_MARKERS.get(m, "x")
def _dn(m): return MODEL_DISPLAY.get(m, m)
def _dm(m): return MATERIAL_DISPLAY.get(m, m)


def _add_id_ood_shading(ax):
    ylim = ax.get_ylim()
    ax.axvspan(min(ALL_RADII) - 0.5, ID_OOD_BOUNDARY, alpha=0.06, color="#0072B2", zorder=0)
    ax.axvspan(ID_OOD_BOUNDARY, max(ALL_RADII) + 0.5, alpha=0.06, color="#D55E00", zorder=0)
    ax.axvline(ID_OOD_BOUNDARY, color="gray", ls="--", lw=1.2, alpha=0.7, zorder=1)
    ax.set_ylim(ylim)


_ANN_BBOX = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9)

# Well-behaved models — the complement of FIG2_EXCLUDE_MODELS. Figure 1's summary
# annotations average only these so the boxes match the per-model degradation ratios
# cited in the manuscript text; DiffCSP/MatterGen normalized ratios are not meaningful
# (relative change atop already-failed baselines) and skew a 5-model average.
WELL_BEHAVED_MODELS = ["adit", "cdvae", "flowmm"]


def _add_id_ood_labels(ax):
    # ID bottom-left, OOD bottom-right (axes coords)
    ax.text(0.02, 0.02, "ID", fontsize=7, ha="left", va="bottom",
            color="#0072B2", alpha=0.5, fontweight="bold", transform=ax.transAxes,
            bbox=_ANN_BBOX)
    ax.text(0.98, 0.02, "OOD", fontsize=7, ha="right", va="bottom",
            color="#D55E00", alpha=0.5, fontweight="bold", transform=ax.transAxes,
            bbox=_ANN_BBOX)


# ── Normalization ────────────────────────────────────────────────────────────


def _compute_id_baseline(
    sample_df: pd.DataFrame, metric_col: str, models: List[str],
) -> Dict[str, float]:
    """Per-model ID mean for normalization. Returns {model: id_mean}."""
    baselines = {}
    for model in models:
        sub = sample_df[(sample_df["model"] == model) & (sample_df["split"] == "id_test")]
        vals = sub[metric_col].dropna().values
        baselines[model] = float(np.mean(vals)) if len(vals) > 0 else 1.0
    return baselines


def _normalize_profile(
    radii: np.ndarray, means: np.ndarray, stds: np.ndarray, baseline: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide means and stds by baseline so ID ≈ 1.0."""
    if baseline < 1e-12:
        baseline = 1.0
    return radii, means / baseline, stds / baseline


# ── Per-radius profiles ─────────────────────────────────────────────────────


def _radius_profile(
    sample_df: pd.DataFrame, metric_col: str, models: List[str],
    normalize: bool = False, baselines: Optional[Dict[str, float]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-radius mean ± std, optionally normalized by ID baseline."""
    profiles = {}
    for model in models:
        sub = sample_df[sample_df["model"] == model]
        if sub.empty:
            continue
        g = sub.groupby("radius")[metric_col].agg(["mean", "std"]).reset_index().sort_values("radius")
        r, m, s = g["radius"].values, g["mean"].values, np.nan_to_num(g["std"].values, nan=0.0)
        if normalize and baselines and model in baselines:
            r, m, s = _normalize_profile(r, m, s, baselines[model])
        profiles[model] = (r, m, s)
    return profiles


def _rotation_profile(
    rot_df: pd.DataFrame, metric_col: str, models: List[str],
    normalize: bool = False, baselines: Optional[Dict[str, float]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    profiles = {}
    for model in models:
        sub = rot_df[rot_df["model"] == model]
        if sub.empty:
            continue
        g = sub.groupby("radius")[metric_col].agg(["mean", "std"]).reset_index().sort_values("radius")
        r, m, s = g["radius"].values, g["mean"].values, np.nan_to_num(g["std"].values, nan=0.0)
        if normalize and baselines and model in baselines:
            r, m, s = _normalize_profile(r, m, s, baselines[model])
        profiles[model] = (r, m, s)
    return profiles


# ── Statistics printer ───────────────────────────────────────────────────────


class StatsPrinter:
    """Collects and prints statistics for each figure/subfigure."""

    def __init__(self):
        self.sections: List[Tuple[str, List[str]]] = []
        self._current: List[str] = []
        self._title: str = ""

    def begin(self, title: str):
        if self._current:
            self.sections.append((self._title, self._current))
        self._title = title
        self._current = []

    def add(self, line: str):
        self._current.append(line)

    def add_profile_stats(
        self, label: str,
        profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        models: List[str],
        id_radii: List[float], ood_radii: List[float],
        unit: str = "",
        normalized: bool = False,
    ):
        """Print per-model ID mean, OOD mean, degradation ratio, frontier radius."""
        suffix = " (normalized)" if normalized else f" ({unit})" if unit else ""
        self.add(f"  {label}{suffix}:")
        self.add(f"    {'Model':<12} {'ID mean':>10} {'OOD mean':>10} {'Degrad':>8} "
                 f"{'ID std':>10} {'OOD std':>10} {'Min r':>8} {'Max r':>8}")
        self.add(f"    {'─'*12} {'─'*10} {'─'*10} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
        all_id, all_ood = [], []
        for model in models:
            if model not in profiles:
                continue
            r, m, s = profiles[model]
            id_mask = np.isin(r, id_radii)
            ood_mask = np.isin(r, ood_radii)
            id_m = m[id_mask] if id_mask.any() else np.array([np.nan])
            ood_m = m[ood_mask] if ood_mask.any() else np.array([np.nan])
            id_s = s[id_mask] if id_mask.any() else np.array([np.nan])
            ood_s = s[ood_mask] if ood_mask.any() else np.array([np.nan])
            id_mean, ood_mean = np.nanmean(id_m), np.nanmean(ood_m)
            degrad = ood_mean / (id_mean + 1e-12)
            all_id.append(id_mean)
            all_ood.append(ood_mean)
            self.add(
                f"    {_dn(model):<12} {id_mean:10.4f} {ood_mean:10.4f} {degrad:8.3f} "
                f"{np.nanmean(id_s):10.4f} {np.nanmean(ood_s):10.4f} "
                f"{np.nanmin(m):8.4f} {np.nanmax(m):8.4f}"
            )
        if all_id:
            self.add(
                f"    {'Overall':<12} {np.mean(all_id):10.4f} {np.mean(all_ood):10.4f} "
                f"{np.mean(all_ood) / (np.mean(all_id) + 1e-12):8.3f}"
            )
        self.add("")

    def add_heatmap_stats(
        self, label: str,
        matrix: np.ndarray,
        materials: List[str],
        models: List[str],
    ):
        """Print heatmap summary: hardest/easiest materials, per-model mean."""
        self.add(f"  {label}:")
        self.add(f"    Overall: min={np.nanmin(matrix):.3f}, max={np.nanmax(matrix):.3f}, "
                 f"mean={np.nanmean(matrix):.3f}, std={np.nanstd(matrix):.3f}")
        # Per model
        self.add("    Per-model mean degradation:")
        for j, model in enumerate(models):
            col = matrix[:, j]
            self.add(f"      {_dn(model):<12}: {np.nanmean(col):.3f} ± {np.nanstd(col):.3f}")
        # Hardest / easiest materials
        mat_means = np.nanmean(matrix, axis=1)
        hard_idx = np.argsort(-mat_means)
        self.add("    Hardest materials (highest degradation):")
        for k in hard_idx[:3]:
            self.add(f"      {materials[k]:<20}: {mat_means[k]:.3f}")
        self.add("    Easiest materials (lowest degradation):")
        for k in hard_idx[-3:]:
            self.add(f"      {materials[k]:<20}: {mat_means[k]:.3f}")
        self.add("")

    def add_scaling_stats(
        self, label: str,
        fit_results: Dict[str, Dict[str, float]],
    ):
        """Print scaling law fit results."""
        self.add(f"  {label}:")
        self.add(f"    {'Model':<12} {'α (slope)':>10} {'R²':>8} {'Intercept':>10} "
                 f"{'N_points':>10}")
        self.add(f"    {'─'*12} {'─'*10} {'─'*8} {'─'*10} {'─'*10}")
        slopes = []
        for model, res in fit_results.items():
            self.add(
                f"    {_dn(model):<12} {res['slope']:10.4f} {res['r2']:8.4f} "
                f"{res['intercept']:10.4f} {res['n_points']:10d}"
            )
            slopes.append(res["slope"])
        if slopes:
            self.add(f"    {'Overall':<12} α range: [{min(slopes):.4f}, {max(slopes):.4f}], "
                     f"mean={np.mean(slopes):.4f}")
        self.add("")

    def flush(self) -> str:
        if self._current:
            self.sections.append((self._title, self._current))
            self._current = []
        lines = []
        for title, content in self.sections:
            lines.append("")
            lines.append("=" * 72)
            lines.append(f"  STATS: {title}")
            lines.append("=" * 72)
            lines.extend(content)
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Hero Figure (2×2)
# ═════════════════════════════════════════════════════════════════════════════


def plot_figure1(
    sample_df: pd.DataFrame,
    rot_df: pd.DataFrame,
    models: List[str],
    normalize: bool = True,
    rmsd_threshold: float = 2.0,
    bond_threshold: float = 0.5,
    stats: Optional[StatsPrinter] = None,
    out_path: Optional[str] = None,
    fmt: str = "pdf",
):
    _style_defaults()
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.5), constrained_layout=True)

    # Baselines for normalization
    rmsd_bl = _compute_id_baseline(sample_df, "rmsd", models) if normalize else None
    bond_bl = _compute_id_baseline(sample_df, "bond_mae", models) if normalize else None
    surf_bl = _compute_id_baseline(sample_df, "surface_rmsd", models) if normalize else None
    int_bl = _compute_id_baseline(sample_df, "interior_rmsd", models) if normalize else None

    rot_bl = None
    if normalize and rot_df is not None and not rot_df.empty:
        rot_bl = {}
        for model in models:
            sub = rot_df[rot_df["model"] == model]
            id_sub = sub[sub["radius"].isin(ID_RADII)]
            vals = id_sub["rot_mean_pairwise_rmsd"].dropna().values
            rot_bl[model] = float(np.mean(vals)) if len(vals) > 0 else 1.0

    norm_label = ""
    threshold_rmsd = rmsd_threshold / np.mean(list(rmsd_bl.values())) if normalize and rmsd_bl else rmsd_threshold
    threshold_bond = bond_threshold / np.mean(list(bond_bl.values())) if normalize and bond_bl else bond_threshold

    # ── (a) RMSD ─────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    profiles = _radius_profile(sample_df, "rmsd", models, normalize, rmsd_bl)
    for model in models:
        if model not in profiles:
            continue
        r, m, s = profiles[model]
        ax.plot(r, m, color=_c(model), marker=_mk(model), ms=3, lw=1.2,
                label=_dn(model), zorder=3)
        ax.fill_between(r, m - s, m + s, alpha=0.12, color=_c(model), zorder=2)
    ax.axhline(threshold_rmsd, color="gray", ls=":", lw=1.2, alpha=0.6)
    if normalize:
        ax.axhline(1.0, color="black", ls="-", lw=1.0, alpha=0.3)
    _add_id_ood_shading(ax)
    _add_id_ood_labels(ax)
    ax.set_xlabel("Radius (Å)")
    ax.set_ylabel(f"RMSD{norm_label}")
    ax.set_title("(a) RMSD vs. radius", fontweight="bold", loc="left")
    _id, _ood = [], []
    for m in WELL_BEHAVED_MODELS:
        if m not in profiles:
            continue
        r, means, _ = profiles[m]
        id_m = np.mean(means[np.isin(r, ID_RADII)]) if np.any(np.isin(r, ID_RADII)) else np.nan
        ood_m = np.mean(means[np.isin(r, OOD_RADII)]) if np.any(np.isin(r, OOD_RADII)) else np.nan
        if not np.isnan(id_m):
            _id.append(id_m)
        if not np.isnan(ood_m):
            _ood.append(ood_m)
    avg_id = np.mean(_id) if _id else np.nan
    avg_ood = np.mean(_ood) if _ood else np.nan
    degrad = avg_ood / (avg_id + 1e-12) if _id and _ood else np.nan
    ann = f"stable models\nOOD/ID: ×{degrad:.2f}"
    ax.text(0.02, 0.98, ann, fontsize=8, va="top", ha="left", transform=ax.transAxes, color="gray",
            bbox=_ANN_BBOX)

    if stats:
        stats.begin("Figure 1(a): RMSD vs. radius")
        stats.add_profile_stats("RMSD", profiles, models, ID_RADII, OOD_RADII,
                                unit="Å", normalized=normalize)

    # ── (b) BondMAE ──────────────────────────────────────────────────────────
    ax = axes[0, 1]
    profiles_b = _radius_profile(sample_df, "bond_mae", models, normalize, bond_bl)
    for model in models:
        if model not in profiles_b:
            continue
        r, m, s = profiles_b[model]
        ax.plot(r, m, color=_c(model), marker=_mk(model), ms=3, lw=1.2,
                label=_dn(model), zorder=3)
        ax.fill_between(r, m - s, m + s, alpha=0.12, color=_c(model), zorder=2)
    ax.axhline(threshold_bond, color="gray", ls=":", lw=1.2, alpha=0.6)
    if normalize:
        ax.axhline(1.0, color="black", ls="-", lw=1.0, alpha=0.3)
    _add_id_ood_shading(ax)
    _add_id_ood_labels(ax)
    ax.set_xlabel("Radius (Å)")
    ax.set_ylabel(f"Bond MAE{norm_label}")
    ax.set_title("(b) Local bond error vs. radius", fontweight="bold", loc="left")
    _id, _ood = [], []
    for m in WELL_BEHAVED_MODELS:
        if m not in profiles_b:
            continue
        r, means, _ = profiles_b[m]
        id_m = np.mean(means[np.isin(r, ID_RADII)]) if np.any(np.isin(r, ID_RADII)) else np.nan
        ood_m = np.mean(means[np.isin(r, OOD_RADII)]) if np.any(np.isin(r, OOD_RADII)) else np.nan
        if not np.isnan(id_m):
            _id.append(id_m)
        if not np.isnan(ood_m):
            _ood.append(ood_m)
    avg_id = np.mean(_id) if _id else np.nan
    avg_ood = np.mean(_ood) if _ood else np.nan
    degrad = avg_ood / (avg_id + 1e-12) if _id and _ood else np.nan
    ann = f"stable models\nOOD/ID: ×{degrad:.2f}"
    ax.text(0.02, 0.98, ann, fontsize=8, va="top", ha="left", transform=ax.transAxes, color="gray",
            bbox=_ANN_BBOX)

    if stats:
        stats.begin("Figure 1(b): Bond MAE vs. radius")
        stats.add_profile_stats("BondMAE", profiles_b, models, ID_RADII, OOD_RADII,
                                unit="Å", normalized=normalize)

    # ── (c) Surface vs Interior ──────────────────────────────────────────────
    ax = axes[0, 2]
    surf_profiles = _radius_profile(sample_df, "surface_rmsd", models, normalize, surf_bl)
    int_profiles = _radius_profile(sample_df, "interior_rmsd", models, normalize, int_bl)
    for model in models:
        if model not in surf_profiles or model not in int_profiles:
            continue
        c = _c(model)
        r_s, m_s, _ = surf_profiles[model]
        r_i, m_i, _ = int_profiles[model]
        ax.plot(r_s, m_s, color=c, lw=1.6, ls="-", marker=_mk(model), ms=3, zorder=3)
        ax.plot(r_i, m_i, color=c, lw=1.6, ls="--", marker=_mk(model), ms=3,
                markerfacecolor="white", alpha=0.8, zorder=3)
        ax.fill_between(r_s, m_i, m_s, alpha=0.08, color=c, zorder=2)
    legend_el = [
        Line2D([0], [0], color="black", lw=1.8, ls="-", label="Surface"),
        Line2D([0], [0], color="black", lw=1.8, ls="--", label="Interior"),
    ]
    ax.legend(handles=legend_el, fontsize=6.5, ncol=1, loc="center right",
              framealpha=0.8, edgecolor="none")
    if normalize:
        ax.axhline(1.0, color="black", ls="-", lw=1.0, alpha=0.3)
    _add_id_ood_shading(ax)
    _add_id_ood_labels(ax)
    ax.set_xlabel("Radius (Å)")
    ax.set_ylabel(f"Regional RMSD{norm_label}")
    ax.set_title("(c) Surface vs. interior error", fontweight="bold", loc="left")
    gaps = []
    for m in models:
        if m not in surf_profiles or m not in int_profiles:
            continue
        r_s, m_s, _ = surf_profiles[m]
        r_i, m_i, _ = int_profiles[m]
        ood_s = m_s[np.isin(r_s, OOD_RADII)]
        ood_i = m_i[np.isin(r_i, OOD_RADII)]
        if len(ood_s) > 0 and len(ood_i) > 0:
            gap = np.mean(ood_s) / (np.mean(ood_i) + 1e-12)
            gaps.append(gap)
    avg_gap = np.mean(gaps) if gaps else np.nan
    min_gap = np.min(gaps) if gaps else np.nan
    max_gap = np.max(gaps) if gaps else np.nan
    ann = f"OOD S/I gap:\navg: {avg_gap:.2f}×\nmin: {min_gap:.2f}×, max: {max_gap:.2f}×"
    ax.text(0.02, 0.98, ann, fontsize=8, va="top", ha="left", transform=ax.transAxes, color="gray",
            bbox=_ANN_BBOX)

    if stats:
        stats.begin("Figure 1(c): Surface vs. interior")
        stats.add_profile_stats("Surface RMSD", surf_profiles, models, ID_RADII, OOD_RADII,
                                unit="Å", normalized=normalize)
        stats.add_profile_stats("Interior RMSD", int_profiles, models, ID_RADII, OOD_RADII,
                                unit="Å", normalized=normalize)
        # Gap analysis
        stats.add("  Surface–Interior gap (OOD mean ratio):")
        for model in models:
            if model in surf_profiles and model in int_profiles:
                r_s, m_s, _ = surf_profiles[model]
                r_i, m_i, _ = int_profiles[model]
                ood_mask_s = np.isin(r_s, OOD_RADII)
                ood_mask_i = np.isin(r_i, OOD_RADII)
                id_mask_s = np.isin(r_s, ID_RADII)
                id_mask_i = np.isin(r_i, ID_RADII)
                if ood_mask_s.any() and ood_mask_i.any():
                    ood_gap = np.mean(m_s[ood_mask_s]) / (np.mean(m_i[ood_mask_i]) + 1e-12)
                    id_gap = np.mean(m_s[id_mask_s]) / (np.mean(m_i[id_mask_i]) + 1e-12)
                    stats.add(f"    {_dn(model):<12}: ID gap={id_gap:.3f}, "
                              f"OOD gap={ood_gap:.3f}, change={ood_gap - id_gap:+.3f}")
        stats.add("")

    # ── (d) Orientation stability ────────────────────────────────────────────
    ax = axes[1, 0]
    rot_profiles = {}
    if rot_df is not None and not rot_df.empty:
        rot_profiles = _rotation_profile(rot_df, "rot_mean_pairwise_rmsd", models,
                                         normalize, rot_bl)
        for model in models:
            if model not in rot_profiles:
                continue
            r, m, s = rot_profiles[model]
            ax.plot(r, m, color=_c(model), marker=_mk(model), ms=3, lw=1.2,
                    label=_dn(model), zorder=3)
            ax.fill_between(r, m - s, m + s, alpha=0.12, color=_c(model), zorder=2)
    if normalize:
        ax.axhline(1.0, color="black", ls="-", lw=1.0, alpha=0.3)
    _add_id_ood_shading(ax)
    _add_id_ood_labels(ax)
    ax.set_xlabel("Radius (Å)")
    ax.set_ylabel(f"Pairwise RMSD{norm_label}")
    ax.set_title("(d) Orientation stability vs. radius", fontweight="bold", loc="left")
    _id, _ood = [], []
    for m in WELL_BEHAVED_MODELS:
        if m not in rot_profiles:
            continue
        r, means, _ = rot_profiles[m]
        id_m = np.mean(means[np.isin(r, ID_RADII)]) if np.any(np.isin(r, ID_RADII)) else np.nan
        ood_m = np.mean(means[np.isin(r, OOD_RADII)]) if np.any(np.isin(r, OOD_RADII)) else np.nan
        if not np.isnan(id_m):
            _id.append(id_m)
        if not np.isnan(ood_m):
            _ood.append(ood_m)
    avg_id = np.mean(_id) if _id else np.nan
    avg_ood = np.mean(_ood) if _ood else np.nan
    degrad = avg_ood / (avg_id + 1e-12) if _id and _ood else np.nan
    ann = f"stable models\nOOD/ID: ×{degrad:.2f}"
    ax.text(0.02, 0.98, ann, fontsize=8, va="top", ha="left", transform=ax.transAxes, color="gray",
            bbox=_ANN_BBOX)

    if stats:
        stats.begin("Figure 1(d): Orientation stability")
        if rot_profiles:
            stats.add_profile_stats("RotMean", rot_profiles, models, ID_RADII, OOD_RADII,
                                    unit="Å", normalized=normalize)
        else:
            stats.add("  (no rotation data)")

    # ── (e) Violin: ID vs OOD RMSD distribution per model ────────────────────
    ax = axes[1, 1]
    violin_data_id = []
    violin_data_ood = []
    violin_labels = []
    for model in models:
        sub = sample_df[sample_df["model"] == model]
        id_vals = sub.loc[sub["split"] == "id_test", "rmsd"].dropna().values
        ood_vals = sub.loc[sub["split"] == "ood_test", "rmsd"].dropna().values
        if normalize and rmsd_bl and model in rmsd_bl:
            bl = rmsd_bl[model] if rmsd_bl[model] > 1e-12 else 1.0
            id_vals = id_vals / bl
            ood_vals = ood_vals / bl
        violin_data_id.append(id_vals)
        violin_data_ood.append(ood_vals)
        violin_labels.append(_dn(model))

    n_models = len(violin_labels)
    positions_id = np.arange(n_models) * 2.5
    positions_ood = positions_id + 0.8

    for i, model in enumerate(models):
        c = _c(model)
        if len(violin_data_id[i]) > 1:
            vp_id = ax.violinplot([violin_data_id[i]], positions=[positions_id[i]],
                                  widths=0.65, showmedians=True, showextrema=False)
            for body in vp_id["bodies"]:
                body.set_facecolor(c)
                body.set_alpha(0.5)
                body.set_edgecolor(c)
            vp_id["cmedians"].set_color(c)
            vp_id["cmedians"].set_linewidth(1.5)
        if len(violin_data_ood[i]) > 1:
            vp_ood = ax.violinplot([violin_data_ood[i]], positions=[positions_ood[i]],
                                   widths=0.65, showmedians=True, showextrema=False)
            for body in vp_ood["bodies"]:
                body.set_facecolor(c)
                body.set_alpha(0.2)
                body.set_edgecolor(c)
                body.set_linestyle("--")
            vp_ood["cmedians"].set_color(c)
            vp_ood["cmedians"].set_linewidth(1.5)
            vp_ood["cmedians"].set_linestyle("--")

    ax.set_xticks((positions_id + positions_ood) / 2)
    ax.set_xticklabels(violin_labels, fontsize=7)
    if normalize:
        ax.axhline(1.0, color="black", ls="-", lw=1.0, alpha=0.3)
    # Legend for solid=ID, light=OOD
    ax.legend(
        handles=[
            matplotlib.patches.Patch(facecolor="gray", alpha=0.5, label="ID"),
            matplotlib.patches.Patch(facecolor="gray", alpha=0.2, label="OOD"),
        ],
        fontsize=6.5, loc="upper left", framealpha=0.8, edgecolor="none",
    )
    ax.set_ylabel(f"RMSD{norm_label}")
    ax.set_title("(e) RMSD distribution: ID vs OOD", fontweight="bold", loc="left")

    if stats:
        stats.begin("Figure 1(e): Violin — RMSD ID vs OOD")
        for i, model in enumerate(models):
            id_v, ood_v = violin_data_id[i], violin_data_ood[i]
            stats.add(f"  {_dn(model)}:")
            if len(id_v) > 0:
                stats.add(f"    ID:  median={np.median(id_v):.4f}, mean={np.mean(id_v):.4f}, "
                          f"std={np.std(id_v):.4f}, Q25={np.percentile(id_v,25):.4f}, "
                          f"Q75={np.percentile(id_v,75):.4f}, max={np.max(id_v):.4f}")
            if len(ood_v) > 0:
                stats.add(f"    OOD: median={np.median(ood_v):.4f}, mean={np.mean(ood_v):.4f}, "
                          f"std={np.std(ood_v):.4f}, Q25={np.percentile(ood_v,25):.4f}, "
                          f"Q75={np.percentile(ood_v,75):.4f}, max={np.max(ood_v):.4f}")
            if len(id_v) > 0 and len(ood_v) > 0:
                stats.add(f"    Shift: median Δ={np.median(ood_v)-np.median(id_v):+.4f}, "
                          f"tail ratio (Q95 OOD/ID)="
                          f"{np.percentile(ood_v,95)/(np.percentile(id_v,95)+1e-12):.3f}")
        stats.add("")

    # ── (f) Dumbbell: multi-metric ID→OOD shift per model ────────────────────
    ax = axes[1, 2]
    metric_specs = [
        ("rmsd", "RMSD", True),
        ("bond_mae", "BondMAE", True),
        ("rg_error", "RgError", True),
        ("coord_preservation", "CoordCorr", False),  # higher is better
    ]
    # Filter to metrics that exist
    metric_specs = [(col, lbl, lower) for col, lbl, lower in metric_specs
                    if col in sample_df.columns]

    if metric_specs:
        dumb_records = []
        for model in models:
            sub = sample_df[sample_df["model"] == model]
            for col, lbl, lower_is_err in metric_specs:
                id_v = sub.loc[sub["split"] == "id_test", col].dropna().values
                ood_v = sub.loc[sub["split"] == "ood_test", col].dropna().values
                if len(id_v) > 0 and len(ood_v) > 0:
                    id_m, ood_m = np.mean(id_v), np.mean(ood_v)
                    # Normalize: ratio to ID mean (so ID=1.0 for all metrics)
                    if abs(id_m) > 1e-12:
                        id_norm, ood_norm = 1.0, ood_m / id_m
                    else:
                        id_norm, ood_norm = 1.0, 1.0
                    # For "higher is better" metrics, invert so that
                    # degradation always means moving RIGHT
                    if not lower_is_err:
                        id_norm, ood_norm = 1.0, id_m / (ood_m + 1e-12)
                    dumb_records.append({
                        "model": model, "metric": lbl,
                        "id_val": id_norm, "ood_val": ood_norm,
                        "shift": ood_norm - id_norm,
                        "raw_id": id_m, "raw_ood": ood_m,
                    })

        if dumb_records:
            dumb_df = pd.DataFrame(dumb_records)

            # Layout: group by model, within each model show metrics
            group_gap = 0.6
            y_positions = []
            y_labels = []
            y_colors = []
            row = 0
            model_centers = {}
            for mi, model in enumerate(models):
                msub = dumb_df[dumb_df["model"] == model]
                if msub.empty:
                    continue
                start = row
                for _, r in msub.iterrows():
                    y_positions.append(row)
                    y_labels.append(r["metric"])
                    y_colors.append(_c(model))
                    row += 1
                model_centers[model] = (start + row - 1) / 2
                row += group_gap  # gap between models

            y_positions = np.array(y_positions)

            idx = 0
            for model in models:
                msub = dumb_df[dumb_df["model"] == model]
                if msub.empty:
                    continue
                c = _c(model)
                for _, r in msub.iterrows():
                    yp = y_positions[idx]
                    ax.plot([r["id_val"], r["ood_val"]], [yp, yp],
                            color=c, lw=2.0, alpha=0.7, zorder=2)
                    ax.scatter(r["id_val"], yp, color=c, marker="o", s=30,
                               zorder=3, edgecolors="white", linewidths=0.4)
                    ax.scatter(r["ood_val"], yp, color=c, marker="D", s=30,
                               zorder=3, edgecolors="white", linewidths=0.4)
                    idx += 1

            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels, fontsize=6.5)

            ax.axvline(1.0, color="black", ls="-", lw=0.8, alpha=0.4,
                       label="ID baseline")
            ax.legend(
                handles=[
                    Line2D([0], [0], marker="o", color="gray", ms=5, lw=0, label="ID (=1.0)"),
                    Line2D([0], [0], marker="D", color="gray", ms=5, lw=0, label="OOD"),
                ],
                fontsize=6.5, loc="upper left", framealpha=0.8, edgecolor="none",
            )
            ax.set_xlabel("Value (ID = 1.0)")
            ax.set_title("(f) Which metric breaks first?", fontweight="bold", loc="left")
            ax.invert_yaxis()
            # Annotation at bottom left
            max_ood = dumb_df["ood_val"].max()
            min_ood = dumb_df["ood_val"].min()
            ann_f = f"ID: 1.0 (baseline)\nOOD/ID min: {min_ood:.2f}×\nOOD/ID max: {max_ood:.2f}×"
            ax.text(0.02, 0.5, ann_f, fontsize=8, va="center", ha="left",
                    transform=ax.transAxes, color="gray", bbox=_ANN_BBOX)

            if stats:
                stats.begin("Figure 1(f): Dumbbell — multi-metric failure sequence")
                stats.add("  All values normalized so ID = 1.0; higher = worse "
                          "(inverted for CoordCorr)")
                stats.add(f"  {'Model':<12} {'Metric':<10} {'ID (raw)':>10} "
                          f"{'OOD (raw)':>10} {'OOD/ID':>8} {'Shift':>8}")
                stats.add(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
                for _, row in dumb_df.iterrows():
                    stats.add(f"  {_dn(row['model']):<12} {row['metric']:<10} "
                              f"{row['raw_id']:10.4f} {row['raw_ood']:10.4f} "
                              f"{row['ood_val']:8.4f} {row['shift']:8.4f}")
                stats.add("")
                # Failure sequence per model
                stats.add("  Failure sequence (metric with largest OOD/ID ratio first):")
                for model in models:
                    msub = dumb_df[dumb_df["model"] == model].sort_values(
                        "ood_val", ascending=False)
                    if not msub.empty:
                        seq = " > ".join(
                            f"{r['metric']}({r['ood_val']:.2f}×)"
                            for _, r in msub.iterrows()
                        )
                        stats.add(f"    {_dn(model):<12}: {seq}")
                stats.add("")

    # ── Shared legend ────────────────────────────────────────────────────────
    handles = [
        Line2D([0], [0], color=_c(m), marker=_mk(m), ms=5, lw=1.2, label=_dn(m))
        for m in models
        if m in _radius_profile(sample_df, "rmsd", models)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(models),
               frameon=True, framealpha=0.9, edgecolor="none",
               bbox_to_anchor=(0.5, 1.04), fontsize=8)

    if out_path:
        fig.savefig(out_path, format=fmt)
        print(f"Saved Figure 1 -> {out_path}")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Material × Model Degradation Heatmap
# ═════════════════════════════════════════════════════════════════════════════


def _compute_degrad_matrix(
    sample_df: pd.DataFrame, metric_col: str,
    materials: List[str], models: List[str],
) -> np.ndarray:
    """Compute degradation ratio matrix [materials × models]."""
    mat = np.full((len(materials), len(models)), np.nan)
    for i, material in enumerate(materials):
        for j, model in enumerate(models):
            sub = sample_df[(sample_df["model"] == model) & (sample_df["material"] == material)]
            id_v = sub.loc[sub["split"] == "id_test", metric_col].dropna().values
            ood_v = sub.loc[sub["split"] == "ood_test", metric_col].dropna().values
            if len(id_v) > 0 and len(ood_v) > 0:
                id_mean = np.mean(id_v)
                if id_mean > 1e-8:
                    mat[i, j] = np.mean(ood_v) / id_mean
    return mat


# Models to exclude from Figure 2 (degradation heatmap) only.
FIG2_EXCLUDE_MODELS = {"mattergen", "diffcsp"}


def plot_figure2(
    sample_df: pd.DataFrame,
    models: List[str],
    metrics: Optional[List[str]] = None,
    stats: Optional[StatsPrinter] = None,
    out_path: Optional[str] = None,
    fmt: str = "pdf",
):
    if metrics is None:
        metrics = ["rmsd", "bond_mae"]
    metrics = [m for m in metrics if m in sample_df.columns]
    if not metrics:
        return None

    models = [m for m in models if m not in FIG2_EXCLUDE_MODELS]
    if not models:
        return None

    _style_defaults()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n + 0.5, 3.8), constrained_layout=True)
    if n == 1:
        axes = [axes]

    titles = {"rmsd": "RMSD degradation", "bond_mae": "Bond MAE degradation",
              "rg_error": "Rg error degradation"}
    materials = sorted(sample_df["material"].unique())

    for pi, mc in enumerate(metrics):
        ax = axes[pi]
        dm = _compute_degrad_matrix(sample_df, mc, materials, models)

        # Sort by difficulty (hardest at top)
        mat_means = np.nanmean(dm, axis=1)
        sort_idx = np.argsort(-mat_means)
        dm = dm[sort_idx]
        materials_sorted = [materials[k] for k in sort_idx]

        vmin = max(0.8, np.nanmin(dm) - 0.05) if not np.all(np.isnan(dm)) else 0.8
        vmax = min(2.5, np.nanmax(dm) + 0.05) if not np.all(np.isnan(dm)) else 2.0
        im = ax.imshow(dm, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax,
                        interpolation="nearest")

        for i in range(len(materials_sorted)):
            for j in range(len(models)):
                v = dm[i, j]
                if not np.isnan(v):
                    color = "white" if v > (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold")

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([_dn(m) for m in models], rotation=45, ha="right")
        ax.set_yticks(range(len(materials_sorted)))
        ax.set_yticklabels([_dm(m) for m in materials_sorted])
        ax.set_title(titles.get(mc, mc), fontweight="bold", fontsize=9)

        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        # if pi == n - 1:
        #     cb.set_label("OOD / ID ratio", fontsize=8)
        cb.ax.axhline(y=1.0, color="black", lw=1, ls="--")

        if stats:
            stats.begin(f"Figure 2 panel {pi+1}: {mc} degradation")
            stats.add_heatmap_stats(mc, dm, materials_sorted, models)

    if out_path:
        fig.savefig(out_path, format=fmt)
        print(f"Saved Figure 2 -> {out_path}")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Scaling Law (log-log)
# ═════════════════════════════════════════════════════════════════════════════


def plot_figure3(
    sample_df: pd.DataFrame,
    models: List[str],
    stats: Optional[StatsPrinter] = None,
    out_path: Optional[str] = None,
    fmt: str = "pdf",
):
    _style_defaults()
    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)

    legend_entries = []
    fit_results: Dict[str, Dict[str, float]] = OrderedDict()

    for model in models:
        sub = sample_df[sample_df["model"] == model]
        if sub.empty or "n_atoms" not in sub.columns:
            continue

        g = sub.groupby("radius").agg(
            n_atoms=("n_atoms", "mean"), rmsd=("rmsd", "mean"),
            rmsd_std=("rmsd", "std"),
        ).reset_index().dropna(subset=["n_atoms", "rmsd"])
        g = g[(g["n_atoms"] > 0) & (g["rmsd"] > 0)].sort_values("n_atoms")

        log_n = np.log10(g["n_atoms"].values)
        log_r = np.log10(g["rmsd"].values)
        c, mk = _c(model), _mk(model)

        ax.scatter(log_n, log_r, color=c, marker=mk, s=20, zorder=4, alpha=0.8)

        id_mask = g["radius"].isin(ID_RADII).values
        ood_mask = g["radius"].isin(OOD_RADII).values

        if id_mask.sum() >= 3:
            sl, ic, rv, _, se = linregress(log_n[id_mask], log_r[id_mask])

            # Solid fit over ID
            x_id = np.linspace(log_n[id_mask].min(), log_n[id_mask].max(), 50)
            ax.plot(x_id, sl * x_id + ic, color=c, lw=1.5, ls="-", zorder=3)

            # Dashed extrapolation into OOD
            if ood_mask.sum() > 0:
                x_ood = np.linspace(log_n[id_mask].max(), log_n.max() + 0.1, 30)
                ax.plot(x_ood, sl * x_ood + ic, color=c, lw=1.2, ls="--", alpha=0.5, zorder=3)

                # Prediction error in OOD
                pred_ood = sl * log_n[ood_mask] + ic
                actual_ood = log_r[ood_mask]
                ood_residual = np.mean(np.abs(actual_ood - pred_ood))
            else:
                ood_residual = np.nan

            fit_results[model] = {
                "slope": sl, "intercept": ic, "r2": rv ** 2,
                "stderr": se, "n_points": int(id_mask.sum()),
                "ood_residual_log": ood_residual,
            }
            label = f"{_dn(model)} (α={sl:.2f}, R²={rv**2:.2f})"
        else:
            label = _dn(model)

        legend_entries.append(
            Line2D([0], [0], color=c, marker=mk, ms=5, lw=1.2, label=label)
        )

    ax.set_xlabel("log₁₀(Atom Count)")
    ax.set_ylabel("log₁₀(RMSD)")
    ax.legend(handles=legend_entries, fontsize=7, loc="upper left",
              framealpha=0.9, edgecolor="none")

    xlim = ax.get_xlim()
    yl = ax.get_ylim()
    ymid = yl[0] + 0.5 * np.ptp(yl)
    ax.text(xlim[0] + 0.12 * np.ptp(xlim), ymid,
            "← smaller (ID)", fontsize=7, color="#0072B2", alpha=0.6, bbox=_ANN_BBOX)
    ax.text(xlim[1] - 0.32 * np.ptp(xlim), ymid,
            "larger (OOD) →", fontsize=7, color="#D55E00", alpha=0.6, bbox=_ANN_BBOX)
    ax.set_title("Scaling law: RMSD ~ N$^α$",
                 fontweight="bold", fontsize=9)

    if stats:
        stats.begin("Figure 3: Scaling law fits")
        stats.add_scaling_stats("Power-law fit (ID only)", fit_results)
        if fit_results:
            stats.add("  OOD extrapolation residuals (mean |log₁₀ actual − log₁₀ predicted|):")
            for model, res in fit_results.items():
                r = res.get("ood_residual_log", np.nan)
                stats.add(f"    {_dn(model):<12}: {r:.4f}" if not np.isnan(r) else
                          f"    {_dn(model):<12}: N/A")
            stats.add("")

    if out_path:
        fig.savefig(out_path, format=fmt)
        print(f"Saved Figure 3 -> {out_path}")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════


def load_data_from_metric_files(results_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load sample and rotation metrics directly from CSV/JSON written by
    compute_metrics_from_predictions. No dependency on analyze_results.
    Returns (sample_df, rot_df, models).
    """
    from scripts.compute_metrics_from_predictions import find_results_dirs

    path = os.path.abspath(results_root)
    eval_check = os.path.join(path, "eval_predictions")
    if os.path.isdir(eval_check) and (
        os.path.isfile(os.path.join(eval_check, "id_test.npz"))
        or os.path.isfile(os.path.join(eval_check, "ood_test.npz"))
    ):
        dirs = [path]
    else:
        dirs = find_results_dirs(path)

    sample_dfs = []
    rot_dfs = []
    for d in dirs:
        sample_path = os.path.join(d, "sample_metrics.csv")
        rot_path = os.path.join(d, "rotation_metrics.csv")
        if os.path.isfile(sample_path):
            try:
                df = pd.read_csv(sample_path)
                if not df.empty:
                    if "model" not in df.columns or "seed" not in df.columns:
                        parts = os.path.normpath(d).replace("\\", "/").split("/")
                        if "model" not in df.columns and len(parts) >= 2 and parts[-1].isdigit():
                            df["model"] = parts[-2]
                        if "seed" not in df.columns and len(parts) >= 1 and parts[-1].isdigit():
                            df["seed"] = int(parts[-1])
                    sample_dfs.append(df)
            except Exception:
                pass
        if os.path.isfile(rot_path):
            try:
                rdf = pd.read_csv(rot_path)
                if not rdf.empty:
                    if "model" not in rdf.columns or "seed" not in rdf.columns:
                        parts = os.path.normpath(d).replace("\\", "/").split("/")
                        if "model" not in rdf.columns and len(parts) >= 2 and parts[-1].isdigit():
                            rdf["model"] = parts[-2]
                        if "seed" not in rdf.columns and len(parts) >= 1 and parts[-1].isdigit():
                            rdf["seed"] = int(parts[-1])
                    rot_dfs.append(rdf)
            except Exception:
                pass

    sample_df = pd.concat(sample_dfs, ignore_index=True) if sample_dfs else pd.DataFrame()
    rot_df = pd.concat(rot_dfs, ignore_index=True) if rot_dfs else pd.DataFrame()
    models = sorted(sample_df["model"].unique().tolist()) if not sample_df.empty else []
    return sample_df, rot_df, models


def load_data(results_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load data for plotting. Uses metric files (sample_metrics.csv, rotation_metrics.csv)
    written by compute_metrics_from_predictions when present; falls back to
    analyze_results.load_all_results for runs that have NPZ but no metrics yet.
    """
    sample_df, rot_df, models = load_data_from_metric_files(results_root)
    if not sample_df.empty:
        return sample_df, rot_df, models
    # Fallback: analyze_results can load from NPZ and compute on the fly
    from scripts.analyze_results import load_all_results

    _, _, _, sample_df = load_all_results(results_root=results_root)
    rot_dfs = []
    task1 = os.path.join(results_root, "task_1")
    if os.path.isdir(task1):
        for model in sorted(os.listdir(task1)):
            md = os.path.join(task1, model)
            if not os.path.isdir(md):
                continue
            for sn in sorted(os.listdir(md)):
                rp = os.path.join(md, sn, "rotation_metrics.csv")
                if os.path.isfile(rp):
                    try:
                        df = pd.read_csv(rp)
                        if "model" not in df.columns:
                            df["model"] = model
                        if "seed" not in df.columns:
                            df["seed"] = int(sn)
                        rot_dfs.append(df)
                    except Exception:
                        pass
    rot_df = pd.concat(rot_dfs, ignore_index=True) if rot_dfs else pd.DataFrame()
    models = sorted(sample_df["model"].unique().tolist()) if not sample_df.empty else []
    return sample_df, rot_df, models


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Generate RADII paper figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--rmsd-threshold", type=float, default=2.0)
    parser.add_argument("--bond-threshold", type=float, default=0.5)
    parser.add_argument("--no-normalize", action="store_true",
                        help="Plot raw values instead of ID-normalized")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    normalize = not args.no_normalize

    print("Loading data...")
    sample_df, rot_df, models = load_data(args.results_dir)
    if sample_df.empty:
        print("No sample metrics found. Run training + evaluation first.")
        return

    print(f"Models: {models}")
    print(f"Samples: {len(sample_df)}")
    print(f"Rotation groups: {len(rot_df)}")
    print(f"Normalization: {'ID-baseline' if normalize else 'raw'}")

    sp = StatsPrinter()

    # Figure 1
    print("\nGenerating Figure 1...")
    plot_figure1(sample_df, rot_df, models, normalize=normalize,
                 rmsd_threshold=args.rmsd_threshold, bond_threshold=args.bond_threshold,
                 stats=sp, out_path=os.path.join(args.out_dir, f"fig1_frontier_profiles.{args.format}"),
                 fmt=args.format)

    # Figure 2
    print("Generating Figure 2...")
    plot_figure2(sample_df, models, metrics=["rmsd", "bond_mae"],
                 stats=sp, out_path=os.path.join(args.out_dir, f"fig2_degradation_heatmap.{args.format}"),
                 fmt=args.format)

    # Figure 3
    print("Generating Figure 3...")
    plot_figure3(sample_df, models,
                 stats=sp, out_path=os.path.join(args.out_dir, f"fig3_scaling_law.{args.format}"),
                 fmt=args.format)

    # Print and save all statistics
    stats_text = sp.flush()
    print(stats_text)

    stats_path = os.path.join(args.out_dir, "figure_statistics.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"\nSaved statistics -> {stats_path}")
    print(f"All figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()