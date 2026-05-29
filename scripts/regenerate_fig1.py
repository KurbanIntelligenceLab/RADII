"""Regenerate Figure 1 -> paper/figs/fig1_frontier_profiles.pdf.

Standalone driver. It loads the per-run metric CSVs directly and calls the existing
``plotters.plot_figure1``. This deliberately avoids ``plotters.main()`` /
``load_data_from_metric_files``, which imports ``scripts.compute_metrics_from_predictions``
-- a module that fails at import time with a stale ``from radii.metrics import ...``
(``train/metrics.py`` does not exist). Importing ``scripts.plotters`` itself is safe:
that broken import is local to ``load_data_from_metric_files`` and is never reached here.

Run from the repository root:  python -m scripts.regenerate_fig1
"""
import glob
import os

import pandas as pd

from scripts.plotters import plot_figure1

RESULTS_ROOT = "results"
OUT_PATH = "paper/figs/fig1_frontier_profiles.pdf"


def load_metric_csvs(results_root: str):
    """Concatenate sample_metrics.csv / rotation_metrics.csv from every run dir.

    Mirrors the file-reading half of ``plotters.load_data_from_metric_files`` without
    importing the broken ``compute_metrics_from_predictions`` module. The CSVs already
    carry ``model`` and ``seed`` columns, so no path-based backfill is needed.
    """
    sample_dfs, rot_dfs = [], []
    for path in sorted(glob.glob(os.path.join(results_root, "**", "sample_metrics.csv"),
                                 recursive=True)):
        df = pd.read_csv(path)
        if not df.empty:
            sample_dfs.append(df)
    for path in sorted(glob.glob(os.path.join(results_root, "**", "rotation_metrics.csv"),
                                 recursive=True)):
        rdf = pd.read_csv(path)
        if not rdf.empty:
            rot_dfs.append(rdf)
    sample_df = pd.concat(sample_dfs, ignore_index=True) if sample_dfs else pd.DataFrame()
    rot_df = pd.concat(rot_dfs, ignore_index=True) if rot_dfs else pd.DataFrame()
    models = sorted(sample_df["model"].unique().tolist()) if not sample_df.empty else []
    return sample_df, rot_df, models


def main():
    sample_df, rot_df, models = load_metric_csvs(RESULTS_ROOT)
    if sample_df.empty:
        raise SystemExit(f"No sample_metrics.csv found under {RESULTS_ROOT}/")
    print(f"Models: {models}  |  samples: {len(sample_df)}  |  rotation rows: {len(rot_df)}")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plot_figure1(sample_df, rot_df, models, normalize=True, out_path=OUT_PATH, fmt="pdf")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
