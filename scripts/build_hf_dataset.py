"""Publish the RADII dataset to HuggingFace Hub as parquet (data hosted on HF).

Converts the local cache (`radii/cache/{samples.npz, metadata.json,
unit_cells.npz}`) to three parquet files (one per split) and uploads them to
the configured HF dataset repo with `HfApi.upload_large_folder` (chunked +
resumable + per-blob retry).

After upload, an outside researcher only needs `pip install datasets`:

    from datasets import load_dataset
    ds = load_dataset("KurbanIntelligenceLab/RADII", split="train").with_format("torch")

No `trust_remote_code`, no `radii` install, no Zenodo download in the user
critical path.

Usage:
    python scripts/build_hf_dataset.py --dry-run                    # validate
    python scripts/build_hf_dataset.py --convert-only               # build parquet, skip upload
    python scripts/build_hf_dataset.py                              # build + upload
    python scripts/build_hf_dataset.py --hf-repo johnpolat/RADII    # override repo

Prerequisite: HF_TOKEN in `.env` (write scope).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
CACHE_DIR = REPO_ROOT / "radii" / "cache"
SAMPLES_NPZ = CACHE_DIR / "samples.npz"
METADATA_JSON = CACHE_DIR / "metadata.json"
UNIT_CELLS_NPZ = CACHE_DIR / "unit_cells.npz"

# Metadata uses 'train', 'ID', 'OOD' as raw labels; HF splits use the
# conventional `train` / `id_test` / `ood_test`.
SPLIT_LABEL_TO_HF = {"train": "train", "ID": "id_test", "OOD": "ood_test"}
SPLITS = tuple(SPLIT_LABEL_TO_HF.values())

DEFAULT_HF_REPO = "KurbanIntelligenceLab/RADII"


# ---------------------------------------------------------------------------
# Dataset card (front-matter triggers HF parquet auto-discovery)
# ---------------------------------------------------------------------------

DATASET_CARD = """\
---
license: cc-by-4.0
language:
- en
pretty_name: RADII
size_categories:
- 10K<n<100K
task_categories:
- other
tags:
- materials-science
- crystal-structures
- generative-models
- benchmark
- kdd-2026
- nanoparticles
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: id_test
    path: id_test.parquet
  - split: ood_test
    path: ood_test.parquet
---

# RADII: Radius-Resolved Benchmark of Nanoparticle Structures

**RADII** measures where graph generative models start to fail as the structures
they generate grow larger. Radius is treated as a continuous scaling knob from
in-distribution to out-of-distribution; the dataset spans 10 materials and 25
radii with leakage-free splits.

- **Paper:** *How Far Can You Grow? Characterizing the Extrapolation Frontier of Graph Generative Models for Materials Science* — KDD '26
- **Code (models, trainer, generation pipeline):** [github.com/KurbanIntelligenceLab/RADII](https://github.com/KurbanIntelligenceLab/RADII) (MIT)
- **Archival DOI:** [10.5281/zenodo.20431021](https://doi.org/10.5281/zenodo.20431021) (CC-BY-4.0)

## Quick start

```bash
pip install datasets
```

```python
from datasets import load_dataset

ds = load_dataset("KurbanIntelligenceLab/RADII", split="train").with_format("torch")
print(ds.features, len(ds), ds[0]["material"], ds[0]["num_atoms"])
```

That's it — no `trust_remote_code`, no Zenodo download, no regeneration.

## Splits

| Split | Size | Radii (Å) |
|---|---|---|
| `train` | 48,000 | 15 values: 8–10, 12, 14, 16, 18, 20, 22–28 |
| `id_test` | 13,500 | 6 values: 11, 13, 15, 17, 19, 21 (interleaved, unseen orientations) |
| `ood_test` | 13,480 | 4 values: 6, 7, 29, 30 (strictly outside training range) |

OOD test structures are ~59% smaller and ~24% larger than the smallest and
largest training structures (atom counts 33–11,298 vs train 81–9,148).

## Materials (10)

Ag, Au, CH₃NH₃PbI₃, Fe₂O₃, MoS₂, PbS, SnO₂, SrTiO₃, TiO₂, ZnO.

## Schema (per row)

| Column | Type | Notes |
|---|---|---|
| `material` | str | One of the 10 above |
| `radius` | int | Truncation radius in Å |
| `rot_idx` | int | Orientation index |
| `split` | str | `train`, `id_test`, or `ood_test` |
| `num_atoms` | int | Length of `pos`/`z` |
| `pos` | list[list[float32]] | `num_atoms × 3` atomic coordinates |
| `z` | list[int8] | `num_atoms` atomic numbers |
| `cell_matrix` | list[list[float32]] | `3 × 3` lattice vectors of the parent unit cell |
| `cell_pos` | list[list[float32]] | Unit-cell basis positions (per-material) |
| `cell_z` | list[int8] | Unit-cell basis atomic numbers |

## Analysis recipes (per-radius, per-material, cross-split)

The metadata columns are first-class, so slicing and grouping are one-liners:

```python
from datasets import load_dataset

# All three splits at once
ds_all = load_dataset("KurbanIntelligenceLab/RADII")

# Per-radius slice — every train sample at R=12
r12 = ds_all["train"].filter(lambda x: x["radius"] == 12)

# Per-material grouping
materials = ds_all["train"].unique("material")  # ['Ag', 'Au', ...]
ag_ood    = ds_all["ood_test"].filter(lambda x: x["material"] == "Ag")

# Cross-split: atom counts by radius
import pandas as pd
for split in ("train", "id_test", "ood_test"):
    df = ds_all[split].select_columns(["radius", "num_atoms"]).to_pandas()
    print(split, df.groupby("radius")["num_atoms"].agg(["mean", "min", "max"]))

# Atom-count envelope check (replicates the paper's 59% / 24% statistic)
train_min, train_max = ds_all["train"]["num_atoms"], ds_all["train"]["num_atoms"]
print("train atom-count range:", min(train_min), "to", max(train_max))
print("OOD  atom-count range:", min(ds_all["ood_test"]["num_atoms"]),
      "to", max(ds_all["ood_test"]["num_atoms"]))
```

## PyTorch collate (variable-length `pos`/`z`)

```python
import torch
from torch.utils.data import DataLoader

def collate(batch):
    return {
        "pos":         [torch.as_tensor(b["pos"]) for b in batch],
        "z":           [torch.as_tensor(b["z"])   for b in batch],
        "cell_matrix": torch.stack([torch.as_tensor(b["cell_matrix"]) for b in batch]),
        "material":    [b["material"] for b in batch],
        "radius":      torch.tensor([b["radius"] for b in batch]),
    }

loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate)
```

For PyTorch Geometric, the `pos`/`z` lists drop straight into
`torch_geometric.data.Batch.from_data_list(...)`.

## Models

The 5 baselines from the paper (ADiT, CDVAE, DiffCSP, FlowMM, MatterGen) live
in the GitHub repo's `radii.models` namespace and follow the standard
HuggingFace model interface (`PyTorchModelHubMixin`):

```python
from radii.models import ADiTUnitCell
from radii.train_config import ModelConfig

m = ADiTUnitCell(**ModelConfig.ADiT.to_dict())
m.save_pretrained("./my_adit")
m2 = ADiTUnitCell.from_pretrained("./my_adit")
# m.push_to_hub("my-username/my-radii-adit") also works
```

See the GitHub README for training (`python -m radii.train --model adit`) and
override flags (`--epochs`, `--lr`, `--from-checkpoint`, `--eval-only`).

## Citation

```bibtex
@article{polat2026far,
  title   = {How Far Can You Grow? Characterizing the Extrapolation Frontier of Graph Generative Models for Materials Science},
  author  = {Polat, Can and Serpedin, Erchin and Kurban, Mustafa and Kurban, Hasan},
  journal = {arXiv preprint arXiv:2602.09309},
  year    = {2026}
}
```

## License

CC-BY-4.0. See [10.5281/zenodo.20431021](https://doi.org/10.5281/zenodo.20431021)
for the archival record terms. Accompanying code is MIT-licensed at the GitHub repo.
"""


# ---------------------------------------------------------------------------
# Parquet conversion
# ---------------------------------------------------------------------------

def load_cache():
    if not SAMPLES_NPZ.exists():
        sys.exit(
            f"ERROR: {SAMPLES_NPZ} not found. Run the generation pipeline first:\n"
            f"  python -m radii.generation.create_radii --raw-data radii_raw.zip --output radii"
        )
    samples = np.load(SAMPLES_NPZ, allow_pickle=True)
    pos_all = samples["pos"]
    z_all = samples["z"]
    offsets = samples["offsets"]

    with METADATA_JSON.open() as f:
        meta = json.load(f)
    sample_meta = meta["samples"]

    if len(sample_meta) + 1 != len(offsets):
        sys.exit(
            f"ERROR: metadata has {len(sample_meta)} samples but offsets has "
            f"length {len(offsets)} (expected {len(sample_meta) + 1})."
        )

    unit_cells = np.load(UNIT_CELLS_NPZ, allow_pickle=True)
    materials_map = {}
    for k in unit_cells.files:
        if k.endswith("_pos"):
            mat = k[:-4]
            materials_map[mat] = {
                "pos": unit_cells[f"{mat}_pos"].astype(np.float32),
                "z": unit_cells[f"{mat}_z"].astype(np.int8),
                "cell_matrix": unit_cells[f"{mat}_cell_matrix"].astype(np.float32),
            }
    if not materials_map:
        sys.exit(f"ERROR: no per-material keys found in {UNIT_CELLS_NPZ}")

    return pos_all, z_all, offsets, sample_meta, materials_map


def iter_rows(pos_all, z_all, offsets, sample_meta, materials_map, hf_split: str):
    raw_split = next(k for k, v in SPLIT_LABEL_TO_HF.items() if v == hf_split)
    for i, m in enumerate(sample_meta):
        if m["split"] != raw_split:
            continue
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e - s != m["num_atoms"]:
            raise ValueError(
                f"Sample {i}: offsets slice {e - s} != metadata num_atoms {m['num_atoms']}"
            )
        mat = m["material"]
        uc = materials_map[mat]
        yield {
            "material": mat,
            "radius": int(m["radius"]),
            "rot_idx": int(m["rot_idx"]),
            "split": hf_split,
            "num_atoms": int(m["num_atoms"]),
            "pos": pos_all[s:e].astype(np.float32).tolist(),
            "z": z_all[s:e].astype(np.int8).tolist(),
            "cell_matrix": uc["cell_matrix"].tolist(),
            "cell_pos": uc["pos"].tolist(),
            "cell_z": uc["z"].tolist(),
        }


def write_parquet(rows: list[dict], out_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path, compression="snappy")
    print(f"  wrote {out_path.name}: {len(rows)} rows, "
          f"{out_path.stat().st_size / 1e6:.2f} MB")


def build_parquets(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {SAMPLES_NPZ.name} + metadata + unit_cells...", flush=True)
    pos_all, z_all, offsets, sample_meta, materials_map = load_cache()
    print(f"  samples: {len(sample_meta)}, total atoms: {pos_all.shape[0]}, "
          f"materials: {len(materials_map)}")

    out: dict[str, Path] = {}
    for split in SPLITS:
        print(f"Building {split}.parquet...", flush=True)
        rows = list(iter_rows(pos_all, z_all, offsets, sample_meta, materials_map, split))
        path = out_dir / f"{split}.parquet"
        write_parquet(rows, path)
        out[split] = path
    return out


# ---------------------------------------------------------------------------
# HF upload (with upload_large_folder + stale RADII.py cleanup)
# ---------------------------------------------------------------------------

def upload(parquet_dir: Path, hf_repo: str) -> str:
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            "ERROR: HF_TOKEN not found in environment. Add to .env or set in shell.\n"
            "  Mint at https://huggingface.co/settings/tokens (write scope)."
        )

    api = HfApi(token=token)
    create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True, token=token)

    (parquet_dir / "README.md").write_text(DATASET_CARD)

    # Best-effort: delete the stale loader from the previous lightweight push so
    # the parquet auto-discovery isn't shadowed by a custom loader script.
    try:
        api.delete_file(
            path_in_repo="RADII.py", repo_id=hf_repo, repo_type="dataset",
            commit_message="Remove stale custom loader (parquet auto-discovery instead)",
        )
        print("  deleted stale RADII.py from repo")
    except Exception as e:
        # 404 if it's not there. Either way, fine.
        print(f"  RADII.py delete skipped ({type(e).__name__}); continuing")

    print(f"Uploading via upload_large_folder to https://huggingface.co/datasets/{hf_repo} ...",
          flush=True)
    api.upload_large_folder(
        folder_path=str(parquet_dir),
        repo_id=hf_repo,
        repo_type="dataset",
        allow_patterns=["*.parquet", "README.md"],
    )
    return f"https://huggingface.co/datasets/{hf_repo}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs only, no parquet write or upload.")
    p.add_argument("--convert-only", action="store_true",
                   help="Build parquet files but skip the HF upload.")
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                   help=f"HF dataset repo id (default: {DEFAULT_HF_REPO})")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "dist" / "hf",
                   help="Where to write the parquet files (default: dist/hf/)")
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH)
    except ImportError:
        pass

    if args.dry_run:
        print("== Dry run: validating inputs ==")
        pos_all, z_all, offsets, sample_meta, materials_map = load_cache()
        per_split = {
            hf: sum(1 for m in sample_meta if m["split"] == raw)
            for raw, hf in SPLIT_LABEL_TO_HF.items()
        }
        print(f"  samples: {len(sample_meta)}, total atoms: {pos_all.shape[0]}")
        print(f"  materials: {sorted(materials_map.keys())}")
        print(f"  per-split row counts: {per_split}")
        # HF auth check
        token = os.environ.get("HF_TOKEN")
        if token:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            try:
                info = api.repo_info(repo_id=args.hf_repo, repo_type="dataset")
                files = [s.rfilename for s in info.siblings]
                print(f"  HF repo {args.hf_repo!r}: exists, files={files}")
            except Exception as e:
                print(f"  HF repo {args.hf_repo!r}: does not yet exist ({e})")
        else:
            print("  HF_TOKEN not set; skipping HF auth check")
        print("OK (dry run; no I/O performed).")
        return 0

    parquets = build_parquets(args.out_dir)

    if args.convert_only:
        print(f"\nParquet files written to {args.out_dir}/. Upload skipped.")
        return 0

    url = upload(args.out_dir, args.hf_repo)
    print(f"\nDone. Visit: {url}")
    print(f"\nTest from a fresh Python:")
    print(f"  pip install datasets")
    print(f"  python -c \"from datasets import load_dataset; "
          f"ds = load_dataset({args.hf_repo!r}, split='train'); print(ds.features, len(ds))\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
