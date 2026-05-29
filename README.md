# RADII

**Radius-resolved benchmark for graph generative models in materials science.**

Where do generative models start to fail as the structures they generate grow larger? RADII makes that question measurable: ~75,000 nanoparticle structures across ten materials, with radius treated as a continuous scaling knob from in-distribution to out-of-distribution. Five state-of-the-art architectures (ADiT, CDVAE, DiffCSP, FlowMM, MatterGen) benchmarked under leakage-free splits. Well-behaved models degrade by ~13% in global positional error beyond training radii; the OOD test radii span structures 59% smaller and 24% larger than the training envelope.

📄 **Paper:** *How Far Can You Grow? Characterizing the Extrapolation Frontier of Graph Generative Models for Materials Science* — KDD '26
📦 **Dataset (Zenodo, CC-BY-4.0):** [10.5281/zenodo.20431021](https://doi.org/10.5281/zenodo.20431021)
🤗 **Dataset (HuggingFace):** [KurbanIntelligenceLab/RADII](https://huggingface.co/datasets/KurbanIntelligenceLab/RADII)
🔬 **Code (MIT):** [github.com/KurbanIntelligenceLab/RADII](https://github.com/KurbanIntelligenceLab/RADII)

---

## Quick start (HuggingFace — recommended for evaluating your own model)

One install command. No `trust_remote_code`. Data hosted as parquet on HF.

```bash
pip install datasets
```
```python
from datasets import load_dataset

ds = load_dataset("KurbanIntelligenceLab/RADII", split="train").with_format("torch")
print(ds.features)        # pos, z, material, radius, rot_idx, num_atoms, cell_matrix, ...
print(len(ds), ds[0])     # 48,000 train; 13,500 ID test; 13,480 OOD test
```

### Per-radius / per-material analysis

The metadata columns (`material`, `radius`, `rot_idx`, `split`, `num_atoms`) are
first-class, so filtering and grouping are one-liners:

```python
ds_all = load_dataset("KurbanIntelligenceLab/RADII")     # all three splits

# Slice by radius (e.g., all train samples at R=12 Å)
r12 = ds_all["train"].filter(lambda x: x["radius"] == 12)

# Group by material
ag_ood = ds_all["ood_test"].filter(lambda x: x["material"] == "Ag")

# Atom-count by radius (the 59%/24% statistic from the paper)
import pandas as pd
for split in ("train", "id_test", "ood_test"):
    df = ds_all[split].select_columns(["radius", "num_atoms"]).to_pandas()
    print(split, df.groupby("radius")["num_atoms"].agg(["min", "max", "mean"]))
```

Each row is one nanoparticle: variable-length `pos` (N×3 atomic coordinates) and `z` (N atomic numbers), plus the source material's unit cell (`cell_matrix`, `cell_pos`, `cell_z`). Plug into your trainer with a ragged collate function:

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
for batch in loader:
    ...
```

For PyG users, the `pos`/`z` lists drop straight into `torch_geometric.data.Batch.from_data_list(...)`.

## Quick start (local pip — for reproducing benchmarks)

```bash
git clone https://github.com/KurbanIntelligenceLab/RADII.git
cd RADII
pip install -e .
```

```python
import radii

# Auto-downloads radii_raw.zip from Zenodo on first use,
# regenerates the full 12 GB cache under ~/.cache/radii/.
ds = radii.RadiiDataset(split="train")
print(len(ds), ds[0])  # torch_geometric.data.Data object
```

Train any of the 5 baselines:

```bash
python -m radii.train --model adit       # or cdvae | diffcsp | flowmm | mattergen
python -m radii.train --model adit --seeds 1 2 3
bash run_all_tasks.sh                    # all 5, sequentially
```

## Models (HuggingFace-compatible API)

Each of the 5 baselines (`ADiTUnitCell`, `CDVAEUnitCell`, `DiffCSPUnitCell`,
`FlowMM`, `MatterGen`) inherits from `huggingface_hub.PyTorchModelHubMixin`, so
the standard HF model interface works out of the box — even though we don't
host any checkpoints on the Hub. You save, load, and (if you want) publish
your own trained variants exactly like a `transformers` model:

```python
from radii.models import ADiTUnitCell
from radii.train_config import ModelConfig

m = ADiTUnitCell(**ModelConfig.ADiT.to_dict())
m.save_pretrained("./my_adit")                # writes model.safetensors + config.json
m2 = ADiTUnitCell.from_pretrained("./my_adit")
# m.push_to_hub("my-username/my-radii-adit")  # opt-in upload to HF Hub
```

### Modify hyperparameters from the CLI

```bash
# Override TrainConfig defaults from flags
python -m radii.train --model adit --epochs 50 --lr 1e-4 --batch-size 64
python -m radii.train --model cdvae --seeds 7

# Point at a different data root (e.g., shared scratch)
python -m radii.train --model flowmm --data-root /scratch/radii --out-dir /scratch/runs/flowmm

# Continue training from an existing checkpoint
python -m radii.train --model adit --from-checkpoint results/task_1/adit/1/best_model.pt --epochs 10

# Skip training entirely — just evaluate a checkpoint on id_test + ood_test
python -m radii.train --model adit \
    --eval-only --checkpoint results/task_1/adit/1/best_model.pt \
    --out-dir /tmp/adit_eval

# After training, also write the HF save_pretrained artifact
python -m radii.train --model adit --save-pretrained-dir ./adit_hf
```

The eval predictions land in `<out-dir>/eval_predictions/{id_test,ood_test}.npz`.
Compute scoring metrics afterwards:

```bash
python scripts/compute_metrics_from_predictions.py <out-dir>
```

## Dataset card

| | |
|---|---|
| **Total structures** | 74,980 |
| **Materials (10)** | Ag, Au, CH₃NH₃PbI₃, Fe₂O₃, MoS₂, PbS, SnO₂, SrTiO₃, TiO₂, ZnO |
| **Radii (25 total)** | Train: 15 radii {8–10, 12, 14, 16, 18, 20, 22–28} Å · ID test: 6 radii {11, 13, 15, 17, 19, 21} Å · OOD test: 4 radii {6, 7, 29, 30} Å |
| **Splits** | 48,000 train / 13,500 ID test / 13,480 OOD test |
| **Atom counts** | Train: 81–9,148. OOD test: 33–11,298 (≈59% smaller, ≈24% larger than the training envelope) |
| **Seed archive** | 3 MB (10 CIFs + per-(material, radius) base XYZ); the full 12 GB cache is regenerated locally |
| **Code license** | MIT |
| **Data license** | CC-BY-4.0 |

The split design is *leakage-free*: no OOD or ID test radius appears during training, and test-rotation orientations are excluded from training orientations via an angular exclusion constraint.

## Reproducibility

The dataset is deterministic given the seed archive. To regenerate locally:

```bash
python -m radii.generation.create_radii --raw-data radii_raw.zip --output radii
```

That same entry point is what `radii.RadiiDataset(...)` calls under the hood on first use. The seed archive on Zenodo (DOI [10.5281/zenodo.20431021](https://doi.org/10.5281/zenodo.20431021)) is the single source of truth.

## Repo layout

```
RADII/
├── src/radii/              # installable package
│   ├── data.py             # RadiiDataset, RadiiDataloader
│   ├── download.py         # Zenodo seed fetcher
│   ├── train.py            # unified trainer (--model adit|cdvae|...)
│   ├── train_config.py     # TrainConfig + per-model ModelConfig
│   ├── metrics.py          # RMSD, BondMAE, CoordCorr, frontier statistics
│   ├── generation/         # seed → full benchmark pipeline
│   └── models/             # 5 baselines (one file each)
├── scripts/                # analysis + HF/Zenodo upload scripts
├── radii/                  # local data cache (gitignored)
├── run_all_tasks.sh        # train all 5 models sequentially
└── pyproject.toml
```

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

- **Code** (this repository, `src/radii/`): MIT — see [LICENSE](LICENSE).
- **Dataset** (Zenodo + HuggingFace artifacts): Creative Commons Attribution 4.0 International (CC-BY-4.0).
