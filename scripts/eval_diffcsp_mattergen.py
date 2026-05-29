"""
Evaluate DiffCSP and MatterGen from existing checkpoints only (no training).

Loads best_model.pt from results/task_1/{diffcsp,mattergen}/{seed}/ and runs
sampling on id_test + ood_test, saving predictions to eval_predictions/ as usual.
"""

import os

import numpy as np
import torch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

from radii.data import RADIIDataloader, get_all_attrs
from radii.models.diffcsp_model import DiffCSPUnitCell
from radii.models.mattergen_model import MatterGen
from radii.train_config import TrainConfig, ModelConfig

TrainConfig.setup_torch([GlobalStorage, DataEdgeAttr, DataTensorAttr])

# Debug: log once per split for first batch (positions shape, finite, min/max)
_DEBUG_BATCH_COUNT = {}


def _debug_sample_output(
    name: str, split: str, output: dict, pos_key: str = "positions"
):
    """Log output keys, positions shape, and finite/min/max for first batch per (name, split)."""
    key = (name, split)
    _DEBUG_BATCH_COUNT[key] = _DEBUG_BATCH_COUNT.get(key, 0) + 1
    if _DEBUG_BATCH_COUNT[key] != 1:
        return
    keys = list(output.keys()) if isinstance(output, dict) else []
    print(f"[DEBUG {name} {split}] output keys: {keys}", flush=True)
    if pos_key not in output:
        print(f"[DEBUG {name} {split}] missing key '{pos_key}' in output", flush=True)
        return
    pos = output[pos_key]
    if torch.is_tensor(pos):
        pos = pos.detach().cpu().numpy()
    pos = np.asarray(pos, dtype=np.float64)
    finite = np.isfinite(pos)
    n_fin = finite.sum()
    n_tot = pos.size
    if n_fin > 0:
        valid = pos[finite]
        print(
            f"[DEBUG {name} {split}] positions shape={pos.shape}, finite={n_fin}/{n_tot}, "
            f"min={valid.min():.4f}, max={valid.max():.4f}",
            flush=True,
        )
    else:
        print(
            f"[DEBUG {name} {split}] positions shape={pos.shape}, finite=0/{n_tot} (all nan/inf)",
            flush=True,
        )


def eval_diffcsp(seed: int):
    """Evaluate DiffCSP from checkpoint."""
    out_dir = os.path.join("results", "task_1", "diffcsp", str(seed))
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[DiffCSP] Skipping seed {seed}: checkpoint not found at {ckpt_path}")
        return

    print(f"\n{'=' * 60}\n[DiffCSP] SEED {seed}\n{'=' * 60}")
    TrainConfig.set_seed(seed)

    def transform(data):
        data.y_pos = data.pos.clone()
        if not hasattr(data, "cell_ptr") or data.cell_ptr is None:
            data.cell_ptr = torch.tensor([0, data.cell_pos.size(0)], dtype=torch.long)
        data.num_atoms = torch.tensor([data.pos.size(0)], dtype=torch.long)
        if not hasattr(data, "lattice") or data.lattice is None:
            pos_min = data.pos.min(dim=0).values
            pos_max = data.pos.max(dim=0).values
            box_size = (pos_max - pos_min).clamp(min=1.0)
            data.lattice = torch.diag(box_size * 1.2).unsqueeze(0)
        return data

    ds = RADIIDataloader(
        root=TrainConfig.DATA_ROOT,
        num_workers=TrainConfig.DATA_NUM_WORKERS,
        transform=transform,
        loaded_frac=TrainConfig.LOADED_FRAC,
    )
    id_test_ds = ds.get_split("id_test")
    ood_test_ds = ds.get_split("ood_test")
    id_test_ds.transform = ood_test_ds.transform = transform

    loaders = {
        "id_test": GeoDataLoader(
            id_test_ds,
            batch_size=TrainConfig.BATCH_SIZE,
            shuffle=False,
            num_workers=TrainConfig.NUM_WORKERS,
        ),
        "ood_test": GeoDataLoader(
            ood_test_ds,
            batch_size=TrainConfig.BATCH_SIZE,
            shuffle=False,
            num_workers=TrainConfig.NUM_WORKERS,
        ),
    }

    model_config = ModelConfig.DiffCSP.to_dict().copy()
    model = DiffCSPUnitCell(**model_config).to(TrainConfig.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=TrainConfig.DEVICE))
    model.eval()

    eval_dir = os.path.join(out_dir, "eval_predictions")
    os.makedirs(eval_dir, exist_ok=True)

    pred_list, meta_list = [], []

    def save_npz(split, pred_list, meta_list):
        out_path = os.path.join(eval_dir, f"{split}.npz")
        pred_cat = np.concatenate(pred_list, axis=0)
        ptr_arr = np.concatenate([[0], np.cumsum([len(p) for p in pred_list])])
        np.savez_compressed(
            out_path,
            pred_pos=pred_cat,
            ptr=ptr_arr,
            materials=np.array(
                [m.get("material", "") for m in meta_list], dtype=object
            ),
            radius=np.array([float(m.get("radius", np.nan)) for m in meta_list]),
            rot_idx=np.array([int(m.get("rot_idx", -1)) for m in meta_list]),
        )
        print(f"Saved {len(pred_list)} predictions to {out_path}", flush=True)

    with torch.no_grad():
        for split in ["id_test", "ood_test"]:
            pred_list, meta_list = [], []
            loader = loaders[split]
            for data in tqdm(loader, desc=f"Eval DiffCSP {split}"):
                data = data.to(TrainConfig.DEVICE)
                if not hasattr(data, "num_atoms") or data.num_atoms is None:
                    data.num_atoms = data.ptr[1:] - data.ptr[:-1]
                output = model.sample(data, num_atoms=data.num_atoms)
                _debug_sample_output("DiffCSP", split, output, pos_key="positions")
                pred_pos = output["positions"].cpu().numpy()
                ptr = data.ptr.cpu().numpy()
                for i in range(len(ptr) - 1):
                    s, e = int(ptr[i]), int(ptr[i + 1])
                    if e - s < 4:
                        continue
                    pred_list.append(pred_pos[s:e])
                    record = get_all_attrs(data, i, ptr)
                    record["split"] = split
                    meta_list.append(record)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            save_npz(split, pred_list, meta_list)

    print(
        f"To compute metrics: python scripts/compute_metrics_from_predictions.py {out_dir}",
        flush=True,
    )


def eval_mattergen(seed: int):
    """Evaluate MatterGen from checkpoint."""
    out_dir = os.path.join("results", "task_1", "mattergen", str(seed))
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[MatterGen] Skipping seed {seed}: checkpoint not found at {ckpt_path}")
        return

    print(f"\n{'=' * 60}\n[MatterGen] SEED {seed}\n{'=' * 60}")
    TrainConfig.set_seed(seed)

    def add_target(data):
        data.y_pos = data.pos.clone()
        data.cell_ptr = torch.tensor([0, data.cell_pos.size(0)], dtype=torch.long)
        return data

    def clean(data):
        data.y_pos = data.pos.clone()
        if not hasattr(data, "cell_ptr"):
            data.cell_ptr = torch.tensor([0, data.cell_pos.size(0)], dtype=torch.long)
        return data

    ds = RADIIDataloader(
        root=TrainConfig.DATA_ROOT,
        num_workers=TrainConfig.DATA_NUM_WORKERS,
        transform=add_target,
        loaded_frac=TrainConfig.LOADED_FRAC,
    )
    id_test_ds = ds.get_split("id_test")
    ood_test_ds = ds.get_split("ood_test")
    for subset in (id_test_ds, ood_test_ds):
        subset.transform = clean

    loaders = {
        "id_test": GeoDataLoader(
            id_test_ds,
            batch_size=TrainConfig.BATCH_SIZE,
            shuffle=False,
            num_workers=TrainConfig.NUM_WORKERS,
        ),
        "ood_test": GeoDataLoader(
            ood_test_ds,
            batch_size=TrainConfig.BATCH_SIZE,
            shuffle=False,
            num_workers=TrainConfig.NUM_WORKERS,
        ),
    }

    model_config = ModelConfig.MatterGen.to_dict().copy()
    model = MatterGen(**model_config).to(TrainConfig.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=TrainConfig.DEVICE))
    model.eval()

    eval_dir = os.path.join(out_dir, "eval_predictions")
    os.makedirs(eval_dir, exist_ok=True)

    num_steps = min(
        TrainConfig.EVAL_NUM_STEPS, getattr(model, "num_diffusion_steps", 1000)
    )

    pred_list, meta_list = [], []

    def save_npz(split, pred_list, meta_list):
        out_path = os.path.join(eval_dir, f"{split}.npz")
        pred_cat = np.concatenate(pred_list, axis=0)
        ptr_arr = np.concatenate([[0], np.cumsum([len(p) for p in pred_list])])
        np.savez_compressed(
            out_path,
            pred_pos=pred_cat,
            ptr=ptr_arr,
            materials=np.array(
                [m.get("material", "") for m in meta_list], dtype=object
            ),
            radius=np.array([float(m.get("radius", np.nan)) for m in meta_list]),
            rot_idx=np.array([int(m.get("rot_idx", -1)) for m in meta_list]),
        )
        print(f"Saved {len(pred_list)} predictions to {out_path}", flush=True)

    with torch.no_grad():
        for split in ["id_test", "ood_test"]:
            pred_list, meta_list = [], []
            loader = loaders[split]
            for data in tqdm(loader, desc=f"Eval MatterGen {split}"):
                data = data.to(TrainConfig.DEVICE)
                if not hasattr(data, "num_atoms") or data.num_atoms is None:
                    data.num_atoms = data.ptr[1:] - data.ptr[:-1]
                out = model.sample(data, num_steps=num_steps)
                _debug_sample_output("MatterGen", split, out, pos_key="pos")
                pred_pos = out["pos"].cpu().numpy()
                ptr = data.ptr.cpu().numpy()
                for i in range(len(ptr) - 1):
                    s, e = int(ptr[i]), int(ptr[i + 1])
                    if e - s < 4:
                        continue
                    pred_list.append(pred_pos[s:e])
                    record = get_all_attrs(data, i, ptr)
                    record["split"] = split
                    meta_list.append(record)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            save_npz(split, pred_list, meta_list)

    print(
        f"To compute metrics: python scripts/compute_metrics_from_predictions.py {out_dir}",
        flush=True,
    )


def main():
    for SEED in TrainConfig.SEEDS:
        if SEED != 1:
            eval_diffcsp(SEED)
        eval_mattergen(SEED)
    print("\n=== DiffCSP + MatterGen evaluation done ===")


if __name__ == "__main__":
    main()
