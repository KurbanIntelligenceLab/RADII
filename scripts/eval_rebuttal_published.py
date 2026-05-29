"""
Evaluate published-sized DiffCSP / MatterGen rebuttal checkpoints end-to-end.

Loads best_model.pt from rebuttal_results/{diffcsp,mattergen}/, instantiates
each model from its accompanying config.json (published sizes, NOT the
budget-controlled ModelConfig.* defaults), samples on id_test + ood_test,
computes the full RADII metric suite in-process, and writes one JSON per
model to rebuttal_results/calculated/{model}.json.

Why this exists: Reviewer XQdQ argued that RADII's DiffCSP/MatterGen RMSD
reflects a capacity shortfall rather than an extrapolation failure. These
checkpoints match the published parameter counts (~12.4M / ~47M), so the
metrics produced here are the direct rebuttal signal.

Usage:
    conda run -n aclWork python -m scripts.eval_rebuttal_published
    conda run -n aclWork python -m scripts.eval_rebuttal_published --model diffcsp
    conda run -n aclWork python -m scripts.eval_rebuttal_published --smoke
"""

import argparse
import datetime
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

from radii.data import RADIIDataloader, get_all_attrs
from radii.models.diffcsp_model import DiffCSPUnitCell
from radii.models.mattergen_model import MatterGen
from radii.metrics import (
    compute_aggregate_metrics,
    compute_rotation_group_metrics,
    compute_sample_metrics,
)
from radii.train_config import TrainConfig

TrainConfig.setup_torch([GlobalStorage, DataEdgeAttr, DataTensorAttr])

REBUTTAL_DIR = "rebuttal_results"
OUTPUT_DIR = os.path.join(REBUTTAL_DIR, "calculated")
_GT_CACHE_DIR = "gt_cache"


# =============================================================================
# GT cache loader (inlined from scripts/compute_metrics_from_predictions.py
# because that file has a broken top-level `from radii.metrics import ...`).
# =============================================================================


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
    data_root: str,
) -> Dict[str, Dict[Tuple[str, float, int], np.ndarray]]:
    cache_dir = os.path.join(data_root, _GT_CACHE_DIR)
    out = {}
    for split in ("id_test", "ood_test"):
        path = os.path.join(cache_dir, f"{split}.npz")
        if not os.path.isfile(path):
            raise SystemExit(
                f"GT cache missing: {path}. Rebuild via analyze_results or "
                f"compute_metrics_from_predictions first."
            )
        data = np.load(path, allow_pickle=True)
        out[split] = _arrays_to_gt_lookup(
            data["materials"],
            data["radius"],
            data["rot_idx"],
            data["ptr"],
            data["gt_pos"],
        )
    return out


# =============================================================================
# Sampling — transforms mirror scripts/eval_diffcsp_mattergen.py
# =============================================================================


def _diffcsp_transform(data):
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


def _mattergen_add_target(data):
    data.y_pos = data.pos.clone()
    data.cell_ptr = torch.tensor([0, data.cell_pos.size(0)], dtype=torch.long)
    return data


def _mattergen_clean(data):
    data.y_pos = data.pos.clone()
    if not hasattr(data, "cell_ptr"):
        data.cell_ptr = torch.tensor([0, data.cell_pos.size(0)], dtype=torch.long)
    return data


_DEBUG_BATCH_COUNT: Dict[Tuple[str, str], int] = {}


def _debug_sample_output(name: str, split: str, output: dict, pos_key: str):
    key = (name, split)
    _DEBUG_BATCH_COUNT[key] = _DEBUG_BATCH_COUNT.get(key, 0) + 1
    if _DEBUG_BATCH_COUNT[key] != 1:
        return
    keys = list(output.keys()) if isinstance(output, dict) else []
    print(f"[DEBUG {name} {split}] output keys: {keys}", flush=True)
    if pos_key not in output:
        print(
            f"[DEBUG {name} {split}] missing key '{pos_key}' in output",
            flush=True,
        )
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
            f"[DEBUG {name} {split}] positions shape={pos.shape}, "
            f"finite={n_fin}/{n_tot}, min={valid.min():.4f}, max={valid.max():.4f}",
            flush=True,
        )
    else:
        print(
            f"[DEBUG {name} {split}] positions shape={pos.shape}, "
            f"finite=0/{n_tot} (all nan/inf)",
            flush=True,
        )


def build_loaders(model_name: str, data_root: str, loaded_frac: float):
    """Build id_test / ood_test DataLoaders with model-specific transforms."""
    if model_name == "diffcsp":
        base_tf = _diffcsp_transform
        split_tf = _diffcsp_transform
        batch_size = 2
    elif model_name == "mattergen":
        base_tf = _mattergen_add_target
        split_tf = _mattergen_clean
        batch_size = 1
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ds = RADIIDataloader(
        root=data_root,
        num_workers=TrainConfig.DATA_NUM_WORKERS,
        transform=base_tf,
        loaded_frac=loaded_frac,
    )
    id_ds = ds.get_split("id_test")
    ood_ds = ds.get_split("ood_test")
    id_ds.transform = split_tf
    ood_ds.transform = split_tf

    return {
        "id_test": GeoDataLoader(
            id_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=TrainConfig.NUM_WORKERS,
        ),
        "ood_test": GeoDataLoader(
            ood_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=TrainConfig.NUM_WORKERS,
        ),
    }


def run_sampling(
    model: torch.nn.Module,
    model_name: str,
    data_root: str,
    loaded_frac: float,
    device: torch.device,
) -> Dict[str, List[Dict]]:
    """Run sampling on id_test + ood_test. Returns {split: [sample_dict]}."""
    loaders = build_loaders(model_name, data_root, loaded_frac)

    if model_name == "mattergen":
        num_steps = min(
            TrainConfig.EVAL_NUM_STEPS, getattr(model, "num_diffusion_steps", 1000)
        )
    else:
        num_steps = None

    predictions: Dict[str, List[Dict]] = {"id_test": [], "ood_test": []}

    with torch.no_grad():
        for split in ("id_test", "ood_test"):
            loader = loaders[split]
            for data in tqdm(loader, desc=f"Sample {model_name} {split}"):
                data = data.to(device)
                if not hasattr(data, "num_atoms") or data.num_atoms is None:
                    data.num_atoms = data.ptr[1:] - data.ptr[:-1]

                if model_name == "diffcsp":
                    output = model.sample(data, num_atoms=data.num_atoms)
                    pos_key = "positions"
                else:
                    output = model.sample(data, num_steps=num_steps)
                    pos_key = "pos"

                _debug_sample_output(model_name, split, output, pos_key=pos_key)

                pred_pos = output[pos_key]
                if torch.is_tensor(pred_pos):
                    pred_pos = pred_pos.detach().cpu().numpy()
                pred_pos = np.asarray(pred_pos, dtype=np.float64)

                ptr = data.ptr.cpu().numpy()
                for i in range(len(ptr) - 1):
                    s, e = int(ptr[i]), int(ptr[i + 1])
                    if e - s < 4:
                        continue
                    record = get_all_attrs(data, i, ptr)
                    sample = {
                        "pred": pred_pos[s:e],
                        "material": str(record.get("material", "")),
                        "radius": float(record.get("radius", float("nan"))),
                        "rot_idx": int(record.get("rot_idx", -1)),
                    }
                    predictions[split].append(sample)

                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()

    return predictions


# =============================================================================
# Metrics (Phase 1/2/3) — matches run_one() logic from
# scripts/compute_metrics_from_predictions.py but operates on in-memory samples.
# =============================================================================


def _compute_one_sample(args):
    sample, gt, split = args
    pred = sample["pred"]
    metrics = compute_sample_metrics(pred, gt, radius=sample["radius"])
    metrics["material"] = sample["material"]
    metrics["radius"] = sample["radius"]
    metrics["rot_idx"] = sample["rot_idx"]
    metrics["split"] = split
    group_key = (sample["material"], sample["radius"])
    return (metrics, group_key, sample["rot_idx"], pred.copy(), gt.copy())


def _compute_one_rotation_group(args):
    group_key, preds_dict, gt_dict = args
    rot_metrics = compute_rotation_group_metrics(preds_dict, gt_dict)
    return (group_key, rot_metrics)


def compute_metrics(
    predictions: Dict[str, List[Dict]],
    gt_lookup: Dict[str, Dict[Tuple[str, float, int], np.ndarray]],
    n_jobs: Optional[int] = None,
) -> Tuple[Dict, Dict, int]:
    """Run Phases 1-3. Returns (aggregate, split_counts, num_rotation_groups)."""
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) - 1)

    split_counts: Dict[str, Dict[str, int]] = {}
    all_sample_metrics: List[Dict] = []
    predictions_by_group: Dict[Tuple[str, float], Dict[int, np.ndarray]] = defaultdict(
        dict
    )
    gt_by_group: Dict[Tuple[str, float], Dict[int, np.ndarray]] = defaultdict(dict)

    phase1_tasks: List[Tuple[Dict, np.ndarray, str]] = []

    print("\n=== Phase 1: Per-sample metrics ===", flush=True)
    for split in ("id_test", "ood_test"):
        samples = predictions.get(split, [])
        lookup = gt_lookup[split]
        matched, skipped = 0, 0
        for sample in samples:
            key = (sample["material"], sample["radius"], sample["rot_idx"])
            if key not in lookup:
                skipped += 1
                continue
            phase1_tasks.append((sample, lookup[key], split))
            matched += 1
        split_counts[split] = {
            "matched": matched,
            "skipped": skipped,
            "total": len(samples),
            "n_nan": 0,
        }
        print(
            f"  {split}: matched={matched}, skipped={skipped}, total={len(samples)}",
            flush=True,
        )

    if phase1_tasks:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            chunksize = max(1, len(phase1_tasks) // (n_jobs * 4))
            for result in pool.map(_compute_one_sample, phase1_tasks, chunksize=chunksize):
                metrics, group_key, rot, pred, gt = result
                all_sample_metrics.append(metrics)
                if not np.isfinite(metrics.get("rmsd", np.nan)):
                    split_counts[metrics["split"]]["n_nan"] += 1
                predictions_by_group[group_key][rot] = pred
                gt_by_group[group_key][rot] = gt
        print(
            f"  Computed {len(all_sample_metrics)} sample metrics ({n_jobs} workers)",
            flush=True,
        )

    print("\n=== Phase 2: Rotation consistency ===", flush=True)
    rotation_group_metrics: Dict[Tuple[str, float], Dict[str, float]] = {}
    phase2_tasks = [
        (group_key, preds_dict, gt_by_group[group_key])
        for group_key, preds_dict in predictions_by_group.items()
        if len(preds_dict) >= 2
    ]
    if phase2_tasks:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            chunksize = max(1, len(phase2_tasks) // (n_jobs * 2))
            for group_key, rot_metrics in pool.map(
                _compute_one_rotation_group, phase2_tasks, chunksize=chunksize
            ):
                if rot_metrics:
                    rotation_group_metrics[group_key] = rot_metrics
    print(
        f"  Computed rotation metrics for {len(rotation_group_metrics)} groups",
        flush=True,
    )

    print("\n=== Phase 3: Aggregate metrics ===", flush=True)
    aggregate = compute_aggregate_metrics(
        all_sample_metrics=all_sample_metrics,
        rotation_group_metrics=rotation_group_metrics,
        id_radii=TrainConfig.ID_RADII,
        ood_radii=TrainConfig.OOD_RADII,
    )

    return aggregate, split_counts, len(rotation_group_metrics)


# =============================================================================
# Model building
# =============================================================================


def build_model(model_name: str, model_cfg: dict, device: torch.device):
    if model_name == "diffcsp":
        return DiffCSPUnitCell(**model_cfg).to(device)
    if model_name == "mattergen":
        return MatterGen(**model_cfg).to(device)
    raise ValueError(f"Unknown model: {model_name}")


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    state_dict = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
        print("  load_state_dict strict=True succeeded", flush=True)
    except RuntimeError as e:
        print(f"  strict=True failed: {e}", flush=True)
        print("  retrying with strict=False", flush=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  missing keys ({len(missing)}): {missing[:10]}...", flush=True)
        if unexpected:
            print(
                f"  unexpected keys ({len(unexpected)}): {unexpected[:10]}...",
                flush=True,
            )


# =============================================================================
# Orchestration
# =============================================================================


def _resolve_device(override: Optional[str]) -> torch.device:
    if override is not None:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def eval_one_model(
    model_name: str,
    loaded_frac: float = 1.0,
    data_root: Optional[str] = None,
    device_override: Optional[str] = None,
) -> None:
    print(f"\n{'=' * 60}\n[{model_name}] evaluating published checkpoint\n{'=' * 60}", flush=True)

    ckpt_dir = os.path.join(REBUTTAL_DIR, model_name)
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    config_path = os.path.join(ckpt_dir, "config.json")

    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")
    if not os.path.isfile(config_path):
        raise SystemExit(f"Missing config:     {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    seed = int(config.get("seed", 1))
    model_cfg = config["model_config"]
    data_root = data_root or TrainConfig.DATA_ROOT

    print(f"  seed={seed}, loaded_frac={loaded_frac}, data_root={data_root}", flush=True)
    print(f"  model_config keys: {sorted(model_cfg.keys())}", flush=True)

    TrainConfig.set_seed(seed)

    device = _resolve_device(device_override)
    print(f"  device={device}", flush=True)
    model = build_model(model_name, model_cfg, device)
    _load_checkpoint(model, ckpt_path, device)
    model.eval()

    total_params_loaded = sum(p.numel() for p in model.parameters())
    expected_params = config.get("total_params")
    print(
        f"  total_params_loaded={total_params_loaded}, expected={expected_params}",
        flush=True,
    )
    if expected_params:
        rel = abs(total_params_loaded - expected_params) / max(1, expected_params)
        if rel > 0.001:
            print(
                f"  WARNING: param-count mismatch {rel * 100:.2f}% "
                f"(loaded={total_params_loaded}, expected={expected_params})",
                flush=True,
            )

    gt_lookup = _load_gt_cache(data_root)
    print(
        f"  GT cache: id_test={len(gt_lookup['id_test'])}, "
        f"ood_test={len(gt_lookup['ood_test'])}",
        flush=True,
    )

    predictions = run_sampling(
        model=model,
        model_name=model_name,
        data_root=data_root,
        loaded_frac=loaded_frac,
        device=device,
    )

    aggregate, split_counts, n_rot_groups = compute_metrics(predictions, gt_lookup)

    num_samples = sum(sc["matched"] for sc in split_counts.values())
    num_steps = (
        min(TrainConfig.EVAL_NUM_STEPS, getattr(model, "num_diffusion_steps", 1000))
        if model_name == "mattergen"
        else config.get("model_config", {}).get("num_diffusion_steps")
    )

    result = {
        "model_name": model_name,
        "model_size_label": "published",
        "checkpoint_path": ckpt_path,
        "config_path": config_path,
        "config": config,
        "seed": seed,
        "loaded_frac": loaded_frac,
        "eval_num_steps": num_steps,
        "timestamp": datetime.datetime.now().isoformat(),
        "device": str(device),
        "total_params_loaded": total_params_loaded,
        "split_counts": split_counts,
        "num_samples": num_samples,
        "num_rotation_groups": n_rot_groups,
        "aggregate": aggregate,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{model_name}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved metrics -> {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["diffcsp", "mattergen", "all"],
        default="all",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run with loaded_frac=0.02 for a quick sanity check.",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--device",
        default=None,
        choices=[None, "cpu", "mps", "cuda"],
        help="Override device (default: auto-detect CUDA > MPS > CPU).",
    )
    args = parser.parse_args()

    loaded_frac = 0.02 if args.smoke else 1.0
    targets = ["diffcsp", "mattergen"] if args.model == "all" else [args.model]

    for name in targets:
        eval_one_model(
            name,
            loaded_frac=loaded_frac,
            data_root=args.data_root,
            device_override=args.device,
        )

    print("\n=== Rebuttal evaluation done ===", flush=True)


if __name__ == "__main__":
    main()
