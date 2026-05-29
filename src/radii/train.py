"""Unified trainer for the 5 RADII benchmark models.

CLI: `python -m radii.train --model {adit|cdvae|diffcsp|flowmm|mattergen} [--seeds 1 2 ...]`

Replaces train/adit_train.py, train/cdvae_train.py, train/diffcsp_train.py,
train/flowmm_train.py, train/mattergen_train.py with a single entrypoint.
Per-model differences (loss API, sample kwargs, optional VAE pretraining) are
dispatched via the TRAINERS table below — no model behavior is changed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

from .data import RADIIDataloader, get_all_attrs
from .models import MODEL_REGISTRY
from .train_config import ModelConfig, TrainConfig

TrainConfig.setup_torch([GlobalStorage, DataEdgeAttr, DataTensorAttr])


# ---------------------------------------------------------------------------
# Per-model dispatch table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainerSpec:
    config_attr: str                      # nested class on ModelConfig (e.g., "ADiT")
    compute_loss: Callable                # (model, data) -> Tensor
    sample: Callable                      # (model, data, eval_kwargs) -> {"positions": ndarray}
    eval_kwargs_factory: Callable         # (full_config_dict) -> dict
    extra_config_keys: tuple[str, ...]    # TrainConfig fields to include in saved config
    vae_pretrain: bool = False            # ADiT only


def _adit_eval_kwargs(cfg: dict) -> dict:
    return {"num_steps": cfg["eval_num_steps"], "guidance_scale": cfg["eval_guidance_scale"]}


def _cdvae_eval_kwargs(cfg: dict) -> dict:
    return {"num_langevin_steps": cfg["eval_langevin_steps"], "step_size": cfg["eval_step_size"]}


def _num_steps_only(cfg: dict) -> dict:
    return {"num_steps": cfg["eval_num_steps"]}


def _empty_kwargs(_: dict) -> dict:
    return {}


TRAINERS: dict[str, TrainerSpec] = {
    "adit": TrainerSpec(
        config_attr="ADiT",
        compute_loss=lambda m, d: m(d, t=None)["loss"],
        sample=lambda m, d, kw: m.sample(d, **kw),
        eval_kwargs_factory=_adit_eval_kwargs,
        extra_config_keys=(
            "vae_pretrain_epochs", "eval_num_steps", "eval_guidance_scale",
        ),
        vae_pretrain=True,
    ),
    "cdvae": TrainerSpec(
        config_attr="CDVAE",
        compute_loss=lambda m, d: m.loss(m(d)),
        sample=lambda m, d, kw: m.sample(d, **kw),
        eval_kwargs_factory=_cdvae_eval_kwargs,
        extra_config_keys=("eval_langevin_steps", "eval_step_size"),
    ),
    "diffcsp": TrainerSpec(
        config_attr="DiffCSP",
        compute_loss=lambda m, d: m.loss(m(d)),
        sample=lambda m, d, kw: m.sample(d, **kw),
        eval_kwargs_factory=_empty_kwargs,
        extra_config_keys=(),
    ),
    "flowmm": TrainerSpec(
        config_attr="FlowMM",
        compute_loss=lambda m, d: m.loss(m(d)),
        sample=lambda m, d, kw: m.sample(d, **kw),
        eval_kwargs_factory=_num_steps_only,
        extra_config_keys=("eval_num_steps",),
    ),
    "mattergen": TrainerSpec(
        config_attr="MatterGen",
        compute_loss=lambda m, d: m.loss(m(d)),
        sample=lambda m, d, kw: m.sample(d, **kw),
        eval_kwargs_factory=_num_steps_only,
        extra_config_keys=("eval_num_steps",),
    ),
}


# ---------------------------------------------------------------------------
# Generic training loop (same shape as adit_train.run_epoch, with per-model loss)
# ---------------------------------------------------------------------------

def run_epoch(model, loader, spec: TrainerSpec, optimizer=None, train=False, device=None):
    model.train() if train else model.eval()
    total_loss, total_samples = 0.0, 0

    pbar = tqdm(loader, desc=("Train" if train else "Val"))
    for data in pbar:
        data = data.to(device)

        if train:
            optimizer.zero_grad()
            loss = spec.compute_loss(model, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=TrainConfig.GRAD_CLIP_NORM
            )
            optimizer.step()
        else:
            with torch.no_grad():
                loss = spec.compute_loss(model, data)

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
        pbar.set_postfix(loss=total_loss / total_samples)

    return total_loss / total_samples


def evaluate_and_save_predictions(model, loader, spec, device, split, out_path, eval_kwargs):
    """Run model sampling and save pred + metadata to NPZ.

    Compute metrics later via `python scripts/compute_metrics_from_predictions.py <out_dir>`.
    """
    model.eval()
    pred_list, meta_list = [], []

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Eval {split}"):
            data = data.to(device)

            if not hasattr(data, "num_atoms") or data.num_atoms is None:
                data.num_atoms = data.ptr[1:] - data.ptr[:-1]

            output = spec.sample(model, data, eval_kwargs)
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

    pred_cat = np.concatenate(pred_list, axis=0)
    ptr_arr = np.concatenate([[0], np.cumsum([len(p) for p in pred_list])])

    np.savez_compressed(
        out_path,
        pred_pos=pred_cat,
        ptr=ptr_arr,
        materials=np.array([m.get("material", "") for m in meta_list], dtype=object),
        radius=np.array([float(m.get("radius", np.nan)) for m in meta_list]),
        rot_idx=np.array([int(m.get("rot_idx", -1)) for m in meta_list]),
    )
    print(f"Saved {len(pred_list)} predictions to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _transform(data):
    """Common preprocessor matching the per-model train scripts."""
    data.y_pos = data.pos.clone()
    if not hasattr(data, "cell_ptr") or data.cell_ptr is None:
        data.cell_ptr = torch.tensor([0, data.cell_pos.size(0)], dtype=torch.long)
    data.num_atoms = torch.tensor([data.pos.size(0)], dtype=torch.long)
    return data


def _build_loaders(seed: int) -> dict[str, GeoDataLoader]:
    ds = RADIIDataloader(
        root=TrainConfig.DATA_ROOT,
        num_workers=TrainConfig.DATA_NUM_WORKERS,
        transform=_transform,
        loaded_frac=TrainConfig.LOADED_FRAC,
    )
    train_ds, val_ds = ds.random_id_splits(TrainConfig.TRAIN_RATIO, seed=seed)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}", flush=True)

    id_test_ds = ds.get_split("id_test")
    ood_test_ds = ds.get_split("ood_test")
    for subset in (train_ds, val_ds, id_test_ds, ood_test_ds):
        subset.transform = _transform

    return {
        "train": GeoDataLoader(train_ds, batch_size=TrainConfig.BATCH_SIZE, shuffle=True,
                               num_workers=TrainConfig.NUM_WORKERS),
        "val": GeoDataLoader(val_ds, batch_size=TrainConfig.BATCH_SIZE, shuffle=False,
                             num_workers=TrainConfig.NUM_WORKERS),
        "id_test": GeoDataLoader(id_test_ds, batch_size=TrainConfig.BATCH_SIZE, shuffle=False,
                                 num_workers=TrainConfig.NUM_WORKERS),
        "ood_test": GeoDataLoader(ood_test_ds, batch_size=TrainConfig.BATCH_SIZE, shuffle=False,
                                  num_workers=TrainConfig.NUM_WORKERS),
    }


def _build_full_config(model_name: str, model_config: dict, seed: int, spec: TrainerSpec) -> dict:
    base = dict(
        model_name=model_name,
        model_config=model_config,
        seed=seed,
        batch_size=TrainConfig.BATCH_SIZE,
        lr=TrainConfig.LR,
        num_epochs=TrainConfig.NUM_EPOCHS,
        train_ratio=TrainConfig.TRAIN_RATIO,
        device=str(TrainConfig.DEVICE),
        data_root=TrainConfig.DATA_ROOT,
        loaded_frac=TrainConfig.LOADED_FRAC,
        grad_clip_norm=TrainConfig.GRAD_CLIP_NORM,
        timestamp=datetime.now().isoformat(),
    )
    for k in spec.extra_config_keys:
        # TrainConfig field naming: vae_pretrain_epochs → VAE_PRETRAIN_EPOCHS
        attr = k.upper()
        if hasattr(TrainConfig, attr):
            base[k] = getattr(TrainConfig, attr)
    return base


def _vae_pretrain(model, loaders, vae_epochs: int, epoch_logs: list) -> None:
    """ADiT-only VAE pretraining phase."""
    vae_optimizer = optim.Adam(model.vae.parameters(), lr=TrainConfig.LR)
    for epoch in range(1, vae_epochs + 1):
        model.train()
        losses = {"total": 0, "recon_atom": 0, "recon_pos": 0, "kl": 0}
        n = 0
        t0 = time.time()
        for data in tqdm(loaders["train"], desc=f"VAE {epoch}"):
            data = data.to(TrainConfig.DEVICE)
            vae_optimizer.zero_grad()
            vae_losses = model.train_vae_step(data)
            vae_losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.vae.parameters(), max_norm=TrainConfig.GRAD_CLIP_NORM
            )
            vae_optimizer.step()
            bs = data.num_graphs
            for k in losses:
                losses[k] += vae_losses[k].item() * bs
            n += bs
        epoch_logs.append({
            "epoch": -vae_epochs + epoch - 1,
            "phase": "vae_pretrain",
            "train_loss": losses["total"] / n,
            "recon_atom": losses["recon_atom"] / n,
            "recon_pos": losses["recon_pos"] / n,
            "kl": losses["kl"] / n,
        })
        elapsed = time.time() - t0
        print(
            f"VAE pretrain epoch {epoch}/{vae_epochs}: loss={losses['total']/n:.6f} ({elapsed:.1f}s)",
            flush=True,
        )


def _run_eval(model, loaders, spec, eval_kwargs, out_dir: str) -> None:
    """Run eval on id_test + ood_test, save predictions npz."""
    eval_dir = os.path.join(out_dir, "eval_predictions")
    os.makedirs(eval_dir, exist_ok=True)
    for split in ["id_test", "ood_test"]:
        out_path = os.path.join(eval_dir, f"{split}.npz")
        evaluate_and_save_predictions(
            model, loaders[split], spec, TrainConfig.DEVICE, split, out_path, eval_kwargs
        )


def train_one_seed(
    model_name: str,
    seed: int,
    *,
    out_dir: str | None = None,
    from_checkpoint: str | None = None,
    save_pretrained_dir: str | None = None,
) -> None:
    spec = TRAINERS[model_name]
    model_cls = MODEL_REGISTRY[model_name]
    config_cls = getattr(ModelConfig, spec.config_attr)

    print(f"\n{'=' * 60}\nMODEL {model_name}  SEED {seed}\n{'=' * 60}", flush=True)
    TrainConfig.set_seed(seed)

    if out_dir is None:
        out_dir = os.path.join("results", "task_1", model_name, str(seed))
    os.makedirs(out_dir, exist_ok=True)

    model_config = config_cls.to_dict().copy()
    full_config = _build_full_config(model_name, model_config, seed, spec)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(full_config, f, indent=2)

    print("Loading dataset..", flush=True)
    loaders = _build_loaders(seed)

    model = model_cls(**model_config).to(TrainConfig.DEVICE)
    if from_checkpoint:
        print(f"Loading initial weights from {from_checkpoint}", flush=True)
        model.load_state_dict(torch.load(from_checkpoint, map_location=TrainConfig.DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=TrainConfig.LR)
    scheduler = TrainConfig.make_scheduler(optimizer)
    epoch_logs: list[dict] = []

    if spec.vae_pretrain and full_config.get("vae_pretrain_epochs", 0) > 0:
        _vae_pretrain(model, loaders, full_config["vae_pretrain_epochs"], epoch_logs)

    print("Starting main training...", flush=True)
    best_val = float("inf")
    for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
        t0 = time.time()
        tl = run_epoch(model, loaders["train"], spec, optimizer, train=True, device=TrainConfig.DEVICE)
        vl = run_epoch(model, loaders["val"], spec, train=False, device=TrainConfig.DEVICE)
        scheduler.step(vl)
        epoch_logs.append({
            "epoch": epoch,
            "phase": "train",
            "train_loss": tl,
            "val_loss": vl,
            "lr": optimizer.param_groups[0]["lr"],
            "time": time.time() - t0,
        })
        print(f"Epoch {epoch}: train={tl:.6f} val={vl:.6f}", flush=True)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

    pd.DataFrame(epoch_logs).to_csv(os.path.join(out_dir, "epochs.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pt"))

    if save_pretrained_dir:
        print(f"Saving HF-format artifact to {save_pretrained_dir}", flush=True)
        model.save_pretrained(save_pretrained_dir)

    # Eval (load best checkpoint first)
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt")))
    eval_kwargs = spec.eval_kwargs_factory(full_config)
    _run_eval(model, loaders, spec, eval_kwargs, out_dir)

    print(f"\nTraining complete. Run metrics with:")
    print(f"  python scripts/compute_metrics_from_predictions.py {out_dir}")


def eval_only(model_name: str, checkpoint: str, seed: int, out_dir: str | None) -> None:
    """Load a checkpoint, skip training, run eval on id_test + ood_test."""
    spec = TRAINERS[model_name]
    model_cls = MODEL_REGISTRY[model_name]
    config_cls = getattr(ModelConfig, spec.config_attr)

    TrainConfig.set_seed(seed)
    if out_dir is None:
        out_dir = os.path.join("results", "task_1", model_name, str(seed), "eval_only")
    os.makedirs(out_dir, exist_ok=True)

    model_config = config_cls.to_dict().copy()
    full_config = _build_full_config(model_name, model_config, seed, spec)

    print(f"\n{'=' * 60}\nEVAL-ONLY  {model_name}  seed={seed}\n  checkpoint: {checkpoint}\n  out_dir: {out_dir}\n{'=' * 60}", flush=True)
    print("Loading dataset..", flush=True)
    loaders = _build_loaders(seed)

    model = model_cls(**model_config).to(TrainConfig.DEVICE)
    # Support both raw state-dict .pt files and HF save_pretrained directories.
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        model = model_cls.from_pretrained(str(ckpt_path)).to(TrainConfig.DEVICE)
    else:
        model.load_state_dict(torch.load(str(ckpt_path), map_location=TrainConfig.DEVICE))

    eval_kwargs = spec.eval_kwargs_factory(full_config)
    _run_eval(model, loaders, spec, eval_kwargs, out_dir)

    print(f"\nEval complete. Predictions at: {out_dir}/eval_predictions/")
    print(f"  Compute metrics: python scripts/compute_metrics_from_predictions.py {out_dir}")


def _apply_overrides(args) -> None:
    """Mutate TrainConfig class attrs from CLI overrides (one-shot at program start)."""
    if args.epochs is not None:
        TrainConfig.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        TrainConfig.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        TrainConfig.LR = args.lr
    if args.data_root is not None:
        TrainConfig.DATA_ROOT = args.data_root


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m radii.train --model adit
  python -m radii.train --model adit --epochs 50 --lr 1e-4
  python -m radii.train --model adit --seeds 1 2 3
  python -m radii.train --model adit --from-checkpoint results/task_1/adit/1/best_model.pt --epochs 10
  python -m radii.train --model adit --eval-only --checkpoint results/task_1/adit/1/best_model.pt
  python -m radii.train --model adit --save-pretrained-dir ./adit_hf
""",
    )
    p.add_argument("--model", required=True, choices=sorted(TRAINERS.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="One or more seeds (overrides TrainConfig.SEEDS)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override TrainConfig.NUM_EPOCHS")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override TrainConfig.BATCH_SIZE")
    p.add_argument("--lr", type=float, default=None,
                   help="Override TrainConfig.LR")
    p.add_argument("--data-root", default=None,
                   help="Override TrainConfig.DATA_ROOT (path to the radii cache dir)")
    p.add_argument("--out-dir", default=None,
                   help="Override default output dir (results/task_1/<model>/<seed>)")
    p.add_argument("--from-checkpoint", default=None,
                   help="Initialize model weights from a .pt before training (still trains)")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load --checkpoint and run eval on id_test + ood_test")
    p.add_argument("--checkpoint", default=None,
                   help="Required by --eval-only: .pt file OR a save_pretrained directory")
    p.add_argument("--save-pretrained-dir", default=None,
                   help="After training, also call model.save_pretrained(DIR)")
    args = p.parse_args()

    _apply_overrides(args)

    if args.eval_only:
        if not args.checkpoint:
            sys.exit("--eval-only requires --checkpoint PATH")
        seed = args.seeds[0] if args.seeds else list(TrainConfig.SEEDS)[0]
        eval_only(args.model, args.checkpoint, seed, args.out_dir)
        return

    seeds = args.seeds if args.seeds is not None else list(TrainConfig.SEEDS)
    for seed in seeds:
        train_one_seed(
            args.model, seed,
            out_dir=args.out_dir,
            from_checkpoint=args.from_checkpoint,
            save_pretrained_dir=args.save_pretrained_dir,
        )


if __name__ == "__main__":
    main()
