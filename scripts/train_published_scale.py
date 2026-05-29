"""
Train DiffCSP or MatterGen at published-scale configurations.

Addresses reviewer concerns (XQdQ 2a, sp5a Point 2, nm9J W4) about
running models at their published capacity rather than the 500K-param
RADII benchmark budget.

Usage:
    # Train DiffCSP at published scale (all 3 seeds)
    python -m scripts.train_published_scale --model diffcsp

    # Train MatterGen at published scale, single seed
    python -m scripts.train_published_scale --model mattergen --seed 1

    # Resume from checkpoint
    python -m scripts.train_published_scale --model diffcsp --seed 1 --resume
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

from radii.data import RADIIDataloader, get_all_attrs
from radii.models.diffcsp_model import DiffCSPUnitCell
from radii.models.mattergen_model import MatterGen
from radii.train_config import TrainConfig

TrainConfig.setup_torch([GlobalStorage, DataEdgeAttr, DataTensorAttr])

# =============================================================================
# Published-scale configs (not modifying train/config.py)
# =============================================================================

# DiffCSP: NeurIPS 2023, MP-20 setting (Jiao et al.)
# Published: CSPNet with hidden=512, 6 layers, 128 Fourier freqs = 12.3M params.
# Our RADII denoiser has slightly different edge features, so hidden_dim=465
# with 6 layers gives 12.4M — matching published param count.
PUBLISHED_DIFFCSP = dict(
    max_atomic_number=100,
    hidden_dim=465,
    num_layers=6,
    num_gaussians=128,
    cutoff_radius=7.0,
    time_dim=256,
    cond_hidden_dim=120,   # unchanged from RADII
    cond_num_layers=2,     # unchanged from RADII
    r_emb_dim=32,          # unchanged from RADII
    num_diffusion_steps=1000,
    beta_start=1e-4,
    beta_end=0.02,
)

# MatterGen: Nature 2025, base model (Zeni et al.)
# Published: 46.8M params with GemNet-T.
# Our RADII adaptation uses simpler equivariant MP (not GemNet-T), so we scale
# hidden_dim and num_layers to match the 46.8M parameter count.
# Note: cutoff_radius stays at 5.0 because EquivariantBlock hardcodes
# GaussianSmearing(0.0, 5.0, ...) — using cutoff>5.0 creates edges with
# zero distance features, causing numerical instability.
# num_gaussians stays at 50 to match the hardcoded smearing range.
PUBLISHED_MATTERGEN = dict(
    hidden_dim=1100,
    num_layers=5,
    num_gaussians=50,      # matches EquivariantBlock's GaussianSmearing(0, 5.0)
    cutoff_radius=5.0,     # matches EquivariantBlock's hardcoded range
    time_dim=256,
    num_atom_types=100,
    num_diffusion_steps=1000,  # published
    cond_hidden_dim=88,    # unchanged from RADII
    cond_num_layers=2,     # unchanged from RADII
)

# Training hyperparameters
# MatterGen at 47M params needs batch_size=1 to fit in A100 80GB
# (large nanoparticles with cutoff=5.0 create ~50 neighbors/atom,
#  hidden_dim=1100 * 5 layers * gradients easily exceeds 80GB at batch=2)
BATCH_SIZE_DIFFCSP = 2
BATCH_SIZE_MATTERGEN = 1
GRAD_ACCUM_DIFFCSP = 8    # effective batch = 16
GRAD_ACCUM_MATTERGEN = 16 # effective batch = 16
LR_DIFFCSP = 1e-4
LR_MATTERGEN = 1e-5       # 47M params without LayerNorm needs gentler LR
WARMUP_EPOCHS = 3          # linear warmup for MatterGen stability
NUM_EPOCHS = 50
LOADED_FRAC = 0.5         # same as original runs
GRAD_CLIP_NORM = 0.5       # published DiffCSP & MatterGen both use 0.5
SEEDS = [1, 2, 3]
EVAL_NUM_STEPS = 100      # sampling steps for evaluation


# =============================================================================
# Training loop with AMP + gradient accumulation
# =============================================================================

def run_epoch_diffcsp(model, loader, optimizer, scaler, train, device, accum_steps=8):
    """Training/val epoch for DiffCSP with AMP and gradient accumulation."""
    model.train() if train else model.eval()
    total_loss, total_lattice, total_coord = 0.0, 0.0, 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=("Train" if train else "Val"))
    for i, data in enumerate(pbar):
        data = data.to(device)

        if train:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(data, t=None)
                losses = model.loss(output)
                loss = losses["total"] / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRAD_CLIP_NORM
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(data, t=None)
                    losses = model.loss(output)

        bs = data.num_graphs
        total_loss += losses["total"].item() * bs
        total_lattice += losses["lattice"].item() * bs
        total_coord += losses["coord"].item() * bs
        total_samples += bs

        pbar.set_postfix(
            loss=total_loss / total_samples,
            lat=total_lattice / total_samples,
            coord=total_coord / total_samples,
        )

    # Flush remaining accumulated gradients
    if train and (i + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return {
        "loss": total_loss / total_samples,
        "lattice": total_lattice / total_samples,
        "coord": total_coord / total_samples,
    }


def run_epoch_mattergen(model, loader, optimizer, _scaler, train, device, accum_steps=16):
    """Training/val epoch for MatterGen with gradient accumulation (fp32).

    Note: MatterGen uses scatter_add_ in message passing which requires
    matching dtypes. bf16 autocast breaks this, so we run in fp32.
    """
    model.train() if train else model.eval()
    total_loss = total_nodes = 0

    pbar = tqdm(loader, desc=("Train" if train else "Val"))
    for i, data in enumerate(pbar):
        data = data.to(device)

        if train:
            output = model(data)
            loss = model.loss(output) / accum_steps
            loss.backward()

            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRAD_CLIP_NORM
                )
                optimizer.step()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                output = model(data)
                loss = model.loss(output)

        n = data.num_nodes
        total_loss += loss.item() * n * (accum_steps if train else 1)
        total_nodes += n
        pbar.set_postfix(loss=total_loss / total_nodes)

    # Flush remaining accumulated gradients
    if train and (i + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / total_nodes


# =============================================================================
# Evaluation (same pattern as existing scripts)
# =============================================================================

def evaluate_diffcsp(model, loader, device, split, out_path):
    """Run DiffCSP sampling and save predictions to NPZ."""
    model.eval()
    pred_list, meta_list = [], []

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Eval {split}"):
            data = data.to(device)
            if not hasattr(data, "num_atoms") or data.num_atoms is None:
                data.num_atoms = data.ptr[1:] - data.ptr[:-1]
            output = model.sample(data, num_atoms=data.num_atoms)
            pred_pos = output["positions"].cpu().numpy()
            ptr = data.ptr.cpu().numpy()
            for j in range(len(ptr) - 1):
                s, e = int(ptr[j]), int(ptr[j + 1])
                if e - s < 4:
                    continue
                pred_list.append(pred_pos[s:e])
                record = get_all_attrs(data, j, ptr)
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


def evaluate_mattergen(model, loader, device, split, out_path, num_steps):
    """Run MatterGen sampling and save predictions to NPZ."""
    model.eval()
    pred_list, meta_list = [], []

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Eval {split}"):
            data = data.to(device)
            if not hasattr(data, "num_atoms") or data.num_atoms is None:
                data.num_atoms = data.ptr[1:] - data.ptr[:-1]
            out = model.sample(
                data,
                num_steps=min(num_steps, getattr(model, "num_diffusion_steps", 1000)),
            )
            pred_pos = out["pos"].cpu().numpy()
            ptr = data.ptr.cpu().numpy()
            for j in range(len(ptr) - 1):
                s, e = int(ptr[j]), int(ptr[j + 1])
                if e - s < 4:
                    continue
                pred_list.append(pred_pos[s:e])
                record = get_all_attrs(data, j, ptr)
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


# =============================================================================
# Data transforms (replicated from existing training scripts)
# =============================================================================

def diffcsp_transform(data):
    """Transform for DiffCSP: add targets, cell_ptr, num_atoms, dummy lattice."""
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


def mattergen_transform(data):
    """Transform for MatterGen: add targets, cell_ptr."""
    data.y_pos = data.pos.clone()
    if not hasattr(data, "cell_ptr") or data.cell_ptr is None:
        data.cell_ptr = torch.tensor([0, data.cell_pos.size(0)], dtype=torch.long)
    return data


def mattergen_clean(data):
    """Clean transform applied to all splits for MatterGen."""
    data.y_pos = data.pos.clone()
    if not hasattr(data, "cell_ptr"):
        data.cell_ptr = torch.tensor([0, data.cell_pos.size(0)], dtype=torch.long)
    return data


# =============================================================================
# Main training function
# =============================================================================

def train_model(model_name, seed, num_epochs, resume):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name} (published scale) | Seed: {seed} | Device: {device}")
    print(f"{'=' * 60}")

    TrainConfig.set_seed(seed)

    # Select config and model class
    if model_name == "diffcsp":
        model_config = PUBLISHED_DIFFCSP.copy()
        model = DiffCSPUnitCell(**model_config).to(device)
        transform = diffcsp_transform
        batch_size = BATCH_SIZE_DIFFCSP
        accum_steps = GRAD_ACCUM_DIFFCSP
        lr = LR_DIFFCSP
        warmup_epochs = 0
        run_epoch = lambda m, l, o, s, t, d: run_epoch_diffcsp(m, l, o, s, t, d, accum_steps)
        evaluate = evaluate_diffcsp
    elif model_name == "mattergen":
        model_config = PUBLISHED_MATTERGEN.copy()
        model = MatterGen(**model_config).to(device)
        transform = mattergen_transform
        batch_size = BATCH_SIZE_MATTERGEN
        accum_steps = GRAD_ACCUM_MATTERGEN
        lr = LR_MATTERGEN
        warmup_epochs = WARMUP_EPOCHS
        run_epoch = lambda m, l, o, s, t, d: run_epoch_mattergen(m, l, o, s, t, d, accum_steps)
        evaluate = lambda m, l, d, s, p: evaluate_mattergen(m, l, d, s, p, EVAL_NUM_STEPS)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Print parameter count
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Output directory
    out_dir = os.path.join("results", "task_1", f"{model_name}_published", str(seed))
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    config = dict(
        model_name=f"{model_name}_published",
        model_config=model_config,
        seed=seed,
        batch_size=batch_size,
        grad_accum_steps=accum_steps,
        effective_batch_size=batch_size * accum_steps,
        lr=lr,
        num_epochs=num_epochs,
        loaded_frac=LOADED_FRAC,
        grad_clip_norm=GRAD_CLIP_NORM,
        mixed_precision="bfloat16",
        eval_num_steps=EVAL_NUM_STEPS,
        total_params=num_params,
        trainable_params=num_trainable,
        device=str(device),
        timestamp=datetime.now().isoformat(),
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load dataset
    ds = RADIIDataloader(
        root=TrainConfig.DATA_ROOT,
        num_workers=TrainConfig.DATA_NUM_WORKERS,
        transform=transform,
        loaded_frac=LOADED_FRAC,
    )
    train_ds, val_ds = ds.random_id_splits(TrainConfig.TRAIN_RATIO, seed=seed)
    id_test_ds = ds.get_split("id_test")
    ood_test_ds = ds.get_split("ood_test")

    clean_fn = mattergen_clean if model_name == "mattergen" else diffcsp_transform
    for subset in (train_ds, val_ds, id_test_ds, ood_test_ds):
        subset.transform = clean_fn

    loaders = {
        "train": GeoDataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        ),
        "val": GeoDataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0
        ),
        "id_test": GeoDataLoader(
            id_test_ds, batch_size=1, shuffle=False, num_workers=0
        ),
        "ood_test": GeoDataLoader(
            ood_test_ds, batch_size=1, shuffle=False, num_workers=0
        ),
    }

    # Optimizer, scheduler, scaler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scaler = torch.amp.GradScaler()

    # Resume from checkpoint
    start_epoch = 1
    best_val = float("inf")
    log = []

    checkpoint_path = os.path.join(out_dir, "checkpoint.pt")
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        log = ckpt.get("log", [])
        print(f"Resumed at epoch {start_epoch}, best_val={best_val:.6f}")

    # Training loop
    print(f"\nStarting training: epochs {start_epoch}-{num_epochs}")
    if warmup_epochs > 0:
        print(f"  LR warmup: {lr/warmup_epochs:.2e} -> {lr:.2e} over {warmup_epochs} epochs")
    for epoch in range(start_epoch, num_epochs + 1):
        # Linear LR warmup
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_lr = lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        t0 = time.time()
        optimizer.zero_grad()

        train_metrics = run_epoch(model, loaders["train"], optimizer, scaler, True, device)
        val_metrics = run_epoch(model, loaders["val"], None, scaler, False, device)

        val_loss = val_metrics if isinstance(val_metrics, float) else val_metrics["loss"]
        train_loss = train_metrics if isinstance(train_metrics, float) else train_metrics["loss"]
        if epoch > warmup_epochs:
            scheduler.step(val_loss)

        epoch_time = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr_now,
            "time": epoch_time,
        }
        if isinstance(train_metrics, dict):
            entry["train_lattice"] = train_metrics.get("lattice", 0)
            entry["train_coord"] = train_metrics.get("coord", 0)
        if isinstance(val_metrics, dict):
            entry["val_lattice"] = val_metrics.get("lattice", 0)
            entry["val_coord"] = val_metrics.get("coord", 0)
        log.append(entry)

        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"train={train_loss:.6f} val={val_loss:.6f} "
            f"lr={lr_now:.2e} time={epoch_time:.1f}s"
        )

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

        # Save checkpoint every epoch (for resume)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val,
                "log": log,
            },
            checkpoint_path,
        )

    # Save training log and final model
    pd.DataFrame(log).to_csv(os.path.join(out_dir, "epochs.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pt"))

    # Evaluation
    print(f"\nEvaluation (loading best model)...")
    model.load_state_dict(
        torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device)
    )
    eval_dir = os.path.join(out_dir, "eval_predictions")
    os.makedirs(eval_dir, exist_ok=True)

    for split in ["id_test", "ood_test"]:
        out_path = os.path.join(eval_dir, f"{split}.npz")
        evaluate(model, loaders[split], device, split, out_path)

    print(f"\nSeed {seed} complete. Results in {out_dir}/")
    return out_dir


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train DiffCSP or MatterGen at published-scale configs"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["diffcsp", "mattergen"],
        help="Which model to train",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed to run (default: run all 3 seeds)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else SEEDS

    for seed in seeds:
        train_model(args.model, seed, args.epochs, args.resume)

        # Marker file after each seed completes
        marker_dir = os.path.join("results", "task_1", f"{args.model}_published")
        Path(os.path.join(marker_dir, f"SEED_{seed}_DONE")).touch()
        print(f"\n>>> SEED_{seed}_DONE marker written to {marker_dir}/")

    print(f"\nAll seeds complete for {args.model}_published!")


if __name__ == "__main__":
    main()
