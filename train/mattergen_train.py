import json
import os
import time
from datetime import datetime

import pandas as pd
import torch
from torch import optim
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

from dataloaders.dataloader import RADIIDataloader, get_all_attrs
from models.mattergen_model import MatterGen
from train.config import TrainConfig, ModelConfig

TrainConfig.setup_torch([GlobalStorage, DataEdgeAttr, DataTensorAttr])


def run_epoch(model, loader, optimizer=None, train=False, device=None):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = total_nodes = 0
    pbar = tqdm(loader, desc=("Train " if train else "Eval  ") + "batch")

    for data in pbar:
        data = data.to(device)

        # MatterGen: forward(data) returns dict; loss(output) gives combined loss
        if train:
            output = model(data)
            loss = model.loss(output)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=TrainConfig.GRAD_CLIP_NORM
            )
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(data)
                loss = model.loss(output)

        n = data.num_nodes
        total_loss += loss.item() * n
        total_nodes += n
        pbar.set_postfix(loss=total_loss / total_nodes)

    return total_loss / total_nodes


def evaluate_and_save_predictions(
    model, loader, device, split, out_path, num_steps=100
):
    """Run model sampling and save pred + metadata to NPZ. Compute metrics later via compute_metrics_from_predictions.py"""
    import numpy as np

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


# Multi-seed loop (hyperparams from TrainConfig)
for SEED in TrainConfig.SEEDS:
    print(f"\n===== Seed {SEED} =====")
    TrainConfig.set_seed(SEED)

    # Output dir per seed
    out_dir = os.path.join("results", "task_1", "mattergen", str(SEED))
    os.makedirs(out_dir, exist_ok=True)

    config = dict(
        model_name="mattergen",
        model_config=ModelConfig.MatterGen.to_dict().copy(),
        seed=SEED,
        batch_size=TrainConfig.BATCH_SIZE,
        lr=TrainConfig.LR,
        num_epochs=TrainConfig.NUM_EPOCHS,
        train_ratio=TrainConfig.TRAIN_RATIO,
        device=str(TrainConfig.DEVICE),
        data_root=TrainConfig.DATA_ROOT,
        loaded_frac=TrainConfig.LOADED_FRAC,
        eval_num_steps=TrainConfig.EVAL_NUM_STEPS,
        grad_clip_norm=TrainConfig.GRAD_CLIP_NORM,
        timestamp=datetime.now().isoformat(),
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load dataset
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
    train_ds, val_ds = ds.random_id_splits(TrainConfig.TRAIN_RATIO, seed=SEED)
    id_test_ds = ds.get_split("id_test")
    ood_test_ds = ds.get_split("ood_test")
    for subset in (train_ds, val_ds, id_test_ds, ood_test_ds):
        subset.transform = clean
    loaders = {
        "train": GeoDataLoader(
            train_ds,
            batch_size=TrainConfig.BATCH_SIZE,
            shuffle=True,
            num_workers=TrainConfig.NUM_WORKERS,
        ),
        "val": GeoDataLoader(
            val_ds,
            batch_size=TrainConfig.BATCH_SIZE,
            shuffle=False,
            num_workers=TrainConfig.NUM_WORKERS,
        ),
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

    model = MatterGen(**ModelConfig.MatterGen.to_dict()).to(TrainConfig.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=TrainConfig.LR)
    scheduler = TrainConfig.make_scheduler(optimizer)

    # Training
    best_val = float("inf")
    log = []
    for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
        print(f"-- Epoch {epoch}/{TrainConfig.NUM_EPOCHS}")
        start_time = time.time()
        tl = run_epoch(model, loaders["train"], optimizer, True, TrainConfig.DEVICE)
        vl = run_epoch(model, loaders["val"], None, False, TrainConfig.DEVICE)
        epoch_duration = time.time() - start_time
        scheduler.step(vl)
        log.append(
            {
                "epoch": epoch,
                "train_loss": tl,
                "val_loss": vl,
                "epoch_duration": epoch_duration,
            }
        )
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
    pd.DataFrame(log).to_csv(os.path.join(out_dir, "training_log.csv"), index=False)

    # Evaluation - save predictions only; run compute_metrics_from_predictions.py later
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt")))
    eval_dir = os.path.join(out_dir, "eval_predictions")
    os.makedirs(eval_dir, exist_ok=True)
    for split in ["id_test", "ood_test"]:
        out_path = os.path.join(eval_dir, f"{split}.npz")
        evaluate_and_save_predictions(
            model,
            loaders[split],
            TrainConfig.DEVICE,
            split,
            out_path,
            num_steps=TrainConfig.EVAL_NUM_STEPS,
        )
    print(
        f"To compute metrics: python -m train.compute_metrics_from_predictions {out_dir}",
        flush=True,
    )
