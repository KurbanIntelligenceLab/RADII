import os
import time
import json
from datetime import datetime

import pandas as pd
import torch
from torch import optim
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

from dataloaders.dataloader import RADIIDataloader, get_all_attrs
from models.diffcsp_model import DiffCSPUnitCell
from train.config import TrainConfig, ModelConfig

TrainConfig.setup_torch([GlobalStorage, DataEdgeAttr, DataTensorAttr])

MODEL_NAME = "diffcsp"


def run_epoch(model, loader, optimizer=None, train=False, device=None):
    """Run one epoch - uses model.forward() and model.loss()."""
    model.train() if train else model.eval()
    total_loss, total_lattice, total_coord = 0.0, 0.0, 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=("Train" if train else "Val"))
    for data in pbar:
        data = data.to(device)

        if train:
            optimizer.zero_grad()
            output = model(data, t=None)
            losses = model.loss(output)
            loss = losses["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=TrainConfig.GRAD_CLIP_NORM
            )
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(data, t=None)
                losses = model.loss(output)
                loss = losses["total"]

        bs = data.num_graphs
        total_loss += loss.item() * bs
        total_lattice += losses["lattice"].item() * bs
        total_coord += losses["coord"].item() * bs
        total_samples += bs

        pbar.set_postfix(
            loss=total_loss / total_samples,
            lat=total_lattice / total_samples,
            coord=total_coord / total_samples,
        )

    return {
        "loss": total_loss / total_samples,
        "lattice": total_lattice / total_samples,
        "coord": total_coord / total_samples,
    }


def evaluate_and_save_predictions(model, loader, device, split, out_path):
    """Run model sampling and save pred + metadata to NPZ. Compute metrics later via compute_metrics_from_predictions.py"""
    import numpy as np

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


def main():
    for SEED in TrainConfig.SEEDS:
        print(f"\n{'=' * 60}\nSEED {SEED}\n{'=' * 60}")
        TrainConfig.set_seed(SEED)

        out_dir = os.path.join("results", "task_1", MODEL_NAME, str(SEED))
        os.makedirs(out_dir, exist_ok=True)

        model_config = ModelConfig.DiffCSP.to_dict().copy()
        config = dict(
            model_name=MODEL_NAME,
            model_config=model_config,
            seed=SEED,
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

        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Data
        def transform(data):
            data.y_pos = data.pos.clone()
            if not hasattr(data, "cell_ptr") or data.cell_ptr is None:
                data.cell_ptr = torch.tensor(
                    [0, data.cell_pos.size(0)], dtype=torch.long
                )
            data.num_atoms = torch.tensor([data.pos.size(0)], dtype=torch.long)
            # Ensure lattice exists (DiffCSP needs it)
            if not hasattr(data, "lattice") or data.lattice is None:
                # Create dummy lattice from bounding box if not present
                pos_min = data.pos.min(dim=0).values
                pos_max = data.pos.max(dim=0).values
                box_size = (pos_max - pos_min).clamp(min=1.0)
                data.lattice = torch.diag(box_size * 1.2).unsqueeze(0)  # [1, 3, 3]
            return data

        ds = RADIIDataloader(
            root=TrainConfig.DATA_ROOT,
            num_workers=TrainConfig.DATA_NUM_WORKERS,
            transform=transform,
            loaded_frac=TrainConfig.LOADED_FRAC,
        )
        train_ds, val_ds = ds.random_id_splits(TrainConfig.TRAIN_RATIO, seed=SEED)
        id_test_ds = ds.get_split("id_test")
        ood_test_ds = ds.get_split("ood_test")

        for subset in (train_ds, val_ds, id_test_ds, ood_test_ds):
            subset.transform = transform

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

        # Model
        model = DiffCSPUnitCell(**model_config).to(TrainConfig.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=TrainConfig.LR)
        scheduler = TrainConfig.make_scheduler(optimizer)

        # Training
        epoch_logs = []
        best_val = float("inf")

        for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
            t0 = time.time()
            train_metrics = run_epoch(
                model,
                loaders["train"],
                optimizer,
                train=True,
                device=TrainConfig.DEVICE,
            )
            val_metrics = run_epoch(
                model, loaders["val"], train=False, device=TrainConfig.DEVICE
            )
            scheduler.step(val_metrics["loss"])

            epoch_logs.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_lattice": train_metrics["lattice"],
                    "train_coord": train_metrics["coord"],
                    "val_loss": val_metrics["loss"],
                    "val_lattice": val_metrics["lattice"],
                    "val_coord": val_metrics["coord"],
                    "lr": optimizer.param_groups[0]["lr"],
                    "time": time.time() - t0,
                }
            )

            print(
                f"Epoch {epoch}: train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f}"
            )

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

        pd.DataFrame(epoch_logs).to_csv(
            os.path.join(out_dir, "epochs.csv"), index=False
        )
        torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pt"))

        # Evaluation - save predictions only; run compute_metrics_from_predictions.py later
        model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt")))
        eval_dir = os.path.join(out_dir, "eval_predictions")
        os.makedirs(eval_dir, exist_ok=True)
        for split in ["id_test", "ood_test"]:
            out_path = os.path.join(eval_dir, f"{split}.npz")
            evaluate_and_save_predictions(
                model, loaders[split], TrainConfig.DEVICE, split, out_path
            )
        print(
            f"To compute metrics: python -m train.compute_metrics_from_predictions {out_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
