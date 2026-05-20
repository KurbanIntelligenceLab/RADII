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
from models.flowmm_model import FlowMM
from train.config import TrainConfig, ModelConfig

TrainConfig.setup_torch([GlobalStorage, DataEdgeAttr, DataTensorAttr])

MODEL_NAME = "flowmm"


def run_epoch(model, loader, optimizer=None, train=False, device=None):
    """Run one epoch - uses model.forward() and model.loss() for flow matching."""
    model.train() if train else model.eval()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=("Train" if train else "Val"))
    for data in pbar:
        data = data.to(device)

        if train:
            optimizer.zero_grad()
            output = model(data)
            loss = model.loss(output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=TrainConfig.GRAD_CLIP_NORM
            )
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(data)
                loss = model.loss(output)

        bs = data.num_graphs
        total_loss += loss.item() * bs
        total_samples += bs

        pbar.set_postfix(loss=total_loss / total_samples)

    return total_loss / total_samples


def evaluate_and_save_predictions(
    model, loader, device, split, out_path, num_steps=100
):
    """Run model sampling and save pred + metadata to NPZ. Compute metrics later via compute_metrics_from_predictions.py"""
    import numpy as np

    model.eval()
    pred_list, meta_list = [], []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader, desc=f"Eval {split}")):
            data = data.to(device)
            if not hasattr(data, "num_atoms") or data.num_atoms is None:
                data.num_atoms = data.ptr[1:] - data.ptr[:-1]
            num_atoms = int(data.ptr[-1].item())
            num_graphs = data.num_graphs
            atoms_per_graph = (data.ptr[1:] - data.ptr[:-1]).tolist()
            mem_mb = (
                torch.cuda.memory_allocated(device) / 1e6
                if device.type == "cuda"
                else 0
            )
            # DEBUG: diagnose OOM - large batches of big graphs cause huge edge counts in radius_graph
            if batch_idx < 3 or max(atoms_per_graph) > 3000:
                print(
                    f"[eval batch {batch_idx}] num_graphs={num_graphs} num_atoms={num_atoms} "
                    f"atoms_per_graph={atoms_per_graph[:5]}{'...' if len(atoms_per_graph) > 5 else ''} "
                    f"mem={mem_mb:.0f}MB",
                    flush=True,
                )
            output = model.sample(data, num_steps=num_steps)
            pred_pos = output["pos"].cpu().numpy()
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

        model_config = ModelConfig.FlowMM.to_dict().copy()
        eval_batch_size = getattr(
            TrainConfig, "EVAL_BATCH_SIZE", TrainConfig.BATCH_SIZE
        )
        config = dict(
            model_name=MODEL_NAME,
            model_config=model_config,
            seed=SEED,
            batch_size=TrainConfig.BATCH_SIZE,
            eval_batch_size=eval_batch_size,
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

        # Data
        def transform(data):
            data.y_pos = data.pos.clone()
            if not hasattr(data, "cell_ptr") or data.cell_ptr is None:
                data.cell_ptr = torch.tensor(
                    [0, data.cell_pos.size(0)], dtype=torch.long
                )
            data.num_atoms = torch.tensor([data.pos.size(0)], dtype=torch.long)
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
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=TrainConfig.NUM_WORKERS,
            ),
            "ood_test": GeoDataLoader(
                ood_test_ds,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=TrainConfig.NUM_WORKERS,
            ),
        }

        # Model
        model = FlowMM(**model_config).to(TrainConfig.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=TrainConfig.LR)
        scheduler = TrainConfig.make_scheduler(optimizer)

        # Training
        epoch_logs = []
        best_val = float("inf")

        for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
            t0 = time.time()
            train_loss = run_epoch(
                model,
                loaders["train"],
                optimizer,
                train=True,
                device=TrainConfig.DEVICE,
            )
            val_loss = run_epoch(
                model, loaders["val"], train=False, device=TrainConfig.DEVICE
            )
            scheduler.step(val_loss)

            epoch_logs.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "time": time.time() - t0,
                }
            )

            print(f"Epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
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
                model,
                loaders[split],
                TrainConfig.DEVICE,
                split,
                out_path,
                num_steps=config["eval_num_steps"],
            )
        print(
            f"To compute metrics: python -m train.compute_metrics_from_predictions {out_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
