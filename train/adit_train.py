import os
import time
import json
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch import optim
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

from dataloaders.dataloader import RADIIDataloader, get_all_attrs
from models.adit_model import ADiTUnitCell
from train.config import TrainConfig, ModelConfig

TrainConfig.setup_torch([GlobalStorage, DataEdgeAttr, DataTensorAttr])

MODEL_NAME = "adit"


def run_epoch(model, loader, optimizer=None, train=False, device=None):
    model.train() if train else model.eval()
    total_loss, total_samples = 0.0, 0

    pbar = tqdm(loader, desc=("Train" if train else "Val"))
    for data in pbar:
        data = data.to(device)

        if train:
            optimizer.zero_grad()
            output = model(data, t=None)
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=TrainConfig.GRAD_CLIP_NORM
            )
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(data, t=None)
                loss = output["loss"]

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
        pbar.set_postfix(loss=total_loss / total_samples)

    return total_loss / total_samples


def evaluate_and_save_predictions(
    model, loader, device, split, out_path, num_steps=100, guidance_scale=1.0
):
    """
    Run model sampling and save predictions + metadata to NPZ.
    Metrics computed separately via compute_metrics_from_predictions.py
    """
    model.eval()
    pred_list, meta_list = [], []

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Eval {split}"):
            data = data.to(device)

            if not hasattr(data, "num_atoms") or data.num_atoms is None:
                data.num_atoms = data.ptr[1:] - data.ptr[:-1]

            output = model.sample(
                data, num_steps=num_steps, guidance_scale=guidance_scale
            )

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
        print(f"\n{'=' * 60}\nSEED {SEED}\n{'=' * 60}", flush=True)
        TrainConfig.set_seed(SEED)

        out_dir = os.path.join("results", "task_1", MODEL_NAME, str(SEED))
        os.makedirs(out_dir, exist_ok=True)

        model_config = ModelConfig.ADiT.to_dict().copy()

        # === ADDED: Include split radii in config for metrics script ===
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
            vae_pretrain_epochs=TrainConfig.VAE_PRETRAIN_EPOCHS,
            eval_num_steps=TrainConfig.EVAL_NUM_STEPS,
            eval_guidance_scale=TrainConfig.EVAL_GUIDANCE_SCALE,
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

        print("Loading dataset..", flush=True)
        ds = RADIIDataloader(
            root=TrainConfig.DATA_ROOT,
            num_workers=TrainConfig.DATA_NUM_WORKERS,
            transform=transform,
            loaded_frac=TrainConfig.LOADED_FRAC,
        )
        train_ds, val_ds = ds.random_id_splits(TrainConfig.TRAIN_RATIO, seed=SEED)
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}", flush=True)

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
        model = ADiTUnitCell(**model_config).to(TrainConfig.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=TrainConfig.LR)
        scheduler = TrainConfig.make_scheduler(optimizer)

        epoch_logs = []

        # VAE pretraining (if applicable)
        if config["vae_pretrain_epochs"] > 0:
            vae_optimizer = optim.Adam(model.vae.parameters(), lr=TrainConfig.LR)

            for epoch in range(1, config["vae_pretrain_epochs"] + 1):
                model.train()
                losses = {"total": 0, "recon_atom": 0, "recon_pos": 0, "kl": 0}
                n = 0
                t0_vae = time.time()

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

                epoch_logs.append(
                    {
                        "epoch": -config["vae_pretrain_epochs"] + epoch - 1,
                        "phase": "vae_pretrain",
                        "train_loss": losses["total"] / n,
                        "recon_atom": losses["recon_atom"] / n,
                        "recon_pos": losses["recon_pos"] / n,
                        "kl": losses["kl"] / n,
                    }
                )
                elapsed = time.time() - t0_vae
                print(
                    f"VAE pretrain epoch {epoch}/{config['vae_pretrain_epochs']}: loss={losses['total'] / n:.6f} ({elapsed:.1f}s)",
                    flush=True,
                )

        # Main training
        print("Starting main training...", flush=True)
        best_val = float("inf")

        for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
            t0 = time.time()
            tl = run_epoch(
                model,
                loaders["train"],
                optimizer,
                train=True,
                device=TrainConfig.DEVICE,
            )
            vl = run_epoch(
                model, loaders["val"], train=False, device=TrainConfig.DEVICE
            )
            scheduler.step(vl)

            epoch_logs.append(
                {
                    "epoch": epoch,
                    "phase": "train",
                    "train_loss": tl,
                    "val_loss": vl,
                    "lr": optimizer.param_groups[0]["lr"],
                    "time": time.time() - t0,
                }
            )

            print(f"Epoch {epoch}: train={tl:.6f} val={vl:.6f}", flush=True)

            if vl < best_val:
                best_val = vl
                torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

        pd.DataFrame(epoch_logs).to_csv(
            os.path.join(out_dir, "epochs.csv"), index=False
        )
        torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pt"))

        # Evaluation
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
                guidance_scale=config["eval_guidance_scale"],
            )

        print("\nTraining complete. Run metrics with:")
        print(f"  python -m train.compute_metrics_from_predictions {out_dir}")


if __name__ == "__main__":
    main()
