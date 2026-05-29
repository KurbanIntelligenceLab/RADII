"""
Count total and trainable parameters for all Task 1 models.

Reads model configs from config.py (same as train scripts). Target ~500-550k params each.

Run from repo root:
  python scripts/count_params.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch

from radii.models.adit_model import ADiTUnitCell
from radii.models.cdvae_model import CDVAEUnitCell
from radii.models.diffcsp_model import DiffCSPUnitCell
from radii.models.flowmm_model import FlowMM
from radii.models.mattergen_model import MatterGen
from radii.train_config import ModelConfig

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def count_params(model: torch.nn.Module) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    device = torch.device("cpu")
    results = []

    for name, ModelClass, config in [
        ("ADiT", ADiTUnitCell, ModelConfig.ADiT.to_dict()),
        ("CDVAE", CDVAEUnitCell, ModelConfig.CDVAE.to_dict()),
        ("DiffCSP", DiffCSPUnitCell, ModelConfig.DiffCSP.to_dict()),
        ("FlowMM", FlowMM, ModelConfig.FlowMM.to_dict()),
        ("MatterGen", MatterGen, ModelConfig.MatterGen.to_dict()),
    ]:
        try:
            model = ModelClass(**config).to(device)
            total, trainable = count_params(model)
            results.append((name, total, trainable))
        except Exception as e:
            results.append((name, None, None))
            print(f"{name} error: {e}", file=sys.stderr)

    print("\n" + "=" * 60)
    print("Task 1 model parameter counts (500-550k each)")
    print("=" * 60)
    print(f"{'Model':<12} {'Total':>18} {'Trainable':>18}")
    print("-" * 60)
    grand_total = grand_trainable = 0
    for name, total, trainable in results:
        if total is not None:
            print(f"{name:<12} {total:>18,} {trainable:>18,}")
            grand_total += total
            grand_trainable += trainable
        else:
            print(f"{name:<12} {'(failed)':>18} {'(failed)':>18}")
    print("-" * 60)
    print(f"{'TOTAL':<12} {grand_total:>18,} {grand_trainable:>18,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
