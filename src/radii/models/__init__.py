"""Generative model registry for RADII.

Each model is a self-contained nn.Module. The dispatcher in `radii.train`
calls model-specific loss/sample adapters; see ADAPTERS in train.py for
the per-model glue.
"""
from __future__ import annotations

from .adit_model import ADiTUnitCell
from .cdvae_model import CDVAEUnitCell
from .diffcsp_model import DiffCSPUnitCell
from .flowmm_model import FlowMM
from .mattergen_model import MatterGen

MODEL_REGISTRY: dict[str, type] = {
    "adit": ADiTUnitCell,
    "cdvae": CDVAEUnitCell,
    "diffcsp": DiffCSPUnitCell,
    "flowmm": FlowMM,
    "mattergen": MatterGen,
}

__all__ = [
    "MODEL_REGISTRY",
    "ADiTUnitCell",
    "CDVAEUnitCell",
    "DiffCSPUnitCell",
    "FlowMM",
    "MatterGen",
]
