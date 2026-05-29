"""RADII: a radius-resolved benchmark for graph generative models in materials science.

Public API:
    RadiiDataset       — PyTorch-Geometric Dataset over nanoparticle structures
    RadiiDataloader    — convenience wrapper with split handling
    MODEL_REGISTRY     — name → model-class map (adit, cdvae, diffcsp, flowmm, mattergen)
    download_seed      — fetch radii_raw.zip from the Zenodo DOI
    create_radii       — regenerate the full benchmark from the seed zip
    __version__

Paper: "How Far Can You Grow? Characterizing the Extrapolation Frontier of Graph
Generative Models for Materials Science" (KDD '26).
Dataset DOI: 10.5281/zenodo.20431021 (CC-BY-4.0)
Code: https://github.com/KurbanIntelligenceLab/RADII (MIT)
"""
from __future__ import annotations

__version__ = "1.0.0"

from .data import RADIIDataset as RadiiDataset
from .data import RADIIDataloader as RadiiDataloader
from .download import download_seed
from .generation import create_radii
from .models import MODEL_REGISTRY

__all__ = [
    "RadiiDataset",
    "RadiiDataloader",
    "MODEL_REGISTRY",
    "download_seed",
    "create_radii",
    "__version__",
]
