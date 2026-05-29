"""Dataset generation pipeline: seed CIFs + base XYZ → full 12 GB cache.

`create_radii(raw_data, output_dir)` consumes either a `radii_raw/` directory
or a `radii_raw.zip` archive and writes the expanded benchmark to `output_dir`.
"""
from __future__ import annotations

from .create_radii import create_radii

__all__ = ["create_radii"]
