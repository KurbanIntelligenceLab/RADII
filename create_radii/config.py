"""
Configuration for RADII benchmark (~100K structures).
Clean scale-based splits for cross-scale generalization evaluation.
"""

from pathlib import Path
from scipy.spatial.transform import Rotation as R

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
RAW_DATA_DIR = Path("radii_raw")
OUTPUT_DIR = Path("radii")

QUATERNIONS_SUBDIR = "crystals"
UNIT_CELLS_SUBDIR = "unit_cells"

# ═══════════════════════════════════════════════════════════════════════════════
# SCALE (RADIUS) CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
r_values = list(range(6, 31))  # σ_6 to σ_30 (25 scales)

# Scale-based splits:
# - Train: mid-range scales → learn scale-invariant representations
# - ID: held-out mid-range scales → test generalization within seen regime
# - OOD: extreme scales (small + large) → test extrapolation beyond training
r_splits = {
    "train": [9, 10, 12, 14, 16, 18, 20, 22, 24, 26],  # 10 scales (mid-range, sampled)
    "ID": [11, 13, 15, 17, 19, 21],  # 6 scales (mid-range, interleaved)
    "OOD": [6, 7, 29, 30],  # 4 scales (extremes)
}

# ═══════════════════════════════════════════════════════════════════════════════
# TARGET COUNTS
# ═══════════════════════════════════════════════════════════════════════════════
TARGET_TOTAL_FILES = 75000

# Split fractions (must sum to 1.0)
SPLIT_FRACTIONS = {
    "train": 0.64,  # 64K structures
    "ID": 0.18,  # 18K structures
    "OOD": 0.18,  # 18K structures
}

# ═══════════════════════════════════════════════════════════════════════════════
# ROTATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
ANGLE_BY_SPLIT = {
    "train": {"base": 16.0},
    "ID": {"base": 14.0},
    "OOD": {"base": 12.0},
}

MAX_ROTS_PER_FILE = {
    "train": {"base": 426},
    "ID": {"base": 299},
    "OOD": {"base": 449},
}
MARGINS = {
    "train": 0.0,
    "ID": 8.0,
    "OOD": 10.0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# ORIENTATION OFFSETS
# ═══════════════════════════════════════════════════════════════════════════════
id_offset_euler = [20, 30, 45]
ood_offset_euler = [50, 70, 90]

ID_OFFSET = R.from_euler("xyz", id_offset_euler, degrees=True)
OOD_OFFSET = R.from_euler("xyz", ood_offset_euler, degrees=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MISC
# ═══════════════════════════════════════════════════════════════════════════════
GLOBAL_SEED = 42
max_workers = 4
coord_tolerance = 1e-6
