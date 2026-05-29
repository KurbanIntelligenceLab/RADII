"""
RADII Benchmark Dataloader with NPZ Caching

Directory structure:
    radii/
    ├── crystals/
    │   └── {material}/
    │       └── R{radius}/
    │           └── xyz/
    │               └── rot_{i}.xyz
    ├── unit_cells/
    │   └── {material}.cif
    └── cache/                    # Auto-created
        ├── samples.npz           # All nanoparticle data
        ├── unit_cells.npz        # All unit cell data
        └── metadata.json         # Index and split info

Each sample contains:
    - Nanoparticle structure (pos, z) from xyz file
    - Unit cell structure (cell_pos, cell_z) from cif file
    - Metadata: material, radius, split, rotation info

Caching:
    - First run: parses all files and creates cache
    - Subsequent runs: loads from cache (10-100x faster)
    - Cache invalidation: delete cache/ directory to rebuild
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Optional, List, Dict, Callable, Tuple
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Optional: ase for CIF parsing (fallback to simple parser if not available)
try:
    from ase.io import read as ase_read

    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    warnings.warn("ASE not installed. Using simple CIF parser (limited support).")

# Shared element symbol -> atomic number
ELEMENT_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
}


def get_all_attrs(data, idx, ptr):
    """Extract ALL available attributes from a data sample."""
    record = {}

    s, e = int(ptr[idx]), int(ptr[idx + 1])
    record["num_atoms"] = e - s

    for key in data.keys():
        if key in ("ptr", "batch", "edge_index", "edge_attr"):
            continue

        val = data[key]
        if val is None:
            continue

        try:
            if isinstance(val, torch.Tensor):
                if val.dim() == 0:
                    record[key] = val.item()
                elif val.size(0) == data.num_graphs:
                    record[key] = (
                        val[idx].item() if val[idx].numel() == 1 else val[idx].tolist()
                    )
            elif isinstance(val, (list, tuple)) and len(val) == data.num_graphs:
                record[key] = val[idx]
            elif isinstance(val, (int, float, str)):
                record[key] = val
        except Exception:
            pass

    return record


# =============================================================================
# Parsers
# =============================================================================


def parse_xyz(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Parse XYZ file."""
    with open(path, "r") as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())
    comment = lines[1].strip()

    metadata = {"comment": comment}
    for match in re.finditer(r"(\w+)=([^\s|]+)", comment):
        key, val = match.groups()
        metadata[key] = val

    elements = []
    coords = []
    for line in lines[2 : 2 + num_atoms]:
        parts = line.split()
        elem = parts[0].capitalize()
        elements.append(elem)
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coords = np.array(coords, dtype=np.float32)
    atomic_numbers = np.array(
        [ELEMENT_TO_Z.get(e, 0) for e in elements], dtype=np.int64
    )
    return coords, atomic_numbers, metadata


def parse_cif_ase(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse CIF using ASE."""
    atoms = ase_read(path)
    coords = atoms.get_positions().astype(np.float32)
    atomic_numbers = atoms.get_atomic_numbers().astype(np.int64)
    cell = atoms.get_cell().array.astype(np.float32)
    return coords, atomic_numbers, cell


def parse_cif_simple(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple CIF parser for basic structures (fallback)."""
    with open(path, "r") as f:
        content = f.read()

    a = float(re.search(r"_cell_length_a\s+([\d.]+)", content).group(1))
    b = float(re.search(r"_cell_length_b\s+([\d.]+)", content).group(1))
    c = float(re.search(r"_cell_length_c\s+([\d.]+)", content).group(1))
    alpha = float(re.search(r"_cell_angle_alpha\s+([\d.]+)", content).group(1))
    beta = float(re.search(r"_cell_angle_beta\s+([\d.]+)", content).group(1))
    gamma = float(re.search(r"_cell_angle_gamma\s+([\d.]+)", content).group(1))

    alpha_r, beta_r, gamma_r = np.radians([alpha, beta, gamma])
    cos_a, cos_b, cos_g = np.cos([alpha_r, beta_r, gamma_r])
    sin_g = np.sin(gamma_r)

    cell = np.array(
        [
            [a, 0, 0],
            [b * cos_g, b * sin_g, 0],
            [
                c * cos_b,
                c * (cos_a - cos_b * cos_g) / sin_g,
                c
                * np.sqrt(
                    1 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g
                )
                / sin_g,
            ],
        ],
        dtype=np.float32,
    )

    atoms = []
    lines = content.split("\n")
    in_atom_block = False
    atom_cols = {}

    for line in lines:
        line = line.strip()
        if line.startswith("_atom_site_"):
            col_name = line.split()[0]
            atom_cols[col_name] = len(atom_cols)
            in_atom_block = True
        elif (
            in_atom_block
            and line
            and not line.startswith("_")
            and not line.startswith("loop_")
        ):
            parts = line.split()
            if len(parts) >= len(atom_cols):
                try:
                    elem = parts[atom_cols.get("_atom_site_type_symbol", 0)]
                    elem = re.sub(r"[^A-Za-z]", "", elem).capitalize()
                    x = float(parts[atom_cols.get("_atom_site_fract_x", 1)])
                    y = float(parts[atom_cols.get("_atom_site_fract_y", 2)])
                    z = float(parts[atom_cols.get("_atom_site_fract_z", 3)])
                    atoms.append((elem, x, y, z))
                except (ValueError, IndexError):
                    continue
        elif in_atom_block and (
            line.startswith("loop_")
            or line.startswith("_")
            and "_atom_site" not in line
        ):
            in_atom_block = False

    if not atoms:
        raise ValueError(f"No atoms found in {path}")

    elements, frac_coords = zip(*[(a[0], a[1:]) for a in atoms])
    frac_coords = np.array(frac_coords, dtype=np.float32)
    cart_coords = frac_coords @ cell

    atomic_numbers = np.array(
        [ELEMENT_TO_Z.get(e, 0) for e in elements], dtype=np.int64
    )

    return cart_coords, atomic_numbers, cell


def parse_cif(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse CIF file, using ASE if available."""
    if HAS_ASE:
        return parse_cif_ase(path)
    return parse_cif_simple(path)


# =============================================================================
# Cache Manager
# =============================================================================


class CacheManager:
    """
    Manages NPZ caching for RADII dataset.

    Cache structure:
        cache/
        ├── samples.npz          # Concatenated pos/z arrays with offsets
        ├── unit_cells.npz       # Unit cell data per material
        └── metadata.json        # Sample metadata and index info
    """

    CACHE_VERSION = "1.0"

    def __init__(self, root: Path):
        self.root = root
        self.cache_dir = root / "cache"
        self.samples_path = self.cache_dir / "samples.npz"
        self.unit_cells_path = self.cache_dir / "unit_cells.npz"
        self.metadata_path = self.cache_dir / "metadata.json"

    def is_valid(self) -> bool:
        """Check if cache exists and is valid."""
        if not all(
            p.exists()
            for p in [self.samples_path, self.unit_cells_path, self.metadata_path]
        ):
            return False

        try:
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)

            # Check version
            if metadata.get("version") != self.CACHE_VERSION:
                return False

            # Check hash matches current directory structure
            current_hash = self._compute_directory_hash()
            if metadata.get("directory_hash") != current_hash:
                return False

            return True
        except Exception:
            return False

    def _compute_directory_hash(self) -> str:
        """Compute hash of directory structure for cache invalidation."""
        hasher = hashlib.md5()

        crystals_dir = self.root / "crystals"
        unit_cells_dir = self.root / "unit_cells"

        # Hash xyz file paths and mtimes
        if crystals_dir.exists():
            for xyz_file in sorted(crystals_dir.rglob("*.xyz")):
                hasher.update(str(xyz_file.relative_to(self.root)).encode())
                hasher.update(str(xyz_file.stat().st_mtime_ns).encode())

        # Hash cif file paths and mtimes
        if unit_cells_dir.exists():
            for cif_file in sorted(unit_cells_dir.glob("*.cif")):
                hasher.update(str(cif_file.relative_to(self.root)).encode())
                hasher.update(str(cif_file.stat().st_mtime_ns).encode())

        return hasher.hexdigest()

    def build_cache(self) -> Dict:
        """Build cache from raw files."""
        print("Building RADII cache (this may take a few minutes on first run)...")

        self.cache_dir.mkdir(exist_ok=True)

        crystals_dir = self.root / "crystals"
        unit_cells_dir = self.root / "unit_cells"

        # =========================
        # Parse all unit cells
        # =========================
        print("  Parsing unit cells...")
        unit_cell_data = {}
        materials = set()

        for cif_file in sorted(unit_cells_dir.glob("*.cif")):
            material = cif_file.stem
            materials.add(material)
            try:
                coords, atomic_numbers, cell_matrix = parse_cif(cif_file)
                unit_cell_data[material] = {
                    "pos": coords,
                    "z": atomic_numbers,
                    "cell_matrix": cell_matrix,
                }
            except Exception as e:
                warnings.warn(f"Failed to parse {cif_file}: {e}")

        # Save unit cells as NPZ
        unit_cells_arrays = {}
        for material, data in unit_cell_data.items():
            unit_cells_arrays[f"{material}_pos"] = data["pos"]
            unit_cells_arrays[f"{material}_z"] = data["z"]
            unit_cells_arrays[f"{material}_cell_matrix"] = data["cell_matrix"]

        np.savez_compressed(self.unit_cells_path, **unit_cells_arrays)

        # =========================
        # Parse all nanoparticles
        # =========================
        print("  Parsing nanoparticles...")

        all_pos = []
        all_z = []
        sample_metadata = []
        offsets = [0]

        for material_dir in sorted(crystals_dir.iterdir()):
            if not material_dir.is_dir():
                continue
            material = material_dir.name

            for r_dir in sorted(material_dir.iterdir()):
                if not r_dir.is_dir():
                    continue

                r_match = re.match(r"R(\d+)", r_dir.name)
                if not r_match:
                    continue
                radius = int(r_match.group(1))

                xyz_dir = r_dir / "xyz"
                if not xyz_dir.exists():
                    continue

                for xyz_file in sorted(xyz_dir.glob("rot_*.xyz")):
                    rot_match = re.match(r"rot_(\d+)\.xyz", xyz_file.name)
                    rot_idx = int(rot_match.group(1)) if rot_match else 0

                    try:
                        coords, atomic_numbers, meta = parse_xyz(xyz_file)

                        all_pos.append(coords)
                        all_z.append(atomic_numbers)
                        offsets.append(offsets[-1] + len(coords))

                        sample_metadata.append(
                            {
                                "material": material,
                                "radius": radius,
                                "rot_idx": rot_idx,
                                "split": meta.get("split", "train"),
                                "num_atoms": len(coords),
                            }
                        )
                    except Exception as e:
                        warnings.warn(f"Failed to parse {xyz_file}: {e}")

        # Concatenate arrays
        all_pos = (
            np.concatenate(all_pos, axis=0)
            if all_pos
            else np.zeros((0, 3), dtype=np.float32)
        )
        all_z = (
            np.concatenate(all_z, axis=0) if all_z else np.zeros((0,), dtype=np.int64)
        )
        offsets = np.array(offsets, dtype=np.int64)

        # Save samples NPZ
        np.savez_compressed(
            self.samples_path,
            pos=all_pos,
            z=all_z,
            offsets=offsets,
        )

        # Save metadata JSON
        metadata = {
            "version": self.CACHE_VERSION,
            "directory_hash": self._compute_directory_hash(),
            "num_samples": len(sample_metadata),
            "num_atoms_total": len(all_pos),
            "materials": sorted(list(materials)),
            "samples": sample_metadata,
        }

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"  Cache built: {len(sample_metadata)} samples, {len(all_pos)} total atoms"
        )

        return metadata

    def load_cache(self) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load cache from disk.

        Returns:
            metadata: dict with sample info
            pos: [N_total, 3] all positions concatenated
            z: [N_total] all atomic numbers concatenated
            offsets: [num_samples + 1] start/end indices
            unit_cells: dict of material -> (pos, z, cell_matrix)
        """
        # Load metadata
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        # Load samples
        samples_data = np.load(self.samples_path)
        pos = samples_data["pos"]
        z = samples_data["z"]
        offsets = samples_data["offsets"]

        # Load unit cells
        unit_cells_data = np.load(self.unit_cells_path)
        unit_cells = {}

        for material in metadata["materials"]:
            unit_cells[material] = {
                "pos": unit_cells_data[f"{material}_pos"],
                "z": unit_cells_data[f"{material}_z"],
                "cell_matrix": unit_cells_data[f"{material}_cell_matrix"],
            }

        return metadata, pos, z, offsets, unit_cells


# =============================================================================
# Dataset with Caching
# =============================================================================


class RADIIDataset(Dataset):
    """
    PyTorch Geometric dataset for RADII benchmark with NPZ caching.

    Args:
        root: Path to radii directory
        split: 'train', 'ID', 'OOD', or None for all
        transform: Optional transform to apply
        pre_filter: Optional filter function
        use_cache: Whether to use/build NPZ cache (default: True)
    """

    def __init__(
        self,
        root: str | Path,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        use_cache: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.pre_filter = pre_filter
        self.use_cache = use_cache

        self.crystals_dir = self.root / "crystals"
        self.unit_cells_dir = self.root / "unit_cells"

        # Initialize cache
        self.cache_manager = CacheManager(self.root)

        if use_cache:
            self._load_with_cache()
        else:
            self._load_without_cache()

        # Apply split filter
        if self.split is not None:
            self.indices = [
                i
                for i, s in enumerate(self._sample_metadata)
                if s["split"] == self.split
            ]
        else:
            self.indices = list(range(len(self._sample_metadata)))

        # Apply pre_filter
        if pre_filter is not None:
            self.indices = [
                i for i in self.indices if pre_filter(self._sample_metadata[i])
            ]

    def _load_with_cache(self):
        """Load data using cache (build if necessary)."""
        if not self.cache_manager.is_valid():
            self.cache_manager.build_cache()

        metadata, pos, z, offsets, unit_cells = self.cache_manager.load_cache()

        self._pos = pos
        self._z = z
        self._offsets = offsets
        self._unit_cells = unit_cells
        self._sample_metadata = metadata["samples"]

    def _load_without_cache(self):
        """Load data without cache (original slow method)."""
        # This is a fallback - uses the old discovery method
        self._unit_cell_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        samples = self._discover_samples()

        # Convert to cached format
        all_pos = []
        all_z = []
        offsets = [0]

        for sample in samples:
            coords, atomic_numbers, _ = parse_xyz(sample["xyz_path"])
            all_pos.append(coords)
            all_z.append(atomic_numbers)
            offsets.append(offsets[-1] + len(coords))

        self._pos = (
            np.concatenate(all_pos, axis=0)
            if all_pos
            else np.zeros((0, 3), dtype=np.float32)
        )
        self._z = (
            np.concatenate(all_z, axis=0) if all_z else np.zeros((0,), dtype=np.int64)
        )
        self._offsets = np.array(offsets, dtype=np.int64)
        self._sample_metadata = [
            {
                "material": s["material"],
                "radius": s["radius"],
                "rot_idx": s["rot_idx"],
                "split": s["split"],
                "num_atoms": self._offsets[i + 1] - self._offsets[i],
            }
            for i, s in enumerate(samples)
        ]

        # Load unit cells
        self._unit_cells = {}
        materials = set(s["material"] for s in samples)
        for material in materials:
            cif_path = self.unit_cells_dir / f"{material}.cif"
            if cif_path.exists():
                coords, atomic_numbers, cell_matrix = parse_cif(cif_path)
                self._unit_cells[material] = {
                    "pos": coords,
                    "z": atomic_numbers,
                    "cell_matrix": cell_matrix,
                }

    def _discover_samples(self) -> List[Dict]:
        """Find all xyz files and extract metadata (used when cache disabled)."""
        samples = []

        for material_dir in sorted(self.crystals_dir.iterdir()):
            if not material_dir.is_dir():
                continue
            material = material_dir.name

            for r_dir in sorted(material_dir.iterdir()):
                if not r_dir.is_dir():
                    continue

                r_match = re.match(r"R(\d+)", r_dir.name)
                if not r_match:
                    continue
                radius = int(r_match.group(1))

                xyz_dir = r_dir / "xyz"
                if not xyz_dir.exists():
                    continue

                for xyz_file in sorted(xyz_dir.glob("rot_*.xyz")):
                    rot_match = re.match(r"rot_(\d+)\.xyz", xyz_file.name)
                    rot_idx = int(rot_match.group(1)) if rot_match else 0

                    sample = {
                        "xyz_path": xyz_file,
                        "material": material,
                        "radius": radius,
                        "rot_idx": rot_idx,
                    }

                    try:
                        _, _, meta = parse_xyz(xyz_file)
                        sample["split"] = meta.get("split", "train")
                    except Exception:
                        sample["split"] = "train"

                    samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        real_idx = self.indices[idx]
        sample = self._sample_metadata[real_idx]

        # Get nanoparticle data from concatenated arrays
        start = self._offsets[real_idx]
        end = self._offsets[real_idx + 1]

        coords = self._pos[start:end]
        atomic_numbers = self._z[start:end]

        # Get unit cell data
        material = sample["material"]
        cell_data = self._unit_cells[material]

        # Create PyG Data object
        data = Data(
            # Nanoparticle
            pos=torch.from_numpy(coords.copy()),
            z=torch.from_numpy(
                atomic_numbers.copy()
            ).flatten(),  # Ensure 1D (flatten avoids scalar for single-atom)
            # Unit cell
            cell_pos=torch.from_numpy(cell_data["pos"].copy()),
            cell_z=torch.from_numpy(
                cell_data["z"].copy()
            ).flatten(),  # Ensure 1D (flatten avoids scalar for single-atom)
            cell_matrix=torch.from_numpy(cell_data["cell_matrix"].copy()),
            # Metadata
            material=material,
            radius=torch.tensor(sample["radius"], dtype=torch.float),
            split=sample["split"],
            rot_idx=sample["rot_idx"],
            # Convenience
            num_atoms=torch.tensor(sample["num_atoms"], dtype=torch.long),
        )

        if self.transform is not None:
            data = self.transform(data)

        return data

    @property
    def samples(self):
        """Return metadata for indexed samples."""
        return [self._sample_metadata[i] for i in self.indices]

    def get_split(self, split: str) -> "RADIIDataset":
        """Return a new dataset filtered to a specific split."""
        return RADIIDataset(
            self.root,
            split=split,
            transform=self.transform,
            pre_filter=self.pre_filter,
            use_cache=self.use_cache,
        )

    def random_split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Tuple["SubsetRADIIDataset", "SubsetRADIIDataset"]:
        """Randomly split dataset into train and validation."""
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.indices))

        n_train = int(len(indices) * train_ratio)
        train_idx = [self.indices[i] for i in indices[:n_train]]
        val_idx = [self.indices[i] for i in indices[n_train:]]

        train_ds = SubsetRADIIDataset(self, train_idx)
        val_ds = SubsetRADIIDataset(self, val_idx)
        return train_ds, val_ds


class SubsetRADIIDataset(Dataset):
    """Subset wrapper for RADIIDataset."""

    def __init__(self, dataset: RADIIDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
        self.transform = dataset.transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        real_idx = self.indices[idx]
        sample = self.dataset._sample_metadata[real_idx]

        start = self.dataset._offsets[real_idx]
        end = self.dataset._offsets[real_idx + 1]

        coords = self.dataset._pos[start:end]
        atomic_numbers = self.dataset._z[start:end]

        material = sample["material"]
        cell_data = self.dataset._unit_cells[material]

        data = Data(
            pos=torch.from_numpy(coords.copy()),
            z=torch.from_numpy(atomic_numbers.copy()).flatten(),  # Ensure 1D
            cell_pos=torch.from_numpy(cell_data["pos"].copy()),
            cell_z=torch.from_numpy(cell_data["z"].copy()).flatten(),  # Ensure 1D
            cell_matrix=torch.from_numpy(cell_data["cell_matrix"].copy()),
            material=material,
            radius=torch.tensor(sample["radius"], dtype=torch.float),
            split=sample["split"],
            rot_idx=sample["rot_idx"],
            num_atoms=torch.tensor(sample["num_atoms"], dtype=torch.long),
        )

        if self.transform is not None:
            data = self.transform(data)

        return data

    @property
    def samples(self):
        return [self.dataset._sample_metadata[i] for i in self.indices]


# =============================================================================
# Convenience loader
# =============================================================================


class RADIIDataloader:
    """
    Wrapper matching existing RADIIDataloader API with caching support.

    Args:
        root: Path to radii directory
        num_workers: Number of workers for DataLoader
        transform: Optional transform to apply to samples
        use_cache: Whether to use NPZ cache (default True)
        loaded_frac: Fraction of rotations to load (0 < loaded_frac <= 1.0).
                     E.g., 0.5 loads half the rotations per material/radius.

    Usage:
        ds = RADIIDataloader(root="radii", num_workers=4, transform=my_transform)
        train_ds, val_ds = ds.random_id_splits(0.8, seed=42)
        id_test_ds = ds.get_split("id_test")
        ood_test_ds = ds.get_split("ood_test")
    """

    def __init__(
        self,
        root: str | Path,
        num_workers: int = 4,
        transform: Optional[Callable] = None,
        use_cache: bool = True,
        loaded_frac: float = 0.5,
    ):
        self.root = Path(root)
        self.num_workers = num_workers
        self.transform = transform
        self.use_cache = use_cache
        self.loaded_frac = loaded_frac

        assert 0 < loaded_frac <= 1.0, "loaded_frac must be in (0, 1]"

        # Build pre_filter for loaded_frac (sample rotations per material/radius)
        pre_filter = None
        if loaded_frac < 1.0:
            pre_filter = self._make_rotation_filter(loaded_frac)

        # Load full dataset (this triggers cache build if needed)
        self._full_dataset = RADIIDataset(
            root,
            split=None,
            transform=transform,
            use_cache=use_cache,
            pre_filter=pre_filter,
        )

        # Create split views (these share the same cached data)
        self._train_dataset = RADIIDataset(
            root,
            split="train",
            transform=transform,
            use_cache=use_cache,
            pre_filter=pre_filter,
        )
        self._id_dataset = RADIIDataset(
            root,
            split="ID",
            transform=transform,
            use_cache=use_cache,
            pre_filter=pre_filter,
        )
        self._ood_dataset = RADIIDataset(
            root,
            split="OOD",
            transform=transform,
            use_cache=use_cache,
            pre_filter=pre_filter,
        )

    def _make_rotation_filter(self, loaded_frac: float) -> Callable:
        """
        Create a filter that keeps only a fraction of rotations per (material, radius).

        Strategy: For each (material, radius) group, keep rotations 0, 1, ..., floor(N * loaded_frac) - 1
        where N is the total number of rotations for that group.
        """
        # First pass: discover all samples to find max rot_idx per (material, radius)
        # We need to know the structure before filtering, so we'll build a lookup
        # This requires reading the cache or discovering samples
        # For simplicity, we'll make the filter stateful: it tracks seen (material, radius)
        # and counts rotations, keeping only the first loaded_frac fraction

        from collections import defaultdict

        max_rot_per_group = defaultdict(
            lambda: None
        )  # (material, radius) -> max_rot_idx

        # We need to know max rotations per group. Since we're building the filter before
        # loading the dataset, we'll do a quick scan of the cache or discovery
        # Actually, simpler approach: the filter will be called on each sample dict
        # We can track (material, radius) -> count and max_count, then accept if rot_idx < max_count * loaded_frac
        # But we don't know max_count in advance. So we need two passes or we need to scan first.

        # Simplest: scan the cache/samples first to find max rot_idx per (material, radius)
        # Then filter keeps rot_idx <= floor(max_rot_idx * loaded_frac)

        cache_path = self.root / "radii_cache.npz"
        if cache_path.exists():
            # Load from cache to get all samples
            data = np.load(cache_path, allow_pickle=True)
            materials = data["material_names"]
            material_idx = data["material_idx"]
            radius = data["radius"]
            rot_idx = data["rot_idx"]

            # Find max rot_idx per (material, radius)
            for i in range(len(material_idx)):
                mat = str(materials[material_idx[i]])
                rad = int(radius[i])
                rot = int(rot_idx[i])
                key = (mat, rad)
                if max_rot_per_group[key] is None or rot > max_rot_per_group[key]:
                    max_rot_per_group[key] = rot
            data.close()
        else:
            # Discovery from disk (slower but works if no cache yet)
            crystals_dir = self.root / "crystals"
            for material_dir in sorted(crystals_dir.iterdir()):
                if not material_dir.is_dir():
                    continue
                material = material_dir.name
                for r_dir in sorted(material_dir.iterdir()):
                    if not r_dir.is_dir():
                        continue
                    r_match = re.match(r"R(\d+)", r_dir.name)
                    if not r_match:
                        continue
                    rad = int(r_match.group(1))
                    xyz_dir = r_dir / "xyz"
                    if not xyz_dir.exists():
                        continue
                    max_rot = -1
                    for xyz_file in xyz_dir.glob("rot_*.xyz"):
                        rot_match = re.match(r"rot_(\d+)\.xyz", xyz_file.name)
                        if rot_match:
                            rot = int(rot_match.group(1))
                            if rot > max_rot:
                                max_rot = rot
                    if max_rot >= 0:
                        max_rot_per_group[(material, rad)] = max_rot

        # Now build the filter: keep rot_idx <= floor(max_rot * loaded_frac)
        def rotation_filter(sample: Dict) -> bool:
            mat = sample["material"]
            rad = sample["radius"]
            rot = sample["rot_idx"]
            key = (mat, rad)
            max_rot = max_rot_per_group.get(key, 0)
            # Keep rotations 0, 1, ..., floor(max_rot * loaded_frac)
            # E.g., if max_rot=9 (rotations 0-9, total 10) and loaded_frac=0.5, keep 0-4 (5 rotations)
            threshold = int(max_rot * loaded_frac)
            return rot <= threshold

        return rotation_filter

    def get_split(self, split: str) -> RADIIDataset:
        """Get predefined split: 'train', 'ID', or 'OOD'."""
        if split.lower() == "train":
            return self._train_dataset
        elif split.lower() in ("id", "id_test"):
            return self._id_dataset
        elif split.lower() in ("ood", "ood_test"):
            return self._ood_dataset
        else:
            raise ValueError(f"Unknown split: {split}")

    def random_id_splits(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Tuple[SubsetRADIIDataset, SubsetRADIIDataset]:
        """Randomly split the predefined TRAIN set into train and val."""
        return self._train_dataset.random_split(train_ratio, seed)

    def __len__(self) -> int:
        return len(self._full_dataset)

    def clear_cache(self):
        """Delete cached files to force rebuild on next load."""
        import shutil

        cache_dir = self.root / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cache cleared: {cache_dir}")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys
    import time

    root = sys.argv[1] if len(sys.argv) > 1 else "radii"

    print(f"Loading RADII dataset from: {root}")
    print("=" * 60)

    # First load (will build cache if needed)
    t0 = time.time()
    ds = RADIIDataloader(root=root, use_cache=True)
    t1 = time.time()
    print(f"\nFirst load time: {t1 - t0:.2f}s")

    print(f"\nTotal samples: {len(ds)}")
    print(f"Train split: {len(ds.get_split('train'))}")
    print(f"ID split: {len(ds.get_split('ID'))}")
    print(f"OOD split: {len(ds.get_split('OOD'))}")

    # Second load (should be fast from cache)
    t0 = time.time()
    ds2 = RADIIDataloader(root=root, use_cache=True)
    t1 = time.time()
    print(f"\nSecond load time (from cache): {t1 - t0:.2f}s")

    # Test random splits
    train_ds, val_ds = ds.random_id_splits(0.8, seed=42)
    print("\nRandom splits of train set (80/20):")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val: {len(val_ds)}")

    # Load a sample
    print("\nSample data:")
    t0 = time.time()
    sample = train_ds[0]
    t1 = time.time()
    print(f"  Load time: {(t1 - t0) * 1000:.2f}ms")
    print(f"  Material: {sample.material}")
    print(f"  Radius: {sample.radius.item()}")
    print(f"  Split: {sample.split}")
    print(f"  Num atoms: {sample.num_atoms.item()}")
    print(f"  Pos shape: {sample.pos.shape}")
    print(f"  Z shape: {sample.z.shape}")
    print(f"  Cell pos shape: {sample.cell_pos.shape}")
    print(f"  Cell matrix shape: {sample.cell_matrix.shape}")

    # Benchmark batch loading
    print("\nBenchmark: Loading 100 samples...")
    t0 = time.time()
    for i in range(min(100, len(train_ds))):
        _ = train_ds[i]
    t1 = time.time()
    print(f"  Total time: {t1 - t0:.2f}s")
    print(f"  Per sample: {(t1 - t0) / min(100, len(train_ds)) * 1000:.2f}ms")
