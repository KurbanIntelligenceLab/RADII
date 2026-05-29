"""Fetch the RADII seed archive from Zenodo and verify its checksum.

Cache layout (XDG-style):
    ~/.cache/radii/
    ├── radii_raw.zip           # downloaded seed (~3 MB)
    └── radii/                  # generated benchmark (~12 GB) — managed by
                                # radii.generation.create_radii(), not this module

Public API:
    download_seed(dest=None) -> Path
        Returns the path to the verified zip. Idempotent: if the cache already
        holds a zip with the expected checksum, returns immediately.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

ZENODO_RECORD_ID = "20431021"
ZENODO_FILE = "radii_raw.zip"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ZENODO_DOWNLOAD = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{ZENODO_FILE}/content"

# SHA256 of the published radii_raw.zip on Zenodo. Computed at download time
# against the Zenodo record's `files[0].checksum` (format "md5:..." per Zenodo's
# legacy schema, or "sha256:..." for newer records). If the published file
# changes (versioned record), update this string.
EXPECTED_SHA256: Optional[str] = None  # populated lazily from Zenodo metadata


def _cache_dir() -> Path:
    base = os.environ.get("RADII_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "radii"
    return Path.home() / ".cache" / "radii"


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _fetch_expected_checksum() -> tuple[str, str]:
    """Get (algorithm, hex) tuple from Zenodo metadata. Returns ('sha256', ...) or ('md5', ...)."""
    import requests

    r = requests.get(ZENODO_API, timeout=30)
    r.raise_for_status()
    record = r.json()
    files = record.get("files", [])
    if not files:
        raise RuntimeError(f"Zenodo record {ZENODO_RECORD_ID} has no files attached")
    # Pick the radii_raw.zip entry
    target = next((f for f in files if f.get("key") == ZENODO_FILE), files[0])
    checksum = target.get("checksum", "")
    if ":" not in checksum:
        raise RuntimeError(f"Unexpected checksum format from Zenodo: {checksum!r}")
    algo, _, hex_ = checksum.partition(":")
    return algo, hex_


def _verify(path: Path) -> None:
    """Verify the downloaded file against Zenodo's published checksum."""
    algo, expected = _fetch_expected_checksum()
    if algo == "sha256":
        actual = _sha256(path)
    elif algo == "md5":
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
    else:
        raise RuntimeError(f"Unsupported checksum algorithm: {algo!r}")
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: expected {algo}:{expected}, "
            f"got {algo}:{actual}. File may be corrupted; delete and re-download."
        )


def download_seed(dest: Path | str | None = None, *, force: bool = False) -> Path:
    """Download radii_raw.zip from Zenodo and return its path.

    Args:
        dest: Directory to cache the zip in. Defaults to ~/.cache/radii/.
        force: If True, re-download even if a cached zip exists.

    Returns:
        Path to the verified radii_raw.zip.
    """
    import requests

    cache = Path(dest) if dest else _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / ZENODO_FILE

    if out.exists() and not force:
        try:
            _verify(out)
            return out
        except Exception as e:
            print(f"Cached {out.name} failed verification ({e}); re-downloading.", file=sys.stderr)

    print(f"Fetching {ZENODO_FILE} from Zenodo record {ZENODO_RECORD_ID}...", file=sys.stderr)
    with requests.get(ZENODO_DOWNLOAD, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = out.with_suffix(out.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        tmp.rename(out)
    _verify(out)
    print(f"  saved to {out} ({out.stat().st_size / 1e6:.2f} MB, verified)", file=sys.stderr)
    return out


__all__ = ["download_seed"]
