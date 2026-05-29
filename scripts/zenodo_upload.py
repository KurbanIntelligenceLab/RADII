"""Upload radii_raw.zip to Zenodo as a draft deposition with a reserved DOI.

Reads ZENODO_TOKEN from /Users/jp/RADII/.env. Creates a new draft on
zenodo.org (production), reserves a DOI, uploads the cleaned seed zip via
the bucket API, sets full metadata (CC-BY-4.0, author list, KDD'26
description), and stops. The user reviews in the Zenodo web UI and clicks
Publish on camera-ready submission day.

Usage (from repo root):
    python scripts/zenodo_upload.py            # full upload, stops at draft
    python scripts/zenodo_upload.py --dry-run  # verify token only, no draft
    python scripts/zenodo_upload.py --community KDD2026   # opt into a community
    python scripts/zenodo_upload.py --draft-id 1234567    # resume an existing draft

Reference: https://developers.zenodo.org/
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
DEFAULT_ZIP = REPO_ROOT / "radii_raw.zip"
ZENODO_BASE = "https://zenodo.org/api"

TITLE = (
    "RADII: Radius-Resolved Benchmark of Nanoparticle Structures "
    "for Generative Models in Materials Science"
)

DESCRIPTION_HTML = """
<p><strong>RADII</strong> is a radius-resolved benchmark of ~75,000
nanoparticle structures (33–11,298 atoms across ten materials) designed
to measure the <em>extrapolation frontier</em> of graph generative models
for materials science. Each structure is a spherical truncation of a
published crystal lattice; the benchmark treats radius as a continuous
scaling knob and provides leakage-free in-distribution and
out-of-distribution splits.</p>

<p>This archive contains the <strong>seed inputs</strong> for the
benchmark: the ten primitive unit cell CIFs and per-(material, radius)
base XYZ structures. The full ~12 GB benchmark (with rotation
augmentation) is reproduced locally by running the generation pipeline
shipped in the GitHub repository.</p>

<p><strong>Paper:</strong> "How Far Can You Grow? Characterizing the
Extrapolation Frontier of Graph Generative Models for Materials Science"
(KDD '26).</p>

<p><strong>Code &amp; generation pipeline:</strong>
<a href="https://github.com/KurbanIntelligenceLab/RADII">github.com/KurbanIntelligenceLab/RADII</a>
(MIT license). To reproduce the full benchmark from this archive:
<code>python -m create_radii.create_radii --raw-data radii_raw.zip --output radii</code>.</p>

<p><strong>Materials included:</strong> Ag, Au, CH₃NH₃PbI₃, Fe₂O₃, MoS₂,
PbS, SnO₂, SrTiO₃, TiO₂, ZnO.</p>
"""

CREATORS = [
    {"name": "Polat, Can", "affiliation": "Texas A&M University"},
    {"name": "Serpedin, Erchin", "affiliation": "Texas A&M University"},
    {
        "name": "Kurban, Mustafa",
        "affiliation": "Ankara University; Texas A&M University at Qatar",
    },
    {
        "name": "Kurban, Hasan",
        "affiliation": "Hamad Bin Khalifa University",
    },
]

KEYWORDS = [
    "KDD",
    "KDD 2026",
    "generative models",
    "materials science",
    "crystal structures",
    "nanoparticles",
    "benchmark",
    "out-of-distribution",
    "extrapolation",
    "graph neural networks",
]

RELATED = [
    {
        "identifier": "https://github.com/KurbanIntelligenceLab/RADII",
        "relation": "isSupplementTo",
        "resource_type": "software",
    }
]


def load_token() -> str:
    load_dotenv(ENV_PATH)
    tok = os.environ.get("ZENODO_TOKEN")
    if not tok:
        sys.exit(
            f"ERROR: ZENODO_TOKEN not found. Expected in {ENV_PATH} as "
            "`ZENODO_TOKEN=<your token>`."
        )
    return tok


def verify_clean_zip(zip_path: Path) -> None:
    if not zip_path.exists():
        sys.exit(f"ERROR: {zip_path} not found.")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    junk = [n for n in names if "__MACOSX" in n or n.endswith(".DS_Store")]
    if junk:
        sys.exit(
            f"ERROR: {zip_path} still contains {len(junk)} macOS junk entries. "
            f"Run: zip -d radii_raw.zip '__MACOSX/*' '*/.DS_Store'"
        )
    print(f"  zip ok: {len(names)} entries, {zip_path.stat().st_size / 1e6:.2f} MB")


def make_metadata(community: str | None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "title": TITLE,
        "upload_type": "dataset",
        "description": DESCRIPTION_HTML.strip(),
        "creators": CREATORS,
        "license": "cc-by-4.0",
        "keywords": KEYWORDS,
        "related_identifiers": RELATED,
        "access_right": "open",
        "prereserve_doi": True,
    }
    if community:
        meta["communities"] = [{"identifier": community}]
    return meta


def dry_run(token: str) -> None:
    """Hit a read endpoint to confirm the token works."""
    print("== Dry run: verifying ZENODO_TOKEN ==")
    r = requests.get(
        f"{ZENODO_BASE}/deposit/depositions",
        params={"access_token": token, "size": 1},
        timeout=30,
    )
    if r.status_code == 200:
        print(f"  ok: GET /deposit/depositions returned {r.status_code}, "
              f"existing drafts visible to this token: {len(r.json())}")
    elif r.status_code == 401:
        sys.exit(f"  FAIL: 401 Unauthorized. Token is invalid or expired.")
    elif r.status_code == 403:
        sys.exit(
            f"  FAIL: 403 Forbidden. Token lacks `deposit:write` scope. "
            f"Mint a new token at https://zenodo.org/account/settings/applications/ "
            f"with deposit:write (and deposit:actions if you want auto-publish later)."
        )
    else:
        sys.exit(f"  FAIL: unexpected status {r.status_code}: {r.text[:400]}")


def create_draft(token: str) -> dict[str, Any]:
    print("== Creating draft on zenodo.org ==")
    r = requests.post(
        f"{ZENODO_BASE}/deposit/depositions",
        params={"access_token": token},
        json={},
        timeout=30,
    )
    if r.status_code != 201:
        sys.exit(f"  FAIL: create returned {r.status_code}: {r.text[:400]}")
    dep = r.json()
    print(f"  created deposition id={dep['id']}")
    return dep


def get_draft(token: str, dep_id: int) -> dict[str, Any]:
    r = requests.get(
        f"{ZENODO_BASE}/deposit/depositions/{dep_id}",
        params={"access_token": token},
        timeout=30,
    )
    if r.status_code != 200:
        sys.exit(f"  FAIL: fetching draft {dep_id}: {r.status_code} {r.text[:400]}")
    return r.json()


def upload_file(token: str, bucket_url: str, zip_path: Path) -> None:
    print(f"== Uploading {zip_path.name} to bucket ==")
    size = zip_path.stat().st_size
    with zip_path.open("rb") as fp:
        r = requests.put(
            f"{bucket_url}/{zip_path.name}",
            params={"access_token": token},
            data=fp,
            headers={"Content-Type": "application/octet-stream"},
            timeout=300,
        )
    if r.status_code not in (200, 201):
        sys.exit(f"  FAIL: upload returned {r.status_code}: {r.text[:400]}")
    print(f"  uploaded {size / 1e6:.2f} MB; checksum {r.json().get('checksum')}")


def set_metadata(token: str, dep_id: int, metadata: dict[str, Any]) -> dict[str, Any]:
    print("== Setting metadata + reserving DOI ==")
    r = requests.put(
        f"{ZENODO_BASE}/deposit/depositions/{dep_id}",
        params={"access_token": token},
        json={"metadata": metadata},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if r.status_code != 200:
        sys.exit(f"  FAIL: set metadata returned {r.status_code}: {r.text[:400]}")
    return r.json()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true", help="Verify token only")
    p.add_argument("--draft-id", type=int, help="Resume an existing draft id")
    p.add_argument("--community", help="Zenodo community identifier")
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="Path to seed zip")
    args = p.parse_args()

    token = load_token()

    if args.dry_run:
        dry_run(token)
        return 0

    verify_clean_zip(args.zip)

    if args.draft_id:
        print(f"== Resuming draft id={args.draft_id} ==")
        dep = get_draft(token, args.draft_id)
    else:
        dep = create_draft(token)

    bucket_url = dep["links"]["bucket"]
    upload_file(token, bucket_url, args.zip)

    metadata = make_metadata(args.community)
    dep = set_metadata(token, dep["id"], metadata)

    reserved = dep["metadata"].get("prereserve_doi", {})
    doi = reserved.get("doi", "<not returned>")
    record_id = reserved.get("recid", dep["id"])
    draft_url = dep["links"].get("html") or f"https://zenodo.org/deposit/{dep['id']}"

    print()
    print("=" * 70)
    print("DRAFT CREATED. NOT YET PUBLISHED.")
    print("=" * 70)
    print(f"  Deposition ID : {dep['id']}")
    print(f"  Reserved DOI  : {doi}")
    print(f"  Record ID     : {record_id}")
    print(f"  Draft URL     : {draft_url}")
    print(f"  License       : cc-by-4.0")
    print(f"  Files         : {[f['filename'] for f in dep.get('files', [])]}")
    print()
    print("NEXT STEPS:")
    print(f"  1. Open the draft URL and review the metadata + file.")
    print(f"  2. Update the paper's abstract with DOI: {doi}")
    print(f"     (replace zenodo.XXXXXXX in paper/sample-sigconf.tex line 124)")
    print(f"  3. When ready for camera-ready submission, click 'Publish'")
    print(f"     in the Zenodo web UI to make the DOI live.")
    print()
    print("DO NOT publish until you are sure the metadata is final. A")
    print("published record cannot be deleted, only updated as new versions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
