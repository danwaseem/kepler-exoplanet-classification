"""Read-only access to the raw KOI table and its checksum."""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from exoplanet.config import RAW_CSV, SOURCE_CSV


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_raw_copy() -> dict:
    """Copy the supplied workspace CSV into data/raw without modifying it.

    The cumulative table is never downloaded. If both files already exist,
    their checksums must match. The original file is never opened for writing.
    """
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE_CSV.exists() and not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw KOI table not found at {SOURCE_CSV}")
    if SOURCE_CSV.exists() and not RAW_CSV.exists():
        shutil.copyfile(SOURCE_CSV, RAW_CSV)
    if SOURCE_CSV.exists() and RAW_CSV.exists():
        source_hash = file_sha256(SOURCE_CSV)
        raw_hash = file_sha256(RAW_CSV)
        if source_hash != raw_hash:
            raise RuntimeError(
                "data/raw CSV does not match the original workspace CSV. "
                "The raw file must stay unchanged."
            )
    else:
        source_hash = file_sha256(RAW_CSV)
        raw_hash = source_hash
    stamped = SOURCE_CSV if SOURCE_CSV.exists() else RAW_CSV
    modified = datetime.fromtimestamp(stamped.stat().st_mtime).astimezone().isoformat(
        timespec="seconds"
    )
    return {
        "sha256": raw_hash,
        "bytes": RAW_CSV.stat().st_size,
        "local_mtime": modified,
        "canonical_path": str(RAW_CSV.relative_to(RAW_CSV.parents[2])),
    }


def load_koi(path: Path | None = None) -> pd.DataFrame:
    """Load the cumulative table. Empty fields become missing values."""
    frame = pd.read_csv(path or RAW_CSV, low_memory=False)
    return frame
