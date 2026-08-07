"""Shared paths and timestamp helpers for the Solana scripts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
VALIDATOR_DISTRIBUTION_STEM = "solana_validator_distribution"
SHARE_WEIGHTS_STEM = "solana_share_weights"
APPROXIMATION_REPORT_STEM = "solana_share_approximation"


def utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp with one-second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def timestamped_data_path(stem: str, suffix: str) -> Path:
    """Build a timestamped path beneath the scripts data directory."""
    return DATA_DIR / f"{stem}_{utc_timestamp()}{suffix}"


def latest_data_path(stem: str, suffix: str) -> Path:
    """Find the newest timestamped data file for a given output type."""
    candidates = sorted(DATA_DIR.glob(f"{stem}_????-??-??T??-??-??Z{suffix}"))
    if not candidates:
        raise FileNotFoundError(
            f"no timestamped {stem} file found in {DATA_DIR}; "
            "run fetch_solana_validator_distribution.py first"
        )
    return candidates[-1]
