#!/usr/bin/env python3
"""Generate identity-free integer weights for Solana validator stakes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from solana_share_approximation import (
    DEFAULT_INPUT,
    DEFAULT_SHARE_COUNTS,
    hamilton_allocate,
    load_validators,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "solana_share_weights.json"


def write_weights_json(
    validators: Sequence[dict[str, Any]],
    share_counts: Sequence[int],
    output: Path,
) -> None:
    """Write positive allocations keyed by requested total weight."""
    weights = {
        str(share_count): [
            allocation.shares
            for allocation in hamilton_allocate(validators, share_count)
            if allocation.shares > 0
        ]
        for share_count in share_counts
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(weights, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Allocate integer weights from a Solana validator distribution "
            "and write them as identity-free JSON."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=(
            "JSON validator dump containing account and activeStake fields "
            f"(default: {DEFAULT_INPUT})"
        ),
    )
    parser.add_argument(
        "--shares",
        type=int,
        nargs="+",
        default=DEFAULT_SHARE_COUNTS,
        metavar="N",
        help="total weights to generate (default: 2048 4096 8192 16384 32768)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"allocation JSON path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validators = load_validators(args.input)
    share_counts = list(dict.fromkeys(args.shares))
    write_weights_json(validators, share_counts, args.output)


if __name__ == "__main__":
    main()
