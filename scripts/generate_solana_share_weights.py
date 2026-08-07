#!/usr/bin/env python3
"""Generate identity-free Aptos-style weights for Solana validator stakes.

For every requested symmetric error around a target stake threshold, this
script selects the nominal resolution M using Aptos' DKG rounding rule, rounds
each validator's stake to an integer weight, and writes only positive weights.

The JSON uses ``share_count`` for W, the actual total integer weight consumed by
the cryptographic scheme.  ``selected_resolution_m`` is also included because
Aptos independently rounds each validator and therefore does not force W == M.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from solana_share_approximation import (
    DEFAULT_SYMMETRIC_ERRORS,
    DEFAULT_TARGET_RECONSTRUCTION_RATIO,
    ErrorAllocation,
    allocate_for_errors,
    fixed_to_decimal_string,
    format_ratio,
    load_validators,
    normalize_errors,
    parse_ratio,
)
from solana_script_utils import (
    SHARE_WEIGHTS_STEM,
    VALIDATOR_DISTRIBUTION_STEM,
    latest_data_path,
    timestamped_data_path,
)


def allocation_to_json(result: ErrorAllocation) -> dict[str, Any]:
    """Convert one selected profile to the public identity-free JSON shape."""
    profile = result.profile
    positive_weights = [
        allocation.weight
        for allocation in result.allocations
        if allocation.weight > 0
    ]

    if sum(positive_weights) != profile.total_weight:
        raise AssertionError("positive weights do not sum to total_weight")

    return {
        "error": format_ratio(result.error),
        "lower_stake_ratio": format_ratio(result.lower_ratio),
        "upper_stake_ratio": format_ratio(result.upper_ratio),
        "selected_resolution_m": profile.selected_m,
        # W: the actual weight total passed to the cryptographic scheme.
        "share_count": profile.total_weight,
        "reconstruction_threshold": profile.reconstruction_threshold,
        "effective_reconstruction_stake_ratio": fixed_to_decimal_string(
            profile.effective_reconstruction_ratio_raw
        ),
        "positive_validator_count": len(positive_weights),
        "weights": positive_weights,
    }


def write_weights_json(
    results: Sequence[ErrorAllocation],
    output: Path,
) -> None:
    """Write all selected profiles and positive identity-free weights."""
    if not results:
        raise ValueError("at least one allocation result is required")

    target_ratio = results[0].target_ratio
    if any(result.target_ratio != target_ratio for result in results):
        raise ValueError("all allocation results must use the same target ratio")

    document = {
        "method": "aptos_dkg_nearest_rounding_q64_64",
        "target_reconstruction_ratio": format_ratio(target_ratio),
        "allocations": [allocation_to_json(result) for result in results],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select Aptos-style weight resolutions for symmetric stake "
            "errors and write positive identity-free weights as JSON."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help=(
            "JSON validator dump containing account and activeStake fields "
            "(default: newest timestamped distribution in scripts/data)"
        ),
    )
    parser.add_argument(
        "--target",
        type=parse_ratio,
        default=parse_ratio(DEFAULT_TARGET_RECONSTRUCTION_RATIO),
        metavar="RATIO",
        help=(
            "center of the symmetric stake interval "
            f"(default: {DEFAULT_TARGET_RECONSTRUCTION_RATIO})"
        ),
    )
    parser.add_argument(
        "--errors",
        nargs="+",
        default=list(DEFAULT_SYMMETRIC_ERRORS),
        metavar="E",
        help=(
            "symmetric errors to generate "
            "(default: 1/4 1/8 1/16 1/32 1/64 1/128)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "allocation JSON path (default: a UTC-timestamped file in "
            "scripts/data)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input or latest_data_path(
        VALIDATOR_DISTRIBUTION_STEM,
        ".json",
    )
    output = args.output or timestamped_data_path(SHARE_WEIGHTS_STEM, ".json")

    try:
        validators = load_validators(input_path)
        errors = normalize_errors(args.errors)
        results = allocate_for_errors(validators, args.target, errors)
        write_weights_json(results, output)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for result in results:
        profile = result.profile
        positive_count = sum(weight > 0 for weight in profile.validator_weights)
        print(
            f"e={format_ratio(result.error)}: "
            f"M={profile.selected_m:,}, W={profile.total_weight:,}, "
            f"q={profile.reconstruction_threshold:,}, "
            f"positive validators={positive_count:,}",
            file=sys.stderr,
        )
    print(f"wrote share weights to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
