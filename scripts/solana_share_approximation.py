#!/usr/bin/env python3
"""Approximate Solana validator stakes with Aptos-style integer weights.

The input must be a JSON array whose records contain at least:

    {"account": "...", "activeStake": 123456789}

For each requested symmetric error ``e`` around a target stake threshold ``tau``,
the script defines:

    secrecy/lower threshold      alpha = tau - e
    reconstruction upper bound  beta  = tau + e

It then applies Aptos-style DKG rounding:

* search a bounded range of nominal resolutions M;
* set stake_per_weight = total_stake / M;
* independently round each validator's ideal weight to the nearest integer,
  with half ties rounded upward;
* let W be the sum of those rounded weights;
* derive one integer reconstruction threshold q from alpha and the aggregate
  upward rounding error;
* accept the smallest M for which stake at most alpha cannot reconstruct and
  stake above beta is guaranteed to reconstruct.

The arithmetic emulates Rust ``fixed::types::U64F64`` (unsigned Q64.64), as
used by Aptos.  Unlike Hamilton allocation, independent nearest rounding does
not force W == M.

Aptos' production code currently applies protocol-specific bounds to alpha and
beta.  This script intentionally exposes the underlying rounding construction
for arbitrary symmetric intervals contained in [0, 1].
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

from solana_script_utils import (
    APPROXIMATION_REPORT_STEM,
    VALIDATOR_DISTRIBUTION_STEM,
    latest_data_path,
    timestamped_data_path,
)


DEFAULT_TARGET_RECONSTRUCTION_RATIO = "1/2"
DEFAULT_SYMMETRIC_ERRORS = ("1/4", "1/8", "1/16", "1/32", "1/64", "1/128")
LAMPORTS_PER_SOL = 1_000_000_000

# Unsigned Q64.64 fixed-point constants.
FIXED_FRACTION_BITS = 64
FIXED_SCALE = 1 << FIXED_FRACTION_BITS
FIXED_ONE = FIXED_SCALE


@dataclass(frozen=True)
class AptosProfile:
    """One candidate or selected Aptos-style integer-weight profile."""

    selected_m: int
    stake_per_weight_raw: int
    validator_weights: tuple[int, ...]
    total_weight: int
    delta_down_raw: int
    delta_up_raw: int
    reconstruction_threshold: int
    stake_gap_raw: int
    effective_reconstruction_ratio_raw: int
    valid: bool


@dataclass(frozen=True)
class Allocation:
    """One validator's stake and assigned integer weight."""

    account: str
    active_stake: int
    weight: int
    true_fraction: float
    approximate_fraction: float

    @property
    def absolute_error(self) -> float:
        """Absolute distribution error as a fraction of total stake."""
        return abs(self.approximate_fraction - self.true_fraction)

    @property
    def relative_error_percent(self) -> float:
        """Absolute error divided by this validator's true stake fraction."""
        if self.true_fraction == 0:
            return 0.0
        return 100.0 * self.absolute_error / self.true_fraction


@dataclass(frozen=True)
class ErrorAllocation:
    """Selected profile and per-validator allocation for one symmetric error."""

    target_ratio: Fraction
    error: Fraction
    lower_ratio: Fraction
    upper_ratio: Fraction
    search_lower_bound_m: int
    search_upper_bound_m: int
    profile: AptosProfile
    allocations: tuple[Allocation, ...]


@dataclass(frozen=True)
class Summary:
    """Human-readable approximation and threshold statistics."""

    error: Fraction
    lower_ratio: Fraction
    upper_ratio: Fraction
    selected_m: int
    total_weight: int
    reconstruction_threshold: int
    threshold_weight_percent: float
    effective_reconstruction_percent: float
    nominal_stake_per_weight_sol: float
    actual_stake_per_weight_sol: float
    percentage_per_weight: float
    validators_with_weight: int
    validators_without_weight: int
    zero_weight_stake_percent: float
    mean_absolute_error_pp: float
    median_absolute_error_pp: float
    maximum_absolute_error_pp: float
    median_relative_error_percent: float
    total_variation_percent: float
    rounding_gain_total: float
    rounding_loss_total: float
    stake_gap_pp: float


# ---------------------------------------------------------------------------
# Ratio and fixed-point helpers
# ---------------------------------------------------------------------------


def parse_ratio(text: str) -> Fraction:
    """Parse an integer, decimal, or fraction such as ``1/2``."""
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError(f"invalid ratio {text!r}") from error
    return value


def format_ratio(value: Fraction) -> str:
    """Return a reduced, stable fraction string."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def normalize_errors(values: Iterable[str | Fraction]) -> list[Fraction]:
    """Parse, validate, and de-duplicate symmetric errors in input order."""
    result: list[Fraction] = []
    seen: set[Fraction] = set()
    for value in values:
        if isinstance(value, Fraction):
            error = value
        else:
            try:
                error = parse_ratio(value)
            except argparse.ArgumentTypeError as parse_error:
                raise ValueError(str(parse_error)) from parse_error
        if error <= 0:
            raise ValueError("symmetric errors must be positive")
        if error not in seen:
            seen.add(error)
            result.append(error)
    if not result:
        raise ValueError("at least one symmetric error is required")
    return result


def validate_target_and_error(target: Fraction, error: Fraction) -> tuple[Fraction, Fraction]:
    """Return alpha and beta after checking that the interval lies in [0, 1]."""
    lower = target - error
    upper = target + error
    if target <= 0 or target >= 1:
        raise ValueError("the target reconstruction ratio must lie strictly between 0 and 1")
    if lower < 0 or upper > 1:
        raise ValueError(
            f"target {format_ratio(target)} with error {format_ratio(error)} "
            "does not define an interval contained in [0, 1]"
        )
    if lower >= upper:
        raise ValueError("the lower stake ratio must be less than the upper ratio")
    return lower, upper


def fixed_from_int(value: int) -> int:
    if value < 0:
        raise ValueError("unsigned fixed-point values cannot be negative")
    return value << FIXED_FRACTION_BITS


def fixed_from_fraction(value: Fraction) -> int:
    if value < 0:
        raise ValueError("unsigned fixed-point values cannot be negative")
    # U64F64 conversion/division truncates toward zero.
    return (value.numerator << FIXED_FRACTION_BITS) // value.denominator


def fixed_add(left: int, right: int) -> int:
    return left + right


def fixed_sub(left: int, right: int) -> int:
    if right > left:
        raise ArithmeticError("unsigned fixed-point subtraction underflow")
    return left - right


def fixed_mul(left: int, right: int) -> int:
    return (left * right) >> FIXED_FRACTION_BITS


def fixed_div(left: int, right: int) -> int:
    if right == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    return (left << FIXED_FRACTION_BITS) // right


def fixed_floor(value: int) -> int:
    return (value >> FIXED_FRACTION_BITS) << FIXED_FRACTION_BITS


def fixed_ceil(value: int) -> int:
    return (
        (value + FIXED_SCALE - 1) >> FIXED_FRACTION_BITS
    ) << FIXED_FRACTION_BITS


def fixed_to_int(value: int) -> int:
    return value >> FIXED_FRACTION_BITS


def fixed_to_float(value: int) -> float:
    return value / FIXED_SCALE


def fixed_to_decimal_string(value: int, digits: int = 30) -> str:
    getcontext().prec = max(50, digits + 10)
    decimal_value = Decimal(value) / Decimal(FIXED_SCALE)
    return format(decimal_value, f".{digits}f")


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_validators(path: Path) -> list[dict[str, Any]]:
    """Load and validate a Scheduler War-style validator JSON array."""
    with path.open("r", encoding="utf-8") as input_file:
        validators = json.load(input_file)

    if not isinstance(validators, list) or not validators:
        raise ValueError("input must be a nonempty JSON array")

    seen_accounts: set[str] = set()
    for index, validator in enumerate(validators):
        if not isinstance(validator, dict):
            raise ValueError(f"record {index} is not a JSON object")

        account = validator.get("account")
        stake = validator.get("activeStake")
        if not isinstance(account, str) or not account:
            raise ValueError(f"record {index} has an invalid account")
        # bool is an int subclass in Python, so reject it explicitly.
        if isinstance(stake, bool) or not isinstance(stake, int) or stake < 0:
            raise ValueError(f"record {index} has an invalid activeStake")
        if account in seen_accounts:
            raise ValueError(f"duplicate account: {account}")
        seen_accounts.add(account)

    if sum(validator["activeStake"] for validator in validators) == 0:
        raise ValueError("total active stake must be positive")

    return validators


# ---------------------------------------------------------------------------
# Aptos-style rounding and M selection
# ---------------------------------------------------------------------------


def total_weight_upper_bound(
    validator_count: int,
    lower_ratio_raw: int,
    upper_ratio_raw: int,
) -> int:
    """Mirror Aptos' analytical upper bound for the search over M."""
    if validator_count <= 0:
        raise ValueError("validator_count must be positive")
    if upper_ratio_raw <= lower_ratio_raw:
        raise ValueError("upper ratio must be greater than lower ratio")

    two = fixed_from_int(2)
    n = fixed_from_int(validator_count)
    numerator = fixed_add(fixed_div(n, two), two)
    denominator = fixed_sub(upper_ratio_raw, lower_ratio_raw)
    return fixed_to_int(fixed_ceil(fixed_div(numerator, denominator)))


def compute_profile(
    stakes: Sequence[int],
    m: int,
    lower_ratio_raw: int,
    upper_ratio_raw: int,
) -> AptosProfile:
    """Compute one Aptos-style profile at nominal resolution M."""
    if m <= 0:
        raise ValueError("M must be positive")
    if not stakes:
        raise ValueError("stakes must not be empty")

    stake_sum = sum(stakes)
    if stake_sum <= 0:
        raise ValueError("total stake must be positive")

    stake_sum_raw = fixed_from_int(stake_sum)
    stake_per_weight_raw = max(
        FIXED_ONE,
        fixed_div(stake_sum_raw, fixed_from_int(m)),
    )
    half = FIXED_ONE // 2

    delta_down_raw = 0
    delta_up_raw = 0
    weights: list[int] = []

    for stake in stakes:
        ideal_weight_raw = fixed_div(
            fixed_from_int(stake),
            stake_per_weight_raw,
        )
        # Aptos: floor(ideal + 1/2), so exact half ties round upward.
        rounded_weight_raw = fixed_floor(fixed_add(ideal_weight_raw, half))
        rounded_weight = fixed_to_int(rounded_weight_raw)
        weights.append(rounded_weight)

        if ideal_weight_raw > rounded_weight_raw:
            delta_down_raw = fixed_add(
                delta_down_raw,
                fixed_sub(ideal_weight_raw, rounded_weight_raw),
            )
        else:
            delta_up_raw = fixed_add(
                delta_up_raw,
                fixed_sub(rounded_weight_raw, ideal_weight_raw),
            )

    total_weight = sum(weights)
    threshold_base_raw = fixed_div(
        fixed_mul(lower_ratio_raw, stake_sum_raw),
        stake_per_weight_raw,
    )
    threshold_raw = fixed_add(
        fixed_ceil(fixed_add(threshold_base_raw, delta_up_raw)),
        FIXED_ONE,
    )
    reconstruction_threshold = fixed_to_int(threshold_raw)

    # The Aptos-style q above is the minimum rounded weight that reconstructs.
    # A coalition with stake ratio at most alpha has rounded weight below q.
    # Conversely, if its ideal weight exceeds q - 1 + delta_down, integrality
    # guarantees that its rounded weight is at least q.  Account for the
    # ceiling and +1 in q explicitly when checking the upper stake bound.
    threshold_cutoff_raw = fixed_add(threshold_base_raw, delta_up_raw)
    secrecy_valid = fixed_from_int(reconstruction_threshold) > threshold_cutoff_raw
    reconstruction_boundary_raw = fixed_add(
        fixed_from_int(reconstruction_threshold - 1),
        delta_down_raw,
    )
    effective_reconstruction_ratio_raw = fixed_div(
        fixed_mul(stake_per_weight_raw, reconstruction_boundary_raw),
        stake_sum_raw,
    )
    stake_gap_raw = fixed_div(
        fixed_mul(
            stake_per_weight_raw,
            fixed_add(delta_down_raw, delta_up_raw),
        ),
        stake_sum_raw,
    )

    return AptosProfile(
        selected_m=m,
        stake_per_weight_raw=stake_per_weight_raw,
        validator_weights=tuple(weights),
        total_weight=total_weight,
        delta_down_raw=delta_down_raw,
        delta_up_raw=delta_up_raw,
        reconstruction_threshold=reconstruction_threshold,
        stake_gap_raw=stake_gap_raw,
        effective_reconstruction_ratio_raw=effective_reconstruction_ratio_raw,
        valid=(
            secrecy_valid
            and reconstruction_threshold <= total_weight
            and effective_reconstruction_ratio_raw <= upper_ratio_raw
        ),
    )


def select_profile(
    stakes: Sequence[int],
    lower_ratio: Fraction,
    upper_ratio: Fraction,
) -> tuple[AptosProfile, int, int]:
    """Select the smallest valid M within Aptos' analytical search bound."""
    if not stakes:
        raise ValueError("stakes must not be empty")

    lower_ratio_raw = fixed_from_fraction(lower_ratio)
    upper_ratio_raw = fixed_from_fraction(upper_ratio)
    if upper_ratio_raw <= lower_ratio_raw:
        raise ValueError("the fixed-point upper ratio must exceed the lower ratio")

    search_lower = len(stakes)
    search_upper = total_weight_upper_bound(
        len(stakes),
        lower_ratio_raw,
        upper_ratio_raw,
    )

    # Nearest-rounding errors are not monotone in M, so a binary search can
    # skip an earlier valid resolution.  The analytical upper bound is small
    # enough for an exhaustive ascending scan at the validator-set sizes used
    # here.
    for m in range(search_lower, search_upper + 1):
        profile = compute_profile(
            stakes,
            m,
            lower_ratio_raw,
            upper_ratio_raw,
        )
        if profile.valid:
            return profile, search_lower, search_upper

    raise RuntimeError(
        "could not find a valid Aptos-style profile in the search range "
        f"[{search_lower}, {search_upper}]"
    )


def allocate_for_error(
    validators: Sequence[dict[str, Any]],
    target_ratio: Fraction,
    error: Fraction,
) -> ErrorAllocation:
    """Select M and produce per-validator Aptos-style integer weights."""
    lower_ratio, upper_ratio = validate_target_and_error(target_ratio, error)
    stakes = [validator["activeStake"] for validator in validators]
    profile, search_lower, search_upper = select_profile(
        stakes,
        lower_ratio,
        upper_ratio,
    )

    total_stake = sum(stakes)
    if profile.total_weight <= 0:
        raise RuntimeError("Aptos-style rounding produced zero total weight")

    allocations = tuple(
        Allocation(
            account=validator["account"],
            active_stake=validator["activeStake"],
            weight=weight,
            true_fraction=validator["activeStake"] / total_stake,
            approximate_fraction=weight / profile.total_weight,
        )
        for validator, weight in zip(validators, profile.validator_weights)
    )

    return ErrorAllocation(
        target_ratio=target_ratio,
        error=error,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
        search_lower_bound_m=search_lower,
        search_upper_bound_m=search_upper,
        profile=profile,
        allocations=allocations,
    )


def allocate_for_errors(
    validators: Sequence[dict[str, Any]],
    target_ratio: Fraction,
    errors: Sequence[Fraction],
) -> list[ErrorAllocation]:
    """Generate selected Aptos-style profiles for all requested errors."""
    return [
        allocate_for_error(validators, target_ratio, error)
        for error in errors
    ]


# ---------------------------------------------------------------------------
# Statistics and report formatting
# ---------------------------------------------------------------------------


def summarize_allocation(result: ErrorAllocation) -> Summary:
    """Compute distribution-quality and Aptos threshold statistics."""
    allocations = result.allocations
    profile = result.profile
    total_stake = sum(allocation.active_stake for allocation in allocations)
    zero_weight_allocations = [
        allocation for allocation in allocations if allocation.weight == 0
    ]
    absolute_errors = [allocation.absolute_error for allocation in allocations]
    relative_errors = [
        allocation.relative_error_percent for allocation in allocations
    ]

    return Summary(
        error=result.error,
        lower_ratio=result.lower_ratio,
        upper_ratio=result.upper_ratio,
        selected_m=profile.selected_m,
        total_weight=profile.total_weight,
        reconstruction_threshold=profile.reconstruction_threshold,
        threshold_weight_percent=(
            100.0 * profile.reconstruction_threshold / profile.total_weight
        ),
        effective_reconstruction_percent=(
            100.0 * fixed_to_float(profile.effective_reconstruction_ratio_raw)
        ),
        nominal_stake_per_weight_sol=(
            total_stake / LAMPORTS_PER_SOL / profile.selected_m
        ),
        actual_stake_per_weight_sol=(
            total_stake / LAMPORTS_PER_SOL / profile.total_weight
        ),
        percentage_per_weight=100.0 / profile.total_weight,
        validators_with_weight=sum(
            allocation.weight > 0 for allocation in allocations
        ),
        validators_without_weight=len(zero_weight_allocations),
        zero_weight_stake_percent=(
            100.0
            * sum(
                allocation.active_stake
                for allocation in zero_weight_allocations
            )
            / total_stake
        ),
        mean_absolute_error_pp=100.0 * statistics.mean(absolute_errors),
        median_absolute_error_pp=100.0 * statistics.median(absolute_errors),
        maximum_absolute_error_pp=100.0 * max(absolute_errors),
        median_relative_error_percent=statistics.median(relative_errors),
        # For probability distributions, TV = (1/2) * L1 distance.
        total_variation_percent=50.0 * sum(absolute_errors),
        rounding_gain_total=fixed_to_float(profile.delta_up_raw),
        rounding_loss_total=fixed_to_float(profile.delta_down_raw),
        stake_gap_pp=100.0 * fixed_to_float(profile.stake_gap_raw),
    )


def format_markdown_table(
    summaries: Sequence[Summary],
    target_ratio: Fraction,
) -> str:
    """Format one comparison column per symmetric error."""
    header = ["Metric"] + [
        f"e = {format_ratio(summary.error)}" for summary in summaries
    ]
    rows = [
        ["Stake interval"]
        + [
            f"[{format_ratio(summary.lower_ratio)}, "
            f"{format_ratio(summary.upper_ratio)}]"
            for summary in summaries
        ],
        ["Selected nominal resolution M"]
        + [f"{summary.selected_m:,}" for summary in summaries],
        ["Actual total cryptographic weight W"]
        + [f"{summary.total_weight:,}" for summary in summaries],
        ["Reconstruction threshold q"]
        + [f"{summary.reconstruction_threshold:,}" for summary in summaries],
        ["q / W"]
        + [f"{summary.threshold_weight_percent:.6f}%" for summary in summaries],
        ["Effective guaranteed reconstruction stake"]
        + [
            f"{summary.effective_reconstruction_percent:.6f}%"
            for summary in summaries
        ],
        ["Nominal stake represented by one M unit"]
        + [
            f"{summary.nominal_stake_per_weight_sol:,.0f} SOL"
            for summary in summaries
        ],
        ["Actual stake represented by one weight"]
        + [
            f"{summary.actual_stake_per_weight_sol:,.0f} SOL"
            for summary in summaries
        ],
        ["Percentage represented by one actual weight"]
        + [f"{summary.percentage_per_weight:.6f}%" for summary in summaries],
        ["Validators receiving positive weight"]
        + [f"{summary.validators_with_weight:,}" for summary in summaries],
        ["Validators receiving zero weight"]
        + [f"{summary.validators_without_weight:,}" for summary in summaries],
        ["Actual stake assigned zero weight"]
        + [f"{summary.zero_weight_stake_percent:.4f}%" for summary in summaries],
        ["Aggregate upward rounding error"]
        + [f"{summary.rounding_gain_total:.6f}" for summary in summaries],
        ["Aggregate downward rounding error"]
        + [f"{summary.rounding_loss_total:.6f}" for summary in summaries],
        ["Aptos stake-domain rounding gap"]
        + [f"{summary.stake_gap_pp:.6f} pp" for summary in summaries],
        ["Mean absolute distribution error"]
        + [f"{summary.mean_absolute_error_pp:.6f} pp" for summary in summaries],
        ["Median absolute distribution error"]
        + [f"{summary.median_absolute_error_pp:.6f} pp" for summary in summaries],
        ["Maximum absolute distribution error"]
        + [f"{summary.maximum_absolute_error_pp:.6f} pp" for summary in summaries],
        ["Median relative distribution error"]
        + [
            f"{summary.median_relative_error_percent:.2f}%"
            for summary in summaries
        ],
        ["Total-variation distance"]
        + [f"{summary.total_variation_percent:.4f}%" for summary in summaries],
    ]

    output = [
        f"Target reconstruction ratio: **{format_ratio(target_ratio)}**",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(summaries)) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Choose Aptos-style integer-weight resolutions for symmetric "
            "stake-threshold errors and print a Markdown comparison table."
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
            "symmetric errors to compare "
            "(default: 1/4 1/8 1/16 1/32 1/64 1/128)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Markdown report path (default: a UTC-timestamped file in "
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

    try:
        validators = load_validators(input_path)
        errors = normalize_errors(args.errors)
        results = allocate_for_errors(validators, args.target, errors)
        summaries = [summarize_allocation(result) for result in results]
        table = format_markdown_table(summaries, args.target)

        print(table)
        output = args.output or timestamped_data_path(
            APPROXIMATION_REPORT_STEM,
            ".md",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"wrote approximation report to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
