#!/usr/bin/env python3
"""Approximate validator stake weights with integer virtual shares.

The input must be a JSON array whose records contain at least:

    {"account": "...", "activeStake": 123456789}

The script uses Hamilton's largest-remainder method, so the assigned integer
weights sum to exactly N for every requested total weight N. It prints the
approximation summary as a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_SHARE_COUNTS = (2048, 4096, 8192, 16384, 32768)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "solana_validator_distribution.json"
LAMPORTS_PER_SOL = 1_000_000_000


@dataclass(frozen=True)
class Allocation:
    account: str
    active_stake: int
    shares: int
    true_fraction: float
    approximate_fraction: float

    @property
    def absolute_error(self) -> float:
        """Absolute error as a fraction of total network stake."""
        return abs(self.approximate_fraction - self.true_fraction)

    @property
    def relative_error_percent(self) -> float:
        """Absolute error divided by this validator's true stake fraction."""
        if self.true_fraction == 0:
            return 0.0
        return 100.0 * self.absolute_error / self.true_fraction


@dataclass(frozen=True)
class Summary:
    share_count: int
    stake_per_share_sol: float
    percentage_per_share: float
    validators_with_shares: int
    validators_without_shares: int
    zero_share_stake_percent: float
    mean_absolute_error_pp: float
    median_absolute_error_pp: float
    maximum_absolute_error_pp: float
    median_relative_error_percent: float
    total_variation_percent: float


def load_validators(path: Path) -> list[dict[str, Any]]:
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


def hamilton_allocate(
    validators: Sequence[dict[str, Any]], share_count: int
) -> list[Allocation]:
    """Allocate exactly share_count shares using largest remainders.

    All quota floors and remainders are calculated with integer arithmetic.
    Ties are broken deterministically by active stake and then account string,
    both in descending order.
    """
    if share_count <= 0:
        raise ValueError("share counts must be positive")

    total_stake = sum(validator["activeStake"] for validator in validators)
    quota_parts: list[tuple[int, int, int]] = []

    for index, validator in enumerate(validators):
        numerator = share_count * validator["activeStake"]
        floor, remainder = divmod(numerator, total_stake)
        quota_parts.append((floor, remainder, index))

    shares = [floor for floor, _, _ in quota_parts]
    unassigned = share_count - sum(shares)

    remainder_order = sorted(
        quota_parts,
        key=lambda item: (
            item[1],
            validators[item[2]]["activeStake"],
            validators[item[2]]["account"],
        ),
        reverse=True,
    )
    for _, _, index in remainder_order[:unassigned]:
        shares[index] += 1

    assert sum(shares) == share_count

    return [
        Allocation(
            account=validator["account"],
            active_stake=validator["activeStake"],
            shares=weight,
            true_fraction=validator["activeStake"] / total_stake,
            approximate_fraction=weight / share_count,
        )
        for validator, weight in zip(validators, shares)
    ]


def summarize(
    validators: Sequence[dict[str, Any]], share_count: int
) -> Summary:
    allocations = hamilton_allocate(validators, share_count)
    total_stake = sum(allocation.active_stake for allocation in allocations)
    zero_share_allocations = [
        allocation for allocation in allocations if allocation.shares == 0
    ]
    absolute_errors = [allocation.absolute_error for allocation in allocations]
    relative_errors = [
        allocation.relative_error_percent for allocation in allocations
    ]

    return Summary(
        share_count=share_count,
        stake_per_share_sol=total_stake / LAMPORTS_PER_SOL / share_count,
        percentage_per_share=100.0 / share_count,
        validators_with_shares=sum(
            allocation.shares > 0 for allocation in allocations
        ),
        validators_without_shares=len(zero_share_allocations),
        zero_share_stake_percent=(
            100.0
            * sum(allocation.active_stake for allocation in zero_share_allocations)
            / total_stake
        ),
        mean_absolute_error_pp=100.0 * statistics.mean(absolute_errors),
        median_absolute_error_pp=100.0 * statistics.median(absolute_errors),
        maximum_absolute_error_pp=100.0 * max(absolute_errors),
        median_relative_error_percent=statistics.median(relative_errors),
        # For probability distributions, TV = (1/2) * L1 distance.
        total_variation_percent=50.0 * sum(absolute_errors),
    )


def format_markdown_table(summaries: Sequence[Summary]) -> str:
    header = ["Metric"] + [
        f"{summary.share_count:,} shares" for summary in summaries
    ]
    rows = [
        ["Stake represented by one share"]
        + [f"{summary.stake_per_share_sol:,.0f} SOL" for summary in summaries],
        ["Percentage represented by one share"]
        + [f"{summary.percentage_per_share:.6f}%" for summary in summaries],
        ["Validators receiving shares"]
        + [f"{summary.validators_with_shares:,}" for summary in summaries],
        ["Validators receiving zero shares"]
        + [f"{summary.validators_without_shares:,}" for summary in summaries],
        ["Actual stake assigned zero shares"]
        + [f"{summary.zero_share_stake_percent:.4f}%" for summary in summaries],
        ["Mean absolute error"]
        + [f"{summary.mean_absolute_error_pp:.6f} pp" for summary in summaries],
        ["Median absolute error"]
        + [f"{summary.median_absolute_error_pp:.6f} pp" for summary in summaries],
        ["Maximum absolute error"]
        + [f"{summary.maximum_absolute_error_pp:.6f} pp" for summary in summaries],
        ["Median relative error"]
        + [
            f"{summary.median_relative_error_percent:.2f}%"
            for summary in summaries
        ],
        ["Total-variation distance"]
        + [f"{summary.total_variation_percent:.4f}%" for summary in summaries],
    ]

    output = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(summaries)) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Approximate validator stake percentages with integer shares and "
            "print a Markdown comparison table."
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
        help="share counts to compare (default: 2048 4096 8192 16384 32768)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validators = load_validators(args.input)
    share_counts = list(dict.fromkeys(args.shares))
    summaries = [summarize(validators, count) for count in share_counts]
    print(format_markdown_table(summaries))


if __name__ == "__main__":
    main()
