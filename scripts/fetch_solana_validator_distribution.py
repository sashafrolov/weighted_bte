#!/usr/bin/env python3
"""Fetch Scheduler War's "All Validators" JSON without browser automation.

Scheduler War embeds the validator records in the server-rendered Next.js page.
This script performs one HTTPS GET, decodes the relevant Next.js Flight payload,
and writes the same validator-object shape exposed by the site's JSON export.

No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_URL = "https://schedulerwar.vercel.app/"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "solana_validator_distribution.json"
MAX_PAGE_BYTES = 20 * 1024 * 1024
USER_AGENT = (
    "Mozilla/5.0 (compatible; SchedulerWarValidatorFetcher/1.0; "
    "+https://schedulerwar.vercel.app/)"
)
REQUIRED_FIELDS = ("account", "name", "activeStake", "softwareClient")


class ScriptCollector(HTMLParser):
    """Collect the text contents of every script element."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._inside_script = False
        self._current: list[str] = []
        self.scripts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() == "script":
            self._inside_script = True
            self._current = []

    def handle_data(self, data: str) -> None:
        if self._inside_script:
            self._current.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._inside_script:
            self.scripts.append("".join(self._current))
            self._inside_script = False
            self._current = []


def fetch_html(url: str, timeout: float) -> str:
    parsed_url = urlparse(url)
    if parsed_url.scheme.lower() != "https":
        raise ValueError("the page URL must use HTTPS")

    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        final_url = response.geturl()
        if urlparse(final_url).scheme.lower() != "https":
            raise ValueError(f"refusing non-HTTPS redirect: {final_url}")

        page_bytes = response.read(MAX_PAGE_BYTES + 1)
        if len(page_bytes) > MAX_PAGE_BYTES:
            raise ValueError(
                f"page exceeds the {MAX_PAGE_BYTES:,}-byte safety limit"
            )

        # urllib's HTTPMessage understands a declared charset. Scheduler War
        # currently serves UTF-8, which is also the HTML5 default in practice.
        charset = response.headers.get_content_charset() or "utf-8"
        return page_bytes.decode(charset)


def iter_next_flight_payloads(script: str) -> Iterable[str]:
    """Yield decoded string payloads from self.__next_f.push(...) calls."""
    marker = "self.__next_f.push("
    decoder = json.JSONDecoder()
    search_from = 0

    while True:
        call_start = script.find(marker, search_from)
        if call_start < 0:
            return

        argument_start = call_start + len(marker)
        try:
            argument, consumed = decoder.raw_decode(script[argument_start:])
        except json.JSONDecodeError:
            # This script block is not in the expected Next.js encoding. Move
            # forward rather than repeatedly finding the same marker.
            search_from = argument_start
            continue

        if (
            isinstance(argument, list)
            and len(argument) >= 2
            and isinstance(argument[1], str)
        ):
            yield argument[1]

        search_from = argument_start + consumed


def extract_validator_array(html: str) -> list[dict[str, Any]]:
    collector = ScriptCollector()
    collector.feed(html)

    property_marker = '"solanaValidators":'
    decoder = json.JSONDecoder()

    for script in collector.scripts:
        if "solanaValidators" not in script:
            continue

        for payload in iter_next_flight_payloads(script):
            property_start = payload.find(property_marker)
            if property_start < 0:
                continue

            array_start = property_start + len(property_marker)
            try:
                value, _ = decoder.raw_decode(payload[array_start:])
            except json.JSONDecodeError as error:
                raise ValueError(
                    "found solanaValidators but could not decode its JSON array"
                ) from error

            if not isinstance(value, list):
                raise ValueError("solanaValidators is not a JSON array")
            return value

    raise ValueError(
        "could not find solanaValidators in the page; the site's rendering "
        "format may have changed"
    )


def validate_and_normalize(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("the validator array is empty")

    result: list[dict[str, Any]] = []
    accounts: set[str] = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"validator {index} is not a JSON object")

        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(
                f"validator {index} is missing fields: {', '.join(missing)}"
            )

        account = record["account"]
        name = record["name"]
        active_stake = record["activeStake"]
        software_client = record["softwareClient"]

        if not isinstance(account, str) or not account:
            raise ValueError(f"validator {index} has an invalid account")
        if name is not None and not isinstance(name, str):
            raise ValueError(f"validator {index} has an invalid name")
        if (
            isinstance(active_stake, bool)
            or not isinstance(active_stake, int)
            or active_stake < 0
        ):
            raise ValueError(f"validator {index} has an invalid activeStake")
        if not isinstance(software_client, str) or not software_client:
            raise ValueError(f"validator {index} has an invalid softwareClient")
        if account in accounts:
            raise ValueError(f"duplicate validator account: {account}")
        accounts.add(account)

        # Select and order the same four fields used by the site's export.
        result.append(
            {
                "account": account,
                "name": name,
                "activeStake": active_stake,
                "softwareClient": software_client,
            }
        )

    return result


def write_json(records: list[dict[str, Any]], output: Path | None) -> None:
    serialized = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        sys.stdout.write(serialized)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the Solana validator array embedded in Scheduler War's "
            "server-rendered webpage."
        )
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"HTTPS page to fetch (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="write JSON to standard output instead of a file",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTPS request timeout in seconds (default: 30)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        print("error: --timeout must be positive", file=sys.stderr)
        return 2

    try:
        html = fetch_html(args.url, args.timeout)
        records = validate_and_normalize(extract_validator_array(html))
        output = None if args.stdout else args.output
        write_json(records, output)
    except (HTTPError, URLError, OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    destination = "stdout" if args.stdout else str(args.output)
    total_stake = sum(record["activeStake"] for record in records)
    print(
        f"wrote {len(records):,} validators ({total_stake:,} raw stake) "
        f"to {destination}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
