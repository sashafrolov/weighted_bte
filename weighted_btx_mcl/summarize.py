#!/usr/bin/env python3
"""Combine complete raw protocol samples; quantiles are descriptive, not CIs."""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results_directory", type=Path)
parser.add_argument("--layouts", default="1x16,4x4",
                    help="comma-separated required layouts in table order (default: 1x16,4x4)")
args = parser.parse_args()
layouts = [layout.strip() for layout in args.layouts.split(",")]
if any(layout not in {"1x16", "4x4", "8x2"} for layout in layouts) or len(set(layouts)) != len(layouts):
    parser.error("--layouts must list distinct layouts from 1x16,4x4,8x2")
directory = args.results_directory
phases = {"encryption", "validation", "share_generation", "acceptance", "preparation",
          "precompute", "opening", "combiner", "end_to_end"}
cases = {(layout, cache) for layout in layouts for cache in ["cold", "cached"]}
groups = defaultdict(list)
dispatches = defaultdict(list)
rounds_by_config = defaultdict(set)
seen_files = set()
sample_counts = set()
thread_counts = set()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def round_number(path):
    match = re.search(r"_r([0-9]+)\.(?:json|csv)$", path.name)
    require(match is not None and int(match[1]) > 0, f"{path.name}: invalid round suffix")
    return int(match[1])

def register(path, config, threads, count):
    round_id = round_number(path)
    require((config, round_id) not in seen_files, f"{path.name}: duplicate configuration/round")
    require(threads > 0 and count > 0, f"{path.name}: threads and samples must be positive")
    seen_files.add((config, round_id))
    rounds_by_config[config].add(round_id)
    sample_counts.add(count)
    thread_counts.add(threads)

def measurement(value, path):
    value = float(value)
    require(math.isfinite(value) and value >= 0, f"{path.name}: invalid elapsed time")
    return value

for path in sorted(directory.glob("blst_*_r*.json")):
    report = json.loads(path.read_text())
    require(report["implementation"] in {"weighted_btx", "weighted_btx_swapped"},
            f"{path.name}: unexpected native implementation")
    orientation = "swapped" if report["implementation"].endswith("_swapped") else "normal"
    config = ("blst", "bls12_381", "native", orientation)
    require(report["total_ciphertexts"] == 16 and report["profile"]["accepted_weight"] == 794
            and len(report["profile"]["selected_parties"]) == 38, f"{path.name}: unexpected workload")
    count, threads = int(report["samples_per_case"]), int(report["threads"])
    seen_cases = set()
    for case in report["cases"]:
        require(case["all_plaintexts_verified"], f"{path.name}: plaintext verification failed")
        cache, layout = case["name"].split("/")
        require((layout, cache) in cases and (layout, cache) not in seen_cases,
                f"{path.name}: unexpected or duplicate case {case['name']}")
        seen_cases.add((layout, cache))
        require(len(case["samples"]) == count, f"{path.name}: inconsistent case sample count")
        for sample in case["samples"]:
            require(set(sample) == {phase + "_us" for phase in phases}, f"{path.name}: incomplete phase set")
            for phase_us, value in sample.items():
                key = (*config, layout, cache, threads, phase_us[:-3])
                groups[key].append(measurement(value, path) / 1000)
    require(seen_cases == cases, f"{path.name}: incomplete case set")
    register(path, config, threads, count)

for path in sorted(directory.glob("mcl_*_r*.csv")):
    with path.open() as file:
        raw = list(csv.DictReader(file))
    require(bool(raw), f"{path.name}: no samples")
    file_configs, file_threads = set(), set()
    indices = defaultdict(set)
    for row in raw:
        config = ("mcl", row["curve"], row["simd"], row["orientation"])
        require(row["curve"] in {"bls12_381", "bn254"} and row["simd"] in {"off", "avx512"}
                and row["orientation"] in {"normal", "swapped"}, f"{path.name}: unexpected configuration")
        layout, cache, phase = row["layout"], row["cache"], row["phase"]
        require((layout, cache) in cases and phase in phases, f"{path.name}: unexpected case/phase")
        threads, sample = int(row["threads"]), int(row["sample"])
        phase_key = (layout, cache, phase)
        require(sample >= 0 and sample not in indices[phase_key], f"{path.name}: duplicate/negative sample index")
        indices[phase_key].add(sample)
        file_configs.add(config)
        file_threads.add(threads)
        key = (*config, layout, cache, threads, phase)
        groups[key].append(measurement(row["milliseconds"], path))
        calls = (int(row["avx_msm"]), int(row["avx_each"]))
        require(min(calls) >= 0, f"{path.name}: negative dispatch count")
        dispatches[key].append(calls)
    require(len(file_configs) == len(file_threads) == 1, f"{path.name}: mixed configurations/threads")
    require(set(indices) == {(layout, cache, phase) for layout, cache in cases for phase in phases},
            f"{path.name}: incomplete case/phase matrix")
    counts = {len(values) for values in indices.values()}
    require(len(counts) == 1, f"{path.name}: unequal sample counts across cases/phases")
    count = counts.pop()
    require(all(values == set(range(count)) for values in indices.values()),
            f"{path.name}: sample indices are not contiguous from zero")
    register(path, file_configs.pop(), file_threads.pop(), count)

require(bool(groups), "no raw protocol benchmark files found")
require(len(sample_counts) == 1, "sample counts differ across files/rounds/configurations")
require(len(thread_counts) == 1, "mixed thread counts: use a separate results directory for each thread budget")
round_sets = {tuple(sorted(rounds)) for rounds in rounds_by_config.values()}
require(len(round_sets) == 1, "round sets differ across configurations")
round_ids = next(iter(round_sets))
per_round, threads = next(iter(sample_counts)), next(iter(thread_counts))
expected_samples = per_round * len(round_ids)
require(all(len(values) == expected_samples for values in groups.values()), "incomplete aggregated sample set")

def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)

fields = ["backend", "curve", "simd", "orientation", "layout", "cache", "threads", "phase", "samples", "median_ms", "p10_ms", "p90_ms", "avx_msm_min", "avx_msm_max", "avx_each_min", "avx_each_max"]
rows = []
for key, values in sorted(groups.items()):
    calls = dispatches.get(key, [(0, 0)])
    rows.append(dict(zip(fields, [*key, len(values), quantile(values, .5), quantile(values, .1), quantile(values, .9),
                                  min(c[0] for c in calls), max(c[0] for c in calls), min(c[1] for c in calls), max(c[1] for c in calls)])))
with (directory / "summary.csv").open("w") as file:
    writer = csv.DictWriter(file, fields)
    writer.writeheader()
    writer.writerows(rows)
(directory / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
lookup = {tuple(row[f] for f in fields[:8]): row for row in rows}
configurations = list(dict.fromkeys(key[:4] for key in groups))
round_label = ", ".join(f"r{number}" for number in round_ids)
round_word = "round" if len(round_ids) == 1 else "rounds"
lines = ["# Full online protocol results", "",
         f"Milliseconds per 16 fresh ciphertexts, median [p10–p90]. {expected_samples} samples per case across {len(round_ids)} {round_word} ({round_label}; {per_round} samples per round). Descriptive quantiles are not confidence intervals.", ""]
if len(round_ids) > 1:
    lines += ["The matrix runner reverses implementation order between odd and even rounds.", ""]
columns = [(cache, layout) for cache in ["cold", "cached"] for layout in layouts]
header = "| Backend / curve / SIMD / groups | " + " | ".join(
    f"{cache.title()} {layout.replace('x', '×')}" for cache, layout in columns) + " |"
separator = "| --- | " + " | ".join("---:" for _ in columns) + " |"
for phase in ["end_to_end", "combiner"]:
    lines += [f"## {phase}", "", header, separator]
    for config in configurations:
        cells = []
        for cache, layout in columns:
            r = lookup[(*config, layout, cache, threads, phase)]
            cells.append(f'{r["median_ms"]:.3f} [{r["p10_ms"]:.3f}–{r["p90_ms"]:.3f}]')
        lines.append("| " + " / ".join(config) + " | " + " | ".join(cells) + " |")
    lines += [""]
lines += ["## SIMD dispatch", "", "Counts are observed calls during the full timed online pipeline, excluding setup and warmups.", "", "| Curve / SIMD / groups / layout / cache | MSM calls per iteration | Batched scalar calls per iteration |", "| --- | ---: | ---: |"]
for row in rows:
    if row["backend"] == "mcl" and row["simd"] == "avx512" and row["phase"] == "end_to_end":
        label = " / ".join(row[k] for k in ["curve", "simd", "orientation", "layout", "cache"])
        lines.append(f'| {label} | {row["avx_msm_min"]}–{row["avx_msm_max"]} | {row["avx_each_min"]}–{row["avx_each_max"]} |')
lines += ["", f"All raw per-phase samples, correctness logs, and build metadata accompany this summary. Setup and networking are excluded; all selected validators are simulated on one host with {threads} execution threads.", ""]
(directory / "SUMMARY.md").write_text("\n".join(lines))
print("\n".join(lines))
