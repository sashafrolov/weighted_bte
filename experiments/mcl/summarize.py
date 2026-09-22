#!/usr/bin/env python3
"""Summarize raw independent rounds without treating kernel rows as protocol time."""
import csv
from pathlib import Path
import statistics
import sys

directory = Path(sys.argv[1])
data = {}
for file in sorted(directory.glob("*_r*.csv")):
    round_id = int(file.stem.rsplit("_r", 1)[1])
    log = file.with_suffix(".log").read_text()
    if "correctness_checks=passed" not in log:
        raise SystemExit(f"Incomplete or failed correctness checks: {file}")
    rows = list(csv.DictReader(file.open()))
    if len(rows) != 14:
        raise SystemExit(f"Expected 14 kernel rows: {file}")
    for row in rows:
        if float(row["median_us"]) <= 0:
            raise SystemExit(f"Invalid timing: {file}")
        key = (row["curve"], row["simd"], int(row["batch"]), row["phase"], row["group"])
        data.setdefault(key, {})[round_id] = row
if not data:
    raise SystemExit("No result CSVs found")


def median(key):
    return statistics.median(float(row["median_us"]) for row in data[key].values())


def ratios(first, second):
    common = data[first].keys() & data[second].keys()
    return [float(data[first][i]["median_us"]) / float(data[second][i]["median_us"]) for i in common]


print("# MCL arithmetic kernel results\n")
print("Single CPU; median of each independent round's sample median. Times are milliseconds.")
print("BN254 below is **MCL BN_SNARK1**, not MCL's separately named BN254.")
print("These rows are operation-shape probes and are not end-to-end protocol results.\n")
print("## Actual AVX-512 dispatch\n")
print("| B | G1 operation | SIMD off ms | AVX-512 ms | Paired speedup range | MSM / mulEach calls per repetition |")
print("| --- | --- | ---: | ---: | ---: | ---: |")
for batch in sorted({k[2] for k in data}):
    for phase in ("partial_decrypt_one_msm", "committee_positive_msms", "group_fft"):
        off = ("bls12_381", "off", batch, phase, "G1")
        on = ("bls12_381", "avx512", batch, phase, "G1")
        rr = ratios(off, on)
        calls = next(iter(data[on].values()))
        print(f"| {batch} | {phase} | {median(off)/1000:.4f} | {median(on)/1000:.4f} | "
              f"{min(rr):.2f}–{max(rr):.2f}× | {float(calls['avx_msm_calls_per_rep']):g} / "
              f"{float(calls['avx_each_calls_per_rep']):g} |")
print("\nRows with zero dispatch are controls, not SIMD speedups.\n")
print("## Same-backend curve comparisons (SIMD off)\n")
print("| B | Operation | Group | BLS12-381 ms | BN_SNARK1 ms | BLS / BN paired ratio |")
print("| --- | --- | --- | ---: | ---: | ---: |")
for batch in sorted({k[2] for k in data}):
    for phase, group in (("partial_decrypt_one_msm", "G1"),
                         ("committee_opening_msms", "G1"),
                         ("committee_opening_msms", "G2"),
                         ("committee_positive_msms", "G1"),
                         ("committee_positive_msms", "G2"),
                         ("opening_multi_pairings", "G1xG2")):
        bls = ("bls12_381", "off", batch, phase, group)
        bn = ("bn254", "off", batch, phase, group)
        rr = ratios(bls, bn)
        print(f"| {batch} | {phase} | {group} | {median(bls)/1000:.4f} | {median(bn)/1000:.4f} | "
              f"{min(rr):.2f}–{max(rr):.2f}× |")
