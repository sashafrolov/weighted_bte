#!/usr/bin/env python3
"""Validate every fresh-input run and aggregate arithmetic means for paper tables."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from run_optimized import CONFIGS, POINTS, ROOT

PHASES = ("encryption", "validation", "share_generation", "acceptance", "preparation",
          "precompute", "opening", "combiner", "decryption_total", "end_to_end")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def log_fields(log, prefix):
    lines = [line for line in log.splitlines() if line.startswith(prefix + " ")]
    require(len(lines) == 1, f"MCL missing/duplicate {prefix} metadata")
    pairs = re.findall(r"(\w+)=([^\s]+)", lines[0])
    require(len(dict(pairs)) == len(pairs), f"MCL duplicate {prefix} field")
    return dict(pairs)


def summarize(out):
    manifest = json.loads((out / "optimized_manifest.json").read_text())
    weights_path = ROOT / manifest["weights"]
    require(hashlib.sha256(weights_path.read_bytes()).hexdigest() == manifest["weight_sha256"], "weight checksum mismatch")
    document = json.loads(weights_path.read_text())
    require(document["target_reconstruction_ratio"] == "2/3", "wrong target threshold")
    for path, digest in manifest.get("profile_sha256", {}).items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, "MCL profile checksum mismatch")
    profiles, selected_weights = {}, {}
    for p in document["allocations"]:
        w = p["weights"]
        text_path = ROOT / f"experiments/paper_tables/profiles/{p['error'].replace('/', 'of')}.txt"
        require(list(map(int, text_path.read_text().split())) == [p["reconstruction_threshold"], *w],
                "MCL profile weights differ from JSON")
        require(sum(w) == p["share_count"] and len(w) == p["positive_validator_count"], "profile count mismatch")
        committee = sorted(range(len(w)), key=lambda i: (-w[i], i))
        accepted, tau = 0, 0
        for i in committee:
            accepted += w[i]
            tau += 1
            if accepted >= p["reconstruction_threshold"]:
                break
        profiles[p["error"]] = {"N": len(w), "W": sum(w), "q": p["reconstruction_threshold"],
                                "tau": tau, "accepted_weight": accepted}
        selected_weights[p["error"]] = [w[i] for i in committee[:tau]]
    require(set(manifest["configurations"]) <= set(CONFIGS) and manifest["configurations"], "unknown configuration")
    expected = {(c, e, m, r) for c in manifest["configurations"] for e, m in POINTS for r in range(1, manifest["rounds"] + 1)}
    actual = {(x["config"], x["error"], x["total"], x["round"]) for x in manifest["runs"]}
    require(actual == expected and len(actual) == len(manifest["runs"]), "incomplete or duplicate configuration matrix")
    values = defaultdict(list)
    simd = defaultdict(list)
    sizes = {}
    raw = []
    for run in manifest["runs"]:
        name, error, total, batch = run["config"], run["error"], run["total"], run["batch"]
        profile = profiles[error]
        backend, curve, mode, orientation, split = CONFIGS[name]
        require(run["setup"] == total and batch == (split or total), "unexpected layout/setup")
        observations = defaultdict(dict)
        counters = {}
        if backend == "blst":
            report = json.loads((out / run["file"]).read_text())
            require(report["schema_version"] == 1 and report["benchmark"] == "paper_tables"
                    and report["implementation"] == ("weighted_btx_swapped" if orientation == "swapped" else "weighted_btx"),
                    "wrong Rust harness identity")
            require(report["backend"] == backend and report["curve"] == curve and report["orientation"] == orientation,
                    "wrong Rust backend identity")
            for key, expected_value in {**profile, "total": total, "batch": batch, "setup": total, "error": error,
                                        "sha256": manifest["weight_sha256"]}.items():
                require(report["inputs"][key] == expected_value, f"Rust {key} mismatch")
            require(report["settings"] == {"threads": manifest["threads"], "samples": manifest["samples_per_round"],
                                            "warmup": manifest["warmup_per_case"]}, "Rust settings mismatch")
            require({c["name"] for c in report["cases"]} == {"cold", "cached"} and len(report["cases"]) == 2, "Rust cases mismatch")
            for case in report["cases"]:
                require(case["all_plaintexts_verified"] and case["batch_size"] == batch and case["chunks"] == total // batch,
                        "Rust correctness/layout mismatch")
                require(case["committee_cached"] == (case["name"] == "cached"), "Rust cache identity mismatch")
                for sample, timings in enumerate(case["samples"]):
                    require(set(timings) == {f"{p}_us" for p in PHASES}, "Rust phases mismatch")
                    observations[(case["name"], sample)] = {p: timings[f"{p}_us"] / 1000 for p in PHASES}
            enc = report["group_encoding_bytes"]
            require(enc == {"g1_compressed_bytes": 48, "g2_compressed_bytes": 96, "scalar_bytes": 32},
                    "Rust serialized element widths mismatch")
            cipher_bytes = enc["g2_compressed_bytes" if orientation == "swapped" else "g1_compressed_bytes"]
            public_bytes = enc["g1_compressed_bytes" if orientation == "swapped" else "g2_compressed_bytes"]
        else:
            log = (out / run["log"]).read_text()
            correctness_lines = [line for line in log.splitlines() if line.startswith("correctness=")]
            expected_correctness = {
                f"correctness=passed layout={total // batch}x{batch} cache={cache} "
                f"iterations={manifest['samples_per_round'] + manifest['warmup_per_case']}"
                for cache in ("cold", "cached")}
            require(len(correctness_lines) == 2 and set(correctness_lines) == expected_correctness,
                    "MCL correctness log incomplete or wrong layout/cache/iterations")
            metadata = log_fields(log, "metadata")
            for key, expected_value in {"curve": curve, "simd": mode, "orientation": orientation,
                                        "mcl_curve": "BN_SNARK1" if curve == "bn254" else "BLS12_381"}.items():
                require(metadata[key] == expected_value, f"MCL {key} identity mismatch")
            for key, expected_value in {"N": profile["N"], "W": profile["W"], "q": profile["q"], "tau": profile["tau"],
                                        "W_T": profile["accepted_weight"], "samples": manifest["samples_per_round"],
                                        "warmup": manifest["warmup_per_case"], "t": profile["q"] - 1,
                                        "total": total, "batch": batch, "setup": total, "chunks": total // batch,
                                        "threads": manifest["threads"], "share_count": profile["tau"] * (total // batch),
                                        "setup_avx_msm": 0, "setup_avx_each": 0}.items():
                require(int(metadata[key]) == expected_value, f"MCL {key} mismatch")
            require(math.isfinite(float(metadata["setup_ms"])) and float(metadata["setup_ms"]) >= 0,
                    "MCL invalid setup timing")
            size_fields = log_fields(log, "sizes")
            cipher_bytes, public_bytes = int(size_fields["cipher_group_bytes"]), int(size_fields["public_group_bytes"])
            g1_bytes, g2_bytes, target_bytes = (32, 64, 384) if curve == "bn254" else (48, 96, 576)
            expected_cipher, expected_public = (g2_bytes, g1_bytes) if orientation == "swapped" else (g1_bytes, g2_bytes)
            core_points, verification_points = (2 * total - 1) * profile["W"], total * profile["N"]
            for key, expected_value in {
                "g1_bytes": g1_bytes, "g2_bytes": g2_bytes, "scalar_bytes": 32,
                "target_full_fp12_bytes": target_bytes, "cipher_group_bytes": expected_cipher,
                "public_group_bytes": expected_public, "core_points": core_points,
                "verification_points": verification_points, "public_array_points": core_points + verification_points,
                "public_array_compressed_bytes": (core_points + verification_points) * expected_public,
                "proof_payload_bytes": expected_cipher + 32,
                "ciphertext_payload_bytes": 2 * expected_cipher + target_bytes + 32,
                "share_payload_bytes": expected_cipher,
                "validator_share_payload_bytes": (total // batch) * expected_cipher,
                "all_share_payload_bytes": profile["tau"] * (total // batch) * expected_cipher,
            }.items():
                require(int(size_fields[key]) == expected_value, f"MCL serialized {key} mismatch")
            if mode == "avx512":
                # The pinned MCL source dispatches G1 mulVec at n>=128. In this
                # matrix B=2 (FFT length 4), tau<128, and ciphertexts are G2:
                # only preparation can dispatch, with no mulEach calls.
                require(curve == "bls12_381" and orientation == "swapped" and batch == 2 and profile["tau"] < 128,
                        "AVX count oracle requires the known B2 swapped workload")
                expected_preparation_msm = ((batch - 1) * int(profile["accepted_weight"] >= 128)
                                            + batch * sum(weight >= 128 for weight in selected_weights[error]))
            with (out / run["file"]).open(newline="") as handle:
                for row in csv.DictReader(handle):
                    for key, expected_value in {"curve": curve, "simd": mode, "orientation": orientation,
                                                "total": total, "batch": batch, "setup": total, "threads": manifest["threads"],
                                                "parties": profile["N"], "total_weight": profile["W"], "required_weight": profile["q"],
                                                "selected_parties": profile["tau"], "accepted_weight": profile["accepted_weight"]}.items():
                        require(row[key] == str(expected_value), f"MCL CSV {key} mismatch")
                    key, phase = (row["cache"], int(row["sample"])), row["phase"]
                    require(phase in PHASES and phase not in observations[key], "duplicate/unknown MCL phase")
                    observations[key][phase] = float(row["milliseconds"])
                    calls = (int(row["avx_msm"]), int(row["avx_each"]))
                    require(min(calls) >= 0, "negative MCL AVX callback count")
                    expected_msm = (expected_preparation_msm if mode == "avx512" and key[0] == "cold"
                                    and phase in {"preparation", "combiner", "decryption_total", "end_to_end"} else 0)
                    require(calls == (expected_msm, 0), f"MCL unexpected AVX callbacks for {key}/{phase}")
                    counters[(*key, phase)] = calls
        require(set(observations) == {(cache, s) for cache in ("cold", "cached") for s in range(manifest["samples_per_round"])},
                "missing or duplicate sample indices")
        size_record = {"cipher_group_bytes": cipher_bytes, "public_group_bytes": public_bytes,
                       "core_key_KiB": (2 * total - 1) * profile["W"] * public_bytes / 1024,
                       "verification_key_KiB": total * profile["N"] * public_bytes / 1024,
                       "minimal_chunk_key_KiB": ((2 * batch - 1) * profile["W"] + batch * profile["N"]) * public_bytes / 1024,
                       "validator_response_bytes": total // batch * cipher_bytes}
        size_record["total_key_KiB"] = size_record["core_key_KiB"] + size_record["verification_key_KiB"]
        size_key = (name, error, total)
        require(size_key not in sizes or sizes[size_key] == size_record, "serialization sizes changed across rounds")
        sizes[size_key] = size_record
        for (cache, sample), timings in sorted(observations.items()):
            require(set(timings) == set(PHASES), "incomplete phase set")
            require(all(math.isfinite(v) and v >= 0 for v in timings.values()), "nonfinite/negative timing")
            require(timings["end_to_end"] >= timings["decryption_total"] >= timings["combiner"], "nested timer mismatch")
            require(sum(timings[p] for p in ("share_generation", "acceptance", "preparation", "precompute", "opening"))
                    <= timings["decryption_total"] + 0.00001, "phase sums exceed total")
            timings.update({"dec_open": timings["precompute"] + timings["opening"],
                            "encryption_per_item": timings["encryption"] / total,
                            "predec_per_validator": timings["share_generation"] / profile["tau"],
                            "validate_per_validator": timings["acceptance"] / profile["tau"]})
            for phase, value in timings.items():
                values[(name, error, total, cache, phase)].append(value)
                raw.append({"config": name, "error": error, "total": total, "batch": batch, "cache": cache,
                            "round": run["round"], "sample": sample, "phase": phase, "ms": value})
                if backend == "mcl" and phase in PHASES:
                    simd[(name, error, total, cache, phase)].append(counters[(cache, sample, phase)])
    rows = []
    for (name, error, total, cache, phase), timings in sorted(values.items()):
        require(len(timings) == manifest["samples_per_round"] * manifest["rounds"], "unequal sample counts")
        row = {"config": name, "error": error, "total": total, "cache": cache, "phase": phase,
               **profiles[error], "samples": len(timings), "mean_ms": statistics.fmean(timings),
               "median_ms": statistics.median(timings), "p10_ms": quantile(timings, .1), "p90_ms": quantile(timings, .9)}
        if (name, error, total, cache, phase) in simd:
            calls = simd[(name, error, total, cache, phase)]
            row.update(avx_msm_min=min(c[0] for c in calls), avx_msm_max=max(c[0] for c in calls),
                       avx_each_min=min(c[1] for c in calls), avx_each_max=max(c[1] for c in calls))
        rows.append(row)
    size_rows = [{"config": key[0], "error": key[1], "total": key[2], **value} for key, value in sorted(sizes.items())]
    for name, entries in (("optimized_summary.csv", rows), ("optimized_raw.csv", raw), ("key_sizes.csv", size_rows)):
        columns = list(dict.fromkeys(k for row in entries for k in row))
        with (out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(entries)
    (out / "optimized_summary.json").write_text(json.dumps({"profiles": profiles, "rows": rows, "sizes": size_rows}, indent=2) + "\n")
    print(f"Validated {len(manifest['runs'])} runs, {len(rows)} phase groups, {len(raw)} sample observations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    summarize(parser.parse_args().out.resolve())
