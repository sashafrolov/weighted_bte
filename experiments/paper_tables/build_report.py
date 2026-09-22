#!/usr/bin/env python3
"""Build readable paper-style tables from validated optimized and legacy results."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from legacy_baselines import FIELDS, load_profiles, parse_output
from run_optimized import CONFIGS, POINTS, ROOT
from summarize_optimized import PHASES, quantile

NAMES = {
    "blst_original": "BLST original",
    "blst_swapped": "BLST swapped",
    "bls_avx_swapped_split2": "BLS AVX swapped / 2",
    "bn_swapped_split2": "BN254 swapped / 2",
    "bn_original_split2": "BN254 original / 2",
    "wpfe": "WPFE (unchanged)",
}
ORDER = list(NAMES)
DERIVED_PHASES = ("dec_open", "encryption_per_item", "predec_per_validator", "validate_per_validator")
EXPECTED_PROFILES = {
    "1/8": dict(N=234, W=764, q=512, tau=59, accepted_weight=515),
    "1/16": dict(N=419, W=1580, q=1060, tau=76, accepted_weight=1062),
    "1/32": dict(N=598, W=3063, q=2044, tau=79, accepted_weight=2046),
    "1/64": dict(N=688, W=6489, q=4327, tau=79, accepted_weight=4333),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path, fields=None):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if fields is not None:
            require(reader.fieldnames == list(fields), f"{path}: unexpected CSV schema")
        return list(reader)


def load_publication_data(results):
    """Require the exact measured matrix and observation counts claimed below.

    The optimized summarizer validates protocol metadata and SIMD callbacks.
    Here we additionally bind its means to every unique raw observation. Legacy
    stdout hashes and a fresh parse bind CSV values and counts to original logs.
    """
    optimized_dir = results / "optimized"
    manifest = json.loads((optimized_dir / "optimized_manifest.json").read_text())
    for key, expected in dict(threads=12, cpus="0-11", rounds=2, samples_per_round=15,
                              warmup_per_case=2).items():
        require(manifest[key] == expected, f"optimized publication setting {key} must be {expected}")
    require(manifest["configurations"] == {name: list(config) for name, config in CONFIGS.items()},
            "optimized publication requires all five configurations")
    weights_path = ROOT / manifest["weights"]
    weights_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    require(weights_hash == manifest["weight_sha256"], "optimized weight checksum mismatch")
    legacy_profiles = load_profiles(weights_path)
    profiles = {e: {"N": p["N"], "W": p["W"], "q": p["q"], "tau": p["selected_parties"],
                    "accepted_weight": p["selected_weight"]} for e, p in legacy_profiles.items()}
    require(profiles == EXPECTED_PROFILES, "publication profiles differ from the stated table captions")
    expected_runs = {(name, e, total, r) for name in CONFIGS for e, total in POINTS for r in (1, 2)}
    actual_runs, run_files = set(), set()
    for run in manifest["runs"]:
        identity = (run["config"], run["error"], run["total"], run["round"])
        require(identity in expected_runs and identity not in actual_runs, "unknown or duplicate optimized run")
        actual_runs.add(identity)
        batch = CONFIGS[run["config"]][-1] or run["total"]
        require(run["setup"] == run["total"] and run["batch"] == batch, "optimized setup/layout mismatch")
        for field in ("file", "log"):
            name = run[field]
            require(name not in run_files and (optimized_dir / name).is_file(), "missing or duplicate optimized raw file")
            run_files.add(name)
    require(actual_runs == expected_runs, "incomplete optimized publication matrix")
    optimized = json.loads((optimized_dir / "optimized_summary.json").read_text())
    require(optimized["profiles"] == profiles, "optimized summary profiles mismatch")
    expected_groups = {(name, e, total, cache, phase) for name in CONFIGS for e, total in POINTS
                       for cache in ("cold", "cached") for phase in (*PHASES, *DERIVED_PHASES)}
    groups = {}
    base_fields = {"config", "error", "total", "cache", "phase", *next(iter(profiles.values())),
                   "samples", "mean_ms", "median_ms", "p10_ms", "p90_ms"}
    for row in optimized["rows"]:
        key = tuple(row[k] for k in ("config", "error", "total", "cache", "phase"))
        require(key in expected_groups and key not in groups, "unknown or duplicate optimized summary group")
        fields = base_fields | ({"avx_msm_min", "avx_msm_max", "avx_each_min", "avx_each_max"}
                               if CONFIGS[key[0]][0] == "mcl" and key[-1] in PHASES else set())
        require(set(row) == fields, f"optimized summary schema mismatch: {key}")
        require(row["samples"] == 30 and all(row[k] == v for k, v in profiles[key[1]].items()),
                "optimized sample count or profile mismatch")
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in ("mean_ms", "median_ms", "p10_ms", "p90_ms")),
                "nonfinite or negative optimized summary")
        groups[key] = row
    require(set(groups) == expected_groups, "incomplete optimized summary: expected 1,120 groups")
    observations, seen = defaultdict(list), set()
    raw_fields = ("config", "error", "total", "batch", "cache", "round", "sample", "phase", "ms")
    for row in read_csv(optimized_dir / "optimized_raw.csv", raw_fields):
        key = (row["config"], row["error"], int(row["total"]), row["cache"], row["phase"])
        identity = (*key, int(row["round"]), int(row["sample"]))
        require(key in expected_groups and identity not in seen and identity[-2] in (1, 2)
                and 0 <= identity[-1] < 15, "unknown, duplicate or out-of-range optimized observation")
        require(int(row["batch"]) == (CONFIGS[key[0]][-1] or key[2]), "optimized raw chunk mismatch")
        value = float(row["ms"])
        require(math.isfinite(value) and value >= 0, "invalid optimized raw time")
        seen.add(identity)
        observations[key].append(value)
    for key, summary in groups.items():
        values = observations[key]
        require(len(values) == 30, f"optimized group lacks 30 unique observations: {key}")
        for field, value in {"mean_ms": statistics.fmean(values), "median_ms": statistics.median(values),
                             "p10_ms": quantile(values, .1), "p90_ms": quantile(values, .9)}.items():
            require(math.isclose(summary[field], value, rel_tol=1e-12, abs_tol=1e-12),
                    f"optimized {field} does not match raw observations: {key}")
    expected_sizes = {(name, e, total) for name in CONFIGS for e, total in POINTS}
    seen_sizes = set()
    for row in optimized["sizes"]:
        key = (row["config"], row["error"], row["total"])
        require(key in expected_sizes and key not in seen_sizes, "unknown or duplicate key-size row")
        seen_sizes.add(key)
        _, curve, _, orientation, split = CONFIGS[key[0]]
        g1, g2 = (32, 64) if curve == "bn254" else (48, 96)
        cipher, public = (g2, g1) if orientation == "swapped" else (g1, g2)
        p, total, batch = profiles[key[1]], key[2], split or key[2]
        expected = {"config": key[0], "error": key[1], "total": total,
                    "cipher_group_bytes": cipher, "public_group_bytes": public,
                    "core_key_KiB": (2 * total - 1) * p["W"] * public / 1024,
                    "verification_key_KiB": total * p["N"] * public / 1024,
                    "minimal_chunk_key_KiB": ((2 * batch - 1) * p["W"] + batch * p["N"]) * public / 1024,
                    "validator_response_bytes": total // batch * cipher}
        expected["total_key_KiB"] = expected["core_key_KiB"] + expected["verification_key_KiB"]
        require(row == expected, "key-size formula or schema mismatch")
    require(seen_sizes == expected_sizes, "incomplete key-size matrix")

    legacy = defaultdict(list)
    for folder, schemes, samples in (("legacy_wpfe", {"wpfe"}, 15), ("legacy_prior", {"pfe", "btx", "beat"}, 5)):
        directory = results / folder
        manifest = json.loads((directory / "legacy_manifest.json").read_text())
        require(manifest["status"] == "complete", f"{folder}: incomplete manifest")
        require(len(manifest["schemes"]) == len(schemes) and set(manifest["schemes"]) == schemes,
                f"{folder}: unexpected schemes")
        for key, expected in dict(threads=12, cpus="0-11", rounds=2, requested_samples=samples,
                                  profiles=legacy_profiles, weights_sha256=weights_hash).items():
            require(manifest[key] == expected, f"{folder}: mismatched {key}")
        expected_runs = {(scheme, e, total, r, sample) for scheme in schemes for e, total in POINTS
                         if scheme == "wpfe" or total == 16 for r in (1, 2)
                         for sample in (range(1, samples + 1) if scheme == "beat" else (0,))}
        rows = read_csv(directory / "legacy.csv", FIELDS)
        require(len(rows) == manifest["metric_rows"], f"{folder}: manifest/CSV row count mismatch")
        by_log, identities = defaultdict(dict), set()
        for row in rows:
            identity = (row["scheme"], row["error"], int(row["total"]), int(row["round"]), int(row["sample"]), row["phase"])
            require(identity not in identities and row["phase"] not in by_log[row["log"]], f"{folder}: duplicate CSV row")
            identities.add(identity)
            by_log[row["log"]][row["phase"]] = row
        actual_runs, logs, total_rows = set(), set(), 0
        phase_samples = defaultdict(int)
        for run in manifest["runs"]:
            scheme, e, total, r, sample = identity = tuple(run[k] for k in ("scheme", "error", "total", "round", "sample"))
            require(identity in expected_runs and identity not in actual_runs and run["status"] == "verified",
                    f"{folder}: incomplete, unknown or duplicate run")
            actual_runs.add(identity)
            repetitions = 1 if scheme == "beat" else samples
            require(run["repetitions"] == repetitions, f"{folder}: repetition mismatch")
            log = run["stdout"]
            require(log not in logs, f"{folder}: duplicate raw stdout filename")
            logs.add(log)
            content = (directory / log).read_bytes()
            require(hashlib.sha256(content).hexdigest() == run["stdout_sha256"], f"{folder}: raw stdout checksum mismatch")
            require((directory / run["stderr"]).is_file(), f"{folder}: missing stderr log")
            metrics = parse_output(content.decode(), scheme, total, legacy_profiles[e], 12, repetitions)
            require(len(metrics) == run["metric_rows"] == {"wpfe": 24, "beat": 22, "btx": 21, "pfe": 21}[scheme],
                    f"{folder}: raw metric count mismatch")
            require(set(by_log[log]) == {metric["phase"] for metric in metrics}, f"{folder}: CSV/raw phases mismatch")
            p, virtual = legacy_profiles[e], scheme in {"pfe", "btx"}
            context = {"scheme": scheme, "error": e, "total": total, "round": r, "sample": sample,
                       "N": p["W"] if virtual else p["N"], "source_N": p["N"], "W": p["W"], "q": p["q"], "t": p["q"] - 1,
                       "selected_parties": p["q"] if virtual else p["selected_parties"],
                       "selected_weight": p["q"] if virtual else p["selected_weight"], "threads": 12, "log": log}
            for metric in metrics:
                expected = {k: str(v) for k, v in {**context, **metric}.items()}
                require(by_log[log][metric["phase"]] == expected, f"{folder}: CSV differs from parsed stdout: {log}/{metric['phase']}")
                key = (scheme, e, total, metric["phase"])
                legacy[key].append((metric["value"], metric["samples"]))
                phase_samples[key] += metric["samples"]
            total_rows += len(metrics)
        require(actual_runs == expected_runs and set(by_log) == logs and total_rows == len(rows),
                f"{folder}: incomplete or extra legacy matrix/CSV rows")
        for (scheme, e, total, phase), count in phase_samples.items():
            expected = (2 if phase in {"setup", "encryption", "encryption_per_item", "fixed_kernel"}
                        or phase.startswith("key_") else 30) if scheme == "wpfe" else 10
            require(count == expected, f"{folder}: wrong observation count for {scheme}/{e}/{total}/{phase}")
    old = {k: sum(v * n for v, n in entries) / sum(n for _, n in entries) for k, entries in legacy.items()}
    return optimized, old


def f(value, digits=3):
    return f"{value:,.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = args.results
    optimized, old = load_publication_data(results)
    p = optimized["profiles"]
    metrics = {(r["config"], r["error"], r["total"], r["cache"], r["phase"]): r["mean_ms"] for r in optimized["rows"]}
    sizes = {(r["config"], r["error"], r["total"]): r for r in optimized["sizes"]}
    def m(config, error, total, phase, cache="cold"):
        return metrics[(config, error, total, cache, phase)]

    def l(scheme, error, total, phase):
        return old[(scheme, error, total, phase)]

    sections = []

    def section(title, caption, headers, rows, notes):
        sections.append(dict(title=title, caption=caption, headers=headers, rows=rows, notes=notes))

    overview = []
    optimized_configs = [name for name, config in CONFIGS.items() if config[0] == "mcl"]

    def fastest_cold(e, total):
        return min(optimized_configs, key=lambda name: m(name, e, total, "decryption_total"))

    for e, total in POINTS:
        original = m("blst_original", e, total, "decryption_total")
        best = fastest_cold(e, total)
        improved = m(best, e, total, "decryption_total")
        overview.append([f(p[e]["W"], 0), str(total), f(original, 2), NAMES[best], f(improved, 2),
                         f(original / improved, 2) + "x", f(m("bn_original_split2", e, total, "decryption_total", "cached"), 2)])
    key_best = fastest_cold("1/16", 16)
    key_cold = m(key_best, "1/16", 16, "decryption_total")
    key_original = m("blst_original", "1/16", 16, "decryption_total")
    key_cached = m("bn_original_split2", "1/16", 16, "decryption_total", "cached")
    section("Optimized weighted BTX: paper-table reproduction",
            f"At W=1,580 and 16 messages, the fastest measured cold variant is {NAMES[key_best]} at {key_cold:.2f} ms "
            f"({key_original/key_cold:.2f}x versus the fresh original BLST baseline). Cached BN254 with original groups takes {key_cached:.2f} ms.",
            ["Weight W", "Messages M", "BLST cold ms", "Best cold variant", "Best cold ms", "Speedup", "BN original cached ms"], overview,
            ["All new measurements: AMD EPYC 9275F, 12 workers pinned to physical cores 0-11. Values are arithmetic means, not medians. WBTX: 30 fresh-input samples per cell, two reversed-order rounds, two warmups per case per round.",
             "Cold DecTotal includes all selected local validators' shares, fresh share acceptance, one committee preparation, cross terms and opening. It excludes encryption, client-proof validation, setup and networking. Cached removes only committee preparation.",
             "M is the number of messages processed. Original and swapped BLST use one batch of M; optimized MCL uses M/2 batches of two. Every run retains setup capacity L=M. The committee is the heaviest-first authorized set.",
             "Best cold selects the lowest arithmetic mean among the three optimized MCL variants at that parameter point. This identifies the fastest measured variant in this run, not statistically established dominance. The cached column always shows BN254 with original groups.",
             "Target stake threshold is 2/3. Saved inputs regenerate W=3,063 and 6,489; the PDF prints 3,060 and 6,486. The corresponding q and N match. We use actual generated weights without modifying them to force the printed totals.",
             "These are complete-implementation comparisons. BN254 changes security assumptions; its path has no AVX-512. MCL and BLST differ in validation and target-group arithmetic. WPFE and prior schemes are unmodified references.",
             "Splitting does not improve every case: cached BLS split batches are 15-20% slower than original BLST at 128/256 messages, with no AVX callbacks in those cached runs. BN254 original groups remain the fastest measured cached variant. The cold BN orientation gap at 128 messages is within observed variation."])

    def phase_rows(points):
        rows = []
        for e, total in points:
            axis = str(total) if len(points) == 5 else f(p[e]["W"], 0)
            for name in ORDER:
                if name == "wpfe":
                    tau = p[e]["tau"]
                    vals = [l(name,e,total,"predec_selected")/tau, l(name,e,total,"acceptance")/tau,
                            l(name,e,total,"preparation"), l(name,e,total,"dec_open"),
                            l(name,e,total,"dec_total_without_validation"), l(name,e,total,"dec_cached_without_validation")]
                else:
                    vals = [m(name,e,total,"predec_per_validator"), m(name,e,total,"validate_per_validator"),
                            m(name,e,total,"preparation"), m(name,e,total,"dec_open"),
                            m(name,e,total,"decryption_total"), m(name,e,total,"decryption_total","cached")]
                rows.append([axis, NAMES[name], *[f(v) for v in vals]])
        return rows

    phase_notes = ["Times in ms. PreDec and Validate are pooled wall times divided by the number of selected real validators, across all chunks. They are amortized rates, not a single validator's standalone latency. DecPrecomp is committee preparation; DecOpen is cross-term precomputation plus opening.",
                   "WBTX DecTotal is directly timed. WPFE uses 30 online repetitions over two encrypted-batch fixtures, and its comparable total is the sum of rounded phases excluding client checks. Its cached value subtracts preparation; it is not a separately timed cached run. Fresh WBTX ciphertexts and proofs change every iteration.",
                   "The paper does not fully specify its timing denominators or sample count. Its MacBook M4 measurements are preserved in paper_reference.json; this report's speedups use only fresh measurements on the dev box."]
    section("Table 5. Decryption across weight profiles", "16 messages; epsilon=1/8, 1/16, 1/32, 1/64. N=234,419,598,688; q=512,1060,2044,4327; selected validators=59,76,79,79.",
            ["Weight W", "Implementation", "PreDec / validator", "Validate / validator", "DecPrecomp", "DecOpen", "DecTotal", "Cached DecTotal"],
            phase_rows(POINTS[:4]), phase_notes)
    batch_points = [("1/16", total) for total in (16,32,64,128,256)]
    section("Table 6. Decryption across message counts", "W=1,580; N=419; q=1,060; 76 selected validators with accepted weight 1,062. All messages are recovered in each sample.",
            ["Messages M", "Implementation", "PreDec / validator", "Validate / validator", "DecPrecomp", "DecOpen", "DecTotal", "Cached DecTotal"],
            phase_rows(batch_points), phase_notes)

    def key_rows(points):
        rows = []
        for e,total in points:
            axis = str(total) if len(points) == 5 else f(p[e]["W"],0)
            for name in ORDER:
                if name == "wpfe":
                    vals = [l(name,e,total,"encryption")/total,
                            l(name,e,total,"key_core_bytes")/1024, l(name,e,total,"key_verification_bytes")/1024,
                            l(name,e,total,"key_decryption_total_bytes")/1024]
                    response = 48
                else:
                    size = sizes[(name,e,total)]
                    vals = [m(name,e,total,"encryption_per_item"), size["core_key_KiB"],size["verification_key_KiB"],size["total_key_KiB"]]
                    response = size["validator_response_bytes"]
                rows.append([axis,NAMES[name],f(vals[0]),*[f(v,1) for v in vals[1:]],f(response,0)])
        return rows
    key_notes = ["Enc is pooled encryption plus proof-generation wall time divided by M. WBTX values use cold-mode fresh samples. WPFE encrypts once per process, so its Enc mean has two observations, not 30.",
                 "Key sizes are exact compressed-group payload counts in KiB (1024 bytes), matching the paper's arithmetic despite its kB label. Core WBTX points=(2L-1)W; verification points=N*L; total is their sum. The encryption key, proof CRS, metadata and wire framing are excluded, as in Table 7.",
                 "Every timed run retains setup L=M. Public point widths are 96/48 bytes for BLS original/swapped and 64/32 bytes for BN254 original/swapped. Share widths are the opposite source group: 48/96 and 32/64 bytes. The last column includes all M/2 shares for a split run."]
    key_headers=["Weight W", "Implementation", "Enc ms/item", "Core dk KiB", "Verify keys KiB", "Total key KiB", "Share bytes / validator"]
    section("Table 7a. Encryption and keys across weights", "16 messages; setup capacity L=16 for every implementation.", key_headers, key_rows(POINTS[:4]), key_notes)
    section("Table 7b. Encryption and keys across message counts", "W=1,580; setup capacity L=M is retained even when the working chunk is two.",
            ["Messages M", *key_headers[1:]], key_rows(batch_points), key_notes)

    prior = []
    for e,total in POINTS[:4]:
        for name in ("pfe","btx","beat"):
            if name == "beat":
                predec = l(name,e,total,"predec_all_validators")/p[e]["N"]
                dec,robust = l(name,e,total,"cold_without_validation"),l(name,e,total,"cold_with_validation")
                units = "per real validator"
            else:
                predec = l(name,e,total,"predec_one_virtual_server")/total
                dec,robust = l(name,e,total,"core_one_server_total"),l(name,e,total,"robust_one_server_total")
                units = "per item / virtual server"
            prior.append([f(p[e]["W"],0),"BEAT++" if name=="beat" else name.upper(),f(l(name,e,total,"encryption")/total),
                          f(predec),f(dec),f(robust),units])
    section("Table 8. Prior-scheme reference measurements", "Unmodified implementations on the same 12-core dev-box allocation; M=16. Ten measured repetitions per point across two rounds.",
            ["Weight W","Scheme","Enc ms/item","PreDec ms / denominator","Source core Dec ms","With checks ms","PreDec denominator"],prior,
            ["These rows preserve the upstream runners' distinct meanings. BTX/PFE use q virtual parties for the selected set, generate their shares serially, divide by q for a one-server average, then divide by M for PreDec/item. Their core total includes that one-server average, precomputation, combination and opening; With checks adds client and combined-share checks.",
             "BEAT++ PreDec divides pooled share-generation time by all N real validators. Its core total includes all-N shares and response checks, selected-committee preparation, cross terms and opening. With checks also includes client-proof validation. These totals are not an identical distributed workload and should not be used for unqualified cross-scheme speedups.",
             "BTX/PFE use fresh full runs with one unmeasured warmup per process. BEAT++ uses ten independent one-run processes without an upstream warmup. Raw logs preserve its threshold/security diagnostic for this two-thirds configuration. These are performance observations, not a security claim."])

    online=[]
    for e,total in POINTS:
        s=sizes[("bn_swapped_split2",e,total)]
        online.append([f(p[e]["W"],0),str(total),f(m("blst_original",e,total,"end_to_end"),2),
                       f(m("bn_swapped_split2",e,total,"end_to_end"),2),
                       f(m("bn_original_split2",e,total,"end_to_end","cached"),2),
                       f(s["minimal_chunk_key_KiB"],1),f(s["validator_response_bytes"],0)])
    section("Full online pipeline and deployment tradeoffs", "Encryption and client-proof validation are included in these additional end-to-end measurements.",
            ["Weight W","Messages M","BLST cold ms","BN swapped cold ms","BN original cached ms","BN swapped L=2 key KiB","BN swapped share bytes / validator"],online,
            ["The online timer includes fresh encryption/proofs, client validation, all local validator shares, acceptance, optional committee preparation, cross terms and all openings. Setup and networking are excluded; all plaintexts are checked after every warmup and measured iteration.",
             "The L=2 key column is a calculated deployment option, not the setup used for the timings: constrain the setup itself to chunks of two, giving (3W+2N) compressed public-key points. This key is independent of M but cannot support a single batch larger than two.",
             "Smaller chunks increase responses: M/2 source-group elements per validator instead of one. The table reports raw group payload only. Cached measurements require the same setup, chunk size and accepted committee; all shares and verification are fresh.",
             "Measured curves: BLS12-381 and MCL BN_SNARK1 (Ethereum BN254). AVX-512 is BLS G1 only and is verified through actual callback counters. The MCL protocol is an experimental trusted-dealer, in-memory implementation with no network/wire decoder."])
    report={"title":"Weighted BTX: optimized paper tables", "subtitle":"Tables 5-8 with fresh dev-box baselines and two-thirds stake profiles", "date":"2026-09-22", "sections":sections}
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    md=["# "+report["title"],"",report["subtitle"]+". Measured "+report["date"]+".",""]
    for i,s in enumerate(sections):
        md += ["## "+s["title"],"",s["caption"],"","| "+" | ".join(s["headers"])+" |", "| "+" | ".join("---" for _ in s["headers"])+" |"]
        md += ["| "+" | ".join(row)+" |" for row in s["rows"]]
        md += [""]
        for note in s["notes"]: md += [note, ""]
        with (args.out/f"table_{i+1}.csv").open("w",newline="") as handle:
            w=csv.writer(handle,lineterminator="\n");w.writerow(s["headers"]);w.writerows(s["rows"])
    md += ["## Reproduction and raw data", "", "See [reproduction instructions](../../README.md), [validated raw results](../../../results/2026-09-22-paper-tables/optimized/optimized_summary.csv), and the original paper's [transcribed tables](../../paper_reference.json).", ""]
    (args.out/"TABLES.md").write_text("\n".join(md))
    print(f"Built {len(sections)} table sections at {args.out}")


if __name__=="__main__":
    main()
