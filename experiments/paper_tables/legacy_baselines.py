#!/usr/bin/env python3
"""Run unchanged legacy paper examples serially, preserving their timing semantics.

These are source-defined measurements, not fresh-input aggregate wall-time peers
of the optimized WBTX runner. See legacy_manifest.json for every denominator,
reuse boundary, and the meaning of the reported/derived CSV rows. Rust prints
only three decimal places: parser-derived values cannot recover lost precision.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
ERRORS = ("1/8", "1/16", "1/32", "1/64")
POINTS = [(error, 16) for error in ERRORS] + [("1/16", size) for size in (32, 64, 128, 256)]
CRATES = {"wpfe": "weighted_pfe", "pfe": "pfe", "btx": "btx", "beat": "weighted_indexed_bte"}
PREFIXES = {"wpfe": "WEIGHTED_PFE", "pfe": "PFE", "btx": "BTX", "beat": "WEIGHTED_INDEXED_BTE"}
# Maps exact stdout labels to distinct CSV phases. Totals retain the source's
# definition; they are not all assigned the generic label "decryption".
LABELS = {
    "wpfe": {
        "trusted weighted key generation": "setup",
        "encrypt batch": "encryption",
        "fixed Cauchy FFT kernel": "fixed_kernel",
        "ctxtCheck(B), batched client proofs": "validation",
        "PreDec(B), all selected parties": "predec_selected",
        "batched response acceptance": "acceptance",
        "committee interpolation / G2 MSMs": "preparation",
        "ciphertext-dependent Cauchy precompute": "precompute",
        "weighted opening / unmask": "opening",
        "full cold-committee path incl checks": "cold_with_validation",
        "full cached-committee path incl checks": "cached_with_validation",
    },
    "beat": {
        "powers-of-tau CRS setup": "crs_setup",
        "weighted master key generation": "setup",
        "fixed G2 FFT kernel": "fixed_kernel",
        "encrypt batch": "encryption",
        "client proof validation": "validation",
        "partial decryptions, all N parties": "predec_all_validators",
        "all-N server proofs / select V": "acceptance",
        "committee interpolation / G2 MSMs": "preparation",
        "cross-term precompute": "precompute",
        "weighted PRF opening / unmask": "opening",
        "post-validation cold-committee path": "cold_without_validation",
        "post-validation cached-committee path": "cached_without_validation",
        "full cold decryption incl client checks": "cold_with_validation",
        "full cached decryption incl client checks": "cached_with_validation",
    },
}
for _scheme, _kernel, _opening in [
    ("btx", "fixed G2 FFT kernel", "open(B)"),
    ("pfe", "fixed PFE FFT/G2 kernel", "open(B), fused 3-pair path"),
]:
    LABELS[_scheme] = {
        "trusted key generation": "setup", "encrypt batch": "encryption",
        _kernel: "fixed_kernel", "ctxtCheck(B)": "validation",
        "precompute(B)": "precompute",
        "partialDecrypt(B), one server": "predec_one_virtual_server",
        "combine(n)": "combine", "serverCheck(B,n)": "server_check",
        _opening: "opening", "core sequential total": "core_one_server_total",
        "robust sequential total": "robust_one_server_total",
    }

SEMANTICS = {
    "wpfe": {
        "inputs": "One fresh setup, encrypted batch and fixed kernel per process; one online warmup, then SAMPLES online repetitions reuse that batch. Ciphertext validation, selected-validator shares, acceptance, committee preparation, cross terms and opening rerun each repetition.",
        "samples": "Online phase rows are an internal mean of SAMPLES repetitions. Setup/encryption/fixed_kernel rows are one observation per process, not SAMPLES observations.",
        "predec_denominator": "selected real validators tau; reported wall time for all selected validators divided by tau, without multiplying by threads",
        "acceptance_denominator": "selected real validators tau; batched acceptance wall time divided by tau",
        "totals": "cold_with_validation includes validation+predec_selected+acceptance+preparation+precompute+opening. cached_with_validation excludes preparation. dec_total_without_validation is a derived WBTX-style sum of the same phases except validation. dec_open is precompute+opening.",
    },
    "btx": {
        "inputs": "One full warmup and SAMPLES full fresh runs per process, including fresh setup/encryption/kernel. Generates q virtual shares serially, then normalizes their wall time by q to a one-server average.",
        "predec_denominator": "Reported total is already divided by q; printed ms/item divides again by batch size. This is time per ciphertext per virtual server, not per real weighted validator.",
        "totals": "core_one_server_total=precompute+one-server PreDec+combine+opening; robust_one_server_total adds client and combined-share checks. Both omit setup/encryption/fixed kernel and omit (q-1) servers' share-generation work. all_virtual_shares_derived and all_local_*_derived multiply the rounded one-server time by q; these are approximate reconstructions, not direct wall timers.",
    },
    "pfe": {
        "inputs": "One full warmup and SAMPLES full fresh runs per process, including fresh setup/encryption/kernel. Generates q virtual shares serially, then normalizes their wall time by q to a one-server average.",
        "predec_denominator": "Reported total is already divided by q; printed ms/item divides again by batch size. This is time per ciphertext per virtual server, not per real weighted validator.",
        "totals": "core_one_server_total=precompute+one-server PreDec+combine+opening; robust_one_server_total adds client and combined-share checks. Both omit setup/encryption/fixed kernel and omit (q-1) servers' share-generation work. all_virtual_shares_derived and all_local_*_derived multiply the rounded one-server time by q; these are approximate reconstructions, not direct wall timers.",
    },
    "beat": {
        "inputs": "One fresh setup/encrypted batch per process, with no upstream warmup. Invoke SAMPLES independent processes per case/round. All N positive-weight validators generate shares and undergo response checks; only the selected committee is retained for opening.",
        "predec_denominator": "All N real positive-weight validators; reported parallel wall time divided by N",
        "totals": "cold_without_validation=predec_all_validators+acceptance+preparation+precompute+opening. cold_with_validation adds client checks; cached variants omit preparation. Setup/encryption/fixed kernel remain outside these totals.",
        "security_note": "The runner reports that its old-paper robustness theorem condition t<floor(W/2) does not hold for the two-thirds threshold; preserve that diagnostic. Measurements are not a security claim.",
    },
}
FIELDS = ["scheme", "error", "total", "round", "sample", "samples", "N", "source_N", "W", "q", "t",
          "selected_parties", "selected_weight", "threads", "phase", "value", "ms", "unit",
          "statistic", "origin", "denominator", "source_label", "log"]
TIMING = re.compile(r"^\s*(.*?)\s+([0-9]+(?:\.[0-9]+)?)\s+ms(?:\s+total\s+([0-9]+(?:\.[0-9]+)?)\s+ms/item)?\s*$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_profiles(path):
    document = json.loads(path.read_text())
    require(document.get("target_reconstruction_ratio") == "2/3", "weights file must target reconstruction ratio 2/3")
    profiles = {}
    for error in ERRORS:
        matches = [p for p in document["allocations"] if p.get("error") == error]
        require(len(matches) == 1, f"expected one profile for {error}")
        p = matches[0]
        w = p["weights"]
        require(w and all(type(v) is int and v > 0 for v in w), f"{error}: positive integer weights required")
        require(sum(w) == p["share_count"] and len(w) == p["positive_validator_count"], f"{error}: weight metadata mismatch")
        require(type(p["reconstruction_threshold"]) is int and 0 < p["reconstruction_threshold"] <= sum(w), f"{error}: invalid threshold")
        selected, weight = [], 0
        for party in sorted(range(len(w)), key=lambda i: (-w[i], i)):
            selected.append(party)
            weight += w[party]
            if weight >= p["reconstruction_threshold"]:
                break
        profiles[error] = {"N": len(w), "W": sum(w), "q": p["reconstruction_threshold"],
                           "selected_parties": len(selected), "selected_weight": weight}
    return profiles


def parse_output(text, scheme, total, profile, threads, repetitions):
    """Validate successful stdout and return typed metric rows (no subprocesses)."""
    require("Decryption successful" in text, "runner did not report successful recovery")
    expected = LABELS[scheme]
    parsed = {}
    for line in text.splitlines():
        match = TIMING.match(line)
        if match and match[1] in expected:
            phase = expected[match[1]]
            require(phase not in parsed, f"duplicate timing phase {phase}")
            parsed[phase] = (float(match[2]), float(match[3]) if match[3] else None, match[1])
    require(set(parsed) == set(expected.values()), f"incomplete {scheme} timing output; missing {set(expected.values())-set(parsed)}")
    virtual = scheme in {"btx", "pfe"}
    if virtual:
        shape = re.search(r"B_max=(\d+), B=(\d+), N=(\d+), t=(\d+), shares=(\d+), Rayon threads=(\d+)", text)
        require(shape and tuple(map(int, shape.groups())) == (total, total, profile["W"], profile["q"] - 1, profile["q"], threads), "virtualized runtime parameters mismatch")
        require(re.search(rf"[Mm]easured repetitions={repetitions},", text), "runtime repetitions mismatch")
    else:
        require(f"B_max=B={total}" in text and f"Rayon threads={threads}" in text, "weighted runtime size/thread mismatch")
        require(re.search(rf"real parties N={profile['N']}, total virtual weight W={profile['W']},", text), "weighted runtime allocation mismatch")
        require(f"minimum reconstruction weight q={profile['q']};" in text, "weighted runtime threshold mismatch")
        committee = re.search(r"minimal-party authorized set: (?:tau=)?(\d+) parties, (?:W_T=|weight )(\d+) >= q=(\d+)", text)
        require(committee and tuple(map(int, committee.groups())) == (profile["selected_parties"], profile["selected_weight"], profile["q"]), "weighted selected committee mismatch")
        if scheme == "wpfe":
            require(f"measured online repetitions={repetitions}," in text, "WPFE runtime repetitions mismatch")
    metrics = []

    def add(phase, value, unit="ms", origin="reported", source_label="", denominator="", samples=None):
        require(math.isfinite(value) and value >= 0, f"invalid metric {phase}")
        n = repetitions if samples is None else samples
        metrics.append({"phase": phase, "value": value, "ms": value if unit in {"ms", "ms/item"} else "", "unit": unit,
                        "origin": origin, "source_label": source_label, "denominator": denominator, "samples": n,
                        "statistic": "single_observation" if n == 1 else "internal_arithmetic_mean"})

    for phase, (milliseconds, per_item, label) in parsed.items():
        n = 1 if scheme == "beat" or (scheme == "wpfe" and phase in {"setup", "encryption", "fixed_kernel"}) else repetitions
        add(phase, milliseconds, source_label=label, samples=n)
        if per_item is not None:
            denominator = profile["selected_parties"] if phase in {"predec_selected", "acceptance"} else total
            if phase == "predec_all_validators": denominator = profile["N"]
            if phase == "combine": denominator = profile["q"]
            # Both stdout values are rounded independently to 0.001 ms.
            require(abs(milliseconds / denominator - per_item) <= .000501 + .000501 / denominator,
                    f"{phase}: unexpected per-item denominator")
            add(phase + "_per_item", per_item, "ms/item", source_label=label, denominator=denominator, samples=n)
    value = lambda phase: parsed[phase][0]
    if scheme in {"wpfe", "beat"}:
        add("dec_open", value("precompute") + value("opening"), origin="sum_of_rounded_phase_means")
        predec = "predec_selected" if scheme == "wpfe" else "predec_all_validators"
        cold = sum(value(p) for p in [predec, "acceptance", "preparation", "precompute", "opening"])
        add("dec_total_without_validation", cold, origin="sum_of_rounded_phase_means")
        add("dec_cached_without_validation", cold - value("preparation"), origin="sum_of_rounded_phase_means")
        if scheme == "wpfe":
            for phase, label in [("key_core_bytes", "core D1/D2/global material"),
                                 ("key_verification_bytes", "party verification keys"),
                                 ("key_decryption_total_bytes", "public decryption key")]:
                size = re.search(rf"^{re.escape(label)}: \d+ G2 points, (\d+) bytes", text, re.MULTILINE)
                require(size is not None, f"missing serialized size {label}")
                add(phase, int(size[1]), "bytes", source_label=label, samples=1)
            sizes = {r["phase"]: r["value"] for r in metrics if r["unit"] == "bytes"}
            require(sizes["key_core_bytes"] == (2 * profile["W"] * total + 1) * 96, "WPFE core key size mismatch")
            require(sizes["key_verification_bytes"] == profile["N"] * total * 96, "WPFE verification key size mismatch")
            require(sizes["key_decryption_total_bytes"] == sizes["key_core_bytes"] + sizes["key_verification_bytes"], "WPFE total key size mismatch")
    else:
        all_shares = value("predec_one_virtual_server") * profile["q"]
        core = all_shares + value("precompute") + value("combine") + value("opening")
        add("all_virtual_shares_derived", all_shares, origin="q_times_rounded_one_server_mean")
        add("all_local_core_derived", core, origin="sum_with_q_times_rounded_one_server_mean")
        add("all_local_robust_derived", core + value("validation") + value("server_check"), origin="sum_with_q_times_rounded_one_server_mean")
    return metrics


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--cpus", default="0-11")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--schemes", nargs="+", choices=CRATES, default=list(CRATES))
    parser.add_argument("--dry-run", action="store_true", help="write the planned manifest without executing binaries")
    parser.add_argument("--parse-only", action="store_true", help="parse existing matching stdout logs without executing binaries")
    args = parser.parse_args()
    if min(args.threads, args.samples, args.rounds) < 1 or len(set(args.schemes)) != len(args.schemes):
        parser.error("positive threads/samples/rounds and distinct schemes required")
    if args.dry_run and args.parse_only:
        parser.error("--dry-run and --parse-only are mutually exclusive")
    weights = args.weights_file.resolve()
    profiles = load_profiles(weights)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "legacy_manifest.json"
    previous = None
    if args.parse_only:
        require(manifest_path.is_file(), "--parse-only requires the original legacy_manifest.json")
        previous = json.loads(manifest_path.read_text())
        require(previous["threads"] == args.threads and previous["cpus"] == args.cpus
                and previous["requested_samples"] == args.samples and previous["rounds"] == args.rounds
                and previous["schemes"] == args.schemes and previous["profiles"] == profiles
                and previous["weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest(),
                "--parse-only arguments differ from the original run manifest")
    else:
        require(not manifest_path.exists() and not (out / "legacy.csv").exists(),
                "existing legacy results found; use a new output directory or --parse-only")
    try: weights_argument = str(weights.relative_to(ROOT))
    except ValueError: weights_argument = str(weights)
    if not args.dry_run and not args.parse_only:
        require(shutil.which("taskset") is not None, "taskset is required for pinned measurements")
        for scheme in args.schemes:
            binary = ROOT / CRATES[scheme] / "target/release/examples/paper_reproduction"
            require(binary.is_file() and os.access(binary, os.X_OK), f"build the release example first: {binary}")
    source_paths = [Path(__file__).resolve()]
    for scheme in args.schemes:
        crate = ROOT / CRATES[scheme]
        source_paths += list((crate / "src").rglob("*.rs")) + [crate / "Cargo.toml", crate / "Cargo.lock", crate / "examples/paper_reproduction.rs"]
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "planned", "threads": args.threads, "cpus": args.cpus,
        "requested_samples": args.samples, "rounds": args.rounds, "schemes": args.schemes,
        "weights_file": weights.name, "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "profiles": profiles, "semantics": {s: SEMANTICS[s] for s in args.schemes},
        "notes": ["All subprocesses execute serially; odd/even rounds reverse scheme and parameter order.",
                  "Upstream stdout is rounded to 0.001 ms; no per-iteration distribution can be recovered from internally averaged output.",
                  "CSV N is actual real/virtual protocol party count; source_N is the positive-weight validator count from the source allocation.",
                  "CSV samples is the number of observations contributing to that row, so WPFE setup/encryption/kernel have samples=1.",
                  "Size rows are exact compressed-group byte counts (96 bytes per G2 point), not in-memory sizes. Their ms field is blank.",
                  "Raw stdout/stderr are retained. No networking, new optimizations, or protocol changes are introduced."],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(source_paths)) if p.is_file()},
        "binary_sha256": {}, "runs": [],
    }
    for scheme in args.schemes:
        binary = ROOT / CRATES[scheme] / "target/release/examples/paper_reproduction"
        if binary.is_file(): manifest["binary_sha256"][str(binary.relative_to(ROOT))] = hashlib.sha256(binary.read_bytes()).hexdigest()
    if previous is not None:
        # Re-parsing must not relabel old measurements with today's source or
        # executable hashes. Keep the original measured provenance intact.
        for key in ["created_utc", "source_sha256", "binary_sha256"]:
            manifest[key] = previous[key]
        manifest["parsed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest["parser_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    jobs = []
    for round_id in range(1, args.rounds + 1):
        schemes = args.schemes if round_id % 2 else list(reversed(args.schemes))
        points = POINTS if round_id % 2 else list(reversed(POINTS))
        for error, total in points:
            for scheme in schemes:
                if scheme != "wpfe" and total != 16: continue
                repeats = args.samples if scheme == "beat" else 1
                for sample in range(1, repeats + 1):
                    jobs.append((scheme, error, total, round_id, sample if scheme == "beat" else 0))
    rows = []
    try:
        for index, (scheme, error, total, round_id, sample) in enumerate(jobs, 1):
            profile = profiles[error]
            crate, prefix = CRATES[scheme], PREFIXES[scheme]
            repetitions = 1 if scheme == "beat" else args.samples
            env = {f"{prefix}_BATCH_SIZE": str(total), f"{prefix}_THREADS": str(args.threads)}
            command = [f"{crate}/target/release/examples/paper_reproduction"]
            if scheme in {"wpfe", "beat"}:
                env[f"{prefix}_WEIGHTS_FILE"] = weights_argument
                command += ["--approximation-error", error]
            else:
                env.update({f"{prefix}_SERVERS": str(profile["W"]), f"{prefix}_THRESHOLD": str(profile["q"] - 1)})
            if scheme != "beat": env[f"{prefix}_REPETITIONS"] = str(repetitions)
            command = ["taskset", "-c", args.cpus] + command
            stem = f"{scheme}_e{error.split('/')[1]}_m{total}_r{round_id}" + (f"_s{sample:03d}" if sample else "")
            stdout_path, stderr_path = out / (stem + ".stdout.log"), out / (stem + ".stderr.log")
            run = {"scheme": scheme, "error": error, "total": total, "round": round_id, "sample": sample,
                   "repetitions": repetitions, "stdout": stdout_path.name, "stderr": stderr_path.name,
                   "command": command, "environment_overrides": env, "status": "planned"}
            manifest["runs"].append(run)
            print(f"[{index}/{len(jobs)}] round={round_id} scheme={scheme} error={error} messages={total} sample={sample or 'internal-mean'} repetitions={repetitions}", flush=True)
            atomic_json(manifest_path, manifest)
            if args.dry_run: continue
            if not args.parse_only:
                require(not stdout_path.exists() and not stderr_path.exists(), f"refusing to overwrite existing run logs for {stem}; use another output directory or --parse-only")
                run["status"] = "running"
                atomic_json(manifest_path, manifest)
                started = time.monotonic()
                with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                    result = subprocess.run(command, cwd=ROOT, env={**os.environ, **env}, stdout=stdout, stderr=stderr)
                run["process_elapsed_seconds"] = time.monotonic() - started
                run["returncode"] = result.returncode
                require(result.returncode == 0, f"{stem} failed; inspect {stderr_path}")
            metrics = parse_output(stdout_path.read_text(), scheme, total, profile, args.threads, repetitions)
            virtual = scheme in {"btx", "pfe"}
            context = {"scheme": scheme, "error": error, "total": total, "round": round_id, "sample": sample,
                       "N": profile["W"] if virtual else profile["N"], "source_N": profile["N"],
                       "W": profile["W"], "q": profile["q"], "t": profile["q"] - 1,
                       "selected_parties": profile["q"] if virtual else profile["selected_parties"],
                       "selected_weight": profile["q"] if virtual else profile["selected_weight"],
                       "threads": args.threads, "log": stdout_path.name}
            rows.extend({**context, **metric} for metric in metrics)
            run["status"] = "verified"
            run["metric_rows"] = len(metrics)
            run["stdout_sha256"] = hashlib.sha256(stdout_path.read_bytes()).hexdigest()
            save_csv(out / "legacy.csv", rows)
            atomic_json(manifest_path, manifest)
            print(f"  verified {len(metrics)} metrics", flush=True)
        manifest["status"] = "planned" if args.dry_run else "complete"
        manifest["metric_rows"] = len(rows)
        atomic_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        if manifest["runs"]: manifest["runs"][-1]["status"] = "failed"
        atomic_json(manifest_path, manifest)
        raise
    print(f"{'Planned' if args.dry_run else 'Completed'} {len(jobs)} serial invocations; {len(rows)} metric rows", flush=True)


if __name__ == "__main__":
    main()
