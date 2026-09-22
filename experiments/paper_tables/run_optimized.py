#!/usr/bin/env python3
"""Run fresh-input weighted-BTX paper-table sweeps serially on one pinned CPU set."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[2]
POINTS = [(f"1/{d}", 16) for d in (8, 16, 32, 64)] + [("1/16", m) for m in (32, 64, 128, 256)]
CONFIGS = {
    "blst_original": ("blst", "bls12_381", "native", "normal", None),
    "blst_swapped": ("blst", "bls12_381", "native", "swapped", None),
    "bls_avx_swapped_split2": ("mcl", "bls12_381", "avx512", "swapped", 2),
    "bn_swapped_split2": ("mcl", "bn254", "off", "swapped", 2),
    "bn_original_split2": ("mcl", "bn254", "off", "normal", 2),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--cpus", default="0-11")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.samples, args.rounds) < 1 or args.warmup < 0:
        parser.error("positive threads/samples/rounds and nonnegative warmup required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    weights = "experiments/paper_tables/profiles/weights_2of3.json"
    profiles = json.loads((ROOT / weights).read_text())["allocations"]
    profile_hashes = {}
    for profile in profiles:
        path = Path(f"experiments/paper_tables/profiles/{profile['error'].replace('/', 'of')}.txt")
        tokens = list(map(int, (ROOT / path).read_text().split()))
        if tokens != [profile["reconstruction_threshold"], *profile["weights"]]:
            raise ValueError(f"MCL text profile does not match allocation JSON: {path}")
        profile_hashes[str(path)] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(), "threads": args.threads, "cpus": args.cpus,
        "samples_per_round": args.samples, "warmup_per_case": args.warmup,
        "rounds": args.rounds, "weights": weights,
        "weight_sha256": hashlib.sha256((ROOT / weights).read_bytes()).hexdigest(),
        "configurations": {name: CONFIGS[name] for name in args.configs},
        "profile_sha256": profile_hashes, "runs": [],
    }
    for r in range(1, args.rounds + 1):
        configs = args.configs if r % 2 else list(reversed(args.configs))
        points = POINTS if r % 2 else list(reversed(POINTS))
        for error, total in points:
            for name in configs:
                backend, curve, simd, orientation, split = CONFIGS[name]
                batch = split or total
                suffix = "json" if backend == "blst" else "csv"
                stem = f"{name}_e{error.split('/')[1]}_m{total}_r{r}"
                if backend == "blst":
                    crate = "weighted_btx" if orientation == "normal" else "weighted_btx_swapped"
                    command = [f"{crate}/target/release/examples/paper_tables", "--weights-file", weights,
                               "--error", error, "--total", str(total), "--batch", str(batch),
                               "--setup", str(total), "--threads", str(args.threads),
                               "--samples", str(args.samples), "--warmup", str(args.warmup)]
                else:
                    profile = f"experiments/paper_tables/profiles/{error.replace('/', 'of')}.txt"
                    command = ["weighted_btx_mcl/build/paper_benchmark", curve, simd, orientation,
                               profile, str(total), str(batch), str(total), str(args.threads),
                               str(args.samples), str(args.warmup)]
                command = ["taskset", "-c", args.cpus] + command
                run = {"config": name, "error": error, "total": total, "batch": batch,
                       "setup": total, "round": r, "file": f"{stem}.{suffix}", "log": f"{stem}.log",
                       "command": command}
                manifest["runs"].append(run)
                print(f"round={r} config={name} error={error} messages={total} chunk={batch}", flush=True)
                if not args.dry_run:
                    with (out / run["file"]).open("w") as stdout, (out / run["log"]).open("w") as stderr:
                        subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True,
                                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                (out / "optimized_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed {len(manifest['runs'])} runs", flush=True)


if __name__ == "__main__":
    main()
