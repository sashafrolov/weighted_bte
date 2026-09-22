#!/usr/bin/env python3
"""Record source/build provenance without reading credentials or environment secrets."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
def command(*args):
    return subprocess.check_output(args, cwd=root, text=True).strip()

sources = set()
for crate in ["weighted_btx", "weighted_btx_swapped"]:
    sources.update((root / crate / "src").rglob("*.rs"))
    sources.update(root / crate / name for name in ["Cargo.toml", "Cargo.lock", "examples/end_to_end.rs"])
for suffix in ["*.hpp", "*.cpp", "*.sh", "*.py"]:
    sources.update((root / "weighted_btx_mcl").glob(suffix))
sources.add(root / "scripts/data/solana_share_weights_2026-08-07T16-16-16Z.json")
hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
          for path in sorted(sources) if path.is_file()}
(out / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
metadata = {"captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "hostname": platform.node(), "platform": platform.platform(),
            "repository_head": command("git", "rev-parse", "HEAD"),
            "benchmark_cpu_affinity": os.environ.get("CPUS", "0-11"),
            "benchmark_threads": int(os.environ.get("THREADS", "12")),
            "rust_flags": "-C target-cpu=native", "rust_profile": "release, thin LTO, codegen-units=1",
            "cxx_flags": "-O3 -DNDEBUG -std=c++17 -DMCL_FP_BIT=384 -DMCL_FR_BIT=256 -lpthread",
            "mcl_revision": command("git", "-C", str(root / "experiments/mcl/build/mcl"), "rev-parse", "HEAD")}
if shutil.which("lscpu"):
    (out / "lscpu.json").write_text(command("lscpu", "-J") + "\n")
for tool in ["rustc", "cargo", "c++"]:
    if shutil.which(tool): metadata[tool] = command(tool, "--version")
for name in ["compiler.txt", "build_settings.txt", "profile.json", "mcl_commit.txt"]:
    path = root / "weighted_btx_mcl/build" / name
    if path.is_file(): shutil.copyfile(path, out / name)
binaries = [root / "weighted_btx_mcl/build/benchmark", root / "weighted_btx_mcl/build/tests"]
binaries += [root / crate / "target/release/examples/end_to_end" for crate in ["weighted_btx", "weighted_btx_swapped"]]
metadata["binary_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in binaries if path.is_file()}
library = root / "experiments/mcl/build/mcl/lib/libmcl.a"
if library.is_file():
    metadata["linked_static_library_sha256"] = {
        str(library.relative_to(root)): hashlib.sha256(library.read_bytes()).hexdigest()
    }
(out / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
