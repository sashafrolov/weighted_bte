#!/usr/bin/env python3
"""Extract the same largest-first threshold committee as the native benchmark."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
source = root / "scripts/data/solana_share_weights_2026-08-07T16-16-16Z.json"
profile = next(p for p in json.loads(source.read_text())["allocations"] if p["error"] == "1/16")
selected = []
accepted = 0
for index, weight in sorted(enumerate(profile["weights"]), key=lambda p: (-p[1], p[0])):
    selected.append((index, weight))
    accepted += weight
    if accepted >= profile["reconstruction_threshold"]:
        break
print("\n".join(str(weight) for _, weight in selected))
print(json.dumps({"profile": "1/16", "total_weight": profile["share_count"],
                  "required_weight": profile["reconstruction_threshold"],
                  "accepted_weight": accepted, "parties": len(selected),
                  "selected_indices": [i for i, _ in selected]}), file=sys.stderr)
