#!/usr/bin/env python3
"""Write all weights and the reconstruction threshold, not only a committee."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
source = root / "scripts/data/solana_share_weights_2026-08-07T16-16-16Z.json"
profile = next(p for p in json.loads(source.read_text())["allocations"] if p["error"] == "1/16")
print(profile["reconstruction_threshold"])
print("\n".join(map(str, profile["weights"])))
print(json.dumps({"source": str(source.relative_to(root)), "error": "1/16",
                  "parties": len(profile["weights"]), "total_weight": sum(profile["weights"]),
                  "required_weight": profile["reconstruction_threshold"]}), file=sys.stderr)
