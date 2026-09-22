#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build=${MCL_BUILD_DIR:-"$here/build"}
results=${1:-"$here/results/$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$results"
cp "$build/profile.json" "$build/mcl_commit.txt" "$build/compiler.txt" "$results/"
if [[ -f "$build/build_settings.txt" ]]; then cp "$build/build_settings.txt" "$results/"; fi
uname -a > "$results/uname.txt"
lscpu > "$results/lscpu.txt"
sha256sum "$here/kernel_bench.cpp" "$build/kernel_bench" > "$results/sha256.txt"
for round in $(seq 1 "${ROUNDS:-2}"); do
  for batch in ${BATCHES:-4 16 64}; do
    # Repeat every on/off pair in reverse order on the next round.
    modes="off avx512"
    if (( round % 2 == 0 )); then modes="avx512 off"; fi
    for mode in $modes; do
      taskset -c "${BENCH_CPU:-0}" "$build/kernel_bench" bls12_381 "$mode" "$batch" \
        "$build/weights.txt" "${SAMPLES:-5}" "${SAMPLE_MS:-50}" \
        > "$results/bls12_381_${mode}_B${batch}_r${round}.csv" 2> "$results/bls12_381_${mode}_B${batch}_r${round}.log"
    done
    taskset -c "${BENCH_CPU:-0}" "$build/kernel_bench" bn254 off "$batch" \
      "$build/weights.txt" "${SAMPLES:-5}" "${SAMPLE_MS:-50}" \
      > "$results/bn254_off_B${batch}_r${round}.csv" 2> "$results/bn254_off_B${batch}_r${round}.log"
  done
done
printf 'Results: %s\n' "$results"
