#!/usr/bin/env bash
# Run only after build.sh and the protocol tests succeed on the target machine.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/.." && pwd)
out=${1:-"$root/experiments/results/2026-09-21-end-to-end-8x2"}
mkdir -p "$out"
samples=${SAMPLES:-15}
warmup=${WARMUP:-2}
threads=${THREADS:-12}
cpus=${CPUS:-0-11}
cd "$root"
native() {
  local orientation=$1 round=$2 crate=weighted_btx
  [[ "$orientation" != swapped ]] || crate=weighted_btx_swapped
  [[ "${SKIP_BLST:-0}" != 1 ]] || return 0
  taskset -c "$cpus" "$root/$crate/target/release/examples/end_to_end" \
    --threads "$threads" --samples "$samples" --warmup "$warmup" \
    > "$out/blst_${orientation}_r${round}.json" 2> "$out/blst_${orientation}_r${round}.log"
}
mcl_case() {
  local curve=$1 simd=$2 orientation=$3 round=$4
  taskset -c "$cpus" "$here/build/benchmark" "$curve" "$simd" "$orientation" \
    "$here/build/profile.txt" "$threads" "$samples" "$warmup" \
    > "$out/mcl_${curve}_${simd}_${orientation}_r${round}.csv" \
    2> "$out/mcl_${curve}_${simd}_${orientation}_r${round}.log"
}
for round in ${ROUNDS:-1 2}; do
  if (( round % 2 )); then
    native normal "$round"
    native swapped "$round"
    for orientation in normal swapped; do
      mcl_case bls12_381 off "$orientation" "$round"
      mcl_case bls12_381 avx512 "$orientation" "$round"
      mcl_case bn254 off "$orientation" "$round"
    done
  else
    for orientation in swapped normal; do
      mcl_case bn254 off "$orientation" "$round"
      mcl_case bls12_381 avx512 "$orientation" "$round"
      mcl_case bls12_381 off "$orientation" "$round"
    done
    native swapped "$round"
    native normal "$round"
  fi
done
python3 "$here/summarize.py" "$out" --layouts 1x16,4x4,8x2
