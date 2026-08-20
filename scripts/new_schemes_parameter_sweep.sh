#!/usr/bin/env bash
# Run the two four-row parameter sweeps for the new weighted schemes.
#
# The paper calls the first scheme weighted BTE; its crate is named
# `weighted_btx` in this repository. The (B=32, error=1/16) point is run in
# both sections so that each paper table has a complete set of four rows.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)

ERRORS=("1/8" "1/16" "1/32" "1/64")
BATCH_SIZES=(32 64 128 256)
FIXED_BATCH_SIZE=32
FIXED_ERROR="1/16"
TOTAL_RUNS=16
RUN_NUMBER=0

SWEEP_THREADS=${SWEEP_THREADS:-12}
SWEEP_REPETITIONS=${SWEEP_REPETITIONS:-1}
SWEEP_DRY_RUN=${SWEEP_DRY_RUN:-0}
if [[ -z "${RUSTFLAGS+x}" ]]; then
    RUSTFLAGS="-C target-cpu=native"
fi
export RUSTFLAGS

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_positive_integer() {
    local name=$1
    local value=$2
    [[ "$value" =~ ^[0-9]+$ ]] ||
        die "$name must be a positive integer (got '$value')"
    ((10#$value > 0)) || die "$name must be a positive integer (got '$value')"
}

select_weights_file() {
    local candidate
    if [[ -n "${SWEEP_WEIGHTS_FILE+x}" ]]; then
        [[ -n "$SWEEP_WEIGHTS_FILE" ]] || die "SWEEP_WEIGHTS_FILE must not be empty"
        candidate=$SWEEP_WEIGHTS_FILE
    else
        local candidates
        shopt -s nullglob
        candidates=("$SCRIPT_DIR"/data/solana_share_weights_*.json)
        shopt -u nullglob
        ((${#candidates[@]} > 0)) ||
            die "no solana_share_weights_*.json file found in $SCRIPT_DIR/data"
        candidate=${candidates[${#candidates[@]}-1]}
    fi

    if [[ "$candidate" != /* ]]; then
        local candidate_dir
        local candidate_name
        candidate_dir=$(dirname "$candidate")
        candidate_name=$(basename "$candidate")
        [[ -d "$candidate_dir" ]] || die "weights directory does not exist: $candidate_dir"
        candidate="$(cd "$candidate_dir" && pwd -P)/$candidate_name"
    fi
    [[ -f "$candidate" ]] || die "weights file does not exist: $candidate"
    WEIGHTS_FILE=$candidate
}

# Sets PROFILE_W, PROFILE_Q, PROFILE_T, and PROFILE_N for one error profile.
load_profile() {
    local approximation_error=$1
    local profile
    if ! profile=$(python3 - "$WEIGHTS_FILE" "$approximation_error" <<'PY'
import json
import sys

path, requested_error = sys.argv[1:]
try:
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"failed to load {path}: {error}")

allocations = document.get("allocations")
if not isinstance(allocations, list):
    raise SystemExit(f"{path}: 'allocations' must be a list")
matches = [entry for entry in allocations
           if isinstance(entry, dict) and entry.get("error") == requested_error]
if len(matches) != 1:
    raise SystemExit(
        f"{path}: expected exactly one allocation for error {requested_error!r}, "
        f"found {len(matches)}"
    )

profile = matches[0]
fields = ("share_count", "reconstruction_threshold", "positive_validator_count")
for field in fields:
    value = profile.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"{path}: {field!r} must be an integer")

weights = profile.get("weights")
if not isinstance(weights, list) or any(
    isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0
    for weight in weights
):
    raise SystemExit(f"{path}: 'weights' must be a list of positive integers")

total_weight = profile["share_count"]
threshold = profile["reconstruction_threshold"]
party_count = profile["positive_validator_count"]
if total_weight <= 0:
    raise SystemExit(f"{path}: share_count must be positive")
if party_count <= 0:
    raise SystemExit(f"{path}: positive_validator_count must be positive")
if not 1 <= threshold <= total_weight:
    raise SystemExit(f"{path}: reconstruction_threshold must be in [1, share_count]")
if party_count != len(weights):
    raise SystemExit(f"{path}: positive_validator_count does not match len(weights)")
if total_weight != sum(weights):
    raise SystemExit(f"{path}: share_count does not match sum(weights)")

print(total_weight, threshold, party_count)
PY
    ); then
        die "could not read approximation profile $approximation_error"
    fi

    read -r PROFILE_W PROFILE_Q PROFILE_N <<< "$profile"
    [[ "$PROFILE_W" =~ ^[0-9]+$ && "$PROFILE_Q" =~ ^[0-9]+$ && "$PROFILE_N" =~ ^[0-9]+$ ]] ||
        die "profile parser returned malformed output for $approximation_error: $profile"
    PROFILE_T=$((PROFILE_Q - 1))
}

print_command() {
    printf 'COMMAND'
    printf ' %q' "$@"
    printf '\n'
}

print_banner() {
    local sweep=$1
    local scheme=$2
    local implementation=$3
    local batch_size=$4
    local approximation_error=$5
    local repetitions=$6

    RUN_NUMBER=$((RUN_NUMBER + 1))
    printf '\n%s\n' '================================================================================'
    printf 'PARAMETERS run=%d/%d sweep=%s scheme=%s implementation=%s B=%s approximation_error=%s parties=%s total_weight=%s q=%s t=%s threads=%s repetitions=%s\n' \
        "$RUN_NUMBER" "$TOTAL_RUNS" "$sweep" "$scheme" "$implementation" \
        "$batch_size" "$approximation_error" "$PROFILE_N" "$PROFILE_W" \
        "$PROFILE_Q" "$PROFILE_T" "$SWEEP_THREADS" "$repetitions"
    printf 'WEIGHTS_FILE %s\n' "$WEIGHTS_FILE"
}

execute() {
    print_command "$@"
    printf '%s\n' '================================================================================'
    if [[ "$SWEEP_DRY_RUN" == "1" ]]; then
        return
    fi
    (cd "$REPO_ROOT" && "$@")
}

run_weighted_bte() {
    local sweep=$1
    local batch_size=$2
    local approximation_error=$3
    local command=(
        env
        "RUSTFLAGS=$RUSTFLAGS"
        "WEIGHTED_BTX_WEIGHTS_FILE=$WEIGHTS_FILE"
        "WEIGHTED_BTX_BATCH_SIZE=$batch_size"
        "WEIGHTED_BTX_THREADS=$SWEEP_THREADS"
        cargo run --quiet --release --locked
        --manifest-path "$REPO_ROOT/weighted_btx/Cargo.toml"
        --example paper_reproduction --
        --approximation-error "$approximation_error"
    )

    print_banner "$sweep" weighted_bte weighted_btx "$batch_size" "$approximation_error" 1
    execute "${command[@]}"
}

run_weighted_pfe() {
    local sweep=$1
    local batch_size=$2
    local approximation_error=$3
    local command=(
        env
        "RUSTFLAGS=$RUSTFLAGS"
        "WEIGHTED_PFE_WEIGHTS_FILE=$WEIGHTS_FILE"
        "WEIGHTED_PFE_BATCH_SIZE=$batch_size"
        "WEIGHTED_PFE_THREADS=$SWEEP_THREADS"
        "WEIGHTED_PFE_REPETITIONS=$SWEEP_REPETITIONS"
        cargo run --quiet --release --locked
        --manifest-path "$REPO_ROOT/weighted_pfe/Cargo.toml"
        --example paper_reproduction --
        --approximation-error "$approximation_error"
    )

    print_banner "$sweep" weighted_pfe weighted_pfe "$batch_size" "$approximation_error" "$SWEEP_REPETITIONS"
    execute "${command[@]}"
}

run_setting() {
    local sweep=$1
    local batch_size=$2
    local approximation_error=$3
    load_profile "$approximation_error"
    run_weighted_bte "$sweep" "$batch_size" "$approximation_error"
    run_weighted_pfe "$sweep" "$batch_size" "$approximation_error"
}

command -v python3 >/dev/null 2>&1 || die "python3 is required to read the allocation profiles"
require_positive_integer SWEEP_THREADS "$SWEEP_THREADS"
require_positive_integer SWEEP_REPETITIONS "$SWEEP_REPETITIONS"
[[ "$SWEEP_DRY_RUN" == "0" || "$SWEEP_DRY_RUN" == "1" ]] ||
    die "SWEEP_DRY_RUN must be 0 or 1 (got '$SWEEP_DRY_RUN')"
if [[ "$SWEEP_DRY_RUN" == "0" ]]; then
    command -v cargo >/dev/null 2>&1 || die "cargo is required to run the paper examples"
fi
select_weights_file

# Validate every requested profile before starting any potentially long run.
for error in "${ERRORS[@]}"; do
    load_profile "$error"
done

printf 'New-scheme approximation-error sweep (fixed B=%d)\n' "$FIXED_BATCH_SIZE"
for error in "${ERRORS[@]}"; do
    run_setting approximation_error "$FIXED_BATCH_SIZE" "$error"
done

printf '\nNew-scheme batch-size sweep (fixed approximation error=%s)\n' "$FIXED_ERROR"
for batch_size in "${BATCH_SIZES[@]}"; do
    run_setting batch_size "$batch_size" "$FIXED_ERROR"
done

printf '\nCompleted %d parameterized runs.\n' "$RUN_NUMBER"
