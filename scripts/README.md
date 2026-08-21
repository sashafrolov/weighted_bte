# Solana validator weight scripts

These scripts download the current Solana validator stake distribution and
approximate it with integer weights.

## Run the paper parameter sweeps

The two executable sweep scripts run the paper-reproduction examples and print
a `PARAMETERS` line plus the exact command before every measurement. Run them
from any directory; they resolve crate and data paths relative to their own
location.

```console
mkdir -p scripts/results
./scripts/new_schemes_parameter_sweep.sh 2>&1 \
  | tee scripts/results/new_schemes_results_2.txt
./scripts/baseline_schemes_parameter_sweep.sh 2>&1 \
  | tee scripts/results/baseline_schemes_results_2.txt
```

`new_schemes_parameter_sweep.sh` runs `weighted_btx` (the repository name for
the weighted-BTE scheme) and `weighted_pfe`. The baseline script runs `btx`,
`pfe`, and `weighted_indexed_bte`. For unweighted BTX and PFE, each unit of
virtual weight becomes one party: `N = W`, and the polynomial-degree threshold
is `t = q - 1`, where `q` is the profile's minimum reconstruction weight.

Each script contains two sweep sections:

- approximation errors `1/8`, `1/16`, `1/32`, and `1/64` at `B=16`; and
- batch sizes `16`, `32`, `64`, `128`, and `256` at approximation error
  `1/16`.

The common `(B=16, error=1/16)` point is deliberately run in both sections so
each output table is self-contained. The new-scheme script makes 18 runs; the
baseline script makes 27. The naïvely expanded baseline rows, especially the
`1/64` profile with thousands of parties, can take substantially longer.

The scripts use 12 threads and one measured repetition by default. Override
those choices, select a specific allocation file, or inspect all commands
without running the benchmarks with:

```console
SWEEP_THREADS=8 \
SWEEP_REPETITIONS=3 \
SWEEP_WEIGHTS_FILE=scripts/data/solana_share_weights_2026-08-07T16-16-16Z.json \
./scripts/new_schemes_parameter_sweep.sh

SWEEP_DRY_RUN=1 ./scripts/baseline_schemes_parameter_sweep.sh
```

`SWEEP_REPETITIONS` applies to examples that support measured repetitions;
the weighted-BTX and indexed weighted-BTE examples perform one run. Unless it
is already set, the scripts use `RUSTFLAGS="-C target-cpu=native"`. Without a
`SWEEP_WEIGHTS_FILE` override, they select the lexicographically newest
`solana_share_weights_*.json` file and preflight every requested profile's
weight and threshold fields before starting the first measurement.

## Download the validator distribution

`fetch_solana_validator_distribution.py` fetches validator records
from https://schedulerwar.vercel.app/ and processes/saves them.

From the repository root, run:

```sh
python3 scripts/fetch_solana_validator_distribution.py
```

By default, this writes a UTC-timestamped file such as
`scripts/data/solana_validator_distribution_2026-08-05T14-30-00Z.json`.
Override the destination with `--output`, or use `--stdout` to emit the JSON to
standard output. Timestamped filenames sort chronologically.

## Generate integer weights

`generate_solana_share_weights.py` reads the downloaded distribution and applies
the Aptos DKG stake-rounding method. For each allowed symmetric error around a
target stake threshold, it selects a nominal resolution, independently rounds
validator weights to the nearest integer, and adjusts the reconstruction
threshold for the aggregate rounding error. By default, it reads the newest
timestamped validator distribution and writes identity-free allocation lists
to a UTC-timestamped file such as
`scripts/data/solana_share_weights_2026-08-05T14-31-00Z.json`.

Once the distribution has been downloaded, run with all defaults:

```sh
python3 scripts/generate_solana_share_weights.py
```

The default target is `1/2`, and the default errors are `1/4`, `1/8`, `1/16`,
`1/32`, `1/64`, and `1/128`. To use a different input, target, errors, or output
file:

```sh
python3 scripts/generate_solana_share_weights.py /tmp/solana-validators.json \
  --target 1/2 \
  --errors 1/16 1/32 1/64 \
  --output /tmp/solana-share-weights.json
```

Each output profile records its allowed error, nominal resolution, computed
total weight, reconstruction threshold, and positive weights. Zero-weight
validators and validator identities are omitted.

## Inspect approximation quality

`solana_share_approximation.py` calculates the same allocations, prints a
Markdown table of approximation-quality metrics, and saves the same table to a
UTC-timestamped file such as
`scripts/data/solana_share_approximation_2026-08-05T14-32-00Z.md`. By default,
it reads the newest timestamped validator distribution. Run the report for the
default input, target, and errors with:

```sh
python3 scripts/solana_share_approximation.py
```

To inspect different errors or another validator distribution:

```sh
python3 scripts/solana_share_approximation.py /tmp/solana-validators.json \
  --target 1/2 \
  --errors 1/16 1/32 1/64 \
  --output /tmp/solana-approximation.md
```
