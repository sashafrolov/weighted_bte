# Solana validator weight scripts

These scripts download the current Solana validator stake distribution and
approximate it with integer weights.

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
