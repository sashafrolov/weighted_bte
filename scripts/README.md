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

By default, this writes `scripts/data/solana_validator_distribution.json`.
Override the destination with `--output`, or use `--stdout` to emit the JSON to
standard output.

## Generate integer weights

`generate_solana_share_weights.py` reads the downloaded distribution and applies
Hamilton's largest-remainder method. For each requested total weight, the
assigned integer weights sum exactly to that total. It writes identity-free
allocation lists to `scripts/data/solana_share_weights.json`.

Once the distribution has been downloaded, run with all defaults:

```sh
python3 scripts/generate_solana_share_weights.py
```

The default total weights are 2,048, 4,096, 8,192, 16,384, and 32,768. To use a
different input, totals, or output file:

```sh
python3 scripts/generate_solana_share_weights.py /tmp/solana-validators.json \
  --shares 1024 2048 4096 \
  --output /tmp/solana-share-weights.json
```

The output JSON contains only positive weights; zero-weight validators and
validator identities are omitted.

## Inspect approximation quality

`solana_share_approximation.py` calculates the same allocations but only prints
a Markdown table of approximation-quality metrics. It does not write a file.
Run the report for the default input and total weights with:

```sh
python3 scripts/solana_share_approximation.py
```

To inspect different totals or another validator distribution:

```sh
python3 scripts/solana_share_approximation.py /tmp/solana-validators.json \
  --shares 1024 2048 4096
```
