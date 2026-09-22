# Optimized paper performance tables

This experiment recreates the numeric performance tables (Tables 5-8) from
the supplied `2026-2053.pdf`, with new weighted-BTX configurations and fresh
baselines on `shared-dev14`. The original paper reports Apple MacBook M4
measurements; speedup calculations here use the fresh dev-box baseline.

## Results

- [Readable tables](output/tables/TABLES.md)
- [PDF tables](output/pdf/weighted-btx-paper-tables.pdf)
- [Raw samples, summaries and provenance](../results/2026-09-22-paper-tables/)
- [Transcribed original paper tables](paper_reference.json)

The new WBTX harnesses process fresh encrypted messages and proofs in every
iteration. They measure original and swapped Rust/BLST, swapped BLS12-381/MCL
with AVX-512, and both BN254/MCL orientations. Optimized MCL configurations
split the workload into chunks of two and reuse one committee preparation.
Every timed configuration retains the same maximum setup capacity `L=M`.
Calculated smaller `L=2` public-key sizes are labeled separately.

All selected validators are simulated locally. We use twelve workers pinned
to physical CPUs 0-11, two rounds in reversed configuration/parameter order,
two warmups per case per round, and 15 measured iterations per round. Reported
WBTX values are arithmetic means over 30 observations; raw samples also support
median and descriptive p10/p90 summaries. Setup and networking are excluded.
All recovered plaintexts are checked after every warmup and measured iteration.

## Inputs and correspondence to the paper

The original 862-validator snapshot is retained in the repository. We regenerate
its Aptos-style rounded weights with a target stake threshold of **2/3**, rather
than using the earlier experiment's one-half-threshold allocation.

| Error | Paper W | Actual regenerated W | N | Required q | Selected validators | Accepted weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/8 | 764 | 764 | 234 | 512 | 59 | 515 |
| 1/16 | 1,580 | 1,580 | 419 | 1,060 | 76 | 1,062 |
| 1/32 | 3,060 | 3,063 | 598 | 2,044 | 79 | 2,046 |
| 1/64 | 6,486 | 6,489 | 688 | 4,327 | 79 | 4,333 |

The last two weight sums differ slightly from the PDF. The generated values
also appear in the repository's September two-thirds-threshold approximation
report, and the PDF itself mentions 6,489 in its discussion. We retain the
actual generated positive weights, not manually adjusted vectors. The selected
committee is heaviest-first with increasing-index tie breaks; all protocol
variants receive the same vectors and `t=q-1`.

Table 5 varies these four profiles at `M=16`. Table 6 varies
`M=16,32,64,128,256` with error `1/16`. `M` denotes total messages: native
BLST handles one batch of `M`, whereas optimized MCL handles `M/2` batches
of two. This distinction is necessary when reproducing the paper's single
batch-size axis after splitting.

## Timing and size definitions

- **PreDec/validator:** aggregate local share-generation wall time divided by
  selected real validator count, including every chunk. This is an amortized
  rate, not the standalone latency of one validator.
- **Validate/validator:** fresh share-acceptance wall time divided by selected
  real validator count, across all chunks.
- **DecPrecomp:** the one committee-dependent preparation reused by all chunks.
- **DecOpen:** ciphertext-dependent cross-term precomputation plus opening.
- **DecTotal:** elapsed time from the start of share generation through final
  opening. It excludes encryption and client-proof validation, following the
  original WBTX runner's phase selection.
- **Cached DecTotal:** a separately measured WBTX run with only committee
  preparation reused. Fresh ciphertexts, proofs, shares, acceptance and cross
  terms are still computed.
- **End-to-end:** fresh encryption/proofs, client validation and DecTotal.
- **Key sizes:** compressed-group payloads in KiB (1024 bytes), matching the
  paper's arithmetic despite its kB label. WBTX core points are `(2L-1)W` and
  verification points are `NL`. Total excludes encryption key/proof CRS/wire
  framing, consistent with the paper's Table 7 columns.

The unchanged comparison runners have different sampling/timing definitions:
WPFE reuses one encrypted batch per process for fifteen online repetitions;
its encryption mean has two observations. Its comparable decryption totals
are derived from printed phase means, excluding client checks. Its cached
estimate subtracts preparation. BTX/PFE report a normalized one-virtual-server
share cost; BEAT++ generates shares/proofs from all real validators. Table 8
preserves their source-defined totals and denominators, without claiming equal
distributed workloads. These prior schemes use ten measured repetitions each.
Their printed timings have 0.001 ms precision. Raw logs and the legacy manifests
record every measurement definition and observation count.

BN254 is MCL's `BN_SNARK1` curve and has no AVX-512 path here. Curve changes
alter security assumptions. Cross-backend timings compare implementations with
different validation and target-group arithmetic paths. AVX-512 callback
counts are checked against the actual selected weights. The MCL implementation
remains an experimental trusted-dealer, in-memory protocol.

## Reproduce

From the repository root, regenerate the JSON allocation if desired:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/generate_solana_share_weights.py \
  scripts/data/solana_validator_distribution_2026-08-06T22-10-53Z.json \
  --target 2/3 --errors 1/8 1/16 1/32 1/64 \
  --output experiments/paper_tables/profiles/weights_2of3.json
```

The accompanying text profiles contain `q` followed by all positive weights.
The runner validates them against the JSON before execution. Build pinned MCL
as described in [the protocol README](../../weighted_btx_mcl/README.md), then:

```sh
bash weighted_btx_mcl/build.sh
RUSTFLAGS='-C target-cpu=native' cargo build --locked --release \
  --manifest-path weighted_btx/Cargo.toml --example paper_tables
RUSTFLAGS='-C target-cpu=native' cargo build --locked --release \
  --manifest-path weighted_btx_swapped/Cargo.toml --example paper_tables
# Build paper_reproduction for weighted_pfe, pfe, btx and weighted_indexed_bte.
```

Run the following serially, on an otherwise idle host. Use a new output directory
for a new measurement; legacy runners refuse to overwrite existing results.

```sh
OUT=experiments/results/2026-09-22-paper-tables
PYTHONDONTWRITEBYTECODE=1 python3 experiments/paper_tables/run_optimized.py \
  --out "$OUT/optimized" --threads 12 --cpus 0-11 --samples 15 --rounds 2
PYTHONDONTWRITEBYTECODE=1 python3 experiments/paper_tables/summarize_optimized.py "$OUT/optimized"
PYTHONDONTWRITEBYTECODE=1 python3 experiments/paper_tables/legacy_baselines.py \
  --weights-file experiments/paper_tables/profiles/weights_2of3.json \
  --out "$OUT/legacy_wpfe" --schemes wpfe --samples 15 --rounds 2
PYTHONDONTWRITEBYTECODE=1 python3 experiments/paper_tables/legacy_baselines.py \
  --weights-file experiments/paper_tables/profiles/weights_2of3.json \
  --out "$OUT/legacy_prior" --schemes pfe btx beat --samples 5 --rounds 2
PYTHONDONTWRITEBYTECODE=1 python3 experiments/paper_tables/build_report.py "$OUT" \
  --out experiments/paper_tables/output/tables
python3 experiments/paper_tables/render_report.py \
  experiments/paper_tables/output/tables/report.json \
  experiments/paper_tables/output/pdf/weighted-btx-paper-tables.pdf
```

PDF rendering requires ReportLab. Raw result parsing and table generation use
only Python's standard library. The strict summary rejects missing/duplicate
samples, incorrect workload identities, invalid sizes and unexpected SIMD
dispatch; publication files contain no connection details.
