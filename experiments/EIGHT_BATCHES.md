# Eight batches of two

Eight batches of two improve the fastest uncached configuration from **28.84 to
26.18 ms** for 16 ciphertexts: **9.2% less elapsed time** than four batches of
four. With committee preparation cached, the fastest configuration improves
from **14.13 to 13.83 ms**, a smaller **2.1%** reduction.

These are fresh measurements of all three layouts in the same matrix, rather
than comparisons against the previous run's medians.

## End-to-end comparison

Median milliseconds for encryption through opening of all 16 ciphertexts:

| Implementation | Groups | Cold 4x4 | Cold 8x2 | Cached 4x4 | Cached 8x2 |
| --- | --- | ---: | ---: | ---: | ---: |
| MCL BN254 | Swapped | 28.839 | **26.177** | 16.457 | 16.049 |
| MCL BN254 | Original | 36.023 | 30.592 | **14.128** | **13.834** |
| MCL BLS12-381, AVX-512 | Swapped | 35.323 | 31.085 | 20.706 | 20.189 |

Swapped groups remain best when preparation is included; original groups remain
best when it is cached. The BLS/AVX swapped case improves by 12.0% uncached.
BN254 does not use an AVX-512 backend in this experiment.

The complete matrix also measures original/swapped native BLST and both MCL
BLS SIMD modes. See [all results and sample ranges](results/2026-09-21-end-to-end-8x2/SUMMARY.md).
One exception to the general improvement is native BLST's swapped cached case:
its whole pipeline changes from 22.75 to 23.01 ms even though its combiner
improves, because validator work increases.

## What changes

The same maximum-batch-16 setup and accepted committee serve all layouts. For
8x2, one batch-size-2 committee preparation is reused across eight chunks. Each
chunk still gets fresh validation, shares, acceptance, cross terms, and opening.

For 38 accepted validators with total weight 794:

| Work across 16 ciphertexts | 4x4 | 8x2 |
| --- | ---: | ---: |
| Per-validator weighted MSMs in the shared preparation | 152 | 76 |
| 794-point positive-offset MSMs in the shared preparation | 3 | 1 |
| Shares produced and checked | 152 | 304 |
| Share-acceptance pairing terms | 20 | 24 |
| Share-acceptance final exponentiations | 4 | 8 |
| MCL cross-term pairing/final-exponentiation calls | 32 | 32 |
| MCL inverse-GT-FFT nontrivial-twiddle powers | 20 | 8 |
| Opening pairing terms | 608 | 608 |

The communication tradeoff is **eight shares per validator instead of four**.
Network time is not included. More operations need not mean higher phase wall
time, because the chunks execute in parallel on the same bounded worker pool.

BN254, swapped groups, cold preparation, phase medians in milliseconds:

| Phase | 4x4 | 8x2 |
| --- | ---: | ---: |
| All local validator shares | 1.789 | 1.823 |
| Share acceptance | 5.317 | 5.210 |
| Shared preparation | 12.417 | 10.162 |
| Cross terms | 1.337 | 0.993 |
| Opening | 6.299 | 6.306 |
| Complete combiner, after shares exist | 25.378 | 22.674 |
| Whole online pipeline | **28.839** | **26.177** |

The cold gain is mainly reduced preparation and smaller FFTs. The cached gain
is small because that preparation saving has already been removed from the
timer. SIMD instrumentation records exactly one vector MSM callback for every
cold swapped BLS/AVX 8x2 iteration, and zero callbacks for cached iterations.

## Method and verification

- `shared-dev14`, AMD EPYC 9275F, 12 workers on physical CPUs 0–11.
- Same repository allocation: 419 validators, weight 1,576, required weight 789;
  heaviest-first committee of 38 validators, accepted weight 794.
- All selected validators simulated locally. Fresh encryption, proofs,
  validation, validator shares, acceptance, and cross terms on every iteration.
- Two rounds with reversed implementation order, two warmups and 15 measured
  iterations per case per round: **30 samples per cell**.
- Setup, plaintext generation, and network transport excluded. Cold means
  committee preparation is timed; it does not mean cold CPU caches.
- Every recovered plaintext checked after every warmup and measured iteration.
  Both native runners also passed local six-case smoke runs. The protocol
  implementation is unchanged and its existing correctness suite includes B=2.
- Strict summary validation checks complete six-case/nine-phase coverage,
  matching sample counts and rounds, and unique contiguous sample indices.

All **432 result groups have 30 samples**. The earlier
[implementation-comparison caveats](END_TO_END.md#validation-and-interpretation)
still apply: MCL and BLST differ in validation and target-group arithmetic paths.
The 4x4-to-8x2 comparisons above hold the backend and curve fixed.

## Reproduce

Build commands are in [the protocol README](../weighted_btx_mcl/README.md).
After building the three runners, execute from the repository root:

```sh
CPUS=0-11 THREADS=12 SAMPLES=15 WARMUP=2 ROUNDS="1 2" \
  bash weighted_btx_mcl/run_matrix.sh \
  "$PWD/experiments/results/2026-09-21-end-to-end-8x2"
```

The result directory contains [phase summaries](results/2026-09-21-end-to-end-8x2/summary.csv),
raw per-iteration files, correctness logs,
[source hashes](results/2026-09-21-end-to-end-8x2/source_hashes.json), and
[environment/binary/library provenance](results/2026-09-21-end-to-end-8x2/environment.json).
The earlier two-layout measurements remain in their original directory.
