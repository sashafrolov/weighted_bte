# Full protocol integration and end-to-end results

The subsequent [eight-batches-of-two comparison](EIGHT_BATCHES.md) reruns this
matrix with an additional 8x2 layout. The figures below preserve the initial
two-layout experiment.

BN254 and BLS12-381 AVX-512 now run through the complete weighted BTX protocol in
[`weighted_btx_mcl`](../weighted_btx_mcl/README.md). This is a separate C++ protocol
implementation, with both original and swapped source groups. It performs actual
key generation, encryption and proofs, client validation, validator shares,
randomized share acceptance with fallback, weighted interpolation, reusable
committee preparation, FFT cross terms, and verified message recovery.

The native Rust/BLST implementations have matching fresh-input end-to-end runners.
The figures below are measured protocol executions, not sums of primitive timings.

## Measurement

- Machine: `shared-dev14`, AMD EPYC 9275F, 24 physical cores; these runs use
  **12 workers pinned to physical CPUs 0–11**.
- Input: the repository's `1/16` allocation, 419 validators, total weight 1,576,
  reconstruction weight 789; the selected 38 validators have weight 794.
  This input targets one-half stake; it is not the paper's two-thirds-stake table.
- Same maximum-batch-16 setup for both layouts: `1x16`, or `4x4` with committee
  preparation reused across all four chunks. Each validator produces four shares
  in the latter case.
- **Online end-to-end:** fresh encryption and proofs, validation, generation of
  every selected validator's shares, acceptance, preparation unless cached,
  cross terms, and opening of all 16 messages. All participants run locally;
  these are aggregate local wall times, not distributed network latency.
- Trusted setup, plaintext construction, warmups, and network transport are
  excluded. Setup is executed and separately timed in every invocation.
  Every recovered plaintext is checked after every iteration's ending timestamp.
- Two rounds reverse implementation order. Each round has two warmups and
  15 measured iterations per case: **30 samples per cell**. Figures are medians;
  descriptive p10/p90 ranges and all raw phase samples are linked below.
- “Cold” means committee preparation is included, not cold CPU caches.
  “Cached” reuses only committee-dependent material; ciphertexts, proofs, shares,
  acceptance results, and cross terms are always fresh.

## Online end-to-end results

Milliseconds for 16 ciphertexts. Original groups put ciphertexts/shares in G1
and weighted public keys in G2; swapped groups reverse those assignments.

| Implementation | Groups | Cold 1x16 | Cold 4x4 | Cached 1x16 | Cached 4x4 |
| --- | --- | ---: | ---: | ---: | ---: |
| BLST BLS12-381 | Original | 154.881 | 66.320 | 28.632 | 20.409 |
| BLST BLS12-381 | Swapped | 84.262 | 42.187 | 32.911 | 22.938 |
| MCL BLS12-381, SIMD off | Original | 128.963 | 58.752 | 22.821 | 19.749 |
| MCL BLS12-381, AVX-512 | Original | 128.028 | 58.317 | 22.862 | 19.620 |
| MCL BLS12-381, SIMD off | Swapped | 75.229 | 40.343 | 24.706 | 20.751 |
| MCL BLS12-381, AVX-512 | Swapped | 64.295 | 35.257 | 24.849 | 20.731 |
| MCL BN254 | Original | 75.012 | 36.136 | 16.471 | **14.141** |
| MCL BN254 | Swapped | 50.527 | **28.628** | 19.248 | 16.397 |

The fastest uncached configuration is **BN254, swapped groups, four batches of
four: 28.63 ms**, a **5.41x** improvement over the original uncached BLST batch
of 16. The fastest cached configuration uses **BN254 with the original groups,
four batches of four: 14.14 ms**, a **2.02x** improvement over the original cached
batch of 16. Swapping reduces expensive preparation but increases the local
encryption/share-generation burden; the preferred orientation changes when
preparation is already cached.

With BLS12-381 retained, swapped groups plus AVX-512 and `4x4` takes **35.26 ms**
uncached. Isolating AVX-512 within the same MCL implementation gives:

- Swapped `1x16`, cold: **75.23 → 64.30 ms**, **1.17x**, or 14.5% lower latency.
- Swapped `4x4`, cold: **40.34 → 35.26 ms**, **1.14x**, or 12.6% lower latency.
- Cached cases and original-group cases do not dispatch these AVX kernels;
  their small on/off timing differences should not be attributed to SIMD.

Within the same MCL code, BN254 versus BLS12-381 with SIMD disabled reduces
swapped/cold `4x4` latency from **40.34 to 28.63 ms**, **1.41x**. BN254 changes the
security parameter; this is not an equal-security curve substitution. The pinned
MCL backend has no BN254 AVX-512 path, so the curve and SIMD results are separate
variants, not a combined BN254+AVX-512 implementation.

## Where the time goes

Phase medians in milliseconds for swapped groups, `4x4`, cold preparation:

| Phase | MCL BLS, SIMD off | MCL BLS, AVX-512 | MCL BN254 |
| --- | ---: | ---: | ---: |
| Encryption and proofs | 1.091 | 1.093 | 0.749 |
| Client validation | 1.009 | 1.009 | 0.911 |
| All local validator shares | 3.242 | 3.239 | 1.785 |
| Share acceptance | 5.168 | 5.170 | 5.291 |
| Shared committee preparation | 19.668 | 14.613 | 12.354 |
| Cross-term computation | 1.715 | 1.702 | 1.337 |
| Opening | 8.402 | 8.429 | 6.230 |
| Whole online pipeline | **40.343** | **35.257** | **28.628** |

Independent phase medians need not sum exactly to the whole-pipeline median.
The AVX gain is concentrated in preparation. Instrumentation confirms exactly
15 vector MSM callbacks plus one batched scalar callback per cold swapped
`1x16` iteration, and three vector MSM callbacks per cold swapped `4x4`
iteration. Cached and original-group cases record zero vector callbacks.

The separately measured combiner begins after all local validator shares exist.
For swapped/cold `4x4`, it takes **35.00 ms** with MCL BLS SIMD off,
**29.92 ms** with AVX-512, and **25.19 ms** with BN254. See the complete combiner
table in the [raw-result summary](results/2026-09-21-end-to-end/SUMMARY.md).

## Validation and interpretation

The integrated protocol passed **8,916 assertions on the dev box**, covering
both curves, both group orientations, serial and four-worker execution, direct
DFT and Lagrange references, literal pairing cross terms, bad-share fallback,
weighted authorization, malformed proofs, cache/context binding, identity
boundaries, and stale curve contexts. Every benchmark iteration also verified
all 16 recovered plaintexts. Earlier tests of the native swapped implementation
remain documented in [the experiment overview](README.md).

Cross-backend results compare complete implementations. MCL keeps explicit
source-group and GT membership checks enabled, while native Rust starts with
typed group objects. MCL also copies mutable MSM inputs, uses full Fp12 transcript
encoding, and applies full final exponentiation before a full GT inverse FFT;
native BLST uses its optimized split-exponentiation and partial-IFFT path.
Both absorb inverse-FFT normalization into reusable committee keys. These
differences prevent interpreting every native-to-MCL speedup as a pure curve or
SIMD effect. The controlled MCL on/off comparisons isolate the vector callbacks.

This is a trusted-dealer, in-memory experimental implementation. It has no wire
decoder or distributed transport and is not wire-compatible with the Rust
prototype. It does not establish a new security proof or production readiness.

## Reproduce and inspect

- [Build, API, test, and benchmark instructions](../weighted_btx_mcl/README.md)
- [Complete medians, p10/p90, combiner timings, and SIMD counts](results/2026-09-21-end-to-end/SUMMARY.md)
- [All phase summaries as CSV](results/2026-09-21-end-to-end/summary.csv)
- [Protocol correctness log](results/2026-09-21-end-to-end/tests-mcl.log)
- [Environment and binary provenance](results/2026-09-21-end-to-end/environment.json)
- [Source hashes](results/2026-09-21-end-to-end/source_hashes.json)

MCL is pinned to `cbb18eb08b86129cf936a6436b5e6c68a2ce8ddf`; its BN254 option
uses `BN_SNARK1`, not MCL's separately named `BN254` parameter.
