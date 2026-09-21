# Weighted BTX experiments

**The latest full-protocol measurements include eight batches of two:** see
[the 8x2 comparison](EIGHT_BATCHES.md). It includes fresh encryption, proofs,
validation, all local validator shares, and opening. The fastest measured
uncached case is 26.18 ms (BN254, swapped groups, 8x2); the fastest cached case
is 13.83 ms (BN254, original groups, 8x2), on 12 physical cores. The initial
[BN254 and AVX-512 integration report](END_TO_END.md) remains available.

These experiments compare group orientation, curve/backend choices, SIMD, and
batch layout. The original `weighted_btx` library implementation is unchanged;
its benchmarks add a combined-phase measurement and a batch-layout comparison.
`weighted_btx_swapped` is a separate
experimental implementation with ciphertexts and shares in G2 and public MSM
material in G1.

## Machine and inputs

For publication, machine-specific checkout paths in archived logs and JSON
are replaced with `<checkout>`, and connection details are omitted. Recorded
timings and source/binary hash values are unchanged.

Archived source hashes describe the files at measurement time. After the
latest 8x2 run, publication cleanup sorted imports in the swapped Rust
implementation and added an early error for an unbuilt custom `MCL_SOURCE`
checkout in the build script; protocol computations are unchanged.

- Host: `shared-dev14`, AMD EPYC 9275F, 24 physical cores.
- Upstream revision: `69a2c90b3797259a0d919f795ef978c27f0fe3ef`.
- Rust: 1.97.0; release build with `RUSTFLAGS="-C target-cpu=native"`.
- Twelve Rayon workers, pinned to physical CPUs 0–11 for the initial runs.
- Repository allocation file:
  `scripts/data/solana_share_weights_2026-08-07T16-16-16Z.json`, error `1/16`.
- 419 represented validators; total weight 1,576; minimum accepted weight 789.
  The heaviest-first committee has 38 validators and weight 794.

This allocation targets one-half stake. It is the repository's supplied input,
not an exact reproduction of the supplied paper's two-thirds-stake table.
The environment log records the allocation file's SHA-256 digest.

## Batch-layout benchmark

```sh
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=12 \
taskset -c 0-11 cargo bench --locked \
  --manifest-path weighted_btx/Cargo.toml --bench batch_layout
```

Change the manifest path to `weighted_btx_swapped/Cargo.toml` for the other
group orientation. `BATCH_LAYOUT_WEIGHTS_FILE` and `BATCH_LAYOUT_ERROR` override
the input file and allocation profile.

Within each group orientation, all layouts process the same 16 ciphertexts
under the same maximum-batch-16 setup. The split layout partitions them into
four batches of four without
re-encryption. It reuses only the committee-dependent preparation, which is
bound to the setup, batch size, and exact accepted validator set. Each chunk
independently verifies its own shares and computes its own cross terms.

The timed combiner includes fresh randomized share verification, preparation
unless cached, cross-term computation, and opening. Trusted key generation,
encryption, client-proof validation, validator share generation, and network
transfer are excluded. Validator share generation is separately measured on
one thread, including all four shares for the split layout. Every case checks
all recovered messages before timing. Cold means committee preparation is
not cached; it does not mean cold CPU caches.

The benchmark runs 30 Criterion samples after a one-second warm-up, with at
least three seconds of measurement. Benchmarks on the dev box run serially.

### Protocol combiner results, 12 physical cores

Milliseconds per 16 ciphertexts, Criterion central estimates:

| Combiner case | Original groups (ms) | Swapped groups (ms) |
| --- | ---: | ---: |
| One batch of 16, uncached preparation | 152.81 | 77.363 |
| Four batches of 4, prepare once | 63.043 | 35.390 |
| Four batches of 4, prepare four times | 145.77 | 70.848 |
| Four batches of 4, prepare once, sequential chunks | 85.081 | 59.885 |
| One batch of 16, cached preparation | 25.422 | 26.913 |
| Four batches of 4, cached preparation | 17.163 | 16.079 |

Splitting alone is 2.42x faster with uncached preparation and 1.48x faster
with cached preparation. Swapping alone is 1.98x faster with uncached
preparation, but is 5.9% slower for the already-cached batch of 16. Combining
the swap and split gives **4.32x** versus the original uncached batch of 16,
and **1.58x** versus the original cached batch of 16.

| Validator work per 16 ciphertexts | Original groups | Swapped groups |
| --- | ---: | ---: |
| Generate one share for B=16 | 0.429 ms | 0.982 ms |
| Generate four shares for B=4 | 0.551 ms | 1.423 ms |
| Compressed response payload, B=16 | 48 bytes | 96 bytes |
| Compressed response payload, 4x4 | 192 bytes | 384 bytes |

Response sizes exclude metadata. This makes the latency/communication
tradeoff explicit: the best measured combiner configuration asks each
validator to do more work and send eight times the baseline share payload.

For the same committee, preparation's per-validator opening MSMs decrease
from `16 * sum_j MSM(w_j)` to `4 * sum_j MSM(w_j)`. The large positive-offset
MSMs decrease from 15 to 3. The total number of online pairing inputs in
cross-term computation and opening stays the same. Share verification adds
three pairing inputs and three final exponentiations across the four chunks.

An actual maximum-batch-4 setup would also reduce public material from 55,560
to 12,708 source-group points for this profile. That storage reduction is
analytical and is not included in these timing measurements.

Raw logs, parsed confidence limits, source hashes, and machine metadata are in
`results/2026-09-21-shared-dev14/`. The original and swapped layout benchmark
sources are identical apart from the crate import. The earlier exploratory
`criterion-before-12.log` uses the upstream small synthetic fixture and is not
part of the table above.

The swapped crate passed 59 tests on the dev box: 52 unit tests, two additional
cross-orientation/serialization tests, and five example input tests. All layout
benchmark cases also verified recovery of all 16 messages before timing.

## Backend comparisons

The native code uses BLST. Enabling `target-cpu=native` is not evidence of an
AVX-512 MSM implementation. An explicit vector backend must be measured with
its dispatch enabled and disabled under otherwise identical conditions.

MCL's BLS12-381 AVX-512 IFMA implementation targets G1. Swapping groups moves
the large committee MSMs into that group, but ciphertext/share operations
move into G2. Therefore group swapping must be measured across the complete
combiner and validator workloads. MCL BN_SNARK1 is the Ethereum/arkworks
BN254 variant; MCL's separate `BN254` name denotes different parameters.
BN254 also changes the security parameter; it is not an equal-security
substitution for BLS12-381.

### Measured SIMD and curve effects

The [MCL harness](mcl/README.md) uses the same 38-validator, weight-794
committee sizes. It runs single-core primitive workloads, not a full protocol
port. Eighteen independent case invocations (252 result rows) passed direct
MSM, FFT, and pairing checks. Two complete rounds reversed SIMD on/off order;
actual vector callback counts and output digests were checked.

- Large BLS12-381 G1 positive-offset MSMs: **1.95x** faster with MCL AVX-512
  enabled, at B=4,16,64. The callback count was exactly B-1 per repetition.
- Dense synthetic G1 FFT at B=64: **2.51x** faster, with six vector calls.
  B=4/16 FFTs and all measured small partial-decryption MSMs did not dispatch
  SIMD and showed no material SIMD gain.
- BN_SNARK1 versus BLS12-381 with MCL SIMD disabled: G1 committee-opening MSMs
  were **1.65–1.66x** faster, G2 committee-opening MSMs **1.86–1.87x**, and
  opening multi-pairings **1.33–1.35x**. These are curve/backend kernel ratios,
  not measured complete-decryption speedups.

See the [full MCL results](mcl/results/shared-dev14-20260921/SUMMARY.md).

### Native BLST comparison

The native kernel example anchors those numbers against the existing backend:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --locked --release \
  --manifest-path weighted_btx_swapped/Cargo.toml --example kernel_comparison
taskset -c 0 weighted_btx_swapped/target/release/examples/kernel_comparison
```

Milliseconds per operation-shaped workload, one CPU, B=16:

| Work | BLST BLS12-381 | MCL BLS SIMD off | MCL BLS AVX-512 | MCL BN_SNARK1 |
| --- | ---: | ---: | ---: | ---: |
| 15 G1 MSMs, each of 794 points | 114.756 | 151.638 | 77.790 | 88.471 |
| 16x38 weighted G1 opening MSMs | 294.257 | 222.170 | 221.842 | 134.175 |
| 16x38 weighted G2 opening MSMs | 766.594 | 542.671 | 541.507 | 290.223 |

Thus the large G1 MSM workload is approximately **1.48x faster than native
BLST** with MCL AVX-512, rather than the 1.95x same-MCL comparison. The small
opening MSM rows have no SIMD dispatch; their backend differences should not
be attributed to AVX-512.

The cross-backend input generators differ, but use the same sizes, full-field
scalars, affine synthetic point rows, and output-normalization requirements.
Native coefficient byte encoding is outside committee timers as it is in the
protocol's reused MSM path. These comparisons omit backend conversion and
integration overhead and are not summed into protocol estimates.

The primitive measurements above preceded the complete
[`weighted_btx_mcl` protocol port](../weighted_btx_mcl/README.md), which now
supports both curves and both group orientations. Its curve-generic target-group
FFT and final exponentiation replace the BLS-specific path; changing a curve
constant alone would be insufficient. See the [fresh-input end-to-end
measurements](END_TO_END.md) for the integrated results. Group swapping also
changes the orientation of the paper's asymmetric hardness assumption. The
prototype tests establish algebraic correctness, not a new security proof.

References:

- [MCL API and curve parameters](https://github.com/herumi/mcl/blob/master/api.md)
- [MCL AVX-512 implementation](https://github.com/herumi/mcl/blob/master/src/msm_avx.cpp)
- [BLST build configuration](https://github.com/supranational/blst/blob/master/build.sh)
- [CFRG pairing-friendly curves, security discussion](https://www.ietf.org/archive/id/draft-irtf-cfrg-pairing-friendly-curves-14.html#appendix-D)
