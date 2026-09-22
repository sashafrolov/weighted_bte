# Full weighted BTX protocol over MCL

This directory implements the complete in-memory weighted BTX pipeline over
MCL: trusted-dealer setup, encryption and Fiat–Shamir proofs, client validation,
validator shares, weighted share acceptance, reusable committee preparation,
FFT cross terms, and message opening. Both source-group orientations work over
BLS12-381 and Ethereum BN254. The earlier [MCL kernel harness](../experiments/mcl/README.md)
measures isolated arithmetic; this implementation runs those operations inside
the protocol and verifies recovered plaintexts.

This is an experimental implementation with a trusted dealer, not an audited
production system. `Material` keeps public setup material and validator secrets
in one object for local experiments. There is no distributed key generation,
network transport, or untrusted wire decoder. Objects and transcripts are not
wire-compatible with the Rust/BLST implementation. Changing curves changes
parameters and security assumptions; correctness tests do not establish a new
security proof.

The end-to-end measurements are recorded in
[the three-layout protocol results summary](../experiments/results/2026-09-21-end-to-end-8x2/SUMMARY.md),
with raw phase samples and correctness logs alongside it. The earlier
[two-layout results](../experiments/results/2026-09-21-end-to-end/SUMMARY.md)
remain separately archived.

The [paper-table reproduction](../experiments/paper_tables/README.md) adds
two-thirds-threshold weight profiles and workloads of 16-256 messages. Its
`paper_benchmark` executable accepts total message count, working chunk size,
and maximum setup size separately, and emits fresh-input phase samples together
with exact public-key and response payload sizes. The paper-table report uses
arithmetic means.

## Curves, groups, and API

MCL is pinned to commit `cbb18eb08b86129cf936a6436b5e6c68a2ce8ddf`.

| CLI curve | MCL parameter |
| --- | --- |
| `bls12_381` | `mcl::BLS12_381` |
| `bn254` | `mcl::BN_SNARK1`, the Ethereum/arkworks BN254 curve |

MCL also has a separately named `BN254` parameter. This implementation uses
`BN_SNARK1` for the `bn254` CLI option.

| API alias / CLI orientation | Ciphertexts, proof commitments, validator shares | Weighted public keys, opening keys, committee FFT kernel |
| --- | --- | --- |
| `wbtx::Normal` / `normal` | G1 | G2 |
| `wbtx::Swapped` / `swapped` | G2 | G1 |

Messages and encrypted payloads are in GT. Pairing calls always pass the actual
G1 operand first and G2 operand second. A validator emits one source-group
share per batch, so four batches require four shares from each validator.

Initialize MCL for one curve before creating objects. Curve initialization,
SIMD callback changes, and parallel-executor changes must happen between
operations, after all worker jobs have joined. The protocol records and checks
curve context as well as setup context.

The main API is in [protocol.hpp](protocol.hpp):

| Method | Work and result |
| --- | --- |
| `keygen(L, weights, t)` | Trusted-dealer setup; authorization requires accepted weight strictly greater than `t` |
| `encrypt(material, message)` | Fresh CSPRNG encryption randomness and Schnorr proof binding both ciphertext components and setup context |
| `validate(material, ciphertexts)` | Check ciphertext proofs and group membership; preserve invalid slots as identity contributions in a `ValidatedBatch` |
| `partial(material, party, batch)` | Compute one validator's batch share, bound to setup and ordered batch digest |
| `accept(material, batch, shares)` | Random-linear batched share verification, individual fallback and blame, duplicate rejection, weighted authorization |
| `prepare(material, accepted)` | Interpolation coefficients, weighted opening MSMs, and committee FFT kernel in a reusable `Committee` |
| `precompute(committee, batch)` | Ciphertext-dependent FFT cross terms, bound to the committee and exact batch |
| `open(committee, accepted, batch, ciphertexts, cross)` | Opening multi-pairings and payload recovery; invalid ciphertext slots return empty optionals |

The implementation also exposes direct proof/share verification helpers.
[protocol_math.hpp](protocol_math.hpp) contains scalar/group/GT FFTs,
product-tree and derivative-FFT interpolation, batch inversion, transcript
helpers, and MSM wrappers. [thread_pool.hpp](thread_pool.hpp) provides a shared
pool whose waiting callers execute queued work, including nested protocol loops.

## What the benchmark measures

[benchmark.cpp](benchmark.cpp) uses the repository's approximation-error `1/16`
allocation: 419 validators, total virtual weight 1,576, and minimum reconstruction
weight 789. Selecting the heaviest validators with increasing-index tie breaks
produces 38 validators with weight 794. Setup uses `L=16` and `t=788`.
[prepare_profile.py](prepare_profile.py) writes the complete weight distribution
and threshold, rather than replacing the committee with a synthetic one.

Each case processes 16 newly encrypted ciphertexts with fresh proofs and fresh
validator shares. It compares one batch of 16, four batches of 4, and eight
batches of 2 using the same setup. The six cases are cold/cached committee
preparation crossed with these three layouts:

- **Cold:** preparation is timed once per iteration. The 4x4 and 8x2 cases reuse
  that result across all four or eight chunks, respectively.
- **Cached:** an untimed fixture supplies only the committee-dependent
  preparation. Ciphertexts, proofs, shares, acceptance results, and cross terms
  are freshly computed during every measured iteration.

Committee reuse requires the same setup, curve, batch size, and canonical
accepted party set. Its context identifier binds the setup, batch size, and
parties; the setup identifier binds curve and orientation. Accepted shares and
cross terms retain their exact ordered-batch digest. Reusing committee MSMs
does not reuse or skip share verification.

| Phase label | Timed work |
| --- | --- |
| `encryption` | Encrypt all 16 messages and generate proofs |
| `validation` | Validate every ciphertext in every chunk |
| `share_generation` | Generate all selected validators' shares for every chunk |
| `acceptance` | Freshly verify and accept each chunk's submitted shares |
| `preparation` | One committee preparation in cold mode; cache lookup in cached mode |
| `precompute` | Compute all chunks' ciphertext-dependent cross terms |
| `opening` | Open all ciphertexts using their accepted shares and cross terms |
| `combiner` | Wall time from availability of all local shares through acceptance, preparation, cross terms, and opening |
| `end_to_end` | Wall time from encryption through completion of all openings |

Phases execute in sequence. Encryption, validator work, chunks, and independent
inner protocol loops share one bounded thread pool. This measures the aggregate
work of all validators simulated on one machine, not distributed validator
latency. Trusted setup, plaintext construction, cached fixtures, warmups, and
network transport are outside the reported samples. Setup timing is logged
separately. Every output is checked against its plaintext after the ending
timestamp, including warmup iterations.

## AVX-512 and comparison boundaries

The pinned backend's optional AVX-512 IFMA callbacks accelerate **BLS12-381 G1**
vector MSMs and batched scalar multiplications. They do not provide a G2 or
BN_SNARK1 AVX-512 path. Swapping groups moves weighted committee MSMs into G1.
Only operations reaching the backend's vector dispatch can benefit; enabling
AVX-512 does not imply that every protocol phase uses it.

`off` clears MCL's `G1::mulVecOpti` and `G1::mulEachOpti` callbacks while retaining
the ordinary backend and all other compiler settings. `avx512` requires both
callbacks to be available and wraps them with atomic dispatch counters. An
unavailable callback is an error, not a silent fallback. CSV `avx_msm` and
`avx_each` columns count calls inside each measured phase. The matrix does not
request an AVX-512 mode for BN254. `target-cpu=native` in a Rust build is not
evidence of these MCL vector operations executing.

The MCL benchmark deliberately uses the public checked APIs. It checks GT
membership of plaintexts and ciphertext payloads and source-group membership
of ciphertext/proof points and submitted shares. It also rechecks material and
curve context. The native Rust benchmark starts with typed, locally generated
BLST objects and does not repeat all these membership checks. Consequently,
cross-backend validation/encryption/acceptance timings include different input
checking costs.

Other implementation differences also remain. MCL copies MSM inputs because
its API may normalize mutable points, and its transcripts serialize full Fp12
payloads. The curve-generic MCL cross-term path applies full final exponentiation
to frequency-domain pairings, then performs a full inverse FFT in GT. The native
BLS12-381 implementation uses split final exponentiation and an inverse FFT
restricted to needed coefficients. The normal MCL orientation caches G2 kernel
line tables; the swapped G1 kernel cannot use that static G2 preparation.

Compare MCL `off` against MCL `avx512` to isolate the enabled vector callbacks.
Compare curves within MCL with the callbacks disabled in both runs to assess
that curve change.
Native BLST versus MCL is a comparison of complete implementations, including
these validation and algorithm choices, rather than a pure curve or SIMD ratio.

## Build, test, and run

Run these commands from the repository root. The complete matrix requires Linux,
`taskset`, an AVX512IFMA-capable x86-64 CPU, a C++17 compiler, make, Python 3, Git,
and Rust/Cargo. A manual `off` run can use another MCL-supported machine. The
scripts build dependencies in local directories without system installation.

```sh
bash weighted_btx_mcl/build.sh
./weighted_btx_mcl/build/tests

cargo test --locked --manifest-path weighted_btx/Cargo.toml --lib --tests --examples
cargo test --locked --manifest-path weighted_btx_swapped/Cargo.toml --lib --tests --examples

CARGO_TARGET_DIR="$PWD/weighted_btx/target" RUSTFLAGS="-C target-cpu=native" \
  cargo build --locked --release --manifest-path weighted_btx/Cargo.toml --example end_to_end
CARGO_TARGET_DIR="$PWD/weighted_btx_swapped/target" RUSTFLAGS="-C target-cpu=native" \
  cargo build --locked --release --manifest-path weighted_btx_swapped/Cargo.toml --example end_to_end
```

`build.sh` builds the pinned static MCL dependency if missing, compiles the
protocol benchmark and tests with `-O3 -DNDEBUG -std=c++17`, and writes profile
and compiler metadata under `weighted_btx_mcl/build/`. Its overrides are
`WBTX_BUILD_DIR`, `MCL_SOURCE`, `CXX`, and dependency-build `JOBS`. `MCL_SOURCE`
must point to an already-built checkout of the pinned revision containing
`lib/libmcl.a`; automatic dependency building applies only to the default path.
The matrix
runner expects the default MCL build directory and the native executable paths
shown above; use the manual command for a custom build directory.

The test binary exercises both curves and orientations serially and through a
four-thread pool. It checks literal interpolation and cross-term formulas,
scalar/G1/G2/GT FFTs against direct transforms, GT powers against generic
exponentiation, malformed proofs and shares, weighted thresholds, identity and
invalid slots, ordered-batch/setup/committee binding, fresh-batch cache reuse,
and nested-pool completion and exception propagation. The matrix does not run
these tests automatically; run them successfully before measurement.

A short manual protocol run is:

```sh
taskset -c 0-11 ./weighted_btx_mcl/build/benchmark \
  bls12_381 off normal weighted_btx_mcl/build/profile.txt 12 2 0 \
  > /tmp/weighted-btx-mcl-smoke.csv
```

Its positional arguments are `curve simd orientation profile threads samples
warmup`. Each invocation runs all three layouts and both cache modes. Select
`bls12_381 avx512 swapped` for the swapped BLS vector implementation or
`bn254 off normal` / `bn254 off swapped` for the BN254 implementations.

Run the full matrix serially on an otherwise idle machine, choosing an available
CPU set and matching thread budget:

```sh
CPUS=0-11 THREADS=12 SAMPLES=15 WARMUP=2 ROUNDS="1 2" \
  bash weighted_btx_mcl/run_matrix.sh \
  "$PWD/experiments/results/2026-09-21-end-to-end-8x2"
```

The defaults match that command. The matrix includes native BLST normal/swapped,
MCL BLS12-381 off/AVX-512 in both orientations, and MCL BN254 in both orientations.
The second round reverses implementation order. `SKIP_BLST=1` skips native runs;
`CPUS`, `THREADS`, `SAMPLES`, `WARMUP`, and `ROUNDS` control the matrix. The native
runner's standalone defaults are 11 samples and 2 warmups, but the matrix passes
its own sample count explicitly.

MCL writes per-iteration phase CSV files and stderr correctness/setup logs;
native runners write JSON with raw samples and phase summaries. The final
`summarize.py` call combines matching raw files into `summary.csv`,
`summary.json`, and `SUMMARY.md`, with median and linearly interpolated p10/p90
values. These percentiles describe sample variation; they are not confidence
intervals. Keep configurations with different run conditions in separate output
directories. To regenerate summaries without running benchmarks:

```sh
python3 weighted_btx_mcl/summarize.py experiments/results/2026-09-21-end-to-end-8x2 \
  --layouts 1x16,4x4,8x2
```

The summarizer's default layouts remain `1x16,4x4` for compatibility with the
archived two-layout run; the matrix runner passes all three explicitly.
