# MCL curve and AVX-512 kernel experiment

The completed shared-dev14 run is in
[results/shared-dev14-20260921/SUMMARY.md](results/shared-dev14-20260921/SUMMARY.md),
with all raw CSVs, correctness logs, compiler/CPU metadata, exact source snapshot,
and build/run logs alongside it. It used AMD EPYC 9275F, CPU 0, and Clang 18.1.3.
All 18 cases passed their correctness checks. Actual dispatch counters and output
digests agreed with the expected SIMD/non-SIMD behavior across both rounds.

This directory measures the arithmetic operation shapes used by weighted BTX.
It is an isolated benchmark harness, **not a second implementation of the
encryption scheme**, and it does not change the native Rust/blst implementation.
There are no proof generation/validation, ciphertext validation, interpolation,
setup, transcript binding, share acceptance, GT inverse FFT, or complete
decryption paths here. Do not sum these rows and call the result an end-to-end
protocol time. In particular, the native implementation uses split final
exponentiation, while the final-exponent row here measures the ordinary primitive.

The comparison uses one pinned backend and the same code/operation sizes for
both curves:

- MCL commit `cbb18eb08b86129cf936a6436b5e6c68a2ce8ddf`.
- BLS12-381: MCL `BLS12_381`.
- Ethereum/arkworks BN254: MCL **`BN_SNARK1`**, whose scalar modulus is
  `21888242871839275222246405745257275088548364400416034343698204186575808495617`.
  MCL's separately named `BN254` is a different curve and is deliberately not used.

Curve migration changes parameters and security assumptions; measurements here
are arithmetic comparisons, not interoperability or security equivalence claims.
The generated points and scalars are deterministic synthetic public inputs.
Points occupy separate rows for the committee operations, but are generated as
an arithmetic sequence to keep untimed input preparation inexpensive. Scalars
are full-field hash-derived values. The harness is not suitable for secret data.

## SIMD control and evidence

`off` clears only `G1::mulVecOpti` and `G1::mulEachOpti`, leaving the ordinary
MCL backend, GLV scalar paths, and all other compilation settings unchanged.
`avx512` requires the callbacks installed by MCL for BLS12-381 on a machine with
AVX512IFMA. It wraps them with call counters; absence is a fatal error, not a
silent fallback. CSV columns record actual dispatch calls per measured repetition.
This measures actual AVX-512 code, not a `target-cpu=native` build assumption.
These callbacks do not accelerate G2 or BN_SNARK1 in this MCL build.

At the pinned revision, `mulVec` dispatches at 128 or more inputs; `mulEach`
dispatches complete blocks of 16 inputs. The group FFT gathers each stage's
nontrivial twiddle multiplications into one `mulEach` call. Both SIMD modes use
this same transform and allocation pattern. Its implementation differs from
the native Rust FFT, so backend-level rows must not be presented as native
protocol speedups. B=4 and B=16 do not reach the FFT vector threshold in this
harness; B=64 does.

## Workload

`prepare_profile.py` reads the repository's supplied stake profile (error 1/16),
with total virtual weight 1,576 and reconstruction requirement 789. It selects
validators by descending weight and increasing-index ties until the threshold
is reached: 38 validators and accepted virtual weight 794. The script writes
the exact selected weights and records the indices in metadata.

Rows correspond to:

| Row | Work per repetition |
| --- | --- |
| `partial_decrypt_one_msm` | Recompute q^1..q^B and one B-point MSM |
| `verification_key_msms` | B MSMs of 38 points, followed by normalization |
| `committee_opening_msms` | For each of B slots, one MSM per validator using that validator's selected weight; normalize outputs |
| `committee_positive_msms` | B−1 MSMs of 794 points |
| `group_fft` | Dense group FFT of size 2B, including copy, scratch allocations, and normalization |
| `opening_multi_pairings` | B independent 38-pair Miller products and final exponentiations |
| `g2_line_preparation` | Prepare 2B G2 points |
| `middle_prepared_miller_loops` | 2B prepared Miller loops, without final exponentiation |
| `middle_final_exponents` | 2B ordinary final exponentiations of fixed Miller outputs |

Both G1 and G2 rows are emitted for all group operations. The current native
orientation uses G1 for ciphertexts and partial shares, G2 for committee keys;
swapping reverses those roles. Pairings still take a G1 and a G2 point in either
orientation. The FFT row is dense and therefore does not reproduce the native
half-zero ciphertext transform or every sparsity detail of the committee kernel.
No row includes multithreading: comparisons are pinned to one CPU.

Before recording results, each invocation checks:

- MSMs of the full committee, each validator weight, party count, and batch size
  against a direct sum of independent scalar multiplications.
- Group FFT against direct 8-point DFT, and full-size inverse roundtrip.
- Multi-pairing against a product of independent pairings.
- Prepared pairing against ordinary pairing.

All correctness checks are outside measured intervals. Outputs are serialized
and consumed outside measured intervals to keep the work observable.

## Run

Requires an x86-64 Linux machine with AVX512IFMA, C++17 compiler, make, Python,
git, and `taskset`. No system installation or administrator privileges are used.

```sh
bash experiments/mcl/build.sh
bash experiments/mcl/run.sh experiments/mcl/results/my-run
python3 experiments/mcl/summarize.py experiments/mcl/results/my-run
```

Defaults: B=4,16,64; two complete rounds, reversing BLS SIMD order in round 2;
five samples per case; adaptive repetitions aiming for at least 50 ms per
sample. Large operations may exceed that floor. CSV p10/p90 are descriptive
sample quantiles, not confidence intervals. Raw independent rounds should be
checked for consistency. Available settings: `BENCH_CPU`, `BATCHES`, `ROUNDS`,
`SAMPLES`, `SAMPLE_MS`, `MCL_BUILD_DIR`, `JOBS`, and `CXX`.

Sources for backend behavior:

- [Pinned MCL curve definitions](https://github.com/herumi/mcl/blob/cbb18eb08b86129cf936a6436b5e6c68a2ce8ddf/include/mcl/curve_type.hpp)
- [Pinned callback registration](https://github.com/herumi/mcl/blob/cbb18eb08b86129cf936a6436b5e6c68a2ce8ddf/src/pairing_impl.hpp)
- [Pinned MSM/mulEach dispatch](https://github.com/herumi/mcl/blob/cbb18eb08b86129cf936a6436b5e6c68a2ce8ddf/include/mcl/ec.hpp)
- [Pinned SIMD compilation flags](https://github.com/herumi/mcl/blob/cbb18eb08b86129cf936a6436b5e6c68a2ce8ddf/Makefile)
