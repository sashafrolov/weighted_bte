# Experimental swapped weighted BTX

This standalone crate swaps the BLS12-381 source groups in `../weighted_btx`.
The scalar field, Shamir sharing, weighted interpolation, target group, split
final exponentiation, and specialized BLS12-381 target-group FFT are unchanged.
It is an implementation experiment, not a new security proof or a compatible
wire format. It uses distinct setup, proof, ordered-batch, and committee-context
domain separators.

| Component | Original | Swapped |
| --- | --- | --- |
| Ciphertext first component | G1 | G2 |
| Schnorr commitment | G1 | G2 |
| Validator's batch share | G1 | G2 |
| Public weighted and verification keys | G2 | G1 |
| Committee FFT kernel and opening keys | G2 | G1 |
| Ciphertext payload / recovered message | GT | GT |

All pairing calls keep BLST's actual argument order `e(G1, G2)`: for example,
share verification is `product_i e(V[j,i], C[i]) = e(G1_generator, sigma[j])`.
Proofs bind both ciphertext components and the setup; share and precomputation
objects retain the original ordered-batch and accepted-committee checks.

## Expected tradeoffs

Public weighted MSMs and the committee FFT move to G1. Client encryption,
proof checks, partial decryption, and the ciphertext FFT move to G2. The
experiment must measure both sides of that trade, including the sum of cross
term computation and opening.

BLST prepares Miller-loop line tables only for G2. The original static G2 FFT
kernel can cache these tables across batches. The swapped static G1 kernel
stores affine points instead and runs unprepared loops against each batch's
transformed G2 ciphertexts. For each frequency the kernel footprint drops
from 19,648 bytes of prepared lines to a 96-byte G1 affine point, but the
per-batch Miller loops regain G2 line computation. Opening keeps BLST's
aggregate multi-Miller loop. Its public API has no aggregate prepared-lines
variant; separate prepared loops would duplicate the loop squarings across
parties. G2-share line caching is consequently not assumed to be a free gain.

Compressed representation sizes (excluding indices, context identifiers, and
container framing):

| Component | Original | Swapped |
| --- | ---: | ---: |
| Each public-key point | 96 bytes | 48 bytes |
| Each batch share's group element | 48 bytes | 96 bytes |
| Schnorr commitment + scalar response | 80 bytes | 128 bytes |
| Canonical ciphertext transcript encoding, nonidentity GT payload | 417 bytes | 513 bytes |

The ciphertext row uses the existing 289-byte canonical GT transcript
encoding for a nonidentity payload; the identity GT element uses one byte,
making those ciphertext encodings 129 and 225 bytes respectively. These are
canonical compressed sizes, not a claim about `bincode`
container sizes. Public affine point storage falls from 192 to 96 bytes per
point. Public-key point count remains `(2L-1)W + LN`.

## Validation and benchmarks

```sh
cargo test --manifest-path weighted_btx_swapped/Cargo.toml --lib --tests --examples
cargo bench --manifest-path weighted_btx_swapped/Cargo.toml --bench phases
RAYON_NUM_THREADS=1 cargo bench --manifest-path weighted_btx_swapped/Cargo.toml --bench batch_layout
```

The 52 copied unit tests cover weighted round trips, FFT middle products
against the literal pairing formula, malformed shares and ciphertexts,
identity points, threshold boundaries, context binding, interpolation, and
BLST arithmetic. Two additional integration tests compare both group
orientations with identical deterministic scalar draws over batch sizes
1, 2, 3, 4, 7, and 16 (including corrupted proofs and shares), and round-trip
serialized setup/ciphertext/share objects. Five copied example tests validate
allocation input handling.

`phases` retains the original individual phases and `precompute_and_open`.
`batch_layout` compares one batch of 16 with four batches of 4 using one L=16
setup and the same 16 ciphertexts. It includes fresh randomized share checks
in each combiner measurement; the cold/reuse case prepares the committee
once and reuses that work across the four small batches. The cached cases
exclude committee preparation. Validator share generation is measured
separately, with all four shares included in the 4x4 measurement. It uses
the same `BATCH_LAYOUT_WEIGHTS_FILE` and `BATCH_LAYOUT_ERROR` controls as the
original benchmark. Run variants serially on the same otherwise idle host.

The paper reproduction example is also copied with the swapped import and
correct G1 public-key size reporting; its existing `WEIGHTED_BTX_*`
environment variables are unchanged.

## Fresh-ciphertext end-to-end comparison

The original and swapped crates each contain `examples/end_to_end.rs`; the
files differ only in their crate import. The runner uses one L=16 setup and
measures 1x16, 4x4, and 8x2, with committee preparation either inside each
iteration or cached before measurement. In the cold split cases, preparation
runs once and its result is reused across all four or eight chunks.

Every iteration newly encrypts 16 messages and creates their proofs, validates
all ciphertexts, generates shares from all selected validators, freshly accepts
those shares, computes cross terms, and opens all messages. Every output is
checked against its plaintext. The single machine simulates all validators
using one global Rayon pool; these are local aggregate timings and exclude
network latency. Plaintext construction, trusted setup, and cached fixtures
are outside the interval. Setup and cached preparation times are reported
separately.

```sh
cargo run --release --manifest-path weighted_btx/Cargo.toml --example end_to_end -- \
  --threads 12 --samples 11 --warmup 2 > original-e2e.json
cargo run --release --manifest-path weighted_btx_swapped/Cargo.toml --example end_to_end -- \
  --threads 12 --samples 11 --warmup 2 > swapped-e2e.json
```

Run the commands serially on the benchmark machine. JSON includes raw samples,
median/p10/p90 for the complete pipeline, the combiner after all shares exist,
and each phase. Percentiles use linear interpolation and describe sample
variation; they are not confidence intervals. `--format csv` emits summary
rows. `--weights-file` and `--approximation-error` override the allocation.
Environment defaults are `E2E_THREADS`, `E2E_SAMPLES`, `E2E_WARMUP`,
`E2E_FORMAT`, `BATCH_LAYOUT_WEIGHTS_FILE`, and `BATCH_LAYOUT_ERROR`.
