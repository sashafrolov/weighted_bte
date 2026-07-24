# BTX reproducibility implementation

This crate implements the construction in
[`papers/btx.pdf`](../papers/btx.pdf), “BTX: Simple and Efficient Batch
Threshold Encryption,” over BLS12-381.

The implementation follows the paper's threshold construction and exposes its
four measured phases:

1. `precompute_batch(B)` computes all cross terms with the FFT middle product.
2. `partial_decrypt(B)` computes one server's succinct G1 share with one MSM.
3. `combine_shares(n)` reconstructs the batch secret with one MSM.
4. `open_batch(B)` performs the final B pairings and unmasks the messages.

It additionally implements client Schnorr proof filtering, direct
server-share verification, the optimistic aggregate server check, and a
fallback that excludes malformed shares.

## Running the paper-parameter example

The largest tested batch in the paper is `B = 512`. The paper does not give a
single headline `(N, t)` committee configuration, so the example documents and
uses an 8-of-16 committee (`N = 16`, `t = 7`):

```console
cargo run --release --example paper_parameters
```

The headline paper result is explicitly single-core. The example defaults to
one Rayon thread for a comparable phase decomposition:

```console
BTX_THREADS=1 cargo run --release --example paper_parameters
```

An optimized local parallel run can be requested separately:

```console
BTX_THREADS=8 cargo run --release --example paper_parameters
```

`BTX_BATCH_SIZE`, `BTX_SERVERS`, and `BTX_THRESHOLD` override the other
parameters. The phase benchmark covers the paper's
`B = 32, 64, 128, 256, 512` series:

```console
RAYON_NUM_THREADS=1 cargo bench --bench phases
```

## Construction details

For a maximum batch size `B_max`, trusted setup samples nonzero `tau` and
publishes:

- `ek = [tau^(B_max+1)]_T`;
- `[tau^i]_2` for `i = 1..2B_max`, excluding `B_max+1`;
- Shamir-share commitments `[share_j(tau^i)]_2` for robustness.

Each server receives independent degree-`t` Shamir shares of
`tau, tau^2, ..., tau^B_max`. A ciphertext is index-free:

```text
ct1 = [r]_1
ct2 = message + r * ek
```

Its position in an ordered batch is chosen only at decryption time. Invalid
proofs become identity coefficients without compacting the batch, preserving
the original slot indices.

For an actual batch of size `B`, the implementation constructs the centered
cyclic kernel

```text
K[0]   = identity
K[d]   = h_{-d}
K[m-d] = h_d          for 1 <= d < B
m      = next_power_of_two(2B)
h_d    = [tau^(B_max+1+d)]_2
```

and computes:

```text
beta = IFFT_GT(pair(FFT_G1(ciphertexts), cached FFT_G2(K)))
```

The first `B` coefficients are exactly the paper's cross terms. This
underspecified indexing step is differential-tested against the literal
quadratic pairing formula for power-of-two and irregular batch sizes.

## Performance engineering

The implementation adapts the tuning techniques from
`silent-threshold-encryption`:

- BLS12-381-specific `blstrs`/BLST arithmetic;
- BLST Pippenger MSMs for partial decryption and threshold combination;
- parallel projective generation followed by batch normalization;
- cached, affine, `G2Prepared` FFT kernels and opening powers;
- one multi-Miller loop/final exponentiation for aggregate verification;
- Rayon parallelism over independent FFT blocks, pairings, and setup work;
- preallocated vectors and positional validity masks;
- movement of the inverse-FFT `1/m` factor into the static G2 kernel;
- a BLST cyclotomic-square, width-5 NAF path for target-group FFT scalar
  multiplication.

The last two optimizations specifically reduce the expensive GT inverse FFT.

### Initial local smoke measurements

On an Apple M4 Pro with Rust nightly 1.98 and `blst 0.3.17`, release-mode
`B = 512`, `N = 16`, `t = 7` measurements produced:

| Phase | 1 Rayon thread | 8 Rayon threads | Paper single-core target |
|---|---:|---:|---:|
| `precompute(B)` | 1158 ms | 307 ms | 491 ms |
| one `partialDecrypt(B)` | 0.71 ms | 0.74 ms | 9.73 ms |
| `combine(8)` | 0.12 ms | 0.16 ms | about 0.31 ms |
| `open(B)` | 164 ms | 19 ms | 87.6 ms |
| core sequential total | 1323 ms | 327 ms | about 598 ms |

The one-thread column uses Criterion medians; the eight-thread column is an
end-to-end orientation run. Absolute single-core parity is not expected on
this machine: the paper uses custom C++ with Clang 21.1.8,
AVX-512/ADX/native vectorization on an Intel Xeon Platinum 8488C. In
particular, the paper does not publish its target-group FFT code, while
`blstrs` does not expose the same vectorized GT backend.

## Paper ambiguities handled here

- The published decryption key contains `2B_max - 1` G2 elements, although one
  prose passage says `2B_max`.
- The useful centered offset range ends at `B_max - 1`; the paper's stated
  `+B_max` endpoint would require an unpublished `tau^(2B_max+1)` power.
- Reconstruction needs `t + 1` shares; one preliminaries sentence says `t`,
  contrary to the algorithms and security definitions.
- The cyclic FFT layout and “truncated inverse FFT” are not specified. This
  crate uses a full, correctness-tested inverse transform and moves its scale
  factor offline.
- The paper benchmarks KDF/unmask work but formally specifies messages in GT
  and does not specify a hybrid format. This crate implements the formal GT
  message space.

## Security and implementation status

This is an academic reproduction prototype, not audited production
cryptography.

- Setup is a trusted-dealer implementation.
- The threshold security statement assumes static corruptions, as in the
  paper.
- Fiat–Shamir Schnorr follows the paper's GGM/ROM discussion. It must not be
  described as a standard plain-ROM simulation-extractable NIZK.
- `serde`/`bincode` derives are intended for trusted local persistence only.
  Untrusted GT wire decoding needs explicit canonical compression and subgroup
  validation.
- BLST MSM and the exposed GT paths should not be assumed side-channel safe
  for production secret processing.
