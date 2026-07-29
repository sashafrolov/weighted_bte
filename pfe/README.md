# PFE reproducibility implementation

This crate implements the partial-fraction batch threshold encryption scheme in
[`papers/pfe.pdf`](../papers/pfe.pdf), “Efficient Batch Threshold Encryption
Using Partial Fraction Techniques,” over BLS12-381.

The implemented protocol is the original two-G1-component construction:

- Construction 1: batch encryption and opening;
- Construction 2: non-interactive threshold pre-decryption;
- Construction 3: robust server-share verification;
- Construction 4: the two-base ciphertext NIZK.

It is the construction called **PFE** in the BTX paper, not PFE's
shorter-ciphertext Construction 5.

## Running the matching example

The example uses exactly the same defaults and threshold convention as the BTX
example:

- `B_max = B = 512`;
- `N = 16`;
- corruption/polynomial-degree threshold `t = 7`;
- `t + 1 = 8` reconstruction shares;
- one Rayon thread.

```console
cargo run --release --example paper_reproduction
```

The settings can be overridden independently:

```console
PFE_BATCH_SIZE=512 \
PFE_SERVERS=16 \
PFE_THRESHOLD=7 \
PFE_THREADS=1 \
cargo run --release --example paper_reproduction
```

`PFE_BATCH_SIZE` must be a power of two. PFE calls an 8-of-16 sharing
configuration “`t = 8`” in the paper, whereas the sibling BTX crate stores the
degree/corruption threshold and calls it `threshold = 7`. This crate follows the
BTX API convention so that the examples are directly comparable.

The phase benchmark covers `B = 32, 64, 128, 256, 512` and reconstruction from
`n = 2, 4, 8, 16` shares:

```console
RAYON_NUM_THREADS=1 cargo bench --bench phases
```

## Protocol

Let the ordered batch slots be the `B`-th roots of unity
`z_i = omega^i`. Define the partial fraction

```text
p_a(x) = 1 / (x + a).
```

Setup chooses a secret `x`, two auxiliary indices `0` and `gamma`, and
publishes

```text
relation_base = [p_0(x) + p_gamma(x) + sum_i p_{z_i}(x)]_1
masking_base  = [p_0(x)]_T
DK_i          = [p_{z_i}(x)]_2.
```

Each scalar `p_{z_i}(x)` is independently Shamir-shared among the servers. A
ciphertext for `message` is

```text
C1 = [r]_1
C2 = r * relation_base
C3 = message + r * masking_base.
```

Construction 4 proves with a two-base Schnorr/DLEQ proof that the encryptor
knows the same `r` in `C1` and `C2`. A server computes one succinct share

```text
SBK_j = sum_i share_{j,i} * C1_i
```

with a G1 MSM. Lagrange interpolation reconstructs

```text
SBK = sum_i p_{z_i}(x) * C1_i.
```

The opening algorithm evaluates the three cross-slot partial-fraction sums in
G1, G2, and GT using FFT convolution, then recovers each one-time pad.

## Correcting the paper's FFT indexing

The PDF says to replace integer slots `{1,...,B}` by
`{omega,...,omega^B}` for a primitive `2B`-th root while retaining the
auxiliary index `-1`. This is not well-defined: `omega^B = -1`, so the last
slot collides with that auxiliary index and a partial-fraction denominator is
zero.

The PFE authors' subsequently released
[reference implementation](https://github.com/entrohpy/batch-enc-partial-fractions)
repairs this by using the complete group of `B`-th roots together with `0` and
an auxiliary `gamma` outside that group. This crate adopts that corrected
convention. It chooses `gamma` as the reciprocal of the first small integer
outside the subgroup—`1/2` for all tested parameters—because the small inverse
also makes the frequency-domain opening cheaper.

PFE setup is exact-size: the encryption key and slot subgroup depend on `B`.
Smaller batches require a separately generated key or valid dummy
ciphertexts; they are not silently compacted.

## Performance engineering

The implementation uses the same basic stack and tuning principles as `btx`:

- BLS12-381-specific `blstrs` and BLST arithmetic;
- BLST Pippenger MSMs for partial decryption, proof batching, and share
  combination;
- batch inversion for setup fractions and Lagrange denominators;
- parallel projective generation followed by batch normalization;
- cached affine and `G2Prepared` opening operands;
- randomized two-MSM batch verification for the client proofs;
- one multipairing/final exponentiation for each robust share check;
- Rayon parallelism over independent pairings and setup work;
- cyclotomic-square, width-5 NAF target-group FFT butterflies;
- fused three-pair opening with one final exponentiation per ciphertext.

There are two additional PFE-specific optimizations.

For the cyclic kernel

```text
d[0] = 0
d[r] = 1 / (1 - omega^(-r)),
```

its Fourier transform has the closed form

```text
FFT(d)[k] = (B - 1) / 2 - k.
```

The implementation therefore multiplies each frequency by the small signed
integer

```text
E[k] = B - 1 - 2k
```

and uses an unnormalized inverse FFT. This avoids full-width 255-bit scalar
multiplication at every frequency. Dedicated signed-small double/add paths are
used in G1, G2, and GT, so negative small values are not accidentally encoded
as expensive near-modulus scalars.

Second, multiplication of each convolution output by
`z_i / gamma - 1` is performed as a frequency shift and a small multiplication.
Together with public keys scaled by `1/(2B)`, this removes all online GT output
scaling and all online G2 scalar multiplication from the opening phase.

## Comparison with the BTX PFE baseline

The BTX paper charges its PFE baseline:

- `B` preprocessing pairings and four FFTs of size `2B`;
- four additional opening pairings per item;
- five pairing terms per item in total.

The corrected implementation released by the PFE authors uses `B`-point
circulant FFTs and algebraically fuses opening to three pairing terms, for four
pairings per item in total. This crate uses that faster path and then applies
the optimizations above. Consequently, it is a reproduction of the PFE
construction at the same parameters, but it intentionally does not reproduce
the slower operation accounting used for BTX's PFE baseline.

A release-mode Criterion run on an Apple M4 Pro, `B = 512`, `N = 16`, `t = 7`,
and one Rayon thread produced:

| Phase | This implementation | BTX paper's PFE target |
|---|---:|---:|
| `precompute(B)` | 872 ms | 817 ms |
| one `partialDecrypt(B)` | 0.707 ms | 9.73 ms |
| `combine(8)` | 0.122 ms | about 0.31 ms |
| `open(B)` | 252 ms | 370 ms |
| summed core phases | 1125 ms | about 1197 ms |

The example's first one-shot end-to-end run was 1113 ms. Absolute comparisons
remain machine- and toolchain-dependent; use the included Criterion harness on
the target machine.

## Verification coverage

The tests include:

- scalar, G1, G2, and optimized GT FFT round trips;
- signed-small GT multiplication against stock `blstrs`;
- the closed-form convolution against a literal quadratic implementation;
- the fused three-pair opening against Construction 1's literal `L1/L2`
  equations;
- full threshold round trips and an arbitrary 8-of-16 subset;
- insufficient, duplicate, and malformed shares;
- optimistic aggregate checking with direct-check recovery;
- proof and ciphertext tampering.

Run them with:

```console
cargo test
cargo check --all-targets
```

## Security and implementation status

This is an academic reproduction prototype, not audited production
cryptography.

- Setup currently uses a trusted dealer.
- The threshold proof assumes static corruptions, as in the paper.
- The paper's Construction 4 transcript proves and hashes only `C1` and `C2`;
  it does **not** bind the masked `C3` component. The released reference code
  behaves the same way. This crate reproduces that relation explicitly, but it
  appears to leave a copy/malleability concern for the claimed CCA property.
- Invalid client proofs reject the complete exact-size batch. The paper
  mentions filtering and reindexing as an extension but does not specify it.
- `serde`/`bincode` derives are suitable only for trusted local persistence;
  production wire decoding needs explicit canonical and subgroup checks.
- The direct BLST GT path relies on `blstrs`' benchmark-only Fp12 export and is
  pinned by `Cargo.lock`.
