# Indexed weighted BTE reproducibility implementation

This crate implements the main indexed weighted batched-threshold-encryption
construction in Figure 2 of
[`papers/weighted_bte_old_paper.pdf`](../papers/weighted_bte_old_paper.pdf)
over BLS12-381. It is the earlier, coordination-dependent design: a ciphertext
chooses one of `n = B_max` indices when it is encrypted, and a decryptable batch
may contain at most one valid ciphertext at each index.

This is an academic reproduction prototype, not audited production
cryptography.

## Construction and API

For positive real-party weights `w_j`, total virtual weight `W`, and corruption
threshold `t < W`, a real-party set is authorized exactly when its total weight
is greater than `t`. Key generation evaluates one degree-`t` Shamir polynomial
on a nonzero root-of-unity domain and assigns `w_j` consecutive virtual points
to party `j`.

The trusted powers-of-tau setup contains `n` G1 points and `2n` G2 points. In
the following formulas `i` is the paper's one-based index; Rust API index `k`
uses exponents `k + 1` for `delta` and `n + 2 + k` for `gamma`. The master
public key additionally contains:

- `delta_i = [msk * tau^i]_1` for every ciphertext index;
- one `[sk_j^-1]_1` verification key per real party; and
- `gamma_(i,j,omega) = [sk_j * f(omega) * tau^(n+1+i)]_2`
  for every index and virtual share owned by party `j`.

The crate exposes the protocol phases directly:

1. `setup` creates the reusable powers-of-tau parameters.
2. `keygen` creates the weighted master public key and one constant-size secret
   key per real party.
3. `encrypt` encrypts an arbitrary byte string at an explicit index.
4. `validate_batch` checks client proofs once, filters invalid ciphertexts, and
   rejects duplicate valid indices.
5. `partial_decrypt` creates one G1 element and one compact DLEq proof for a
   party, independent of its weight and the batch size.
6. `accept_decryption_shares` verifies the submitted proofs, discards malformed
   shares, and deterministically selects the largest-weight responses until it
   has a minimum-cardinality authorized subset `V`.
7. `prepare_decryption` computes the accepted virtual-domain interpolation and
   index-specific G2 opening keys. This result can be cached for the same
   accepted committee and occupied-index set.
8. `precompute_batch` evaluates all ciphertext cross terms with an adaptive
   direct/middle-product path.
9. `open_batch` performs one real-party multi-pairing per valid ciphertext and
   unmasks the byte messages.

`decrypt` is a convenience wrapper for the public combiner phases.

## Tests and benchmarks

```console
cargo test --all-targets --manifest-path weighted_indexed_bte/Cargo.toml
RAYON_NUM_THREADS=1 cargo bench \
  --manifest-path weighted_indexed_bte/Cargo.toml --bench phases
```

The benchmark command above matches the old paper's single-core setting;
change `RAYON_NUM_THREADS` to measure parallel throughput. The tests cover
heterogeneous weighted authorization, malformed client and
server proofs, duplicate indices, invalid-index filtering, arbitrary byte
messages, setup separation, threshold failure, and differential comparisons
of both optimized cross-term paths against the paper's direct formula.
The low-level tests inherited from `btx` also cover scalar/G1/G2/GT FFTs, raw
BLST MSMs and batch normalization, split final exponentiation, and truncated
cyclotomic inverse transforms.

## Solana paper-reproduction example

The example reads the newest `solana_share_weights_*.json` file from
`scripts/data`, selects the profile configured by the top-level
`APPROXIMATION_ERROR` constant, and assigns its integer weights to the real
parties. The constant defaults to `"1/64"`.

```console
RUSTFLAGS="-C target-cpu=native" \
WEIGHTED_INDEXED_BTE_BATCH_SIZE=32 \
WEIGHTED_INDEXED_BTE_THREADS=12 \
cargo run --release --manifest-path weighted_indexed_bte/Cargo.toml \
  --example paper_reproduction
```

Runtime choices are controlled by:

- `WEIGHTED_INDEXED_BTE_WEIGHTS_FILE`;
- `WEIGHTED_INDEXED_BTE_BATCH_SIZE` (also `n = B_max`, minimum 2); and
- `WEIGHTED_INDEXED_BTE_THREADS`.

The example reports setup, key generation, encryption, both proof-validation
phases, committee interpolation/MSMs, the fixed FFT kernel, ciphertext cross
terms, opening, and cold- versus cached-committee decryption totals. It also
reports compressed serialized sizes rather than Rust in-memory sizes. It
generates and validates all `N` party responses as charged by Table 2, then
retains the deterministic minimum-cardinality `N'` subset for interpolation
and opening; both response-volume figures are printed.

For index space `n`, real-party count `N`, and total virtual weight `W`, the
paper's serialized public-material formula is:

```text
pp:   n * |G1| + 2n * |G2|
mpk:  (n + N) * |G1| + nW * |G2|
total: (2n + N) * |G1| + n(W + 2) * |G2|
```

Compressed BLS12-381 encodings have `|G1| = 48`, `|G2| = 96`, and
`|Zp| = 32` bytes. The Fiat-Shamir DLEq proofs need no additional structured
CRS. The example reports that separately as zero bytes.

## Performance engineering

The implementation uses the same primitive stack and optimization approach as
the `btx` and `weighted_btx` crates:

- `blstrs`/BLST BLS12-381 arithmetic, raw BLST Pippenger MSMs, and raw batch
  normalization;
- root-of-unity Shamir evaluation by a scalar FFT;
- a product-tree/FFT Lagrange path for Solana-sized accepted virtual sets;
- index-major contiguous `gamma` blocks so each committee/index MSM reads one
  contiguous range;
- parallel G2 MSMs for committee preparation and one allocation-free raw
  multi-Miller loop per output;
- a reusable centered G2 FFT kernel and sparse ciphertext input vector for all
  cross terms at once; the middle-product form needs
  `next_power_of_two(2n)` transform slots, avoiding the old draft's full
  roughly-`3n` convolution without changing its outputs, and the kernel can be
  retained across committee-key rotations that reuse the same powers-of-tau
  CRS;
- an adaptive direct multi-pairing fallback when very few of the `n` indices
  are occupied, avoiding a full `2n` transform for sparse batches;
- split easy/hard final exponentiation, one-inversion batched easy parts,
  truncated cyclotomic inverse FFTs, and the BLS-x/Frobenius target-group
  scalar decomposition inherited from `btx`; and
- Rayon parallelism at independent proof, MSM, FFT-stage, pairing, and output
  boundaries while avoiding nested worker pools.

## Paper issues and implementation resolutions

- **Masked payload is not proof-bound.** Figure 2's client DLEq statement binds
  the two G1 elements but not the masked payload. An attacker can change only
  that payload, reuse the valid proof, and submit the modified ciphertext to
  the batch-decryption oracle, contradicting the stated CCA game. This crate
  binds the index and complete masked payload into the Fiat-Shamir challenge.
  Proof size and group arithmetic are unchanged.
- **Nonzero sampling.** Setup samples `tau`, every `sk_j`, and encryption
  randomness `r` from `Zp`, even though powers are intended to be a CRS,
  `sk_j` is inverted, and zero randomness exposes its message pad. This crate
  samples `tau`, `msk`, every party scalar, and `r` from `Zp*`; the difference
  from uniform `Zp` is negligible.
- **Evaluation domain.** The paper samples `Omega` as a subset of `Zp`; Shamir
  reconstruction at zero requires that zero not be a share point. This crate
  uses distinct nonzero roots of unity.
- **One-index setup.** Encryption requires a helper `u != i`, so Figure 2 is
  undefined for `n = 1`. `setup` requires `n >= 2`.
- **Duplicate indices.** The construction's set `E` and cross-term formula
  require distinct valid indices, but the pseudocode does not state the
  rejection rule. This crate rejects a batch with duplicate proof-valid
  indices instead of silently producing incorrect openings.
- **Lagrange arity.** `Lagrange(Omega', t)` is ambiguous when the accepted set
  contains more than `t + 1` virtual shares. This crate uses the unique
  all-point Lagrange coefficients at zero for the full accepted set.
- **Hash input and byte messages.** Figure 2 types `H` over a source group but
  hashes a target-group pairing value. This crate uses a domain-separated
  SHA-256 counter-mode KDF over its canonical, identity-safe GT encoding,
  allowing arbitrary-length byte messages.
- **Invalid ciphertexts.** Invalid proofs and out-of-domain indices remain
  explicit and open as `None`; valid entries are not compacted because their
  chosen indices are protocol data. This avoids letting one malformed wire
  value suppress unrelated valid slots.
- **Authorization wording.** The formal access structure authorizes weight
  strictly greater than `t`; some security-game lines compare party count to
  `t`. This crate consistently applies total accepted weight `> t`.
- **Exact-half Solana profile.** With the current `1/64` allocation,
  `W = 6486`, `q = 3244`, and therefore `t = q - 1 = 3243`. This is the
  configuration analogous to the other crates, but it does not satisfy the
  old draft's stronger robustness premise `t < floor(W/2)` because
  `floor(W/2) = 3243`. The example prints this explicitly; algebraic
  correctness and the paper's more general CCA threshold condition still
  permit the configuration.
- **Proof-system assumption.** Compact Fiat-Shamir Chaum-Pedersen proofs match
  the paper's concrete size table, but they should not be treated as a proven
  simulation-extractable NIZK in the plain random-oracle model. This crate does
  not claim to close that proof-level gap.
- **Security-game/reduction typos.** The robustness game exposes all party
  keys, its corruption guard uses party count rather than combined weight, and
  the page-19 reduction names the ciphertext's index component where its
  second G1 component is needed. These are treated as draft-description issues;
  the implementation follows the Figure 2 algebra and weighted access
  structure.

The trusted dealer is intentional. Replacing it with a distributed setup or
DKG is separate protocol work. The derived `serde`/`bincode` representations
are intended for authenticated local persistence; untrusted deserialized CRS
or key blobs require explicit dimension, identifier, and subgroup validation.
