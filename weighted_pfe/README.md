# Partial-fraction weighted BTE

This crate implements Construction 2 of
[`papers/weighted_batch_threshold_encryption.pdf`](../papers/weighted_batch_threshold_encryption.pdf):
the partial-fraction weighted batch threshold encryption scheme. It uses
BLS12-381 through `blstrs` and BLST, assigns multiple virtual Shamir shares to
each weighted real party, and keeps that party's pre-decryption response to one
G1 element for the complete batch, independent of its weight.

This is a research and reproduction implementation. It has not been audited
and should not be used as production cryptography. In particular, the current
paper draft does not contain a complete CCA-security statement or proof for
Construction 2; see [Security status and paper issues](#security-status-and-paper-issues).

## Construction mapping

Let party `j` have positive integer weight `w_j` and own a contiguous set
`Omega_j` of that many virtual Shamir evaluation points. Let

```text
W = sum_j w_j
```

be the total virtual weight. The corruption threshold is `t < W`; a committee
is authorized exactly when its weight is greater than `t`, so the minimum
reconstruction weight is `q = t + 1`.

Setup samples a degree-`t` polynomial `Z` with `Z(0) = z`. For

```text
p_a(X)       = 1 / (X + a)
g_{j,i}(X)   = p_{rho_j + alpha_i}(X)
u_{j,i}(X)   = g_{j,i}(X)^2
               + p_{-1}(X) / (alpha_i + 1)
               + p_0(X) / alpha_i,
```

the implementation maps the paper's objects as follows.

| Paper object or phase | Implementation |
|---|---|
| `ek = [z p_0(x)]_T` | `EncryptionKey` |
| `sk_j = (g_{j,i}(x))_i` | `PartySecretKey`, exactly `B` scalars regardless of `w_j` |
| `vk_{j,i} = [g_{j,i}(x)]_2` | `PublicDecryptionKey::verification_keys`, `N * B` G2 points |
| `D_{omega,i} = [Z(omega) g_{owner(omega),i}(x)]_2` | the slot-major `d1` table, `W * B` G2 points |
| `U_{omega,i} = [Z(omega) u_{owner(omega),i}(x)]_2` | the slot-major `d2` table, `W * B` G2 points |
| `V = [z(p_{-1}(x) + p_0(x))]_2` | `global_key`, one G2 point |
| `ct_i = ([r_i]_1, m_i + r_i ek, pi_i)` | `Ciphertext { first, second, proof }` |
| `sigma_j = sum_i g_{j,i}(x) ct_i[1]` | `partial_decrypt`, one size-`B` G1 MSM |
| ciphertext and share checks | `validate_batch`, `verify_decryption_share`, and `accept_decryption_shares` |
| weighted interpolation and opening keys | `prepare_decryption` |
| the two Cauchy cross-term families | `precompute_batch` |
| recovery of all `m_i` | `open_batch` |

`keygen` and `keygen_with_rng` implement a trusted-dealer setup. Virtual
points are a prefix of a nonzero root-of-unity domain whose size is the next
power of two at least `W`; evaluating `Z` uses a scalar FFT. The internal FFT
domain padding does not add virtual shares or public-key entries.

The batch size `B` must be a nonzero power of two. This construction is
exact-size: its `alpha_i` values, every party secret key, and both `W * B`
public tables depend on `B`. Every call to `validate_batch` therefore expects
exactly `B` ciphertexts. A shorter logical batch needs valid dummy
ciphertexts, or a separately generated setup; the crate does not silently
compact or reindex a batch. If any client proof is invalid, validation rejects
the whole batch, matching Construction 2's `PreDec` pseudocode.

## Public API

The usual phased flow is:

1. Generate material with `keygen(B, &party_weights, t)`.
2. Encrypt each target-group message with `encrypt` or `encrypt_with_rng`.
3. Call `validate_batch` once to batch-check the client proofs and cache the
   ordered G1 inputs.
4. Have each participating real party call `partial_decrypt` with its
   `PartySecretKey` and the validated batch.
5. Call `accept_decryption_shares`. It first performs a randomized aggregate
   check, falls back to direct checks only if the aggregate fails, records bad
   parties, and selects a deterministic minimum-cardinality authorized subset
   (largest weights first, with party index as the tie-break).
6. Build the setup-specific `CauchyKernel`, then call `prepare_decryption` for
   the accepted committee and `precompute_batch` for the ciphertext batch.
7. Recover all messages with `open_batch`.

`decrypt` is a convenience wrapper for the combiner side of steps 3 and 5--7
when decryption shares have already been produced. The explicit phased API is
needed for normal distributed use because producing a share requires the
`ValidatedBatch` returned in step 3.

All messages are formal `Gt` elements, as in the paper. The crate does not
define a byte-message hybrid encryption format, KDF, or authenticated-data
encoding.

## Running tests, benchmarks, and the Solana reproduction

From the repository root:

```console
cargo test --all-targets --manifest-path weighted_pfe/Cargo.toml
```

The tests cover heterogeneous weighted authorization, malformed-share blame
and fallback, cross-setup/batch/committee binding, serialization dimensions,
proof tampering, optimized-versus-quadratic Cauchy transforms, and optimized
opening versus the displayed Construction 2 formula.

The Criterion benchmark separates client checking, one-party partial
decryption, batched share acceptance, committee preparation, Cauchy
precomputation, opening, and setup. It measures `B = 32, 64, 128, 256, 512`:

```console
RUSTFLAGS="-C target-cpu=native" \
RAYON_NUM_THREADS=12 \
cargo bench --manifest-path weighted_pfe/Cargo.toml --bench phases
```

The reproduction example reads a generated Solana allocation from
`scripts/data`, maps its `share_count` to `W`, maps its positive `weights`
entries to real parties, and passes `reconstruction_threshold - 1` as `t`.
Its top-level `APPROXIMATION_ERROR` constant selects the allocation profile and
defaults to `"1/64"`.

```console
RUSTFLAGS="-C target-cpu=native" \
WEIGHTED_PFE_BATCH_SIZE=32 \
WEIGHTED_PFE_THREADS=12 \
WEIGHTED_PFE_REPETITIONS=1 \
cargo run --release --manifest-path weighted_pfe/Cargo.toml \
  --example paper_reproduction
```

The runtime overrides are:

- `WEIGHTED_PFE_WEIGHTS_FILE`: an explicit allocation JSON; otherwise the
  lexicographically newest `scripts/data/solana_share_weights_*.json` is used;
- `WEIGHTED_PFE_BATCH_SIZE`: exact power-of-two batch size;
- `WEIGHTED_PFE_THREADS`: Rayon worker count;
- `WEIGHTED_PFE_REPETITIONS`: number of measured repetitions.

The current generated data targets a `1/2` stake threshold. Its four profiles
used by the paper experiments are:

| Error | Real parties `N` | Actual virtual weight `W` | Minimum reconstruction weight `q` | Scheme threshold `t` |
|---:|---:|---:|---:|---:|
| `1/8` | 234 | 764 | 368 | 367 |
| `1/16` | 419 | 1,576 | 789 | 788 |
| `1/32` | 598 | 3,060 | 1,531 | 1,530 |
| `1/64` | 688 | 6,486 | 3,244 | 3,243 |

For the default `1/64` profile, the deterministic heaviest-first accepted set
currently contains 40 real parties with combined weight 3,255. Allocation
generation and its threshold semantics live in `scripts`; this crate consumes
the generated counts rather than recomputing them.

## Serialized sizes

Size reporting uses compressed/canonical wire sizes, not Rust in-memory
layouts:

```text
BLS12-381 scalar     32 bytes
compressed G1       48 bytes
compressed G2       96 bytes
paper-model GT      288 bytes
tagged crate GT     289 bytes
Schnorr proof        80 bytes = one compressed G1 + one scalar
```

BLST's torus compression represents a nonidentity GT element in 288 bytes.
The crate's canonical GT encoding adds a one-byte identity/nonidentity tag so
that the identity also has a unique safe representation. The encryption key
is nonidentity by construction, while caller-supplied messages and resulting
ciphertext components may include the identity; size accessors report the
actual 289-byte encoding in either case.

For `N` real parties, total virtual weight `W`, batch size `B`, and accepted
real-party count `tau`, the paper's Table 3 and this implementation give:

| Quantity | Group/scalar count | Serialized bytes |
|---|---:|---:|
| Encryption key | one GT | 289 in this crate; 288 in the paper model |
| One party secret key | `B` scalars | `32B` |
| One party verification key | `B` G2 | `96B` |
| All verification keys | `NB` G2 | `96NB` |
| Core `D`, `U`, `V` material | `(2WB + 1)` G2 | `96(2WB + 1)` |
| Public decryption key, including verification keys | `((2W + N)B + 1)` G2 | `96((2W + N)B + 1)` |
| Non-NIZK public parameters | preceding G2 material plus one GT | add 289 bytes in this crate |
| Ciphertext, excluding proof (Table 3) | one G1 and one tagged GT | 337 in this crate; 336 in the paper model |
| Ciphertext, including this proof | preceding ciphertext, one G1, one scalar | 417 in this crate; 416 with the paper-model GT |
| One party response | one G1 | 48 |
| Accepted responses | `tau` G1 | `48 tau` |
| Recovered batch | `B` tagged GT values | `289B` |

`PublicDecryptionKey::serialized_size_bytes()` deliberately reports compressed
G2 material only. It excludes Rust metadata, the encryption key, and the
NIZK proof-system setup. `EncryptionKey::serialized_size_bytes()` and
`Ciphertext::serialized_size_bytes()` use the crate's tagged canonical GT
encoding.

At the reproduction defaults `B = 32`, `N = 688`, and `W = 6,486`:

```text
core D/U/V material       415,105 G2 = 39,850,080 bytes
party verification keys   22,016 G2 =  2,113,536 bytes
public decryption key     437,121 G2 = 41,963,616 bytes
total non-NIZK public parameters       41,963,905 bytes
```

The implementation uses Fiat--Shamir Schnorr and has no structured NIZK CRS,
so its structured CRS size is **0 bytes**. The ordinary fixed group generator
is implicit. This is an implementation choice; the paper assumes an abstract
online simulation-extractable `Pi_DL` and excludes its unspecified CRS from
Table 3.

## Performance engineering

The implementation follows the optimization level of the sibling `btx` and
`weighted_btx` crates:

- BLS12-381-specific `blstrs`/BLST arithmetic, raw BLST Pippenger MSMs, and
  raw batch normalization;
- slot-major contiguous G2 tables and bounded one-slot-at-a-time setup, which
  avoids retaining all `2WB` points in projective form simultaneously;
- batch inversion for all setup fractions and a scalar FFT for Shamir
  evaluation;
- a product-tree/FFT path for Lagrange coefficients at Solana-sized accepted
  weights, rather than quadratic interpolation;
- randomized aggregate client-proof and server-share verification, with
  direct share verification used only for blame after aggregate failure;
- committee-dependent G2 MSMs and prepared static G2 line tables cached by
  `prepare_decryption`;
- raw multi-Miller loops, split easy/hard final exponentiation, and a shared
  inversion for batched easy final exponentiations on the one-thread path;
- a cyclotomic target-group Cauchy transform that keeps pairing results after
  only the easy final exponentiation, fuses both opening terms there, and then
  pays just one hard final exponentiation per recovered ciphertext;
- Rayon parallelism over independent setup blocks, MSMs, Miller loops, final
  exponentiations, and opening outputs.

The main Construction 2 specialization concerns the public slot points. The
paper permits arbitrary distinct `alpha_i` outside `{0, -1}`. This crate uses

```text
alpha_i = c + omega^i,
```

where `omega^i` ranges over the complete `B`-th roots of unity and `c` is the
first small shift for which every point avoids `0` and `-1`. The affine shift
preserves every difference `alpha_k - alpha_i`, turning both cross-slot
Cauchy sums into cyclic FFT convolutions without changing the displayed
decryption algebra.

For the cyclic Cauchy kernel, the spectrum is available in closed form. The
implementation multiplies frequency `k` by the small signed integer

```text
B - 1 - 2k
```

instead of doing a full-width scalar multiplication at every frequency. The
transform returns a `2B`-scaled sum, and `prepare_decryption` moves the inverse
scale into committee-dependent G2 operands. Consequently, neither online
Cauchy result needs a full output scaling pass.

`precompute_batch` computes the G1 Cauchy sums and the cyclotomic target-group
Cauchy sums in parallel. It also performs the `B` ordinary pairings against
the aggregate `D` operands. Their hard final exponentiations are deferred.
`open_batch` then performs `B` multipairings of arity `tau` and the remaining
`B` ordinary mask pairings. For each output it combines all three easy-final-
exponentiated terms before one hard final exponentiation. Thus the optimized
path follows Table 3's pairing inputs,

```text
B * MP(tau) + 2B ordinary pairings.
```

while reducing the target Cauchy/opening path from `3B` separate hard final
exponentiations to `B`.

## Deviations and resolved ambiguities

- **Ciphertext-proof transcript.** Construction 2 writes
  `Pi_DL.Prove(ct[1]; r)` and verifies only `ct[1]`. In the paper's CCA game,
  however, changing `ct[2]` while reusing the proof would turn a permitted
  decryption query into a direct transformation of the challenge ciphertext.
  This implementation intentionally binds compressed `ct[1]`, the complete
  canonical `ct[2]`, and the setup identifier into the Fiat--Shamir transcript.
  This strengthening does not change proof size or group-operation counts.
- **Setup binding.** A SHA-256 setup identifier commits to the public
  dimensions, weights, domains, encryption key, and G2 material. Ciphertext,
  validated-batch, response, and committee precomputations are checked against
  it to prevent accidental cross-setup reuse. It is associated context rather
  than an extra ciphertext field.
- **Structured `alpha_i`.** The affine root-of-unity specialization described
  above is required for the implemented `O(B log B)` cyclic Cauchy path. The
  paper gives only arbitrary distinct points and does not state the structure
  needed by its FFT cost row.
- **Sampling `rho_j`.** The paper requires all `rho_j + alpha_i` labels to be
  globally distinct and every fraction denominator to be nonzero, but does
  not give a sampling algorithm. Setup implements explicit rejection
  sampling for both collision and pole conditions.
- **Nondegenerate randomness.** Setup samples nonzero `z`, chooses
  `x` outside `{0, 1}`, and encryption samples nonzero `r`. These checks rule
  out degenerate public keys, ciphertexts, and fraction poles that the
  pseudocode's unrestricted field sampling would otherwise permit.
- **Exact-size padding.** The syntax says “up to” the maximum batch size while
  Construction 2 and its Cauchy formulas operate on a full padded batch. Since
  no dummy-padding convention is specified, the crate requires exactly the
  configured power-of-two `B` and leaves logical padding to the caller.
- **Accepted committee.** After share verification, the paper allows any
  authorized accepted subset. The crate chooses the minimum number of real
  parties deterministically by weight, reducing the arity of each opening
  multipairing.

## Security status and paper issues

- Section 4.2, “Distributed Key Generation for Construction 2,” is empty, so
  this crate implements trusted setup only.
- Section 4.4, “Assumption and CCA Security,” is empty. Section 4.5's Lemma 2
  refers to **“Assumption ??”** and to handles `H_{j,i}` that are not defined
  by Construction 2. There is no formal Construction 2 assumption or CCA
  theorem in the current draft. The generic-group sketch is useful evidence,
  but it is not a complete CCA proof.
- Fiat--Shamir Schnorr is compact and has no structured CRS, but it is not a
  standard plain-random-oracle online simulation-extractable NIZK of the kind
  required by the paper's abstract `Pi_DL`. The implementation therefore does
  not claim to instantiate the paper's full proof-system assumption.
- Section 4.3 says that `B * MP(tau) + 2B * P` is `tau + 1` pairing inputs per
  ciphertext. The arithmetic gives `tau + 2`, and Table 3 correctly retains
  the two ordinary pairings. The implementation follows Table 3.
- The regenerated Solana allocation file targets `1/2` and currently has
  experimental `W` values `764`, `1,576`, `3,060`, and `6,486`. Section 5's
  stale prose/table values are `764`, `1,580`, `3,063`, and `6,489`.
  Appendix A's Table 6 caption also still says target threshold `2/3`, although
  the current data and experiment text use `1/2`. Reproduction output follows
  the generated JSON, not those stale values.
- `serde`/`bincode` support is intended for trusted local persistence. A
  production protocol needs bounded, versioned decoding with explicit
  canonical and subgroup validation before accepting untrusted serialized
  keys or ciphertexts.
- The BLST MSM and low-level GT paths should not be assumed side-channel safe
  for processing production secrets. No independent security review has been
  performed.
