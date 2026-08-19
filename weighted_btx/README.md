# Weighted BTX reproducibility implementation

This crate implements Construction 1 of
[`papers/weighted_batch_threshold_encryption.pdf`](../papers/weighted_batch_threshold_encryption.pdf)
over BLS12-381. It virtualizes the Shamir shares of BTX while keeping each
real party's pre-decryption message to one G1 element, independent of that
party's weight.

This is an academic reproduction prototype, not audited production
cryptography.

## Construction and API

For positive real-party weights `w_j`, total virtual weight `W`, and corruption
threshold `t < W`, a real-party set is authorized exactly when its summed
weight is greater than `t`. Setup gives party `j` a single secret scalar `q_j`
and assigns it `w_j` nonzero Shamir evaluation points.

For maximum batch size `L`, setup publishes:

- `ek = [z]_T`;
- verification powers `[q_j^i]_2` for `i = 1..L`;
- `[Z(omega) q_owner(omega)^d]_2` for every virtual point and
  `d = -L..-1, 1..L-1`.

An index-free ciphertext is

```text
ct[1] = [r]_1
ct[2] = message + r * ek.
```

For an ordered batch, party `j` sends the one-element pre-decryption key

```text
sigma_j = sum_i q_j^i * ct_i[1].
```

The public API exposes the construction's phases:

1. `validate_batch(&decryption_key, ...)` checks client proofs in the expected
   setup context and preserves positional indices.
2. `partial_decrypt` computes one real party's short share with a G1 MSM.
3. `accept_decryption_shares` batch-verifies submitted shares, directly checks
   them only on failure, discards malformed shares, and enforces accepted
   *weight* rather than party count.
4. `prepare_decryption` computes the accepted virtual-domain Lagrange
   coefficients, weighted G2 MSMs, and committee-dependent FFT kernel.
5. `precompute_batch` evaluates all cross terms with the BTX middle product.
6. `open_batch` performs one real-party multi-pairing per output and unmasks
   the valid messages.

`decrypt` is a convenience wrapper for the public combiner phases. As in the
plain [`btx`](../btx) crate, messages are formal target-group elements (`Gt`);
the draft does not specify a hybrid byte-message format or KDF.

## Running tests and benchmarks

```console
cargo test --manifest-path weighted_btx/Cargo.toml
cargo bench --manifest-path weighted_btx/Cargo.toml --bench phases
```

The tests include heterogeneous weighted authorization, malformed-party
fallback and blaming, invalid client slots, irregular batch sizes, and a
differential comparison of the FFT middle product with the literal quadratic
pairing formula. The copied low-level BTX tests continue to cover scalar/G1/G2/
GT FFTs, split final exponentiation, truncated inverse transforms, batch
normalization, and parallel equivalence.

## Solana allocation example

The requested example reads an identity-free positive-weight allocation from
`scripts/data`, derives the real-party count and each party's virtual weight,
and executes the scheme:

```console
cargo run --release --manifest-path weighted_btx/Cargo.toml \
  --example paper_reproduction -- \
  --approximation-error 1/16
```

It selects the newest `solana_share_weights_*.json`, then selects one entry of
its `allocations` array using the required `--approximation-error`
command-line argument. The example reads the selected entry's `share_count` as
`W` and its `weights` as the real parties' virtual weights; it deliberately does
not substitute the separate nominal resolution `selected_resolution_m`.

The scripts default to target reconstruction ratio `1/2`; selecting `1/16`
uses the stake interval `[7/16, 9/16]`. The generated
`reconstruction_threshold` is the minimum reconstructing weight `q`. Since
weighted BTX authorizes a set when its weight is strictly greater than `t`, the
example passes `t = q - 1` to key generation. For the current `1/16` profile
this gives `M = 1612`, `W = 1576`, `q = 789`, `t = 788`, and 419
positive-weight parties. Zero-weight validators are intentionally omitted by
the generator because Construction 1 requires every represented party to have
positive weight.

The paper does not specify a batch size or thread count for this experiment.
The runnable defaults are therefore actual/max batch size 8 and one Rayon
thread. Runtime input and performance choices can be changed with:

- `--approximation-error <ERROR>` (required, after Cargo's `--` separator);
- `WEIGHTED_BTX_WEIGHTS_FILE`;
- `WEIGHTED_BTX_BATCH_SIZE`;
- `WEIGHTED_BTX_THREADS`.

The example greedily chooses the heaviest real parties until their weight is
strictly greater than `t`, reports both real-party and virtual-share counts,
and checks every recovered message.

Public material is deliberately large. With the current `1/16` profile's
`N = 419`, `W = 1576`, and the example's `L = 8`, the construction has 26,992
G2 public-key points. At `L = 512`, it would have 1,826,776 G2 points including
verification keys, roughly 335 MiB in this crate's 192-byte in-memory affine
representation before allocator and working memory.

## Performance engineering

The implementation uses the same primitive stack and low-level tuning as the
plain BTX crate:

- `blstrs`/BLST BLS12-381 arithmetic and BLST Pippenger MSMs;
- raw BLST batch normalization;
- single-task raw BLST Pippenger inside Rayon-parallel decryption loops,
  avoiding nested BLST and Rayon worker pools;
- affine G2 Pippenger directly over the flattened public key, avoiding a
  projective round trip;
- sparse BLST point blocks for accepted virtual indices, avoiding a copied
  `W_T`-point base array for every positive offset;
- exponent-major, contiguous public-key arrays with bounded setup temporaries;
- root-of-unity virtual points and one scalar FFT to evaluate the degree-`t`
  Shamir polynomial during setup;
- a product-tree/FFT Lagrange path instead of quadratic interpolation at
  Solana-sized weights;
- reuse of the negative-power opening MSMs when building the middle-product
  kernel, so only the positive offsets need the additional `W_T`-sized MSMs;
- fixed-size, 64-byte-aligned BLST line tables for the reusable transformed
  kernel;
- one allocation-free raw multi-Miller loop per output for the party-dependent
  opening keys, avoiding hundreds of MiB of cached line tables;
- split easy/hard final exponentiation, one-inversion batched easy parts,
  truncated cyclotomic inverse FFTs, and the BLS-x/Frobenius target-group
  scalar decomposition inherited from `btx`;
- adaptive Rayon FFT scheduling and positional validity masks.

The paper's cost table does not include the work to derive Lagrange
coefficients for an accepted virtual subset. That work is material for large
`W`, which is why this crate implements it quasi-linearly rather than copying
plain BTX's small-committee quadratic routine.

## Paper issues and implementation resolutions

- **Ciphertext proof statement.** Construction 1 on page 6 writes a proof only
  for `ct[1]`, but the CCA reduction on page 9 simulates a proof for
  `(ct[1], ct[2])`. A proof that does not bind `ct[2]` permits second-component
  modification. This crate follows source BTX and binds both components in a
  fresh weighted-BTX transcript domain.
- **Setup context.** The paper works in a single generated-key context and
  does not make ciphertexts key-committing. This crate additionally binds a
  setup-context digest derived from the public material into the Fiat–Shamir
  transcript and requires `validate_batch` to receive the expected public
  decryption key. The digest is out-of-band associated context—not ciphertext
  metadata—so Table 1's
  ciphertext size and the index-free/no-epoch wire format are unchanged. This
  is misuse resistance for the phased API, not a claimed repair to the
  construction's algebraic correctness model.
- **Positive decryption-key endpoint.** The setup line, algebra, and
  `(2L-1)W` storage table use positive exponents only through `L-1`; the output
  line on page 6 accidentally ends at `+L`. This crate uses `1..L-1`.
- **Invalid ciphertext behavior.** Construction 1 says `PreDec` aborts if any
  client proof fails, while source BTX filters invalid positions and returns
  bottom only for those slots. This crate adopts the source BTX behavior:
  invalid slots become identities without compaction and open as `None`. This
  avoids one malformed ciphertext denying service to the whole batch.
- **Batches shorter than `L`.** The draft algorithms index a full `L`-element
  batch and leave padding semantics as a TODO. This crate supports every
  actual `1 <= B <= L` using powers `1..B`; FFT zero-padding is internal
  computation and does not add protocol ciphertexts or encryption-time
  indices.
- **Pairing count.** Section 3.3 claims `|T|+1` pairings per ciphertext, but its
  own Table 2 has `B * MP(|T|) + m * P`, where
  `m = next_power_of_two(2B)`. The resulting input count is
  `|T| + m/B` per ciphertext (`|T|+2` for power-of-two `B`), which is what the
  implemented middle product performs.
- **Evaluation domain.** Preliminaries correctly require disjoint nonzero
  points, while the setup pseudocode only writes a subset of `F_p`. This crate
  uses distinct roots of unity and never assigns the secret point zero.
- **Authorization wording.** Definitions 1–3 authorize exactly `weight > t`,
  but the consistency game later says “at least t,” and page 5 says a
  committee must have total weight “at least W.” This crate consistently uses
  the formal access structure `weight > t`.
- **Simulation extractability wording.** The formal definition excludes a
  previously simulated statement, but the proof later invokes extraction for
  a new proof on that same statement. Those are different SE notions. The
  implementation does not claim to repair that proof-level gap.

## Security and implementation status

- Setup is a trusted-dealer implementation of Construction 1. Section 3.2's
  specialized DKG protocol is separate future integration work.
- The threshold security statement assumes static corruptions, as in the
  draft.
- Fiat-Shamir Schnorr matches the source BTX implementation route and transcript
  binding. It is not a standard plain-ROM simulation-extractable NIZK; the
  paper's GGM/ROM discussion or a stronger proof system is needed for that
  claim.
- `serde`/`bincode` derives are for trusted local persistence only. Untrusted
  GT wire decoding needs explicit canonical compression and subgroup checks,
  and deserialized private collection sizes must be validated.
- BLST MSM and the exposed GT paths should not be assumed side-channel safe for
  production secret processing.
