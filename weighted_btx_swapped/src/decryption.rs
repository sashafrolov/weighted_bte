//! Weighted BTX batch validation, share filtering, FFT middle products, and opening.

use blstrs::{G1Affine, G1Projective, G2Affine, G2Projective, Gt, Scalar};
use ff::Field;
use group::{prime::PrimeCurveAffine, Group};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    blst_utils::{
        batch_normalize_g1, batch_normalize_g2, g1_multi_exp_affine_bytes,
        g1_multi_exp_affine_indexed, g2_multi_exp_affine_bytes, scalars_to_le_bytes,
    },
    encryption::Ciphertext,
    error::{Error, Result},
    fft::Radix2Domain,
    final_exponentiation::{
        batch_easy_final_exponentiation, easy_final_exponentiation, full_final_exponentiation,
        hard_final_exponentiation, unprepared_multi_miller_loop,
    },
    interpolation::lagrange_at_zero,
    setup::{PartySecretKey, PublicDecryptionKey},
};

const BATCH_DIGEST_DOMAIN: &[u8] = b"WEIGHTED-BTX-SWAPPED-ORDERED-BATCH-v1";
const DECRYPTION_CONTEXT_DOMAIN: &[u8] = b"WEIGHTED-BTX-SWAPPED-DECRYPTION-CONTEXT-v1";

/// An ordered ciphertext batch with client proofs checked exactly once.
#[derive(Clone, Debug)]
pub struct ValidatedBatch {
    setup_id: [u8; 32],
    batch_size: usize,
    digest: [u8; 32],
    valid: Box<[bool]>,
    first_affine: Box<[G2Affine]>,
    first_projective: Box<[G2Projective]>,
}

/// One real party's constant-size pre-decryption key for an entire batch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecryptionShare {
    pub party_index: usize,
    pub sigma: G2Projective,
    pub batch_digest: [u8; 32],
    pub setup_id: [u8; 32],
}

/// Verified shares whose combined virtual weight is authorized.
#[derive(Clone, Debug)]
pub struct AcceptedDecryptionShares {
    setup_id: [u8; 32],
    batch_size: usize,
    digest: [u8; 32],
    accepted_weight: usize,
    shares: Box<[DecryptionShare]>,
    rejected_parties: Box<[usize]>,
}

/// Committee-dependent G1 work reused for batches with the same size and
/// accepted real-party set.
#[derive(Clone, Debug)]
pub struct DecryptionPrecomputation {
    setup_id: [u8; 32],
    context_id: [u8; 32],
    batch_size: usize,
    accepted_weight: usize,
    party_indices: Box<[usize]>,
    transform_size: usize,
    domain: Radix2Domain,
    transformed_kernel: Box<[G1Affine]>,
    /// Slot-major `A[j,i] = sum_{omega in Omega_j} lambda_omega D[omega,-i]`.
    opening_keys: Box<[G1Affine]>,
}

/// Ciphertext-dependent cross terms for one ordered batch.
#[derive(Clone, Debug)]
pub struct BatchPrecomputation {
    context_id: [u8; 32],
    batch_size: usize,
    digest: [u8; 32],
    beta: Box<[Gt]>,
}

pub fn validate_batch(
    decryption_key: &PublicDecryptionKey,
    ciphertexts: &[Ciphertext],
) -> Result<ValidatedBatch> {
    ValidatedBatch::from_ciphertexts(decryption_key, ciphertexts)
}

impl ValidatedBatch {
    /// Verify every proof while preserving the original slot positions.
    pub fn verify(
        decryption_key: &PublicDecryptionKey,
        ciphertexts: &[Ciphertext],
    ) -> Result<Self> {
        Self::from_ciphertexts(decryption_key, ciphertexts)
    }

    fn from_ciphertexts(
        decryption_key: &PublicDecryptionKey,
        ciphertexts: &[Ciphertext],
    ) -> Result<Self> {
        validate_batch_size(decryption_key, ciphertexts.len())?;
        let setup_id = decryption_key.setup_id();

        let valid = ciphertexts
            .par_iter()
            .map(|ciphertext| ciphertext.verify_for_setup(setup_id))
            .collect::<Vec<_>>();

        // An invalid slot contributes the identity everywhere.  It must not
        // be removed: q^i and the middle-product offsets are positional.
        let first_affine = ciphertexts
            .iter()
            .zip(&valid)
            .map(|(ciphertext, is_valid)| {
                if *is_valid {
                    ciphertext.first
                } else {
                    G2Affine::identity()
                }
            })
            .collect::<Vec<_>>();
        let first_projective = first_affine
            .iter()
            .copied()
            .map(G2Projective::from)
            .collect::<Vec<_>>();

        Ok(Self {
            setup_id,
            batch_size: ciphertexts.len(),
            digest: batch_digest(ciphertexts),
            valid: valid.into_boxed_slice(),
            first_affine: first_affine.into_boxed_slice(),
            first_projective: first_projective.into_boxed_slice(),
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn setup_id(&self) -> [u8; 32] {
        self.setup_id
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn valid_mask(&self) -> &[bool] {
        &self.valid
    }

    pub fn valid_count(&self) -> usize {
        self.valid.iter().filter(|valid| **valid).count()
    }
}

impl AcceptedDecryptionShares {
    pub fn accepted_weight(&self) -> usize {
        self.accepted_weight
    }

    pub fn party_count(&self) -> usize {
        self.shares.len()
    }

    pub fn party_indices(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.shares.iter().map(|share| share.party_index)
    }

    pub fn rejected_parties(&self) -> &[usize] {
        &self.rejected_parties
    }

    pub fn shares(&self) -> &[DecryptionShare] {
        &self.shares
    }
}

impl DecryptionPrecomputation {
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn accepted_weight(&self) -> usize {
        self.accepted_weight
    }

    pub fn party_count(&self) -> usize {
        self.party_indices.len()
    }

    pub fn transform_size(&self) -> usize {
        self.transform_size
    }
}

/// Compute one party's short pre-decryption key
/// `sigma_j = sum_i q_j^i C_i`.
pub fn partial_decrypt(
    party_key: &PartySecretKey,
    batch: &ValidatedBatch,
) -> Result<DecryptionShare> {
    if batch.setup_id != party_key.setup_id() {
        return Err(Error::MismatchedSetup);
    }
    if batch.batch_size > party_key.max_batch_size() {
        return Err(Error::BatchTooLarge {
            batch_size: batch.batch_size,
            max_batch_size: party_key.max_batch_size(),
        });
    }

    let q = party_key.secret_scalar();
    let mut power = q;
    let mut scalar_bytes = Vec::with_capacity(batch.batch_size.saturating_mul(32));
    for _ in 0..batch.batch_size {
        scalar_bytes.extend_from_slice(&power.to_bytes_le());
        power *= q;
    }

    Ok(DecryptionShare {
        party_index: party_key.party_index,
        sigma: g2_multi_exp_affine_bytes(&batch.first_affine, &scalar_bytes),
        batch_digest: batch.digest,
        setup_id: party_key.setup_id(),
    })
}

/// Verify the paper's per-party equation with one multi-pairing.
pub fn verify_decryption_share(
    decryption_key: &PublicDecryptionKey,
    batch: &ValidatedBatch,
    share: &DecryptionShare,
) -> Result<bool> {
    validate_batch_size(decryption_key, batch.batch_size)?;
    if batch.setup_id != decryption_key.setup_id() {
        return Err(Error::MismatchedSetup);
    }
    if share.batch_digest != batch.digest || share.setup_id != decryption_key.setup_id() {
        return Ok(false);
    }
    if share.party_index >= decryption_key.party_count() {
        return Err(Error::InvalidPartyIndex(share.party_index));
    }

    let valid_slots = batch
        .valid
        .iter()
        .enumerate()
        .filter_map(|(slot, valid)| valid.then_some(slot))
        .collect::<Vec<_>>();

    let mut left = valid_slots
        .iter()
        .map(|slot| batch.first_affine[*slot])
        .collect::<Vec<_>>();
    left.push(G2Affine::from(-share.sigma));

    let mut right = valid_slots
        .iter()
        .map(|slot| decryption_key.verification_key(share.party_index, *slot + 1))
        .collect::<Result<Vec<_>>>()?;
    right.push(G1Affine::generator());

    Ok(full_final_exponentiation(unprepared_multi_miller_loop(&right, &left)) == Gt::identity())
}

/// Verify, blame, and discard malformed party shares.  The accepting set is
/// sorted by party index so its virtual interpolation domain is canonical.
pub fn accept_decryption_shares(
    decryption_key: &PublicDecryptionKey,
    batch: &ValidatedBatch,
    shares: &[DecryptionShare],
) -> Result<AcceptedDecryptionShares> {
    validate_batch_size(decryption_key, batch.batch_size)?;
    if batch.setup_id != decryption_key.setup_id() {
        return Err(Error::MismatchedSetup);
    }
    if shares.len() > decryption_key.party_count() {
        return Err(Error::TooManyShares {
            supplied: shares.len(),
            party_count: decryption_key.party_count(),
        });
    }

    let mut seen = vec![false; decryption_key.party_count()];
    let mut candidates = Vec::with_capacity(shares.len());
    let mut rejected_parties = Vec::new();
    for share in shares {
        if share.party_index >= decryption_key.party_count() {
            return Err(Error::InvalidPartyIndex(share.party_index));
        }
        if seen[share.party_index] {
            return Err(Error::DuplicatePartyIndex(share.party_index));
        }
        seen[share.party_index] = true;
        if share.batch_digest == batch.digest && share.setup_id == decryption_key.setup_id() {
            candidates.push(share.clone());
        } else {
            rejected_parties.push(share.party_index);
        }
    }

    let aggregate_valid = verify_decryption_shares_batched(decryption_key, batch, &candidates)?;
    let mut accepted = if aggregate_valid {
        candidates
    } else {
        let checks = candidates
            .par_iter()
            .map(|share| verify_decryption_share(decryption_key, batch, share))
            .collect::<Vec<_>>();
        let mut accepted = Vec::new();
        for (share, check) in candidates.into_iter().zip(checks) {
            if check? {
                accepted.push(share);
            } else {
                rejected_parties.push(share.party_index);
            }
        }
        accepted
    };

    accepted.sort_unstable_by_key(|share| share.party_index);
    rejected_parties.sort_unstable();
    let mut accepted_weight = 0usize;
    for share in &accepted {
        accepted_weight = accepted_weight
            .checked_add(decryption_key.party_weight(share.party_index)?)
            .ok_or(Error::WeightOverflow)?;
    }
    if accepted_weight <= decryption_key.threshold_weight() {
        return Err(Error::InsufficientWeight {
            accepted: accepted_weight,
            required: decryption_key.required_weight(),
            rejected_parties,
        });
    }

    Ok(AcceptedDecryptionShares {
        setup_id: decryption_key.setup_id(),
        batch_size: batch.batch_size,
        digest: batch.digest,
        accepted_weight,
        shares: accepted.into_boxed_slice(),
        rejected_parties: rejected_parties.into_boxed_slice(),
    })
}

/// Random-linear batch verification.  A failed optimistic check falls back to
/// direct checks in [`accept_decryption_shares`] so malformed parties can be
/// blamed accurately.
fn verify_decryption_shares_batched(
    decryption_key: &PublicDecryptionKey,
    batch: &ValidatedBatch,
    shares: &[DecryptionShare],
) -> Result<bool> {
    if shares.is_empty() {
        return Ok(true);
    }

    let mut rng = rand_core::OsRng;
    let challenges = (0..shares.len())
        .map(|_| loop {
            let challenge = Scalar::random(&mut rng);
            if !bool::from(challenge.is_zero()) {
                break challenge;
            }
        })
        .collect::<Vec<_>>();
    let challenge_bytes = scalars_to_le_bytes(&challenges);

    let share_points = shares.iter().map(|share| share.sigma).collect::<Vec<_>>();
    let mut share_affine = vec![G2Affine::default(); share_points.len()];
    batch_normalize_g2(&share_points, &mut share_affine);
    let aggregate_sigma = g2_multi_exp_affine_bytes(&share_affine, &challenge_bytes);
    let party_indices = shares
        .iter()
        .map(|share| share.party_index)
        .collect::<Vec<_>>();
    let valid_slots = batch
        .valid
        .iter()
        .enumerate()
        .filter_map(|(slot, valid)| valid.then_some(slot))
        .collect::<Vec<_>>();

    let aggregate_verification_projective = valid_slots
        .par_iter()
        .map(|slot| {
            let power = decryption_key.verification_power(*slot + 1)?;
            Ok(g1_multi_exp_affine_indexed(
                power,
                &party_indices,
                &challenge_bytes,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut aggregate_verification =
        vec![G1Affine::default(); aggregate_verification_projective.len()];
    batch_normalize_g1(
        &aggregate_verification_projective,
        &mut aggregate_verification,
    );

    let mut left = valid_slots
        .iter()
        .map(|slot| batch.first_affine[*slot])
        .collect::<Vec<_>>();
    left.push(G2Affine::from(-aggregate_sigma));
    aggregate_verification.push(G1Affine::generator());

    Ok(
        full_final_exponentiation(unprepared_multi_miller_loop(&aggregate_verification, &left))
            == Gt::identity(),
    )
}

/// Build the accepted committee's weighted interpolation material and FFT
/// kernel.  This is independent of ciphertext contents and can be reused for
/// another batch of the same size and accepted party set.
pub fn prepare_decryption(
    decryption_key: &PublicDecryptionKey,
    accepted: &AcceptedDecryptionShares,
) -> Result<DecryptionPrecomputation> {
    validate_batch_size(decryption_key, accepted.batch_size)?;
    if accepted.setup_id != decryption_key.setup_id() {
        return Err(Error::MismatchedSetup);
    }
    if accepted.accepted_weight <= decryption_key.threshold_weight() {
        return Err(Error::InsufficientWeight {
            accepted: accepted.accepted_weight,
            required: decryption_key.required_weight(),
            rejected_parties: Vec::new(),
        });
    }

    let party_indices = accepted.party_indices().collect::<Vec<_>>();
    let context_id = decryption_context_id(
        decryption_key.setup_id(),
        accepted.batch_size,
        &party_indices,
    );
    let mut selected_virtual_indices = Vec::with_capacity(accepted.accepted_weight);
    let mut coefficient_offsets = Vec::with_capacity(party_indices.len() + 1);
    coefficient_offsets.push(0);
    for party_index in &party_indices {
        selected_virtual_indices.extend(decryption_key.party_range(*party_index)?);
        coefficient_offsets.push(selected_virtual_indices.len());
    }
    debug_assert_eq!(selected_virtual_indices.len(), accepted.accepted_weight);

    let coefficients = lagrange_at_zero(decryption_key.domain_points(), &selected_virtual_indices)?;
    let coefficient_bytes = scalars_to_le_bytes(&coefficients);
    let opening_count = accepted
        .batch_size
        .checked_mul(party_indices.len())
        .ok_or(Error::InvalidBatchSize(accepted.batch_size))?;

    let opening_projective = (0..opening_count)
        .into_par_iter()
        .map(|flat_index| {
            let slot = flat_index / party_indices.len();
            let party_position = flat_index % party_indices.len();
            let party_index = party_indices[party_position];
            let range = decryption_key.party_range(party_index)?;
            let coefficient_range =
                coefficient_offsets[party_position]..coefficient_offsets[party_position + 1];
            let coefficient_byte_range = coefficient_range.start * 32..coefficient_range.end * 32;
            let power = decryption_key.negative_power(slot + 1)?;
            Ok(g1_multi_exp_affine_bytes(
                &power[range],
                &coefficient_bytes[coefficient_byte_range],
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    let doubled = accepted
        .batch_size
        .checked_mul(2)
        .ok_or(Error::InvalidBatchSize(accepted.batch_size))?;
    let transform_size = doubled
        .checked_next_power_of_two()
        .ok_or(Error::InvalidBatchSize(accepted.batch_size))?;
    let domain = Radix2Domain::new(transform_size)?;
    let mut kernel = vec![G1Projective::identity(); transform_size];

    // Negative H[-d] values are free once the per-party opening MSMs exist.
    for (distance, kernel_slot) in kernel
        .iter_mut()
        .enumerate()
        .take(accepted.batch_size)
        .skip(1)
    {
        let start = (distance - 1) * party_indices.len();
        *kernel_slot = opening_projective[start..start + party_indices.len()]
            .iter()
            .copied()
            .sum();
    }

    // Only positive offsets need an additional size-|Omega_T| MSM.  This is
    // the reuse responsible for the `(B-1) MSM_1(W_T)` row in Table 2.
    let positive = (1..accepted.batch_size)
        .into_par_iter()
        .map(|distance| {
            let power = decryption_key.positive_power(distance)?;
            Ok((
                distance,
                g1_multi_exp_affine_indexed(power, &selected_virtual_indices, &coefficient_bytes),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    for (distance, point) in positive {
        kernel[transform_size - distance] = point;
    }

    domain.fft(&mut kernel);
    let inverse_size = domain.size_inverse();
    kernel
        .par_iter_mut()
        .for_each(|point| *point *= inverse_size);

    let mut kernel_affine = vec![G1Affine::default(); transform_size];
    batch_normalize_g1(&kernel, &mut kernel_affine);
    // The fixed kernel is now in G1. BLST prepares Miller-loop lines only
    // for G2, so retain compact affine points and use an unprepared loop on
    // the ciphertext-dependent G2 operand for each batch.
    let transformed_kernel = kernel_affine.into_boxed_slice();

    let mut opening_keys = vec![G1Affine::default(); opening_projective.len()];
    batch_normalize_g1(&opening_projective, &mut opening_keys);

    Ok(DecryptionPrecomputation {
        setup_id: decryption_key.setup_id(),
        context_id,
        batch_size: accepted.batch_size,
        accepted_weight: accepted.accepted_weight,
        party_indices: party_indices.into_boxed_slice(),
        transform_size,
        domain,
        transformed_kernel,
        opening_keys: opening_keys.into_boxed_slice(),
    })
}

/// Compute the cross terms `beta_i` with the same split-final-exponentiation
/// middle product used by the optimized plain BTX implementation.
pub fn precompute_batch(
    precomputation: &DecryptionPrecomputation,
    batch: &ValidatedBatch,
) -> Result<BatchPrecomputation> {
    ensure_precomputation_batch(precomputation, batch)?;

    let mut transformed_ciphertexts = vec![G2Projective::identity(); precomputation.transform_size];
    transformed_ciphertexts[..batch.batch_size].copy_from_slice(&batch.first_projective);
    precomputation.domain.fft(&mut transformed_ciphertexts);

    let mut transformed_affine = vec![G2Affine::default(); precomputation.transform_size];
    batch_normalize_g2(&transformed_ciphertexts, &mut transformed_affine);

    let miller_results = transformed_affine
        .par_iter()
        .zip(precomputation.transformed_kernel.par_iter())
        .map(|(ciphertext, kernel)| {
            unprepared_multi_miller_loop(
                std::slice::from_ref(kernel),
                std::slice::from_ref(ciphertext),
            )
        })
        .collect::<Vec<_>>();
    let mut convolution = if rayon::current_num_threads() == 1 {
        batch_easy_final_exponentiation(&miller_results)
    } else {
        miller_results
            .into_par_iter()
            .map(easy_final_exponentiation)
            .collect()
    };

    precomputation
        .domain
        .ifft_cyclotomic_prefix_unscaled(&mut convolution, batch.batch_size);
    let beta = convolution[..batch.batch_size]
        .par_iter()
        .zip(batch.valid.par_iter())
        .map(|(value, valid)| {
            if *valid {
                hard_final_exponentiation(*value)
            } else {
                Gt::identity()
            }
        })
        .collect::<Vec<_>>();

    Ok(BatchPrecomputation {
        context_id: precomputation.context_id,
        batch_size: batch.batch_size,
        digest: batch.digest,
        beta: beta.into_boxed_slice(),
    })
}

/// Evaluate the |T|-ary opening multi-pairing for every valid slot and unmask
/// the messages.
pub fn open_batch(
    precomputation: &DecryptionPrecomputation,
    accepted: &AcceptedDecryptionShares,
    batch: &ValidatedBatch,
    ciphertexts: &[Ciphertext],
    cross_terms: &BatchPrecomputation,
) -> Result<Vec<Option<Gt>>> {
    ensure_precomputation_batch(precomputation, batch)?;
    if accepted.batch_size != batch.batch_size {
        return Err(Error::MismatchedBatchSize {
            expected: batch.batch_size,
            actual: accepted.batch_size,
        });
    }
    if accepted.digest != batch.digest {
        return Err(Error::MismatchedBatchDigest);
    }
    if ciphertexts.len() != batch.batch_size {
        return Err(Error::MismatchedBatchSize {
            expected: batch.batch_size,
            actual: ciphertexts.len(),
        });
    }
    if batch_digest(ciphertexts) != batch.digest {
        return Err(Error::MismatchedBatchDigest);
    }
    if cross_terms.batch_size != batch.batch_size || cross_terms.digest != batch.digest {
        return Err(Error::MismatchedBatchDigest);
    }
    if cross_terms.context_id != precomputation.context_id {
        return Err(Error::MismatchedCommittee);
    }

    let accepted_indices = accepted.party_indices().collect::<Vec<_>>();
    if accepted.setup_id != precomputation.setup_id {
        return Err(Error::MismatchedSetup);
    }
    if accepted_indices.as_slice() != &*precomputation.party_indices {
        return Err(Error::MismatchedCommittee);
    }

    let share_projective = accepted
        .shares
        .iter()
        .map(|share| share.sigma)
        .collect::<Vec<_>>();
    let mut share_affine = vec![G2Affine::default(); share_projective.len()];
    batch_normalize_g2(&share_projective, &mut share_affine);
    let party_count = share_affine.len();

    let valid_slots = batch
        .valid
        .iter()
        .enumerate()
        .filter_map(|(slot, valid)| valid.then_some(slot))
        .collect::<Vec<_>>();
    let alpha_miller = valid_slots
        .par_iter()
        .map(|slot| {
            let start = slot * party_count;
            unprepared_multi_miller_loop(
                &precomputation.opening_keys[start..start + party_count],
                &share_affine,
            )
        })
        .collect::<Vec<_>>();
    let finalized_alpha = if rayon::current_num_threads() == 1 {
        batch_easy_final_exponentiation(&alpha_miller)
            .into_iter()
            .map(hard_final_exponentiation)
            .collect::<Vec<_>>()
    } else {
        alpha_miller
            .into_par_iter()
            .map(full_final_exponentiation)
            .collect::<Vec<_>>()
    };
    let mut alpha = vec![None; batch.batch_size];
    for (slot, value) in valid_slots.into_iter().zip(finalized_alpha) {
        alpha[slot] = Some(value);
    }

    Ok((0..batch.batch_size)
        .into_par_iter()
        .map(|slot| {
            alpha[slot].map(|value| ciphertexts[slot].second - value + cross_terms.beta[slot])
        })
        .collect())
}

/// Convenience wrapper around all public decryption phases.
pub fn decrypt(
    decryption_key: &PublicDecryptionKey,
    ciphertexts: &[Ciphertext],
    shares: &[DecryptionShare],
) -> Result<Vec<Option<Gt>>> {
    validate_batch_size(decryption_key, ciphertexts.len())?;
    let batch = validate_batch(decryption_key, ciphertexts)?;
    let accepted = accept_decryption_shares(decryption_key, &batch, shares)?;
    let fixed = prepare_decryption(decryption_key, &accepted)?;
    let cross_terms = precompute_batch(&fixed, &batch)?;
    open_batch(&fixed, &accepted, &batch, ciphertexts, &cross_terms)
}

fn validate_batch_size(decryption_key: &PublicDecryptionKey, batch_size: usize) -> Result<()> {
    if batch_size == 0 {
        return Err(Error::BatchIsEmpty);
    }
    if batch_size > decryption_key.max_batch_size() {
        return Err(Error::BatchTooLarge {
            batch_size,
            max_batch_size: decryption_key.max_batch_size(),
        });
    }
    Ok(())
}

fn ensure_precomputation_batch(
    precomputation: &DecryptionPrecomputation,
    batch: &ValidatedBatch,
) -> Result<()> {
    if precomputation.batch_size != batch.batch_size {
        return Err(Error::MismatchedBatchSize {
            expected: precomputation.batch_size,
            actual: batch.batch_size,
        });
    }
    if precomputation.setup_id != batch.setup_id {
        return Err(Error::MismatchedSetup);
    }
    Ok(())
}

fn batch_digest(ciphertexts: &[Ciphertext]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(ciphertexts.len() * 513);
    bytes.extend_from_slice(BATCH_DIGEST_DOMAIN);
    bytes.extend_from_slice(&(ciphertexts.len() as u64).to_le_bytes());
    for ciphertext in ciphertexts {
        ciphertext.append_canonical(&mut bytes);
    }
    Sha256::digest(bytes).into()
}

fn decryption_context_id(
    setup_id: [u8; 32],
    batch_size: usize,
    party_indices: &[usize],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(DECRYPTION_CONTEXT_DOMAIN);
    hasher.update(setup_id);
    hasher.update((batch_size as u64).to_le_bytes());
    hasher.update((party_indices.len() as u64).to_le_bytes());
    for party_index in party_indices {
        hasher.update((*party_index as u64).to_le_bytes());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use pairing::Engine;

    use super::*;
    use crate::{blst_utils::g1_multi_exp_affine, encrypt, keygen};

    const WEIGHTS: &[usize] = &[1, 3, 2, 4, 1];
    const THRESHOLD: usize = 5;

    fn fixture(
        max_batch_size: usize,
        batch_size: usize,
        party_indices: &[usize],
    ) -> (
        crate::KeyMaterial,
        Vec<Gt>,
        Vec<Ciphertext>,
        ValidatedBatch,
        AcceptedDecryptionShares,
        DecryptionPrecomputation,
    ) {
        let material = keygen(max_batch_size, WEIGHTS, THRESHOLD).unwrap();
        let messages = (0..batch_size)
            .map(|index| Gt::generator() * Scalar::from((index + 17) as u64))
            .collect::<Vec<_>>();
        let ciphertexts = messages
            .iter()
            .map(|message| encrypt(&material.encryption_key, *message))
            .collect::<Vec<_>>();
        let batch = validate_batch(&material.decryption_key, &ciphertexts).unwrap();
        let shares = party_indices
            .iter()
            .map(|index| partial_decrypt(&material.party_keys[*index], &batch).unwrap())
            .collect::<Vec<_>>();
        let accepted = accept_decryption_shares(&material.decryption_key, &batch, &shares).unwrap();
        let fixed = prepare_decryption(&material.decryption_key, &accepted).unwrap();
        (material, messages, ciphertexts, batch, accepted, fixed)
    }

    fn naive_beta(
        decryption_key: &PublicDecryptionKey,
        batch: &ValidatedBatch,
        accepted: &AcceptedDecryptionShares,
    ) -> Vec<Gt> {
        let party_indices = accepted.party_indices().collect::<Vec<_>>();
        let selected = party_indices
            .iter()
            .flat_map(|party| decryption_key.party_range(*party).unwrap())
            .collect::<Vec<_>>();
        let coefficients = lagrange_at_zero(decryption_key.domain_points(), &selected).unwrap();

        (0..batch.batch_size)
            .map(|output_slot| {
                (0..batch.batch_size)
                    .filter(|input_slot| *input_slot != output_slot && batch.valid[*input_slot])
                    .map(|input_slot| {
                        let offset = input_slot as isize - output_slot as isize;
                        let power = if offset < 0 {
                            decryption_key.negative_power((-offset) as usize).unwrap()
                        } else {
                            decryption_key.positive_power(offset as usize).unwrap()
                        };
                        let bases = selected
                            .iter()
                            .map(|index| power[*index])
                            .collect::<Vec<_>>();
                        let aggregate = g1_multi_exp_affine(&bases, &coefficients);
                        blstrs::Bls12::pairing(
                            &G1Affine::from(aggregate),
                            &G2Affine::from(batch.first_projective[input_slot]),
                        )
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn weighted_threshold_end_to_end_with_shuffled_real_parties() {
        let (_, messages, ciphertexts, batch, accepted, fixed) = fixture(8, 7, &[3, 1]);
        assert_eq!(accepted.accepted_weight(), 7);
        assert_eq!(accepted.party_indices().collect::<Vec<_>>(), vec![1, 3]);
        let cross_terms = precompute_batch(&fixed, &batch).unwrap();
        let opened = open_batch(&fixed, &accepted, &batch, &ciphertexts, &cross_terms).unwrap();
        assert_eq!(opened, messages.into_iter().map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn fft_middle_product_matches_literal_weighted_formula() {
        for batch_size in [1, 2, 3, 7, 8] {
            let (material, _, _, batch, accepted, fixed) = fixture(8, batch_size, &[1, 2, 3]);
            let fast = precompute_batch(&fixed, &batch).unwrap();
            let slow = naive_beta(&material.decryption_key, &batch, &accepted);
            assert_eq!(&*fast.beta, slow.as_slice(), "batch size {batch_size}");
        }
    }

    #[test]
    fn fewer_heavy_parties_can_outweigh_more_light_parties() {
        let material = keygen(4, WEIGHTS, THRESHOLD).unwrap();
        let ciphertext = encrypt(&material.encryption_key, Gt::generator());
        let batch = validate_batch(&material.decryption_key, &[ciphertext]).unwrap();

        let heavy = [1usize, 3]
            .iter()
            .map(|index| partial_decrypt(&material.party_keys[*index], &batch).unwrap())
            .collect::<Vec<_>>();
        assert!(accept_decryption_shares(&material.decryption_key, &batch, &heavy).is_ok());

        let light = [0usize, 2, 4]
            .iter()
            .map(|index| partial_decrypt(&material.party_keys[*index], &batch).unwrap())
            .collect::<Vec<_>>();
        assert!(matches!(
            accept_decryption_shares(&material.decryption_key, &batch, &light),
            Err(Error::InsufficientWeight { .. })
        ));
    }

    #[test]
    fn malformed_share_is_blamed_and_honest_authorized_remainder_opens() {
        let (material, messages, ciphertexts, batch, _, _) = fixture(8, 4, &[1, 2, 3]);
        let mut shares = [1usize, 2, 3]
            .iter()
            .map(|index| partial_decrypt(&material.party_keys[*index], &batch).unwrap())
            .collect::<Vec<_>>();
        shares[1].sigma += G2Projective::generator();

        let accepted = accept_decryption_shares(&material.decryption_key, &batch, &shares).unwrap();
        assert_eq!(accepted.rejected_parties(), &[2]);
        assert_eq!(accepted.accepted_weight(), 7);
        let fixed = prepare_decryption(&material.decryption_key, &accepted).unwrap();
        let cross_terms = precompute_batch(&fixed, &batch).unwrap();
        let opened = open_batch(&fixed, &accepted, &batch, &ciphertexts, &cross_terms).unwrap();
        assert_eq!(opened, messages.into_iter().map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn malformed_share_can_drop_set_below_weight_threshold() {
        let material = keygen(4, WEIGHTS, THRESHOLD).unwrap();
        let ciphertext = encrypt(&material.encryption_key, Gt::generator());
        let batch = validate_batch(&material.decryption_key, &[ciphertext]).unwrap();
        let mut shares = [1usize, 2, 4]
            .iter()
            .map(|index| partial_decrypt(&material.party_keys[*index], &batch).unwrap())
            .collect::<Vec<_>>();
        shares[0].sigma += G2Projective::generator();
        match accept_decryption_shares(&material.decryption_key, &batch, &shares) {
            Err(Error::InsufficientWeight {
                rejected_parties, ..
            }) => assert_eq!(rejected_parties, vec![1]),
            other => panic!("expected insufficient weight with blame, got {other:?}"),
        }
    }

    #[test]
    fn invalid_client_proof_preserves_slot_and_returns_bottom() {
        let (material, messages, mut ciphertexts, _, _, _) = fixture(8, 4, &[1, 3]);
        ciphertexts[2].proof.response += Scalar::ONE;
        let batch = validate_batch(&material.decryption_key, &ciphertexts).unwrap();
        assert_eq!(batch.valid_mask(), &[true, true, false, true]);
        let shares = [1usize, 3]
            .iter()
            .map(|index| partial_decrypt(&material.party_keys[*index], &batch).unwrap())
            .collect::<Vec<_>>();
        let accepted = accept_decryption_shares(&material.decryption_key, &batch, &shares).unwrap();
        let fixed = prepare_decryption(&material.decryption_key, &accepted).unwrap();
        let cross_terms = precompute_batch(&fixed, &batch).unwrap();
        let opened = open_batch(&fixed, &accepted, &batch, &ciphertexts, &cross_terms).unwrap();
        for slot in 0..4 {
            assert_eq!(opened[slot], (slot != 2).then_some(messages[slot]));
        }
    }

    #[test]
    fn entirely_invalid_batch_accepts_identity_shares_and_returns_only_bottom() {
        let (material, _, mut ciphertexts, _, _, _) = fixture(4, 3, &[1, 3]);
        for ciphertext in &mut ciphertexts {
            ciphertext.proof.response += Scalar::ONE;
        }
        let batch = validate_batch(&material.decryption_key, &ciphertexts).unwrap();
        assert_eq!(batch.valid_count(), 0);
        let shares = [1usize, 3]
            .iter()
            .map(|index| partial_decrypt(&material.party_keys[*index], &batch).unwrap())
            .collect::<Vec<_>>();
        assert!(shares
            .iter()
            .all(|share| bool::from(share.sigma.is_identity())));
        let accepted = accept_decryption_shares(&material.decryption_key, &batch, &shares).unwrap();
        let fixed = prepare_decryption(&material.decryption_key, &accepted).unwrap();
        let cross_terms = precompute_batch(&fixed, &batch).unwrap();
        assert_eq!(
            open_batch(&fixed, &accepted, &batch, &ciphertexts, &cross_terms).unwrap(),
            vec![None; 3]
        );
    }

    #[test]
    fn one_party_one_weight_one_ciphertext_boundary_opens() {
        let material = keygen(1, &[1], 0).unwrap();
        let message = Gt::generator() * Scalar::from(99u64);
        let ciphertexts = vec![encrypt(&material.encryption_key, message)];
        let batch = validate_batch(&material.decryption_key, &ciphertexts).unwrap();
        let shares = vec![partial_decrypt(&material.party_keys[0], &batch).unwrap()];
        assert!(verify_decryption_share(&material.decryption_key, &batch, &shares[0]).unwrap());
        let accepted = accept_decryption_shares(&material.decryption_key, &batch, &shares).unwrap();
        let fixed = prepare_decryption(&material.decryption_key, &accepted).unwrap();
        assert_eq!(fixed.transform_size(), 2);
        let cross_terms = precompute_batch(&fixed, &batch).unwrap();
        assert_eq!(
            open_batch(&fixed, &accepted, &batch, &ciphertexts, &cross_terms).unwrap(),
            vec![Some(message)]
        );
    }

    #[test]
    fn opening_rejects_ciphertexts_changed_after_validation() {
        let (_, _, mut ciphertexts, batch, accepted, fixed) = fixture(4, 3, &[1, 3]);
        let cross_terms = precompute_batch(&fixed, &batch).unwrap();
        ciphertexts[0].second += Gt::generator();

        assert!(matches!(
            open_batch(&fixed, &accepted, &batch, &ciphertexts, &cross_terms),
            Err(Error::MismatchedBatchDigest)
        ));
    }

    #[test]
    fn phased_objects_are_bound_to_setup_and_accepted_committee() {
        let (material, _, ciphertexts, batch, accepted_a, fixed_a) = fixture(4, 3, &[1, 3]);
        let cross_terms_a = precompute_batch(&fixed_a, &batch).unwrap();

        let shares_b = [0usize, 2, 3]
            .iter()
            .map(|index| partial_decrypt(&material.party_keys[*index], &batch).unwrap())
            .collect::<Vec<_>>();
        let accepted_b =
            accept_decryption_shares(&material.decryption_key, &batch, &shares_b).unwrap();
        let fixed_b = prepare_decryption(&material.decryption_key, &accepted_b).unwrap();
        assert!(matches!(
            open_batch(&fixed_b, &accepted_b, &batch, &ciphertexts, &cross_terms_a),
            Err(Error::MismatchedCommittee)
        ));

        let other_material = keygen(4, WEIGHTS, THRESHOLD).unwrap();
        assert_ne!(
            material.decryption_key.setup_id(),
            other_material.decryption_key.setup_id()
        );
        assert!(matches!(
            prepare_decryption(&other_material.decryption_key, &accepted_a),
            Err(Error::MismatchedSetup)
        ));
    }

    #[test]
    fn ciphertext_proofs_and_phased_batches_are_setup_bound() {
        let material_a = keygen(2, WEIGHTS, THRESHOLD).unwrap();
        let material_b = keygen(2, WEIGHTS, THRESHOLD).unwrap();
        let ciphertexts = vec![encrypt(&material_a.encryption_key, Gt::generator())];

        let batch_a = validate_batch(&material_a.decryption_key, &ciphertexts).unwrap();
        assert_eq!(batch_a.valid_mask(), &[true]);
        assert!(matches!(
            partial_decrypt(&material_b.party_keys[1], &batch_a),
            Err(Error::MismatchedSetup)
        ));

        let batch_b = validate_batch(&material_b.decryption_key, &ciphertexts).unwrap();
        assert_eq!(batch_b.valid_mask(), &[false]);
        let shares_b = [1usize, 3]
            .iter()
            .map(|index| partial_decrypt(&material_b.party_keys[*index], &batch_b).unwrap())
            .collect::<Vec<_>>();
        let opened = decrypt(&material_b.decryption_key, &ciphertexts, &shares_b).unwrap();
        assert_eq!(opened, vec![None]);
    }

    #[test]
    fn oversized_batch_and_share_list_are_rejected_at_the_boundary() {
        let material = keygen(1, WEIGHTS, THRESHOLD).unwrap();
        let ciphertexts = vec![
            encrypt(&material.encryption_key, Gt::generator()),
            encrypt(&material.encryption_key, Gt::generator()),
        ];
        assert!(matches!(
            validate_batch(&material.decryption_key, &ciphertexts),
            Err(Error::BatchTooLarge { .. })
        ));

        let ciphertexts = vec![encrypt(&material.encryption_key, Gt::generator())];
        let batch = validate_batch(&material.decryption_key, &ciphertexts).unwrap();
        let share = partial_decrypt(&material.party_keys[0], &batch).unwrap();
        let shares = vec![share; material.decryption_key.party_count() + 1];
        assert!(matches!(
            accept_decryption_shares(&material.decryption_key, &batch, &shares),
            Err(Error::TooManyShares { .. })
        ));
    }

    #[test]
    fn duplicate_and_wrong_batch_shares_are_not_counted_twice() {
        let (material, _, mut ciphertexts, batch, _, _) = fixture(4, 2, &[1, 3]);
        let share = partial_decrypt(&material.party_keys[1], &batch).unwrap();
        assert!(matches!(
            accept_decryption_shares(
                &material.decryption_key,
                &batch,
                &[share.clone(), share.clone()]
            ),
            Err(Error::DuplicatePartyIndex(1))
        ));

        ciphertexts.swap(0, 1);
        let reordered = validate_batch(&material.decryption_key, &ciphertexts).unwrap();
        assert!(matches!(
            accept_decryption_shares(&material.decryption_key, &reordered, &[share]),
            Err(Error::InsufficientWeight { .. })
        ));
    }
}
