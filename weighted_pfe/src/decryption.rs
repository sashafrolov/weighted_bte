//! Robust weighted decryption and FFT-accelerated Cauchy sums.

use blstrs::{G1Affine, G1Projective, G2Affine, G2Projective, Gt, Scalar};
use ff::Field;
use group::{prime::PrimeCurveAffine, Curve, Group};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    blst_utils::{
        batch_normalize_g1, batch_normalize_g2, g1_multi_exp_affine_bytes,
        g2_multi_exp_affine_bytes, g2_multi_exp_affine_indexed, scalars_to_le_bytes,
    },
    encryption::Ciphertext,
    error::{Error, Result},
    fft::{cyclotomic_mul_scalar_decomposed, mul_cyclotomic_signed_small, Radix2Domain},
    final_exponentiation::{
        batch_easy_final_exponentiation, easy_final_exponentiation, full_final_exponentiation,
        hard_final_exponentiation, unprepared_multi_miller_loop, CyclotomicFp12, PreparedG2Lines,
        RawMillerResult,
    },
    interpolation::lagrange_at_zero,
    proof,
    setup::{EncryptionKey, PartySecretKey, PublicDecryptionKey},
};

const BATCH_DIGEST_DOMAIN: &[u8] = b"WEIGHTED-PFE-ORDERED-BATCH-v1";
const COMMITTEE_CONTEXT_DOMAIN: &[u8] = b"WEIGHTED-PFE-COMMITTEE-v1";

#[derive(Clone, Debug)]
pub struct ValidatedBatch {
    setup_id: [u8; 32],
    batch_size: usize,
    digest: [u8; 32],
    first_affine: Box<[G1Affine]>,
    first_projective: Box<[G1Projective]>,
}

/// One real party's constant-size response for the complete padded batch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecryptionShare {
    pub party_index: usize,
    pub sigma: G1Projective,
    pub batch_digest: [u8; 32],
    pub setup_id: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct AcceptedDecryptionShares {
    setup_id: [u8; 32],
    batch_size: usize,
    digest: [u8; 32],
    accepted_weight: usize,
    shares: Box<[DecryptionShare]>,
    rejected_parties: Box<[usize]>,
}

/// Fixed Cauchy transform data for one setup's affine-shifted slot domain.
#[derive(Clone, Debug)]
pub struct CauchyKernel {
    setup_id: [u8; 32],
    batch_size: usize,
    domain: Radix2Domain,
    /// Twice the closed-form Cauchy-kernel spectrum: `B - 1 - 2k`.
    frequency_coefficients: Box<[i64]>,
    two_batch_small: i64,
}

/// Weighted interpolation and G2 MSM results for one accepted real committee.
#[derive(Clone, Debug)]
pub struct DecryptionPrecomputation {
    setup_id: [u8; 32],
    context_id: [u8; 32],
    batch_size: usize,
    accepted_weight: usize,
    party_indices: Box<[usize]>,
    /// Slot-major `sum_{w in Omega_j} lambda_w D[w,i] / (2B)`.
    scaled_d1_by_party: Box<[G2Affine]>,
    /// Sum of each slot's party entries above.
    scaled_d1_total: Box<[PreparedG2Lines]>,
    /// `alpha_i(alpha_i+1) D2_i - alpha_i V`.
    mask_keys: Box<[PreparedG2Lines]>,
}

/// Ciphertext-dependent source- and target-group Cauchy sums.
#[derive(Clone, Debug)]
pub struct BatchPrecomputation {
    setup_id: [u8; 32],
    context_id: [u8; 32],
    batch_size: usize,
    digest: [u8; 32],
    /// `2B * sum_{k != i} ct[k,1] / (alpha_k-alpha_i)`.
    scaled_first_cauchy: Box<[G1Projective]>,
    /// `sum_{k != i} e(ct[k,1],D1_k)/(alpha_k-alpha_i)`.
    pairing_cauchy: Box<[CyclotomicFp12]>,
}

pub fn validate_batch(
    encryption_key: &EncryptionKey,
    ciphertexts: &[Ciphertext],
) -> Result<ValidatedBatch> {
    ValidatedBatch::from_ciphertexts(encryption_key, ciphertexts)
}

impl ValidatedBatch {
    pub fn verify(encryption_key: &EncryptionKey, ciphertexts: &[Ciphertext]) -> Result<Self> {
        Self::from_ciphertexts(encryption_key, ciphertexts)
    }

    fn from_ciphertexts(
        encryption_key: &EncryptionKey,
        ciphertexts: &[Ciphertext],
    ) -> Result<Self> {
        if ciphertexts.is_empty() {
            return Err(Error::BatchIsEmpty);
        }
        if ciphertexts.len() != encryption_key.batch_size() {
            return Err(Error::MismatchedBatchSize {
                expected: encryption_key.batch_size(),
                actual: ciphertexts.len(),
            });
        }
        let first_affine = ciphertexts
            .iter()
            .map(|ciphertext| ciphertext.first)
            .collect::<Vec<_>>();
        let second = ciphertexts
            .iter()
            .map(|ciphertext| ciphertext.second)
            .collect::<Vec<_>>();
        let proofs = ciphertexts
            .iter()
            .map(|ciphertext| ciphertext.proof.clone())
            .collect::<Vec<_>>();
        if !proof::verify_batch(
            &first_affine,
            &second,
            &proofs,
            encryption_key.setup_id(),
            &mut rand_core::OsRng,
        ) {
            return Err(Error::InvalidProof);
        }
        let first_projective = first_affine
            .iter()
            .copied()
            .map(G1Projective::from)
            .collect::<Vec<_>>();
        Ok(Self {
            setup_id: encryption_key.setup_id(),
            batch_size: ciphertexts.len(),
            digest: batch_digest(ciphertexts),
            first_affine: first_affine.into_boxed_slice(),
            first_projective: first_projective.into_boxed_slice(),
        })
    }

    pub fn setup_id(&self) -> [u8; 32] {
        self.setup_id
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
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

    pub fn shares(&self) -> &[DecryptionShare] {
        &self.shares
    }

    pub fn rejected_parties(&self) -> &[usize] {
        &self.rejected_parties
    }
}

impl CauchyKernel {
    pub fn new(decryption_key: &PublicDecryptionKey) -> Result<Self> {
        let batch_size = decryption_key.batch_size();
        let domain = Radix2Domain::new(batch_size)?;
        let frequency_coefficients = (0..batch_size)
            .map(|index| batch_size as i64 - 1 - 2 * index as i64)
            .collect::<Vec<_>>();
        let doubled_batch = batch_size
            .checked_mul(2)
            .ok_or(Error::ParameterSizeOverflow)?;
        let two_batch_small =
            i64::try_from(doubled_batch).map_err(|_| Error::ParameterSizeOverflow)?;
        Ok(Self {
            setup_id: decryption_key.setup_id(),
            batch_size,
            domain,
            frequency_coefficients: frequency_coefficients.into_boxed_slice(),
            two_batch_small,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

impl DecryptionPrecomputation {
    pub fn accepted_weight(&self) -> usize {
        self.accepted_weight
    }

    pub fn party_count(&self) -> usize {
        self.party_indices.len()
    }
}

/// Compute `sigma_j = sum_i g_{j,i} ct_i[1]` with one affine G1 MSM.
pub fn partial_decrypt(
    party_key: &PartySecretKey,
    batch: &ValidatedBatch,
) -> Result<DecryptionShare> {
    if party_key.setup_id() != batch.setup_id {
        return Err(Error::MismatchedSetup);
    }
    if party_key.batch_size() != batch.batch_size {
        return Err(Error::MismatchedBatchSize {
            expected: party_key.batch_size(),
            actual: batch.batch_size,
        });
    }
    let fractions = party_key.fractions()?;
    Ok(DecryptionShare {
        party_index: party_key.party_index,
        sigma: g1_multi_exp_affine_bytes(&batch.first_affine, &scalars_to_le_bytes(fractions)),
        batch_digest: batch.digest,
        setup_id: batch.setup_id,
    })
}

/// Verify `e(sigma_j,[1]_2) = product_i e(ct_i[1],vk_{j,i})`.
pub fn verify_decryption_share(
    decryption_key: &PublicDecryptionKey,
    batch: &ValidatedBatch,
    share: &DecryptionShare,
) -> Result<bool> {
    ensure_key_batch(decryption_key, batch)?;
    if share.party_index >= decryption_key.party_count() {
        return Err(Error::InvalidPartyIndex(share.party_index));
    }
    if share.setup_id != batch.setup_id || share.batch_digest != batch.digest {
        return Ok(false);
    }
    let mut left = batch.first_affine.to_vec();
    left.push((-share.sigma).to_affine());
    let mut right = (0..batch.batch_size)
        .map(|slot| decryption_key.verification_key(share.party_index, slot))
        .collect::<Result<Vec<_>>>()?;
    right.push(G2Affine::generator());
    Ok(full_final_exponentiation(unprepared_multi_miller_loop(&left, &right)) == Gt::identity())
}

/// Batch-check submitted responses, blame on failure, then retain a
/// deterministic minimum-cardinality authorized real-party subset.
pub fn accept_decryption_shares(
    decryption_key: &PublicDecryptionKey,
    batch: &ValidatedBatch,
    shares: &[DecryptionShare],
) -> Result<AcceptedDecryptionShares> {
    ensure_key_batch(decryption_key, batch)?;
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
        if share.setup_id == batch.setup_id && share.batch_digest == batch.digest {
            candidates.push(share.clone());
        } else {
            rejected_parties.push(share.party_index);
        }
    }

    let aggregate_valid = verify_decryption_shares_batched(decryption_key, batch, &candidates)?;
    let mut valid_shares = if aggregate_valid {
        candidates
    } else {
        let checks = candidates
            .par_iter()
            .map(|share| verify_decryption_share(decryption_key, batch, share))
            .collect::<Vec<_>>();
        let mut valid = Vec::new();
        for (share, check) in candidates.into_iter().zip(checks) {
            if check? {
                valid.push(share);
            } else {
                rejected_parties.push(share.party_index);
            }
        }
        valid
    };

    let valid_weight = valid_shares.iter().try_fold(0usize, |sum, share| {
        sum.checked_add(decryption_key.party_weight(share.party_index)?)
            .ok_or(Error::WeightOverflow)
    })?;
    rejected_parties.sort_unstable();
    if valid_weight <= decryption_key.threshold_weight() {
        return Err(Error::InsufficientWeight {
            accepted: valid_weight,
            required: decryption_key.required_weight(),
            rejected_parties,
        });
    }

    valid_shares.sort_unstable_by(|left, right| {
        decryption_key.party_weights()[right.party_index]
            .cmp(&decryption_key.party_weights()[left.party_index])
            .then_with(|| left.party_index.cmp(&right.party_index))
    });
    let mut selected = Vec::new();
    let mut accepted_weight = 0usize;
    for share in valid_shares {
        if accepted_weight > decryption_key.threshold_weight() {
            break;
        }
        accepted_weight = accepted_weight
            .checked_add(decryption_key.party_weight(share.party_index)?)
            .ok_or(Error::WeightOverflow)?;
        selected.push(share);
    }
    selected.sort_unstable_by_key(|share| share.party_index);

    Ok(AcceptedDecryptionShares {
        setup_id: batch.setup_id,
        batch_size: batch.batch_size,
        digest: batch.digest,
        accepted_weight,
        shares: selected.into_boxed_slice(),
        rejected_parties: rejected_parties.into_boxed_slice(),
    })
}

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
    let share_projective = shares.iter().map(|share| share.sigma).collect::<Vec<_>>();
    let mut share_affine = vec![G1Affine::default(); shares.len()];
    batch_normalize_g1(&share_projective, &mut share_affine);
    let aggregate_sigma = g1_multi_exp_affine_bytes(&share_affine, &challenge_bytes);
    let party_indices = shares
        .iter()
        .map(|share| share.party_index)
        .collect::<Vec<_>>();
    let aggregate_verification_projective = (0..batch.batch_size)
        .into_par_iter()
        .map(|slot| {
            Ok(g2_multi_exp_affine_indexed(
                decryption_key.verification_slot(slot)?,
                &party_indices,
                &challenge_bytes,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut aggregate_verification =
        vec![G2Affine::default(); aggregate_verification_projective.len()];
    batch_normalize_g2(
        &aggregate_verification_projective,
        &mut aggregate_verification,
    );

    let mut left = batch.first_affine.to_vec();
    left.push((-aggregate_sigma).to_affine());
    aggregate_verification.push(G2Affine::generator());
    Ok(
        full_final_exponentiation(unprepared_multi_miller_loop(&left, &aggregate_verification))
            == Gt::identity(),
    )
}

/// Interpolate the accepted virtual shares and construct the paper's D/U
/// opening operands. The result is reusable for the same accepted committee.
pub fn prepare_decryption(
    decryption_key: &PublicDecryptionKey,
    accepted: &AcceptedDecryptionShares,
) -> Result<DecryptionPrecomputation> {
    if decryption_key.setup_id() != accepted.setup_id
        || decryption_key.batch_size() != accepted.batch_size
    {
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
    let context_id = committee_context_id(accepted.setup_id, &party_indices);
    let mut selected_virtual_indices = Vec::with_capacity(accepted.accepted_weight);
    let mut coefficient_offsets = Vec::with_capacity(party_indices.len() + 1);
    coefficient_offsets.push(0);
    for party_index in &party_indices {
        selected_virtual_indices.extend(decryption_key.party_range(*party_index)?);
        coefficient_offsets.push(selected_virtual_indices.len());
    }
    let coefficients = lagrange_at_zero(decryption_key.domain_points(), &selected_virtual_indices)?;
    let coefficient_bytes = scalars_to_le_bytes(&coefficients);
    let inverse_two_batch =
        Option::<Scalar>::from(two_batch_scalar(decryption_key.batch_size())?.invert())
            .expect("2B is nonzero in the scalar field");
    let scaled_coefficients = coefficients
        .iter()
        .map(|coefficient| *coefficient * inverse_two_batch)
        .collect::<Vec<_>>();
    let scaled_coefficient_bytes = scalars_to_le_bytes(&scaled_coefficients);
    let party_count = party_indices.len();
    let opening_count = decryption_key
        .batch_size()
        .checked_mul(party_count)
        .ok_or(Error::ParameterSizeOverflow)?;

    let scaled_d1_projective = (0..opening_count)
        .into_par_iter()
        .map(|flat_index| {
            let slot = flat_index / party_count;
            let party_position = flat_index % party_count;
            let party_index = party_indices[party_position];
            let party_range = decryption_key.party_range(party_index)?;
            let coefficient_range =
                coefficient_offsets[party_position]..coefficient_offsets[party_position + 1];
            let byte_range = coefficient_range.start * 32..coefficient_range.end * 32;
            Ok(g2_multi_exp_affine_bytes(
                &decryption_key.d1_slot(slot)?[party_range],
                &scaled_coefficient_bytes[byte_range],
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let scaled_d1_total_projective = scaled_d1_projective
        .par_chunks(party_count)
        .map(|party_keys| party_keys.iter().copied().sum())
        .collect::<Vec<G2Projective>>();

    let d2_total_projective = (0..decryption_key.batch_size())
        .into_par_iter()
        .map(|slot| {
            Ok(g2_multi_exp_affine_indexed(
                decryption_key.d2_slot(slot)?,
                &selected_virtual_indices,
                &coefficient_bytes,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mask_projective = d2_total_projective
        .par_iter()
        .enumerate()
        .map(|(slot, d2)| {
            let alpha = decryption_key.alpha_points()[slot];
            let product = alpha * (alpha + Scalar::ONE);
            *d2 * product - G2Projective::from(decryption_key.global_key()) * alpha
        })
        .collect::<Vec<_>>();

    let mut all_projective = Vec::with_capacity(opening_count + 2 * decryption_key.batch_size());
    all_projective.extend_from_slice(&scaled_d1_projective);
    all_projective.extend_from_slice(&scaled_d1_total_projective);
    all_projective.extend_from_slice(&mask_projective);
    let mut all_affine = vec![G2Affine::default(); all_projective.len()];
    batch_normalize_g2(&all_projective, &mut all_affine);
    let mask_keys_affine = all_affine.split_off(opening_count + decryption_key.batch_size());
    let scaled_d1_total_affine = all_affine.split_off(opening_count);
    let scaled_d1_by_party = all_affine;
    let (scaled_d1_total, mask_keys) = rayon::join(
        || {
            scaled_d1_total_affine
                .par_iter()
                .map(PreparedG2Lines::from_affine)
                .collect::<Vec<_>>()
        },
        || {
            mask_keys_affine
                .par_iter()
                .map(PreparedG2Lines::from_affine)
                .collect::<Vec<_>>()
        },
    );

    Ok(DecryptionPrecomputation {
        setup_id: accepted.setup_id,
        context_id,
        batch_size: accepted.batch_size,
        accepted_weight: accepted.accepted_weight,
        party_indices: party_indices.into_boxed_slice(),
        scaled_d1_by_party: scaled_d1_by_party.into_boxed_slice(),
        scaled_d1_total: scaled_d1_total.into_boxed_slice(),
        mask_keys: mask_keys.into_boxed_slice(),
    })
}

/// Compute the source Cauchy sums and the reusable target-group cross terms.
pub fn precompute_batch(
    kernel: &CauchyKernel,
    decryption: &DecryptionPrecomputation,
    batch: &ValidatedBatch,
) -> Result<BatchPrecomputation> {
    ensure_precomputation_batch(kernel, decryption, batch)?;
    let (scaled_first_cauchy, pairing_cauchy) = rayon::join(
        || {
            cauchy_scaled_group(
                &kernel.domain,
                &kernel.frequency_coefficients,
                batch.first_projective.to_vec(),
            )
        },
        || {
            let miller = batch
                .first_affine
                .par_iter()
                .zip(decryption.scaled_d1_total.par_iter())
                .map(|(first, key)| key.miller_loop(first))
                .collect::<Vec<_>>();
            let pairings = easy_finalize_miller_batch(miller);
            cauchy_scaled_cyclotomic(&kernel.domain, &kernel.frequency_coefficients, pairings)
        },
    );
    Ok(BatchPrecomputation {
        setup_id: batch.setup_id,
        context_id: decryption.context_id,
        batch_size: batch.batch_size,
        digest: batch.digest,
        scaled_first_cauchy: scaled_first_cauchy.into_boxed_slice(),
        pairing_cauchy: pairing_cauchy.into_boxed_slice(),
    })
}

/// Evaluate the B arity-|T| pairing products and remove each one-time pad.
pub fn open_batch(
    decryption_key: &PublicDecryptionKey,
    kernel: &CauchyKernel,
    decryption: &DecryptionPrecomputation,
    accepted: &AcceptedDecryptionShares,
    batch: &ValidatedBatch,
    ciphertexts: &[Ciphertext],
    precomputation: &BatchPrecomputation,
) -> Result<Vec<Gt>> {
    ensure_opening_bindings(
        decryption_key,
        kernel,
        decryption,
        accepted,
        batch,
        ciphertexts,
        precomputation,
    )?;
    let party_count = accepted.shares.len();
    let scaled_shares = accepted
        .shares
        .iter()
        .map(|share| mul_group_signed_small(share.sigma, kernel.two_batch_small))
        .collect::<Vec<_>>();
    let left_projective = (0..batch.batch_size * party_count)
        .into_par_iter()
        .map(|flat_index| {
            let slot = flat_index / party_count;
            let party_position = flat_index % party_count;
            scaled_shares[party_position] - precomputation.scaled_first_cauchy[slot]
        })
        .collect::<Vec<_>>();
    let mut left_affine = vec![G1Affine::default(); left_projective.len()];
    batch_normalize_g1(&left_projective, &mut left_affine);

    let mut miller_results = (0..batch.batch_size)
        .into_par_iter()
        .map(|slot| {
            let start = slot * party_count;
            unprepared_multi_miller_loop(
                &left_affine[start..start + party_count],
                &decryption.scaled_d1_by_party[start..start + party_count],
            )
        })
        .collect::<Vec<_>>();
    miller_results.extend(
        batch
            .first_affine
            .par_iter()
            .zip(decryption.mask_keys.par_iter())
            .map(|(first, key)| key.miller_loop(first))
            .collect::<Vec<_>>(),
    );
    let easy = easy_finalize_miller_batch(miller_results);
    let (first_terms, mask_pairings) = easy.split_at(batch.batch_size);

    Ok((0..batch.batch_size)
        .into_par_iter()
        .map(|slot| {
            let alpha = decryption_key.alpha_points()[slot];
            let alpha_product = alpha * (alpha + Scalar::ONE);
            let cauchy_and_first = first_terms[slot].product(&precomputation.pairing_cauchy[slot]);
            let subtracted = cyclotomic_mul_scalar_decomposed(cauchy_and_first, alpha_product);
            let combined = mask_pairings[slot].product(&subtracted.inverse());
            let mask = hard_final_exponentiation(combined);
            ciphertexts[slot].second - mask
        })
        .collect())
}

pub fn decrypt(
    encryption_key: &EncryptionKey,
    decryption_key: &PublicDecryptionKey,
    ciphertexts: &[Ciphertext],
    shares: &[DecryptionShare],
) -> Result<Vec<Gt>> {
    if encryption_key.setup_id() != decryption_key.setup_id() {
        return Err(Error::MismatchedSetup);
    }
    let batch = validate_batch(encryption_key, ciphertexts)?;
    let accepted = accept_decryption_shares(decryption_key, &batch, shares)?;
    let kernel = CauchyKernel::new(decryption_key)?;
    let decryption = prepare_decryption(decryption_key, &accepted)?;
    let precomputation = precompute_batch(&kernel, &decryption, &batch)?;
    open_batch(
        decryption_key,
        &kernel,
        &decryption,
        &accepted,
        &batch,
        ciphertexts,
        &precomputation,
    )
}

fn easy_finalize_miller_batch(inputs: Vec<RawMillerResult>) -> Vec<CyclotomicFp12> {
    if rayon::current_num_threads() == 1 {
        batch_easy_final_exponentiation(&inputs)
    } else {
        inputs
            .into_par_iter()
            .map(easy_final_exponentiation)
            .collect()
    }
}

/// Return `2B * sum_{k != i} values[k]/(alpha_k-alpha_i)` for every i.
fn cauchy_scaled_group<G>(
    domain: &Radix2Domain,
    frequency_coefficients: &[i64],
    mut values: Vec<G>,
) -> Vec<G>
where
    G: Group<Scalar = Scalar> + Send + Sync,
{
    debug_assert_eq!(values.len(), domain.size());
    domain.fft(&mut values);
    values
        .par_iter_mut()
        .zip(frequency_coefficients.par_iter())
        .for_each(|(value, coefficient)| *value = mul_group_signed_small(*value, *coefficient));
    let spectrum = values;
    let mut remapped = vec![G::identity(); spectrum.len()];
    remapped
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, value)| {
            let previous = (index + spectrum.len() - 1) % spectrum.len();
            *value = -spectrum[previous];
        });
    domain.ifft_unscaled(&mut remapped);
    remapped
}

fn cauchy_scaled_cyclotomic(
    domain: &Radix2Domain,
    frequency_coefficients: &[i64],
    mut values: Vec<CyclotomicFp12>,
) -> Vec<CyclotomicFp12> {
    debug_assert_eq!(values.len(), domain.size());
    domain.fft_cyclotomic(&mut values);
    values
        .par_iter_mut()
        .zip(frequency_coefficients.par_iter())
        .for_each(|(value, coefficient)| {
            *value = mul_cyclotomic_signed_small(*value, *coefficient)
        });
    let spectrum = values;
    let mut remapped = vec![CyclotomicFp12::identity(); spectrum.len()];
    remapped
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, value)| {
            let previous = (index + spectrum.len() - 1) % spectrum.len();
            *value = spectrum[previous].inverse();
        });
    domain.ifft_cyclotomic_unscaled(&mut remapped);
    remapped
}

fn mul_group_signed_small<G>(value: G, coefficient: i64) -> G
where
    G: Group,
{
    if coefficient == 0 {
        return G::identity();
    }
    let mut magnitude = coefficient.unsigned_abs();
    let mut base = value;
    let mut result = G::identity();
    while magnitude != 0 {
        if magnitude & 1 == 1 {
            result += base;
        }
        magnitude >>= 1;
        if magnitude != 0 {
            base = base.double();
        }
    }
    if coefficient < 0 {
        -result
    } else {
        result
    }
}

fn ensure_key_batch(decryption_key: &PublicDecryptionKey, batch: &ValidatedBatch) -> Result<()> {
    if decryption_key.setup_id() != batch.setup_id {
        return Err(Error::MismatchedSetup);
    }
    if decryption_key.batch_size() != batch.batch_size {
        return Err(Error::MismatchedBatchSize {
            expected: decryption_key.batch_size(),
            actual: batch.batch_size,
        });
    }
    Ok(())
}

fn ensure_precomputation_batch(
    kernel: &CauchyKernel,
    decryption: &DecryptionPrecomputation,
    batch: &ValidatedBatch,
) -> Result<()> {
    if kernel.setup_id != batch.setup_id || decryption.setup_id != batch.setup_id {
        return Err(Error::MismatchedSetup);
    }
    if kernel.batch_size != batch.batch_size || decryption.batch_size != batch.batch_size {
        return Err(Error::MismatchedBatchSize {
            expected: kernel.batch_size,
            actual: batch.batch_size,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ensure_opening_bindings(
    decryption_key: &PublicDecryptionKey,
    kernel: &CauchyKernel,
    decryption: &DecryptionPrecomputation,
    accepted: &AcceptedDecryptionShares,
    batch: &ValidatedBatch,
    ciphertexts: &[Ciphertext],
    precomputation: &BatchPrecomputation,
) -> Result<()> {
    ensure_key_batch(decryption_key, batch)?;
    ensure_precomputation_batch(kernel, decryption, batch)?;
    if accepted.setup_id != batch.setup_id || precomputation.setup_id != batch.setup_id {
        return Err(Error::MismatchedSetup);
    }
    if accepted.batch_size != batch.batch_size
        || ciphertexts.len() != batch.batch_size
        || precomputation.batch_size != batch.batch_size
    {
        return Err(Error::MismatchedBatchSize {
            expected: batch.batch_size,
            actual: ciphertexts.len(),
        });
    }
    if accepted.digest != batch.digest
        || precomputation.digest != batch.digest
        || batch_digest(ciphertexts) != batch.digest
    {
        return Err(Error::MismatchedBatchDigest);
    }
    let party_indices = accepted.party_indices().collect::<Vec<_>>();
    if party_indices.as_slice() != &*decryption.party_indices
        || committee_context_id(accepted.setup_id, &party_indices) != decryption.context_id
        || precomputation.context_id != decryption.context_id
    {
        return Err(Error::MismatchedCommittee);
    }
    Ok(())
}

fn two_batch_scalar(batch_size: usize) -> Result<Scalar> {
    let doubled = batch_size
        .checked_mul(2)
        .ok_or(Error::ParameterSizeOverflow)?;
    let value = u64::try_from(doubled).map_err(|_| Error::ParameterSizeOverflow)?;
    Ok(Scalar::from(value))
}

fn committee_context_id(setup_id: [u8; 32], party_indices: &[usize]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(COMMITTEE_CONTEXT_DOMAIN);
    hasher.update(setup_id);
    hasher.update((party_indices.len() as u64).to_le_bytes());
    for party_index in party_indices {
        hasher.update((*party_index as u64).to_le_bytes());
    }
    hasher.finalize().into()
}

fn batch_digest(ciphertexts: &[Ciphertext]) -> [u8; 32] {
    let mut transcript = Vec::new();
    transcript.extend_from_slice(&(ciphertexts.len() as u64).to_le_bytes());
    for ciphertext in ciphertexts {
        ciphertext.append_canonical(&mut transcript);
    }
    Sha256::new()
        .chain_update(BATCH_DIGEST_DOMAIN)
        .chain_update(transcript)
        .finalize()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{encrypt, keygen, KeyMaterial};
    use blstrs::{Bls12, G2Projective};
    use pairing::Engine;

    struct Fixture {
        material: KeyMaterial,
        messages: Vec<Gt>,
        ciphertexts: Vec<Ciphertext>,
        batch: ValidatedBatch,
        accepted: AcceptedDecryptionShares,
        kernel: CauchyKernel,
        decryption: DecryptionPrecomputation,
        batch_precomputation: BatchPrecomputation,
    }

    fn fixture(batch_size: usize) -> Fixture {
        let material = keygen(batch_size, &[1, 3, 2, 4], 4).unwrap();
        let messages = (0..batch_size)
            .map(|slot| Gt::generator() * Scalar::from((slot + 11) as u64))
            .collect::<Vec<_>>();
        let ciphertexts = messages
            .iter()
            .map(|message| encrypt(&material.encryption_key, *message))
            .collect::<Vec<_>>();
        let batch = validate_batch(&material.encryption_key, &ciphertexts).unwrap();
        let shares = material
            .party_keys
            .iter()
            .map(|party| partial_decrypt(party, &batch).unwrap())
            .collect::<Vec<_>>();
        let accepted = accept_decryption_shares(&material.decryption_key, &batch, &shares).unwrap();
        let kernel = CauchyKernel::new(&material.decryption_key).unwrap();
        let decryption = prepare_decryption(&material.decryption_key, &accepted).unwrap();
        let batch_precomputation = precompute_batch(&kernel, &decryption, &batch).unwrap();
        Fixture {
            material,
            messages,
            ciphertexts,
            batch,
            accepted,
            kernel,
            decryption,
            batch_precomputation,
        }
    }

    fn open_fixture(fixture: &Fixture) -> Vec<Gt> {
        open_batch(
            &fixture.material.decryption_key,
            &fixture.kernel,
            &fixture.decryption,
            &fixture.accepted,
            &fixture.batch,
            &fixture.ciphertexts,
            &fixture.batch_precomputation,
        )
        .unwrap()
    }

    #[test]
    fn heterogeneous_weighted_protocol_round_trips_all_small_batch_sizes() {
        for batch_size in [1usize, 2, 4, 8, 16] {
            let fixture = fixture(batch_size);
            assert_eq!(fixture.accepted.party_count(), 2);
            assert_eq!(fixture.accepted.accepted_weight(), 7);
            assert_eq!(open_fixture(&fixture), fixture.messages, "B={batch_size}");
        }
    }

    #[test]
    fn fused_opening_round_trips_with_single_thread_batched_easy_parts() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let fixture = fixture(8);
            assert_eq!(open_fixture(&fixture), fixture.messages);
        });
    }

    #[test]
    fn non_greedy_authorized_committee_reconstructs_the_same_messages() {
        let fixture = fixture(4);
        // Weights 1 + 4 meet q=5 exactly, but this is not the default
        // heaviest-first committee (weights 4 + 3).
        let shares = [0usize, 3]
            .into_iter()
            .map(|party| partial_decrypt(&fixture.material.party_keys[party], &fixture.batch))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let accepted =
            accept_decryption_shares(&fixture.material.decryption_key, &fixture.batch, &shares)
                .unwrap();
        assert_eq!(accepted.party_indices().collect::<Vec<_>>(), [0, 3]);
        assert_eq!(accepted.accepted_weight(), 5);

        let decryption = prepare_decryption(&fixture.material.decryption_key, &accepted).unwrap();
        let batch_precomputation =
            precompute_batch(&fixture.kernel, &decryption, &fixture.batch).unwrap();
        let decrypted = open_batch(
            &fixture.material.decryption_key,
            &fixture.kernel,
            &decryption,
            &accepted,
            &fixture.batch,
            &fixture.ciphertexts,
            &batch_precomputation,
        )
        .unwrap();
        assert_eq!(decrypted, fixture.messages);
    }

    #[test]
    fn optimized_cauchy_transforms_match_literal_quadratic_sums() {
        for batch_size in [1usize, 2, 4, 8, 16] {
            let material = keygen(batch_size, &[1, 2], 1).unwrap();
            let key = &material.decryption_key;
            let kernel = CauchyKernel::new(key).unwrap();
            let g1_values = (0..batch_size)
                .map(|index| G1Projective::generator() * Scalar::from((index + 3) as u64))
                .collect::<Vec<_>>();
            let raw_pairings = (0..batch_size)
                .map(|index| {
                    let left =
                        (G1Projective::generator() * Scalar::from((index + 5) as u64)).to_affine();
                    let right = (G2Projective::generator() * Scalar::from((2 * index + 7) as u64))
                        .to_affine();
                    unprepared_multi_miller_loop(&[left], &[right])
                })
                .collect::<Vec<_>>();
            let easy_pairings = raw_pairings
                .iter()
                .copied()
                .map(easy_final_exponentiation)
                .collect::<Vec<_>>();
            let gt_values = raw_pairings
                .into_iter()
                .map(full_final_exponentiation)
                .collect::<Vec<_>>();
            let actual_g1 = cauchy_scaled_group(
                &kernel.domain,
                &kernel.frequency_coefficients,
                g1_values.clone(),
            );
            let actual_gt = cauchy_scaled_cyclotomic(
                &kernel.domain,
                &kernel.frequency_coefficients,
                easy_pairings,
            )
            .into_iter()
            .map(hard_final_exponentiation)
            .collect::<Vec<_>>();
            for output in 0..batch_size {
                let mut expected_g1 = G1Projective::identity();
                let mut expected_gt = Gt::identity();
                for input in 0..batch_size {
                    if input == output {
                        continue;
                    }
                    let inverse = Option::<Scalar>::from(
                        (key.alpha_points()[input] - key.alpha_points()[output]).invert(),
                    )
                    .unwrap();
                    expected_g1 += g1_values[input] * inverse;
                    expected_gt += gt_values[input] * inverse;
                }
                let two_batch = Scalar::from(kernel.two_batch_small as u64);
                assert_eq!(actual_g1[output], expected_g1 * two_batch);
                assert_eq!(actual_gt[output], expected_gt * two_batch);
            }
        }
    }

    #[test]
    fn optimized_opening_matches_construction_five_displayed_formula() {
        let fixture = fixture(8);
        let key = &fixture.material.decryption_key;
        let optimized = open_fixture(&fixture);

        let party_indices = fixture.accepted.party_indices().collect::<Vec<_>>();
        let mut selected_virtual_indices = Vec::new();
        for party_index in &party_indices {
            selected_virtual_indices.extend(key.party_range(*party_index).unwrap());
        }
        let coefficients =
            lagrange_at_zero(key.domain_points(), &selected_virtual_indices).unwrap();
        let mut coefficient_by_virtual = vec![Scalar::ZERO; key.total_weight()];
        for (virtual_index, coefficient) in
            selected_virtual_indices.iter().copied().zip(coefficients)
        {
            coefficient_by_virtual[virtual_index] = coefficient;
        }

        let literal = (0..fixture.batch.batch_size)
            .map(|output| {
                let alpha_i = key.alpha_points()[output];
                let mut excluded_first_sum = G1Projective::identity();
                for input in 0..fixture.batch.batch_size {
                    if input == output {
                        continue;
                    }
                    let inverse =
                        Option::<Scalar>::from((key.alpha_points()[input] - alpha_i).invert())
                            .unwrap();
                    excluded_first_sum += fixture.batch.first_projective[input] * inverse;
                }

                let mut l_sum = Gt::identity();
                for share in fixture.accepted.shares() {
                    let party_range = key.party_range(share.party_index).unwrap();
                    let party_d1 = |slot: usize| {
                        key.d1_slot(slot).unwrap()[party_range.clone()]
                            .iter()
                            .zip(&coefficient_by_virtual[party_range.clone()])
                            .fold(G2Projective::identity(), |sum, (point, coefficient)| {
                                sum + G2Projective::from(*point) * coefficient
                            })
                    };
                    let first_pair = Bls12::pairing(
                        &(share.sigma - excluded_first_sum).to_affine(),
                        &party_d1(output).to_affine(),
                    );
                    let mut second_pairs = Gt::identity();
                    for input in 0..fixture.batch.batch_size {
                        if input == output {
                            continue;
                        }
                        let inverse =
                            Option::<Scalar>::from((key.alpha_points()[input] - alpha_i).invert())
                                .unwrap();
                        second_pairs += Bls12::pairing(
                            &fixture.batch.first_affine[input],
                            &party_d1(input).to_affine(),
                        ) * inverse;
                    }
                    l_sum += first_pair + second_pairs;
                }

                let d2 = selected_virtual_indices.iter().copied().fold(
                    G2Projective::identity(),
                    |sum, virtual_index| {
                        sum + G2Projective::from(key.d2_slot(output).unwrap()[virtual_index])
                            * coefficient_by_virtual[virtual_index]
                    },
                );
                let d2_pair = Bls12::pairing(&fixture.batch.first_affine[output], &d2.to_affine());
                let global_pair =
                    Bls12::pairing(&fixture.batch.first_affine[output], &key.global_key());
                let mask =
                    (d2_pair - l_sum) * (alpha_i * (alpha_i + Scalar::ONE)) - global_pair * alpha_i;
                fixture.ciphertexts[output].second - mask
            })
            .collect::<Vec<_>>();
        assert_eq!(optimized, literal);
        assert_eq!(literal, fixture.messages);
    }

    #[test]
    fn malformed_shares_are_blamed_and_authorization_is_weight_based() {
        let fixture = fixture(4);
        let material = &fixture.material;
        let batch = &fixture.batch;
        let honest = partial_decrypt(&material.party_keys[3], batch).unwrap();
        assert!(verify_decryption_share(&material.decryption_key, batch, &honest).unwrap());

        let mut invalid = honest.clone();
        invalid.sigma += G1Projective::generator();
        assert!(!verify_decryption_share(&material.decryption_key, batch, &invalid).unwrap());

        let one = partial_decrypt(&material.party_keys[3], batch).unwrap();
        assert!(matches!(
            accept_decryption_shares(&material.decryption_key, batch, &[one]),
            Err(Error::InsufficientWeight {
                accepted: 4,
                required: 5,
                ..
            })
        ));

        let mut shares = material
            .party_keys
            .iter()
            .map(|party| partial_decrypt(party, batch).unwrap())
            .collect::<Vec<_>>();
        shares[0].sigma += G1Projective::generator();
        let accepted = accept_decryption_shares(&material.decryption_key, batch, &shares).unwrap();
        assert_eq!(accepted.rejected_parties(), &[0]);
        assert!(accepted.accepted_weight() > material.decryption_key.threshold_weight());

        let duplicate = partial_decrypt(&material.party_keys[1], batch).unwrap();
        assert!(matches!(
            accept_decryption_shares(
                &material.decryption_key,
                batch,
                &[duplicate.clone(), duplicate]
            ),
            Err(Error::DuplicatePartyIndex(1))
        ));
    }

    #[test]
    fn batch_validation_and_convenience_decryption_cover_the_full_pipeline() {
        let fixture = fixture(4);
        assert_eq!(
            decrypt(
                &fixture.material.encryption_key,
                &fixture.material.decryption_key,
                &fixture.ciphertexts,
                fixture.accepted.shares(),
            )
            .unwrap(),
            fixture.messages
        );

        assert!(matches!(
            validate_batch(&fixture.material.encryption_key, &fixture.ciphertexts[..3]),
            Err(Error::MismatchedBatchSize {
                expected: 4,
                actual: 3
            })
        ));

        let mut changed_payload = fixture.ciphertexts.clone();
        changed_payload[0].second += Gt::generator();
        assert!(matches!(
            validate_batch(&fixture.material.encryption_key, &changed_payload),
            Err(Error::InvalidProof)
        ));

        let mut changed_proof = fixture.ciphertexts.clone();
        changed_proof[0].proof.response += Scalar::ONE;
        assert!(matches!(
            validate_batch(&fixture.material.encryption_key, &changed_proof),
            Err(Error::InvalidProof)
        ));
    }

    #[test]
    fn phased_objects_are_bound_to_setup_batch_and_committee() {
        let primary = fixture(4);
        let other = fixture(4);
        assert!(matches!(
            precompute_batch(&primary.kernel, &primary.decryption, &other.batch),
            Err(Error::MismatchedSetup)
        ));

        let mut changed = primary.ciphertexts.clone();
        changed[0].second += Gt::generator();
        assert!(matches!(
            open_batch(
                &primary.material.decryption_key,
                &primary.kernel,
                &primary.decryption,
                &primary.accepted,
                &primary.batch,
                &changed,
                &primary.batch_precomputation,
            ),
            Err(Error::MismatchedBatchDigest)
        ));
    }
}
