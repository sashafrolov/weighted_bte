//! Validation and optimized decryption for the indexed weighted BTE scheme.
//!
//! The construction has two distinct kinds of reusable work.  The FFT kernel
//! depends only on the powers-of-tau public parameters, while the weighted
//! opening keys depend on the accepted real-party set and the set of indices
//! occupied by valid ciphertexts.  Ciphertext contents and server responses
//! remain bound to a validated batch digest throughout the online phases.

use blstrs::{G1Affine, G1Projective, G2Affine, G2Projective, Gt};
use group::{prime::PrimeCurveAffine, Curve, Group};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    blst_utils::{
        batch_normalize_g1, batch_normalize_g2, g2_multi_exp_affine_bytes, scalars_to_le_bytes,
    },
    encryption::{xor_with_mask, Ciphertext},
    error::{Error, Result},
    fft::Radix2Domain,
    final_exponentiation::{
        batch_easy_final_exponentiation, easy_final_exponentiation, full_final_exponentiation,
        hard_final_exponentiation, unprepared_multi_miller_loop, PreparedG2Lines,
    },
    interpolation::lagrange_at_zero,
    proof::{DleqProof, ProofPurpose},
    setup::{MasterPublicKey, PartySecretKey, PublicParameters},
};

const BATCH_DIGEST_DOMAIN: &[u8] = b"WEIGHTED-INDEXED-BTE-BATCH-v1";
const SERVER_PROOF_CONTEXT_DOMAIN: &[u8] = b"WEIGHTED-INDEXED-BTE-SERVER-CONTEXT-v1";
const COMMITTEE_CONTEXT_DOMAIN: &[u8] = b"WEIGHTED-INDEXED-BTE-COMMITTEE-v1";

/// A ciphertext batch whose client proofs have each been checked once.
///
/// Invalid proofs retain their original output slots but contribute neither
/// to the aggregate first component nor to the sparse punctured-key vector.
#[derive(Clone, Debug)]
pub struct ValidatedBatch {
    setup_id: [u8; 32],
    crs_id: [u8; 32],
    index_space: usize,
    batch_size: usize,
    digest: [u8; 32],
    valid: Box<[bool]>,
    /// A valid ciphertext's index in its original slot; invalid slots are
    /// `None`.
    valid_index_by_slot: Box<[Option<usize>]>,
    /// Position within `used_indices` for each valid original slot.
    used_position_by_slot: Box<[Option<usize>]>,
    /// Valid occupied indices in ascending order.
    used_indices: Box<[usize]>,
    aggregate_first: G1Affine,
    /// Sparse, index-addressed punctured keys; unoccupied indices are the
    /// identity.
    punctured_by_index: Box<[G1Projective]>,
    punctured_affine_by_index: Box<[G1Affine]>,
}

/// One real party's constant-size response for an entire validated batch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecryptionShare {
    pub party_index: usize,
    pub sigma: G1Affine,
    pub proof: DleqProof,
    pub batch_digest: [u8; 32],
    pub setup_id: [u8; 32],
}

/// Valid server responses whose combined virtual weight exceeds the
/// corruption threshold.
#[derive(Clone, Debug)]
pub struct AcceptedDecryptionShares {
    setup_id: [u8; 32],
    index_space: usize,
    batch_size: usize,
    digest: [u8; 32],
    accepted_weight: usize,
    used_indices: Box<[usize]>,
    shares: Box<[DecryptionShare]>,
    rejected_parties: Box<[usize]>,
}

/// Fixed powers-of-tau middle-product kernel.
///
/// The transformed and prepared G2 points can be reused for every batch under
/// the same powers-of-tau CRS and index space, including across committee-key
/// rotations.
#[derive(Clone, Debug)]
pub struct IndexedMiddleProductKernel {
    crs_id: [u8; 32],
    index_space: usize,
    transform_size: usize,
    domain: Radix2Domain,
    transformed_kernel: Box<[PreparedG2Lines]>,
    /// Direct-path powers for positive/negative nonzero offsets. Slot zero is
    /// the identity in both arrays.
    positive_powers: Box<[G2Affine]>,
    negative_powers: Box<[G2Affine]>,
}

/// Weighted interpolation material for a fixed accepted party/index set.
#[derive(Clone, Debug)]
pub struct DecryptionPrecomputation {
    setup_id: [u8; 32],
    context_id: [u8; 32],
    index_space: usize,
    accepted_weight: usize,
    party_indices: Box<[usize]>,
    used_indices: Box<[usize]>,
    /// Used-index-major opening keys.  Every contiguous party-sized block
    /// corresponds to one entry of `used_indices`.
    opening_keys: Box<[G2Affine]>,
}

/// Ciphertext-dependent cross terms aligned to the original ciphertext
/// slots.
#[derive(Clone, Debug)]
pub struct BatchPrecomputation {
    setup_id: [u8; 32],
    index_space: usize,
    batch_size: usize,
    digest: [u8; 32],
    beta: Box<[Gt]>,
}

pub fn validate_batch(
    public_key: &MasterPublicKey,
    ciphertexts: &[Ciphertext],
) -> Result<ValidatedBatch> {
    ValidatedBatch::from_ciphertexts(public_key, ciphertexts)
}

impl ValidatedBatch {
    pub fn verify(public_key: &MasterPublicKey, ciphertexts: &[Ciphertext]) -> Result<Self> {
        Self::from_ciphertexts(public_key, ciphertexts)
    }

    fn from_ciphertexts(public_key: &MasterPublicKey, ciphertexts: &[Ciphertext]) -> Result<Self> {
        validate_batch_size(public_key.index_space_size(), ciphertexts.len())?;
        let index_space = public_key.index_space_size();
        let setup_id = public_key.setup_id();
        let crs_id = public_key.crs_id();

        // Out-of-domain indices fail `verify` and are filtered like other
        // malformed ciphertexts, preserving availability for valid slots.
        // In-domain duplicate indices are rejected only when both proofs are
        // valid, since an invalid ciphertext does not occupy an index.
        let valid = ciphertexts
            .par_iter()
            .map(|ciphertext| ciphertext.verify(public_key))
            .collect::<Vec<_>>();

        let mut slot_by_index = vec![None; index_space];
        for (slot, (ciphertext, is_valid)) in ciphertexts.iter().zip(&valid).enumerate() {
            if !*is_valid {
                continue;
            }
            if slot_by_index[ciphertext.index].replace(slot).is_some() {
                return Err(Error::DuplicateCiphertextIndex(ciphertext.index));
            }
        }

        let used_indices = slot_by_index
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| slot.is_some().then_some(index))
            .collect::<Vec<_>>();
        let mut used_position_by_index = vec![None; index_space];
        for (position, index) in used_indices.iter().copied().enumerate() {
            used_position_by_index[index] = Some(position);
        }

        let valid_index_by_slot = ciphertexts
            .iter()
            .zip(&valid)
            .map(|(ciphertext, is_valid)| is_valid.then_some(ciphertext.index))
            .collect::<Vec<_>>();
        let used_position_by_slot = valid_index_by_slot
            .iter()
            .map(|index| index.and_then(|index| used_position_by_index[index]))
            .collect::<Vec<_>>();

        let aggregate_first = ciphertexts
            .iter()
            .zip(&valid)
            .filter(|(_, is_valid)| **is_valid)
            .map(|(ciphertext, _)| G1Projective::from(ciphertext.first))
            .sum::<G1Projective>()
            .to_affine();

        let mut punctured_by_index = vec![G1Projective::identity(); index_space];
        let mut punctured_affine_by_index = vec![G1Affine::identity(); index_space];
        for index in &used_indices {
            let slot = slot_by_index[*index].expect("used indices have an occupied slot");
            let punctured = ciphertexts[slot].punctured;
            punctured_by_index[*index] = G1Projective::from(punctured);
            punctured_affine_by_index[*index] = punctured;
        }

        Ok(Self {
            setup_id,
            crs_id,
            index_space,
            batch_size: ciphertexts.len(),
            digest: batch_digest(ciphertexts),
            valid: valid.into_boxed_slice(),
            valid_index_by_slot: valid_index_by_slot.into_boxed_slice(),
            used_position_by_slot: used_position_by_slot.into_boxed_slice(),
            used_indices: used_indices.into_boxed_slice(),
            aggregate_first,
            punctured_by_index: punctured_by_index.into_boxed_slice(),
            punctured_affine_by_index: punctured_affine_by_index.into_boxed_slice(),
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
        self.used_indices.len()
    }

    pub fn used_indices(&self) -> &[usize] {
        &self.used_indices
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

    pub fn used_indices(&self) -> &[usize] {
        &self.used_indices
    }
}

impl IndexedMiddleProductKernel {
    /// Transform and prepare the fixed centered powers used by every indexed
    /// batch under this powers-of-tau CRS. The result remains reusable across
    /// committee/key rotations that retain the same public parameters.
    pub fn new(public_parameters: &PublicParameters) -> Result<Self> {
        let index_space = public_parameters.index_space_size();
        let doubled = index_space
            .checked_mul(2)
            .ok_or(Error::InvalidBatchSize(index_space))?;
        let transform_size = doubled
            .checked_next_power_of_two()
            .ok_or(Error::InvalidBatchSize(index_space))?;
        let domain = Radix2Domain::new(transform_size)?;
        let mut kernel = vec![G2Projective::identity(); transform_size];
        let mut positive_powers = vec![G2Affine::identity(); index_space];
        let mut negative_powers = vec![G2Affine::identity(); index_space];

        // Circular convolution is collision-free because m >= 2n.  K[0] is
        // deliberately the identity: the missing centered power excludes a
        // ciphertext's puncture point from its own cross term.
        for distance in 1..index_space {
            let positive = public_parameters.centered_power(distance as isize)?;
            let negative = public_parameters.centered_power(-(distance as isize))?;
            positive_powers[distance] = positive;
            negative_powers[distance] = negative;
            kernel[distance] = G2Projective::from(positive);
            kernel[transform_size - distance] = G2Projective::from(negative);
        }

        domain.fft(&mut kernel);
        let inverse_size = domain.size_inverse();
        kernel
            .par_iter_mut()
            .for_each(|point| *point *= inverse_size);

        let mut kernel_affine = vec![G2Affine::identity(); transform_size];
        batch_normalize_g2(&kernel, &mut kernel_affine);
        let transformed_kernel = kernel_affine
            .par_iter()
            .map(PreparedG2Lines::from_affine)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self {
            crs_id: public_parameters.crs_id(),
            index_space,
            transform_size,
            domain,
            transformed_kernel,
            positive_powers: positive_powers.into_boxed_slice(),
            negative_powers: negative_powers.into_boxed_slice(),
        })
    }

    pub fn index_space(&self) -> usize {
        self.index_space
    }

    pub fn transform_size(&self) -> usize {
        self.transform_size
    }
}

impl DecryptionPrecomputation {
    pub fn accepted_weight(&self) -> usize {
        self.accepted_weight
    }

    pub fn party_count(&self) -> usize {
        self.party_indices.len()
    }

    pub fn used_indices(&self) -> &[usize] {
        &self.used_indices
    }
}

/// Produce `sigma_j = q_j * sum ct[first]` and prove the equality of
/// discrete logarithms against the party's public inverse key.
pub fn partial_decrypt(
    party_key: &PartySecretKey,
    batch: &ValidatedBatch,
) -> Result<DecryptionShare> {
    if party_key.setup_id() != batch.setup_id {
        return Err(Error::MismatchedSetup);
    }
    let scalar = party_key.inverse_scalar();
    let sigma = (G1Projective::from(batch.aggregate_first) * scalar).to_affine();
    let context = server_proof_context(batch.digest, party_key.party_index);
    let proof = DleqProof::create(
        ProofPurpose::Server,
        batch.setup_id,
        &context,
        G1Affine::generator(),
        party_key.public_inverse(),
        batch.aggregate_first,
        sigma,
        scalar,
        &mut rand_core::OsRng,
    );

    Ok(DecryptionShare {
        party_index: party_key.party_index,
        sigma,
        proof,
        batch_digest: batch.digest,
        setup_id: batch.setup_id,
    })
}

/// Verify one party response against the validated batch and public inverse
/// key. This is exposed for callers that need an explicit blame check.
pub fn verify_decryption_share(
    public_key: &MasterPublicKey,
    batch: &ValidatedBatch,
    share: &DecryptionShare,
) -> Result<bool> {
    if public_key.setup_id() != batch.setup_id || public_key.index_space_size() != batch.index_space
    {
        return Err(Error::MismatchedSetup);
    }
    if share.party_index >= public_key.party_count() {
        return Err(Error::InvalidPartyIndex(share.party_index));
    }
    if share.setup_id != batch.setup_id || share.batch_digest != batch.digest {
        return Ok(false);
    }

    let context = server_proof_context(batch.digest, share.party_index);
    Ok(share.proof.verify(
        ProofPurpose::Server,
        batch.setup_id,
        &context,
        G1Affine::generator(),
        public_key.inverse_key(share.party_index)?,
        batch.aggregate_first,
        share.sigma,
    ))
}

/// Directly verify every server proof in parallel, blame malformed responses,
/// and select a deterministic minimum-cardinality authorized subset.
///
/// Figure 2 first validates the responder set `S` and then chooses
/// `V subset S` with weight greater than `t`. We greedily take the largest
/// public weights (party-index tie break), which minimizes the number of
/// pairing inputs used during opening, then restore party-index order for a
/// canonical interpolation domain.
pub fn accept_decryption_shares(
    public_key: &MasterPublicKey,
    batch: &ValidatedBatch,
    shares: &[DecryptionShare],
) -> Result<AcceptedDecryptionShares> {
    if public_key.setup_id() != batch.setup_id || public_key.index_space_size() != batch.index_space
    {
        return Err(Error::MismatchedSetup);
    }
    if shares.len() > public_key.party_count() {
        return Err(Error::TooManyShares {
            supplied: shares.len(),
            party_count: public_key.party_count(),
        });
    }

    let mut seen = vec![false; public_key.party_count()];
    for share in shares {
        if share.party_index >= public_key.party_count() {
            return Err(Error::InvalidPartyIndex(share.party_index));
        }
        if seen[share.party_index] {
            return Err(Error::DuplicatePartyIndex(share.party_index));
        }
        seen[share.party_index] = true;
    }

    let checks = shares
        .par_iter()
        .map(|share| verify_decryption_share(public_key, batch, share))
        .collect::<Vec<_>>();
    let mut valid_shares = Vec::with_capacity(shares.len());
    let mut rejected_parties = Vec::new();
    for (share, check) in shares.iter().cloned().zip(checks) {
        if check? {
            valid_shares.push(share);
        } else {
            rejected_parties.push(share.party_index);
        }
    }

    rejected_parties.sort_unstable();
    let mut valid_weight = 0usize;
    for share in &valid_shares {
        valid_weight = valid_weight
            .checked_add(public_key.party_weight(share.party_index)?)
            .ok_or(Error::WeightOverflow)?;
    }
    if valid_weight <= public_key.threshold_weight() {
        return Err(Error::InsufficientWeight {
            accepted: valid_weight,
            required: public_key.required_weight(),
            rejected_parties,
        });
    }

    valid_shares.sort_unstable_by(|left, right| {
        public_key.party_weights()[right.party_index]
            .cmp(&public_key.party_weights()[left.party_index])
            .then_with(|| left.party_index.cmp(&right.party_index))
    });
    let mut accepted = Vec::new();
    let mut accepted_weight = 0usize;
    for share in valid_shares {
        if accepted_weight > public_key.threshold_weight() {
            break;
        }
        accepted_weight = accepted_weight
            .checked_add(public_key.party_weights()[share.party_index])
            .ok_or(Error::WeightOverflow)?;
        accepted.push(share);
    }
    debug_assert!(accepted_weight > public_key.threshold_weight());
    accepted.sort_unstable_by_key(|share| share.party_index);

    Ok(AcceptedDecryptionShares {
        setup_id: batch.setup_id,
        index_space: batch.index_space,
        batch_size: batch.batch_size,
        digest: batch.digest,
        accepted_weight,
        used_indices: batch.used_indices.clone(),
        shares: accepted.into_boxed_slice(),
        rejected_parties: rejected_parties.into_boxed_slice(),
    })
}

/// Build the weighted opening keys for one accepted party/index set.
pub fn prepare_decryption(
    public_key: &MasterPublicKey,
    accepted: &AcceptedDecryptionShares,
) -> Result<DecryptionPrecomputation> {
    if public_key.setup_id() != accepted.setup_id
        || public_key.index_space_size() != accepted.index_space
    {
        return Err(Error::MismatchedSetup);
    }
    if accepted.accepted_weight <= public_key.threshold_weight() {
        return Err(Error::InsufficientWeight {
            accepted: accepted.accepted_weight,
            required: public_key.required_weight(),
            rejected_parties: Vec::new(),
        });
    }

    let party_indices = accepted.party_indices().collect::<Vec<_>>();
    let context_id =
        committee_context_id(accepted.setup_id, &party_indices, &accepted.used_indices);
    let mut selected_virtual_indices = Vec::with_capacity(accepted.accepted_weight);
    let mut coefficient_offsets = Vec::with_capacity(party_indices.len() + 1);
    coefficient_offsets.push(0);
    for party_index in &party_indices {
        selected_virtual_indices.extend(public_key.party_range(*party_index)?);
        coefficient_offsets.push(selected_virtual_indices.len());
    }
    debug_assert_eq!(selected_virtual_indices.len(), accepted.accepted_weight);

    let coefficients = lagrange_at_zero(public_key.domain_points(), &selected_virtual_indices)?;
    let coefficient_bytes = scalars_to_le_bytes(&coefficients);
    let opening_count = accepted
        .used_indices
        .len()
        .checked_mul(party_indices.len())
        .ok_or(Error::ParameterSizeOverflow)?;
    let opening_projective = (0..opening_count)
        .into_par_iter()
        .map(|flat_index| {
            let used_position = flat_index / party_indices.len();
            let party_position = flat_index % party_indices.len();
            let party_index = party_indices[party_position];
            let party_range = public_key.party_range(party_index)?;
            let coefficient_range =
                coefficient_offsets[party_position]..coefficient_offsets[party_position + 1];
            let byte_range = coefficient_range.start * 32..coefficient_range.end * 32;
            let gamma = public_key.gamma_block(accepted.used_indices[used_position])?;
            Ok(g2_multi_exp_affine_bytes(
                &gamma[party_range],
                &coefficient_bytes[byte_range],
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut opening_keys = vec![G2Affine::identity(); opening_projective.len()];
    batch_normalize_g2(&opening_projective, &mut opening_keys);

    Ok(DecryptionPrecomputation {
        setup_id: accepted.setup_id,
        context_id,
        index_space: accepted.index_space,
        accepted_weight: accepted.accepted_weight,
        party_indices: party_indices.into_boxed_slice(),
        used_indices: accepted.used_indices.clone(),
        opening_keys: opening_keys.into_boxed_slice(),
    })
}

/// Compute every indexed cross term with an adaptive direct/middle-product
/// path. Small sparse batches avoid paying for the full index-space FFT.
pub fn precompute_batch(
    kernel: &IndexedMiddleProductKernel,
    batch: &ValidatedBatch,
) -> Result<BatchPrecomputation> {
    if kernel.crs_id != batch.crs_id || kernel.index_space != batch.index_space {
        return Err(Error::MismatchedSetup);
    }

    let valid_count = batch.used_indices.len();
    let direct_terms = valid_count.saturating_mul(valid_count.saturating_sub(1));
    let beta = if direct_terms <= kernel.transform_size / 2 {
        direct_cross_terms(kernel, batch)
    } else {
        fft_cross_terms(kernel, batch)
    };

    Ok(BatchPrecomputation {
        setup_id: batch.setup_id,
        index_space: batch.index_space,
        batch_size: batch.batch_size,
        digest: batch.digest,
        beta: beta.into_boxed_slice(),
    })
}

fn fft_cross_terms(kernel: &IndexedMiddleProductKernel, batch: &ValidatedBatch) -> Vec<Gt> {
    let mut transformed = vec![G1Projective::identity(); kernel.transform_size];
    transformed[..batch.index_space].copy_from_slice(&batch.punctured_by_index);
    kernel.domain.fft(&mut transformed);

    let mut transformed_affine = vec![G1Affine::identity(); kernel.transform_size];
    batch_normalize_g1(&transformed, &mut transformed_affine);
    let miller_results = transformed_affine
        .par_iter()
        .zip(kernel.transformed_kernel.par_iter())
        .map(|(left, right)| right.miller_loop(left))
        .collect::<Vec<_>>();
    let mut convolution = if rayon::current_num_threads() == 1 {
        batch_easy_final_exponentiation(&miller_results)
    } else {
        miller_results
            .into_par_iter()
            .map(easy_final_exponentiation)
            .collect()
    };
    kernel
        .domain
        .ifft_cyclotomic_prefix_unscaled(&mut convolution, batch.index_space);

    // Applying the hard exponent only at occupied valid indices is important
    // for sparse batches: all other convolution coefficients remain internal
    // cyclotomic intermediates.
    batch
        .valid_index_by_slot
        .par_iter()
        .map(|index| {
            index
                .map(|index| hard_final_exponentiation(convolution[index]))
                .unwrap_or_else(Gt::identity)
        })
        .collect()
}

fn direct_cross_terms(kernel: &IndexedMiddleProductKernel, batch: &ValidatedBatch) -> Vec<Gt> {
    batch
        .valid_index_by_slot
        .par_iter()
        .map(|output_index| {
            let Some(output_index) = *output_index else {
                return Gt::identity();
            };
            let mut left = Vec::with_capacity(batch.used_indices.len().saturating_sub(1));
            let mut right = Vec::with_capacity(batch.used_indices.len().saturating_sub(1));
            for input_index in batch.used_indices.iter().copied() {
                if input_index == output_index {
                    continue;
                }
                left.push(batch.punctured_affine_by_index[input_index]);
                let distance = output_index.abs_diff(input_index);
                right.push(if output_index > input_index {
                    kernel.positive_powers[distance]
                } else {
                    kernel.negative_powers[distance]
                });
            }
            full_final_exponentiation(unprepared_multi_miller_loop(&left, &right))
        })
        .collect()
}

/// Evaluate the accepted-party multi-pairing at every valid ciphertext index
/// and remove the resulting one-time pad.
pub fn open_batch(
    precomputation: &DecryptionPrecomputation,
    accepted: &AcceptedDecryptionShares,
    batch: &ValidatedBatch,
    ciphertexts: &[Ciphertext],
    cross_terms: &BatchPrecomputation,
) -> Result<Vec<Option<Box<[u8]>>>> {
    ensure_opening_bindings(precomputation, accepted, batch, ciphertexts, cross_terms)?;

    let share_affine = accepted
        .shares
        .iter()
        .map(|share| share.sigma)
        .collect::<Vec<_>>();
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
            let used_position = batch.used_position_by_slot[*slot]
                .expect("valid slots have an opening-key position");
            let start = used_position * party_count;
            unprepared_multi_miller_loop(
                &share_affine,
                &precomputation.opening_keys[start..start + party_count],
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
            alpha[slot].map(|alpha| {
                let pad = alpha - cross_terms.beta[slot];
                xor_with_mask(
                    &pad,
                    batch.setup_id,
                    ciphertexts[slot].index,
                    &ciphertexts[slot].masked_message,
                )
            })
        })
        .collect())
}

/// Convenience wrapper around every public decryption phase.
pub fn decrypt(
    public_parameters: &PublicParameters,
    public_key: &MasterPublicKey,
    ciphertexts: &[Ciphertext],
    shares: &[DecryptionShare],
) -> Result<Vec<Option<Box<[u8]>>>> {
    let batch = validate_batch(public_key, ciphertexts)?;
    let accepted = accept_decryption_shares(public_key, &batch, shares)?;
    if public_parameters.crs_id() != public_key.crs_id() {
        return Err(Error::MismatchedSetup);
    }
    let kernel = IndexedMiddleProductKernel::new(public_parameters)?;
    let precomputation = prepare_decryption(public_key, &accepted)?;
    let cross_terms = precompute_batch(&kernel, &batch)?;
    open_batch(
        &precomputation,
        &accepted,
        &batch,
        ciphertexts,
        &cross_terms,
    )
}

fn ensure_opening_bindings(
    precomputation: &DecryptionPrecomputation,
    accepted: &AcceptedDecryptionShares,
    batch: &ValidatedBatch,
    ciphertexts: &[Ciphertext],
    cross_terms: &BatchPrecomputation,
) -> Result<()> {
    if precomputation.setup_id != batch.setup_id
        || accepted.setup_id != batch.setup_id
        || cross_terms.setup_id != batch.setup_id
        || precomputation.index_space != batch.index_space
        || accepted.index_space != batch.index_space
        || cross_terms.index_space != batch.index_space
    {
        return Err(Error::MismatchedSetup);
    }
    if accepted.batch_size != batch.batch_size
        || cross_terms.batch_size != batch.batch_size
        || ciphertexts.len() != batch.batch_size
    {
        return Err(Error::MismatchedBatchSize {
            expected: batch.batch_size,
            actual: ciphertexts.len(),
        });
    }
    if accepted.digest != batch.digest
        || cross_terms.digest != batch.digest
        || batch_digest(ciphertexts) != batch.digest
    {
        return Err(Error::MismatchedBatchDigest);
    }

    let party_indices = accepted.party_indices().collect::<Vec<_>>();
    let expected_context =
        committee_context_id(accepted.setup_id, &party_indices, &accepted.used_indices);
    if precomputation.context_id != expected_context
        || precomputation.party_indices.as_ref() != party_indices.as_slice()
        || precomputation.used_indices.as_ref() != accepted.used_indices.as_ref()
        || accepted.used_indices.as_ref() != batch.used_indices.as_ref()
    {
        return Err(Error::MismatchedCommittee);
    }
    Ok(())
}

fn validate_batch_size(index_space: usize, batch_size: usize) -> Result<()> {
    if batch_size == 0 {
        return Err(Error::BatchIsEmpty);
    }
    if batch_size > index_space {
        return Err(Error::BatchTooLarge {
            batch_size,
            max_batch_size: index_space,
        });
    }
    Ok(())
}

fn batch_digest(ciphertexts: &[Ciphertext]) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(BATCH_DIGEST_DOMAIN);
    bytes.extend_from_slice(&(ciphertexts.len() as u64).to_le_bytes());
    for ciphertext in ciphertexts {
        ciphertext.append_canonical(&mut bytes);
    }
    Sha256::digest(bytes).into()
}

fn server_proof_context(batch_digest: [u8; 32], party_index: usize) -> Vec<u8> {
    let mut context = Vec::with_capacity(SERVER_PROOF_CONTEXT_DOMAIN.len() + 32 + 8);
    context.extend_from_slice(SERVER_PROOF_CONTEXT_DOMAIN);
    context.extend_from_slice(&batch_digest);
    context.extend_from_slice(&(party_index as u64).to_le_bytes());
    context
}

fn committee_context_id(
    setup_id: [u8; 32],
    party_indices: &[usize],
    used_indices: &[usize],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(COMMITTEE_CONTEXT_DOMAIN);
    hasher.update(setup_id);
    hasher.update((party_indices.len() as u64).to_le_bytes());
    for party_index in party_indices {
        hasher.update((*party_index as u64).to_le_bytes());
    }
    hasher.update((used_indices.len() as u64).to_le_bytes());
    for index in used_indices {
        hasher.update((*index as u64).to_le_bytes());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use blstrs::{Bls12, Scalar};
    use ff::Field;
    use pairing::Engine;

    use super::*;
    use crate::{encrypt, keygen, setup};

    const WEIGHTS: &[usize] = &[1, 3, 2, 4, 1];
    const THRESHOLD: usize = 5;

    struct Fixture {
        parameters: PublicParameters,
        material: crate::KeyMaterial,
        messages: Vec<Box<[u8]>>,
        ciphertexts: Vec<Ciphertext>,
        batch: ValidatedBatch,
        accepted: AcceptedDecryptionShares,
        fixed: IndexedMiddleProductKernel,
        committee: DecryptionPrecomputation,
    }

    fn fixture(index_space: usize, indices: &[usize], parties: &[usize]) -> Fixture {
        let parameters = setup(index_space).unwrap();
        let material = keygen(&parameters, WEIGHTS, THRESHOLD).unwrap();
        let messages = indices
            .iter()
            .enumerate()
            .map(|(slot, index)| {
                format!("indexed message at slot {slot}, domain index {index}")
                    .into_bytes()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>();
        let ciphertexts = messages
            .iter()
            .zip(indices)
            .map(|(message, index)| {
                encrypt(&parameters, &material.public_key, message, *index).unwrap()
            })
            .collect::<Vec<_>>();
        let batch = validate_batch(&material.public_key, &ciphertexts).unwrap();
        let shares = parties
            .iter()
            .map(|party| partial_decrypt(&material.party_keys[*party], &batch).unwrap())
            .collect::<Vec<_>>();
        let accepted = accept_decryption_shares(&material.public_key, &batch, &shares).unwrap();
        let fixed = IndexedMiddleProductKernel::new(&parameters).unwrap();
        let committee = prepare_decryption(&material.public_key, &accepted).unwrap();
        Fixture {
            parameters,
            material,
            messages,
            ciphertexts,
            batch,
            accepted,
            fixed,
            committee,
        }
    }

    fn open_fixture(fixture: &Fixture) -> Vec<Option<Box<[u8]>>> {
        let cross = precompute_batch(&fixture.fixed, &fixture.batch).unwrap();
        open_batch(
            &fixture.committee,
            &fixture.accepted,
            &fixture.batch,
            &fixture.ciphertexts,
            &cross,
        )
        .unwrap()
    }

    fn direct_beta(
        parameters: &PublicParameters,
        batch: &ValidatedBatch,
        ciphertexts: &[Ciphertext],
    ) -> Vec<Gt> {
        batch
            .valid_index_by_slot
            .iter()
            .map(|output_index| {
                let Some(output_index) = *output_index else {
                    return Gt::identity();
                };
                ciphertexts
                    .iter()
                    .zip(&batch.valid_index_by_slot)
                    .filter_map(|(ciphertext, input_index)| {
                        let input_index = (*input_index)?;
                        if input_index == output_index {
                            return None;
                        }
                        let offset = isize::try_from(output_index).unwrap()
                            - isize::try_from(input_index).unwrap();
                        Some(Bls12::pairing(
                            &ciphertext.punctured,
                            &parameters.centered_power(offset).unwrap(),
                        ))
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn weighted_threshold_and_sparse_distinct_indices_open() {
        let fixture = fixture(8, &[7, 1, 4, 2], &[3, 1]);
        assert_eq!(fixture.batch.used_indices(), &[1, 2, 4, 7]);
        assert_eq!(fixture.accepted.accepted_weight(), 7);
        assert_eq!(
            fixture.accepted.party_indices().collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(
            open_fixture(&fixture),
            fixture
                .messages
                .iter()
                .cloned()
                .map(Some)
                .collect::<Vec<_>>()
        );

        let all_shares = fixture
            .material
            .party_keys
            .iter()
            .map(|party_key| partial_decrypt(party_key, &fixture.batch).unwrap())
            .collect::<Vec<_>>();
        let selected =
            accept_decryption_shares(&fixture.material.public_key, &fixture.batch, &all_shares)
                .unwrap();
        assert_eq!(selected.party_indices().collect::<Vec<_>>(), vec![1, 3]);
        assert_eq!(selected.accepted_weight(), 7);

        let light_shares = [0usize, 2, 4]
            .iter()
            .map(|party| {
                partial_decrypt(&fixture.material.party_keys[*party], &fixture.batch).unwrap()
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            accept_decryption_shares(&fixture.material.public_key, &fixture.batch, &light_shares,),
            Err(Error::InsufficientWeight { .. })
        ));
    }

    #[test]
    fn duplicate_valid_indices_are_rejected_and_out_of_range_indices_are_filtered() {
        let parameters = setup(4).unwrap();
        let material = keygen(&parameters, WEIGHTS, THRESHOLD).unwrap();
        let duplicate = vec![
            encrypt(&parameters, &material.public_key, b"first", 2).unwrap(),
            encrypt(&parameters, &material.public_key, b"second", 2).unwrap(),
        ];
        assert!(matches!(
            validate_batch(&material.public_key, &duplicate),
            Err(Error::DuplicateCiphertextIndex(2))
        ));

        let mut out_of_range = vec![
            encrypt(&parameters, &material.public_key, b"honest", 1).unwrap(),
            encrypt(&parameters, &material.public_key, b"malformed", 3).unwrap(),
        ];
        out_of_range[1].index = 4;
        let batch = validate_batch(&material.public_key, &out_of_range).unwrap();
        assert_eq!(batch.valid_mask(), &[true, false]);
        assert_eq!(batch.used_indices(), &[1]);
    }

    #[test]
    fn invalid_client_proofs_preserve_positions_and_do_not_claim_indices() {
        let mut fixture = fixture(8, &[0, 3, 6, 7], &[1, 3]);
        fixture.ciphertexts[1].proof.response += Scalar::ONE;
        // The invalid ciphertext may collide with a valid one without making
        // the valid index set ambiguous.
        fixture.ciphertexts[1].index = 0;

        let batch = validate_batch(&fixture.material.public_key, &fixture.ciphertexts).unwrap();
        assert_eq!(batch.valid_mask(), &[true, false, true, true]);
        assert_eq!(batch.used_indices(), &[0, 6, 7]);
        let shares = [1usize, 3]
            .iter()
            .map(|party| partial_decrypt(&fixture.material.party_keys[*party], &batch).unwrap())
            .collect::<Vec<_>>();
        let accepted =
            accept_decryption_shares(&fixture.material.public_key, &batch, &shares).unwrap();
        let committee = prepare_decryption(&fixture.material.public_key, &accepted).unwrap();
        let cross = precompute_batch(&fixture.fixed, &batch).unwrap();
        let opened =
            open_batch(&committee, &accepted, &batch, &fixture.ciphertexts, &cross).unwrap();
        assert_eq!(opened[0].as_deref(), Some(fixture.messages[0].as_ref()));
        assert!(opened[1].is_none());
        assert_eq!(opened[2].as_deref(), Some(fixture.messages[2].as_ref()));
        assert_eq!(opened[3].as_deref(), Some(fixture.messages[3].as_ref()));
    }

    #[test]
    fn malformed_and_replayed_server_shares_are_blamed() {
        let fixture = fixture(8, &[0, 2, 5], &[1, 2, 3]);
        let mut shares = [1usize, 2, 3]
            .iter()
            .map(|party| {
                partial_decrypt(&fixture.material.party_keys[*party], &fixture.batch).unwrap()
            })
            .collect::<Vec<_>>();
        shares[1].proof.response += Scalar::ONE;
        let accepted =
            accept_decryption_shares(&fixture.material.public_key, &fixture.batch, &shares)
                .unwrap();
        assert_eq!(accepted.rejected_parties(), &[2]);
        assert_eq!(accepted.accepted_weight(), 7);

        let other_ciphertexts = vec![encrypt(
            &fixture.parameters,
            &fixture.material.public_key,
            b"different batch",
            1,
        )
        .unwrap()];
        let other_batch = validate_batch(&fixture.material.public_key, &other_ciphertexts).unwrap();
        assert!(matches!(
            accept_decryption_shares(
                &fixture.material.public_key,
                &other_batch,
                &[shares[0].clone(), shares[2].clone()],
            ),
            Err(Error::InsufficientWeight { .. })
        ));
    }

    #[test]
    fn adaptive_cross_term_paths_match_direct_formula() {
        for index_space in [2usize, 4, 8] {
            let indices = (0..index_space).collect::<Vec<_>>();
            let fixture = fixture(index_space, &indices, &[1, 3]);
            let fast = precompute_batch(&fixture.fixed, &fixture.batch).unwrap();
            let slow = direct_beta(&fixture.parameters, &fixture.batch, &fixture.ciphertexts);
            assert_eq!(&*fast.beta, slow.as_slice(), "index space {index_space}");
        }

        // This shape takes the adaptive direct path rather than the FFT path.
        let sparse = fixture(16, &[0, 7, 15], &[1, 3]);
        let fast = precompute_batch(&sparse.fixed, &sparse.batch).unwrap();
        let slow = direct_beta(&sparse.parameters, &sparse.batch, &sparse.ciphertexts);
        assert_eq!(&*fast.beta, slow.as_slice());
    }

    #[test]
    fn phased_objects_reject_cross_setup_batch_and_committee_reuse() {
        let fixture = fixture(8, &[0, 3, 6], &[1, 3]);
        let rotated_material = keygen(&fixture.parameters, WEIGHTS, THRESHOLD).unwrap();
        let rotated_ciphertexts = vec![encrypt(
            &fixture.parameters,
            &rotated_material.public_key,
            b"same CRS, rotated committee key",
            1,
        )
        .unwrap()];
        let rotated_batch =
            validate_batch(&rotated_material.public_key, &rotated_ciphertexts).unwrap();
        assert!(precompute_batch(&fixture.fixed, &rotated_batch).is_ok());

        let other_parameters = setup(8).unwrap();
        let other_material = keygen(&other_parameters, WEIGHTS, THRESHOLD).unwrap();
        let other_ciphertexts =
            vec![encrypt(&other_parameters, &other_material.public_key, b"other", 0).unwrap()];
        let other_batch = validate_batch(&other_material.public_key, &other_ciphertexts).unwrap();
        assert!(matches!(
            precompute_batch(&fixture.fixed, &other_batch),
            Err(Error::MismatchedSetup)
        ));

        let cross = precompute_batch(&fixture.fixed, &fixture.batch).unwrap();
        let reordered = vec![
            fixture.ciphertexts[2].clone(),
            fixture.ciphertexts[1].clone(),
            fixture.ciphertexts[0].clone(),
        ];
        let reordered_batch = validate_batch(&fixture.material.public_key, &reordered).unwrap();
        let reordered_shares = [1usize, 3]
            .iter()
            .map(|party| {
                partial_decrypt(&fixture.material.party_keys[*party], &reordered_batch).unwrap()
            })
            .collect::<Vec<_>>();
        let reordered_accepted = accept_decryption_shares(
            &fixture.material.public_key,
            &reordered_batch,
            &reordered_shares,
        )
        .unwrap();
        // The committee precomputation itself is reusable because the party
        // and used-index sets are unchanged, but the stale accepted shares and
        // cross terms remain batch-digest bound.
        assert!(matches!(
            open_batch(
                &fixture.committee,
                &reordered_accepted,
                &reordered_batch,
                &reordered,
                &cross,
            ),
            Err(Error::MismatchedBatchDigest)
        ));

        let alternate_shares = [0usize, 2, 3]
            .iter()
            .map(|party| {
                partial_decrypt(&fixture.material.party_keys[*party], &fixture.batch).unwrap()
            })
            .collect::<Vec<_>>();
        let alternate = accept_decryption_shares(
            &fixture.material.public_key,
            &fixture.batch,
            &alternate_shares,
        )
        .unwrap();
        assert!(matches!(
            open_batch(
                &fixture.committee,
                &alternate,
                &fixture.batch,
                &fixture.ciphertexts,
                &cross,
            ),
            Err(Error::MismatchedCommittee)
        ));
    }
}
