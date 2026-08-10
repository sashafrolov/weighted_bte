//! Trusted setup and weighted Shamir key generation for Construction 2.

use std::{collections::HashSet, mem::size_of, ops::Range};

use blstrs::{G2Affine, G2Projective, Gt, Scalar};
use ff::{BatchInvert, Field};
use group::Group;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    blst_utils::batch_normalize_g2,
    encoding::append_gt,
    error::{Error, Result},
    fft::Radix2Domain,
};

const SETUP_ID_DOMAIN: &[u8] = b"WEIGHTED-PFE-SETUP-BLS12381-v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncryptionKey {
    batch_size: usize,
    element: Gt,
    setup_id: [u8; 32],
}

/// Public weighted decryption material.
///
/// The three large arrays are slot-major.  A complete virtual-share block is
/// contiguous, which lets committee preparation pass it directly to BLST's
/// affine Pippenger implementation without copying points.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicDecryptionKey {
    setup_id: [u8; 32],
    batch_size: usize,
    threshold_weight: usize,
    total_weight: usize,
    domain_size: usize,
    party_weights: Box<[usize]>,
    party_offsets: Box<[usize]>,
    evaluation_domain: Box<[Scalar]>,
    alpha_points: Box<[Scalar]>,
    alpha_shift_small: u64,
    /// `vk[j,i] = [g_{j,i}(x)]_2`, laid out as `i * N + j`.
    verification_keys: Box<[G2Affine]>,
    /// `D[w,i] = [Z(w) g_{owner(w),i}(x)]_2`, laid out as `i * W + w`.
    d1: Box<[G2Affine]>,
    /// `U[w,i] = [Z(w) u_{owner(w),i}(x)]_2`, laid out as `i * W + w`.
    d2: Box<[G2Affine]>,
    /// `V = [z (p_{-1}(x) + p_0(x))]_2`.
    global_key: G2Affine,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartySecretKey {
    pub party_index: usize,
    setup_id: [u8; 32],
    batch_size: usize,
    /// `(g_{j,i}(x))_i`; its length is B regardless of party weight.
    fractions: Box<[Scalar]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyMaterial {
    pub encryption_key: EncryptionKey,
    pub decryption_key: PublicDecryptionKey,
    pub party_keys: Vec<PartySecretKey>,
}

#[derive(Clone, Debug)]
pub struct KeygenConfig {
    pub batch_size: usize,
    pub party_weights: Vec<usize>,
    pub threshold_weight: usize,
}

impl KeygenConfig {
    pub fn new(batch_size: usize, party_weights: &[usize], threshold_weight: usize) -> Self {
        Self {
            batch_size,
            party_weights: party_weights.to_vec(),
            threshold_weight,
        }
    }
}

pub fn keygen(
    batch_size: usize,
    party_weights: &[usize],
    threshold_weight: usize,
) -> Result<KeyMaterial> {
    let mut rng = rand_core::OsRng;
    keygen_with_rng(
        KeygenConfig::new(batch_size, party_weights, threshold_weight),
        &mut rng,
    )
}

pub fn keygen_with_rng<R: rand_core::RngCore + rand_core::CryptoRng>(
    config: KeygenConfig,
    rng: &mut R,
) -> Result<KeyMaterial> {
    let dimensions = validate_config(&config)?;
    let party_count = config.party_weights.len();
    let batch_size = config.batch_size;
    let sharing_domain =
        Radix2Domain::new(dimensions.domain_size).map_err(|_| Error::InvalidEvaluationDomain {
            total_weight: dimensions.total_weight,
        })?;

    // Z is shared over a nonzero root-of-unity domain. Sampling its constant
    // term from F_p^* avoids the degenerate public zero encryption key.
    let master_secret = sample_nonzero(rng);
    let mut polynomial = vec![Scalar::ZERO; dimensions.domain_size];
    polynomial[0] = master_secret;
    polynomial[1..=config.threshold_weight]
        .iter_mut()
        .for_each(|coefficient| *coefficient = Scalar::random(&mut *rng));
    sharing_domain.fft(&mut polynomial);
    polynomial.truncate(dimensions.total_weight);
    let share_evaluations = polynomial;

    let evaluation_domain = root_domain_prefix(dimensions.domain_size, dimensions.total_weight)?;
    let party_offsets = prefix_offsets(&config.party_weights)?;
    let owners = owners_from_offsets(&party_offsets);

    // The paper leaves alpha_i abstract. An affine shift of a complete
    // root-of-unity domain preserves every alpha_k-alpha_i denominator while
    // making all Cauchy sums cyclic FFT convolutions.
    let (alpha_points, alpha_shift_small) = shifted_alpha_points(batch_size)?;

    let x = loop {
        let candidate = Scalar::random(&mut *rng);
        if !bool::from(candidate.is_zero()) && candidate != Scalar::ONE {
            break candidate;
        }
    };
    let mut global_denominators = [x - Scalar::ONE, x];
    global_denominators.iter_mut().batch_invert();
    let p_minus_one = global_denominators[0];
    let p_zero = global_denominators[1];

    // Rejection sampling concretely instantiates the paper's requirement that
    // every rho_j+alpha_i be globally distinct and every hidden denominator
    // be nonzero.  Collisions occur only with negligible probability, but the
    // explicit check keeps the implementation valid for deterministic RNGs.
    let mut occupied_labels = HashSet::with_capacity(dimensions.verification_count);
    let mut party_shifts = Vec::with_capacity(party_count);
    for _ in 0..party_count {
        loop {
            let rho = Scalar::random(&mut *rng);
            let labels = alpha_points
                .iter()
                .map(|alpha| rho + alpha)
                .collect::<Vec<_>>();
            if labels.iter().any(|label| bool::from((x + label).is_zero()))
                || labels
                    .iter()
                    .any(|label| occupied_labels.contains(&label.to_bytes_le()))
            {
                continue;
            }
            for label in labels {
                occupied_labels.insert(label.to_bytes_le());
            }
            party_shifts.push(rho);
            break;
        }
    }

    let mut fractions = Vec::with_capacity(dimensions.verification_count);
    for rho in &party_shifts {
        fractions.extend(alpha_points.iter().map(|alpha| x + rho + alpha));
    }
    debug_assert!(fractions.iter().all(|value| !bool::from(value.is_zero())));
    fractions.iter_mut().batch_invert();

    let mut alpha_inverses = alpha_points.clone();
    let mut alpha_plus_one_inverses = alpha_points
        .iter()
        .map(|alpha| alpha + Scalar::ONE)
        .collect::<Vec<_>>();
    debug_assert!(alpha_inverses
        .iter()
        .chain(&alpha_plus_one_inverses)
        .all(|value| !bool::from(value.is_zero())));
    alpha_inverses.iter_mut().batch_invert();
    alpha_plus_one_inverses.iter_mut().batch_invert();
    let mut u_values = vec![Scalar::ZERO; dimensions.verification_count];
    u_values
        .par_chunks_mut(batch_size)
        .zip(fractions.par_chunks(batch_size))
        .for_each(|(party_u, party_g)| {
            for slot in 0..batch_size {
                party_u[slot] = party_g[slot].square()
                    + p_minus_one * alpha_plus_one_inverses[slot]
                    + p_zero * alpha_inverses[slot];
            }
        });

    // Verification keys are small compared with D/U, but retaining the same
    // slot-major layout makes randomized all-party verification contiguous.
    let mut verification_keys = Vec::with_capacity(dimensions.verification_count);
    for slot in 0..batch_size {
        let scalars = (0..party_count)
            .map(|party| fractions[party * batch_size + slot])
            .collect::<Vec<_>>();
        append_g2_block(&mut verification_keys, &scalars);
    }

    // Generate one D and one U block at a time. At Solana parameters a single
    // all-projective allocation would consume hundreds of MiB; bounded block
    // generation retains only 2W projective points before normalization.
    let mut d1 = Vec::with_capacity(dimensions.share_material_count);
    let mut d2 = Vec::with_capacity(dimensions.share_material_count);
    for slot in 0..batch_size {
        let (d1_scalars, d2_scalars) = rayon::join(
            || {
                share_evaluations
                    .par_iter()
                    .enumerate()
                    .map(|(virtual_index, share)| {
                        *share * fractions[owners[virtual_index] * batch_size + slot]
                    })
                    .collect::<Vec<_>>()
            },
            || {
                share_evaluations
                    .par_iter()
                    .enumerate()
                    .map(|(virtual_index, share)| {
                        *share * u_values[owners[virtual_index] * batch_size + slot]
                    })
                    .collect::<Vec<_>>()
            },
        );
        let (d1_block, d2_block) = rayon::join(
            || scalars_to_g2_affine(&d1_scalars),
            || scalars_to_g2_affine(&d2_scalars),
        );
        d1.extend(d1_block);
        d2.extend(d2_block);
    }

    let global_key = (G2Projective::generator() * (master_secret * (p_minus_one + p_zero))).into();
    let encryption_element = Gt::generator() * (master_secret * p_zero);
    let setup_id = setup_identifier(
        batch_size,
        config.threshold_weight,
        dimensions.domain_size,
        alpha_shift_small,
        &config.party_weights,
        &evaluation_domain,
        &alpha_points,
        &encryption_element,
        &verification_keys,
        &d1,
        &d2,
        global_key,
    );

    let party_keys = fractions
        .chunks(batch_size)
        .enumerate()
        .map(|(party_index, party_fractions)| PartySecretKey {
            party_index,
            setup_id,
            batch_size,
            fractions: party_fractions.to_vec().into_boxed_slice(),
        })
        .collect::<Vec<_>>();
    let encryption_key = EncryptionKey {
        batch_size,
        element: encryption_element,
        setup_id,
    };
    let decryption_key = PublicDecryptionKey {
        setup_id,
        batch_size,
        threshold_weight: config.threshold_weight,
        total_weight: dimensions.total_weight,
        domain_size: dimensions.domain_size,
        party_weights: config.party_weights.into_boxed_slice(),
        party_offsets: party_offsets.into_boxed_slice(),
        evaluation_domain: evaluation_domain.into_boxed_slice(),
        alpha_points: alpha_points.into_boxed_slice(),
        alpha_shift_small,
        verification_keys: verification_keys.into_boxed_slice(),
        d1: d1.into_boxed_slice(),
        d2: d2.into_boxed_slice(),
        global_key,
    };

    Ok(KeyMaterial {
        encryption_key,
        decryption_key,
        party_keys,
    })
}

impl EncryptionKey {
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn element(&self) -> Gt {
        self.element
    }

    pub fn setup_id(&self) -> [u8; 32] {
        self.setup_id
    }

    /// Identity-safe canonical encoding used by transcripts and persistence.
    pub fn serialized_size_bytes(&self) -> usize {
        let mut encoded = Vec::new();
        append_gt(&self.element, &mut encoded);
        encoded.len()
    }
}

impl PublicDecryptionKey {
    pub fn setup_id(&self) -> [u8; 32] {
        self.setup_id
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn party_count(&self) -> usize {
        self.party_weights.len()
    }

    pub fn total_weight(&self) -> usize {
        self.total_weight
    }

    pub fn threshold_weight(&self) -> usize {
        self.threshold_weight
    }

    pub fn required_weight(&self) -> usize {
        self.threshold_weight + 1
    }

    pub fn party_weights(&self) -> &[usize] {
        &self.party_weights
    }

    pub fn party_weight(&self, party_index: usize) -> Result<usize> {
        self.party_weights
            .get(party_index)
            .copied()
            .ok_or(Error::InvalidPartyIndex(party_index))
    }

    pub fn domain_points(&self) -> &[Scalar] {
        &self.evaluation_domain
    }

    pub fn domain_size(&self) -> usize {
        self.domain_size
    }

    pub fn alpha_points(&self) -> &[Scalar] {
        &self.alpha_points
    }

    pub fn alpha_shift_small(&self) -> u64 {
        self.alpha_shift_small
    }

    pub fn verification_key_count(&self) -> usize {
        self.verification_keys.len()
    }

    pub fn d1_point_count(&self) -> usize {
        self.d1.len()
    }

    pub fn d2_point_count(&self) -> usize {
        self.d2.len()
    }

    pub fn core_g2_point_count(&self) -> usize {
        self.d1.len() + self.d2.len() + 1
    }

    pub fn g2_point_count(&self) -> usize {
        self.verification_keys.len() + self.core_g2_point_count()
    }

    pub fn serialized_size_bytes(&self) -> usize {
        self.g2_point_count() * G2Affine::default().to_compressed().len()
    }

    pub(crate) fn party_range(&self, party_index: usize) -> Result<Range<usize>> {
        if party_index >= self.party_count() {
            return Err(Error::InvalidPartyIndex(party_index));
        }
        Ok(self.party_offsets[party_index]..self.party_offsets[party_index + 1])
    }

    pub(crate) fn verification_key(&self, party_index: usize, slot: usize) -> Result<G2Affine> {
        if party_index >= self.party_count() {
            return Err(Error::InvalidPartyIndex(party_index));
        }
        if slot >= self.batch_size {
            return Err(Error::InvalidBatchSize(slot + 1));
        }
        Ok(self.verification_keys[slot * self.party_count() + party_index])
    }

    pub(crate) fn verification_slot(&self, slot: usize) -> Result<&[G2Affine]> {
        if slot >= self.batch_size {
            return Err(Error::InvalidBatchSize(slot + 1));
        }
        let start = slot * self.party_count();
        Ok(&self.verification_keys[start..start + self.party_count()])
    }

    pub(crate) fn d1_slot(&self, slot: usize) -> Result<&[G2Affine]> {
        self.share_slot(&self.d1, slot)
    }

    pub(crate) fn d2_slot(&self, slot: usize) -> Result<&[G2Affine]> {
        self.share_slot(&self.d2, slot)
    }

    pub(crate) fn global_key(&self) -> G2Affine {
        self.global_key
    }

    fn share_slot<'a>(&self, material: &'a [G2Affine], slot: usize) -> Result<&'a [G2Affine]> {
        if slot >= self.batch_size {
            return Err(Error::InvalidBatchSize(slot + 1));
        }
        let start = slot * self.total_weight;
        Ok(&material[start..start + self.total_weight])
    }
}

impl PartySecretKey {
    pub fn setup_id(&self) -> [u8; 32] {
        self.setup_id
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn scalar_count(&self) -> usize {
        self.fractions.len()
    }

    pub(crate) fn fractions(&self) -> &[Scalar] {
        &self.fractions
    }
}

#[derive(Clone, Copy, Debug)]
struct ValidatedDimensions {
    total_weight: usize,
    domain_size: usize,
    verification_count: usize,
    share_material_count: usize,
}

fn validate_config(config: &KeygenConfig) -> Result<ValidatedDimensions> {
    Radix2Domain::new(config.batch_size)?;
    if config.party_weights.is_empty() {
        return Err(Error::EmptyCommittee);
    }
    for (party_index, weight) in config.party_weights.iter().copied().enumerate() {
        if weight == 0 {
            return Err(Error::InvalidPartyWeight {
                party_index,
                weight,
            });
        }
    }
    let total_weight = config
        .party_weights
        .iter()
        .try_fold(0usize, |sum, weight| sum.checked_add(*weight))
        .ok_or(Error::TotalWeightOverflow)?;
    if config.threshold_weight >= total_weight {
        return Err(Error::InvalidCommittee {
            party_count: config.party_weights.len(),
            total_weight,
            threshold_weight: config.threshold_weight,
        });
    }
    let domain_size = total_weight
        .checked_next_power_of_two()
        .ok_or(Error::InvalidEvaluationDomain { total_weight })?;
    Radix2Domain::new(domain_size).map_err(|_| Error::InvalidEvaluationDomain { total_weight })?;
    let verification_count = checked_count(config.batch_size, config.party_weights.len())?;
    let share_material_count = checked_count(config.batch_size, total_weight)?;
    checked_allocation::<G2Affine>(verification_count)?;
    checked_allocation::<G2Affine>(share_material_count)?;
    checked_allocation::<Scalar>(verification_count)?;
    checked_allocation::<Scalar>(domain_size)?;
    Ok(ValidatedDimensions {
        total_weight,
        domain_size,
        verification_count,
        share_material_count,
    })
}

fn checked_count(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or(Error::ParameterSizeOverflow)
}

fn checked_allocation<T>(count: usize) -> Result<()> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or(Error::ParameterSizeOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(Error::ParameterSizeOverflow);
    }
    Ok(())
}

fn root_domain_prefix(domain_size: usize, length: usize) -> Result<Vec<Scalar>> {
    let domain = Radix2Domain::new(domain_size)?;
    let mut points = vec![Scalar::ZERO; domain_size];
    if domain_size == 1 {
        points[0] = Scalar::ONE;
    } else {
        points[1] = Scalar::ONE;
        domain.fft(&mut points);
    }
    points.truncate(length);
    Ok(points)
}

fn shifted_alpha_points(batch_size: usize) -> Result<(Vec<Scalar>, u64)> {
    let roots = root_domain_prefix(batch_size, batch_size)?;
    for candidate in 2u64.. {
        let shift = Scalar::from(candidate);
        let points = roots.iter().map(|root| *root + shift).collect::<Vec<_>>();
        if points
            .iter()
            .all(|alpha| !bool::from(alpha.is_zero()) && *alpha + Scalar::ONE != Scalar::ZERO)
        {
            return Ok((points, candidate));
        }
    }
    unreachable!("the scalar field contains a valid affine subgroup shift")
}

fn prefix_offsets(weights: &[usize]) -> Result<Vec<usize>> {
    let mut offsets: Vec<usize> = Vec::with_capacity(weights.len() + 1);
    offsets.push(0);
    for weight in weights {
        let next = offsets
            .last()
            .copied()
            .and_then(|offset| offset.checked_add(*weight))
            .ok_or(Error::TotalWeightOverflow)?;
        offsets.push(next);
    }
    Ok(offsets)
}

fn owners_from_offsets(offsets: &[usize]) -> Vec<usize> {
    let total_weight = offsets.last().copied().unwrap_or_default();
    let mut owners = vec![0usize; total_weight];
    for party in 0..offsets.len().saturating_sub(1) {
        owners[offsets[party]..offsets[party + 1]].fill(party);
    }
    owners
}

fn append_g2_block(output: &mut Vec<G2Affine>, scalars: &[Scalar]) {
    output.extend(scalars_to_g2_affine(scalars));
}

fn scalars_to_g2_affine(scalars: &[Scalar]) -> Vec<G2Affine> {
    let projective = scalars
        .par_iter()
        .map(|scalar| G2Projective::generator() * scalar)
        .collect::<Vec<_>>();
    let mut affine = vec![G2Affine::default(); projective.len()];
    batch_normalize_g2(&projective, &mut affine);
    affine
}

#[allow(clippy::too_many_arguments)]
fn setup_identifier(
    batch_size: usize,
    threshold_weight: usize,
    domain_size: usize,
    alpha_shift_small: u64,
    party_weights: &[usize],
    evaluation_domain: &[Scalar],
    alpha_points: &[Scalar],
    encryption_element: &Gt,
    verification_keys: &[G2Affine],
    d1: &[G2Affine],
    d2: &[G2Affine],
    global_key: G2Affine,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SETUP_ID_DOMAIN);
    hasher.update((batch_size as u64).to_le_bytes());
    hasher.update((threshold_weight as u64).to_le_bytes());
    hasher.update((domain_size as u64).to_le_bytes());
    hasher.update(alpha_shift_small.to_le_bytes());
    hasher.update((party_weights.len() as u64).to_le_bytes());
    for weight in party_weights {
        hasher.update((*weight as u64).to_le_bytes());
    }
    for point in evaluation_domain {
        hasher.update(point.to_bytes_le());
    }
    for alpha in alpha_points {
        hasher.update(alpha.to_bytes_le());
    }
    let mut target = Vec::with_capacity(289);
    append_gt(encryption_element, &mut target);
    hasher.update((target.len() as u64).to_le_bytes());
    hasher.update(target);
    for point in verification_keys {
        hasher.update(point.to_compressed());
    }
    for point in d1 {
        hasher.update(point.to_compressed());
    }
    for point in d2 {
        hasher.update(point.to_compressed());
    }
    hasher.update(global_key.to_compressed());
    hasher.finalize().into()
}

fn sample_nonzero<R: rand_core::RngCore + rand_core::CryptoRng>(rng: &mut R) -> Scalar {
    loop {
        let candidate = Scalar::random(&mut *rng);
        if !bool::from(candidate.is_zero()) {
            return candidate;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blstrs::G1Affine;
    use group::{prime::PrimeCurveAffine, Curve};

    #[test]
    fn key_material_has_the_paper_dimensions_and_constant_party_key_size() {
        let material = keygen(8, &[1, 3, 2], 3).unwrap();
        let key = &material.decryption_key;
        assert_eq!(key.party_count(), 3);
        assert_eq!(key.total_weight(), 6);
        assert_eq!(key.verification_key_count(), 24);
        assert_eq!(key.d1_point_count(), 48);
        assert_eq!(key.d2_point_count(), 48);
        assert_eq!(key.core_g2_point_count(), 97);
        assert_eq!(key.g2_point_count(), 121);
        assert_eq!(key.serialized_size_bytes(), 121 * 96);
        assert_eq!(material.encryption_key.serialized_size_bytes(), 289);
        assert!(material
            .party_keys
            .iter()
            .all(|key| key.scalar_count() == 8));
        assert_eq!(key.d1_slot(7).unwrap().len(), 6);
        assert_eq!(key.d2_slot(7).unwrap().len(), 6);
        assert_eq!(key.verification_slot(7).unwrap().len(), 3);
    }

    #[test]
    fn alpha_points_and_verification_keys_match_secret_fractions() {
        let material = keygen(16, &[2, 1, 4], 3).unwrap();
        let key = &material.decryption_key;
        let mut seen = HashSet::new();
        for alpha in key.alpha_points() {
            assert!(!bool::from(alpha.is_zero()));
            assert_ne!(*alpha, -Scalar::ONE);
            assert!(seen.insert(alpha.to_bytes_le()));
        }
        for party in &material.party_keys {
            for (slot, fraction) in party.fractions().iter().enumerate() {
                assert_eq!(
                    key.verification_key(party.party_index, slot).unwrap(),
                    (G2Projective::generator() * fraction).to_affine()
                );
            }
        }
    }

    #[test]
    fn setup_is_context_bound_and_round_trips_through_bincode() {
        let first = keygen(4, &[1, 2, 3], 2).unwrap();
        let second = keygen(4, &[1, 2, 3], 2).unwrap();
        assert_ne!(
            first.encryption_key.setup_id(),
            second.encryption_key.setup_id()
        );
        assert_eq!(
            first.encryption_key.setup_id(),
            first.decryption_key.setup_id()
        );
        assert!(first
            .party_keys
            .iter()
            .all(|party| party.setup_id() == first.decryption_key.setup_id()));

        let encoded = bincode::serialize(&first).unwrap();
        let decoded: KeyMaterial = bincode::deserialize(&encoded).unwrap();
        assert_eq!(
            decoded.decryption_key.setup_id(),
            first.decryption_key.setup_id()
        );
        assert_eq!(decoded.decryption_key.d1, first.decryption_key.d1);
        assert_eq!(decoded.decryption_key.d2, first.decryption_key.d2);
    }

    #[test]
    fn rejects_invalid_dimensions_and_weights() {
        assert!(matches!(
            keygen(0, &[1], 0),
            Err(Error::InvalidBatchSize(0))
        ));
        assert!(matches!(
            keygen(3, &[1], 0),
            Err(Error::InvalidBatchSize(3))
        ));
        assert!(matches!(keygen(4, &[], 0), Err(Error::EmptyCommittee)));
        assert!(matches!(
            keygen(4, &[1, 0], 0),
            Err(Error::InvalidPartyWeight { party_index: 1, .. })
        ));
        assert!(matches!(
            keygen(4, &[1, 2], 3),
            Err(Error::InvalidCommittee { .. })
        ));
    }

    #[test]
    fn singleton_batch_is_well_defined() {
        let material = keygen(1, &[1, 2], 1).unwrap();
        assert_eq!(material.decryption_key.alpha_points().len(), 1);
        assert_eq!(material.decryption_key.g2_point_count(), 2 * 3 + 2 + 1);
        assert!(!bool::from(
            material.decryption_key.global_key().is_identity()
        ));
        let _: G1Affine = G1Affine::generator();
    }
}
