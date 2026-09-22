//! Trusted-dealer setup and weighted Shamir sharing for weighted BTX.

use core::{mem::size_of, ops::Range};

use blstrs::{G1Affine, G1Projective, Gt, Scalar};
use ff::Field;
use group::Group;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    blst_utils::batch_normalize_g1,
    encoding::append_gt,
    error::{Error, Result},
    fft::Radix2Domain,
};

const SETUP_ID_DOMAIN: &[u8] = b"WEIGHTED-BTX-SWAPPED-SETUP-ID-v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncryptionKey {
    max_batch_size: usize,
    element: Gt,
    setup_id: [u8; 32],
}

/// Public weighted-BTX decryption material.
///
/// Every large G1 array is exponent-major. One contiguous block therefore
/// contains all parties (for a verification-key power) or all virtual shares
/// (for a decryption-key power). This is the layout used by the online MSMs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicDecryptionKey {
    setup_id: [u8; 32],
    max_batch_size: usize,
    threshold_weight: usize,
    total_weight: usize,
    domain_size: usize,
    party_weights: Box<[usize]>,
    /// Prefix offsets into `evaluation_domain`; length is `party_count + 1`.
    party_offsets: Box<[usize]>,
    /// The first `total_weight` points of a radix-2 root-of-unity domain.
    evaluation_domain: Box<[Scalar]>,
    /// `[(q_j)^i]_1`, indexed as `(i - 1) * party_count + j`.
    verification_keys: Box<[G1Affine]>,
    /// `[Z(omega) q_owner(omega)^(-i)]_1`, in `i`-major W-sized blocks.
    negative_material: Box<[G1Affine]>,
    /// `[Z(omega) q_owner(omega)^i]_1`, for `i = 1` through `L - 1`, in W-sized blocks.
    positive_material: Box<[G1Affine]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartySecretKey {
    pub party_index: usize,
    setup_id: [u8; 32],
    max_batch_size: usize,
    q: Scalar,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyMaterial {
    pub encryption_key: EncryptionKey,
    pub decryption_key: PublicDecryptionKey,
    pub party_keys: Vec<PartySecretKey>,
}

#[derive(Clone, Debug)]
pub struct KeygenConfig {
    pub max_batch_size: usize,
    pub party_weights: Vec<usize>,
    /// A set is authorized when its total weight is strictly greater than this.
    pub threshold_weight: usize,
}

impl KeygenConfig {
    pub fn new(max_batch_size: usize, party_weights: &[usize], threshold_weight: usize) -> Self {
        Self {
            max_batch_size,
            party_weights: party_weights.to_vec(),
            threshold_weight,
        }
    }
}

pub fn keygen(
    max_batch_size: usize,
    party_weights: &[usize],
    threshold_weight: usize,
) -> Result<KeyMaterial> {
    let mut rng = rand_core::OsRng;
    keygen_with_rng(
        KeygenConfig::new(max_batch_size, party_weights, threshold_weight),
        &mut rng,
    )
}

pub fn keygen_with_rng<R: rand_core::RngCore + rand_core::CryptoRng>(
    config: KeygenConfig,
    rng: &mut R,
) -> Result<KeyMaterial> {
    let dimensions = validate_config(&config)?;
    let party_count = config.party_weights.len();

    let domain =
        Radix2Domain::new(dimensions.domain_size).map_err(|_| Error::InvalidEvaluationDomain {
            total_weight: dimensions.total_weight,
        })?;

    // Evaluate one random degree-t Shamir polynomial at every virtual point
    // with a scalar FFT. Only the first W roots are assigned to parties when
    // the enclosing radix-2 domain is larger than W.
    let mut polynomial = vec![Scalar::ZERO; dimensions.domain_size];
    polynomial[..=config.threshold_weight]
        .iter_mut()
        .for_each(|coefficient| *coefficient = Scalar::random(&mut *rng));
    let z = polynomial[0];
    domain.fft(&mut polynomial);
    polynomial.truncate(dimensions.total_weight);
    let z_evaluations = polynomial;

    let mut evaluation_domain = vec![Scalar::ZERO; dimensions.domain_size];
    if dimensions.domain_size == 1 {
        evaluation_domain[0] = Scalar::ONE;
    } else {
        // FFT evaluation of X gives 1, omega, omega^2, ... in exactly the same
        // ordering used for the Shamir-polynomial evaluations above.
        evaluation_domain[1] = Scalar::ONE;
        domain.fft(&mut evaluation_domain);
    }
    evaluation_domain.truncate(dimensions.total_weight);

    let party_offsets = prefix_offsets(&config.party_weights)?;
    debug_assert_eq!(party_offsets.last().copied(), Some(dimensions.total_weight));
    let owner_by_virtual_point = owners_from_offsets(&party_offsets);

    let mut party_keys = (0..party_count)
        .map(|party_index| PartySecretKey {
            party_index,
            setup_id: [0u8; 32],
            max_batch_size: config.max_batch_size,
            q: sample_nonzero(rng),
        })
        .collect::<Vec<_>>();

    // These O(NL) scalar tables are small relative to the O(WL) public key and
    // let each output block be generated without retaining another giant
    // scalar or projective array.
    let mut positive_q_powers = vec![Scalar::ZERO; dimensions.verification_count];
    let mut negative_q_powers = vec![Scalar::ZERO; dimensions.verification_count];
    for party_key in &party_keys {
        let inverse = Option::<Scalar>::from(party_key.q.invert())
            .expect("party setup samples q from the nonzero field elements");
        let mut positive = party_key.q;
        let mut negative = inverse;
        for power_index in 0..config.max_batch_size {
            let index = power_index * party_count + party_key.party_index;
            positive_q_powers[index] = positive;
            negative_q_powers[index] = negative;
            positive *= party_key.q;
            negative *= inverse;
        }
    }

    let mut verification_keys = Vec::with_capacity(dimensions.verification_count);
    for power_index in 0..config.max_batch_size {
        append_g1_block(
            &mut verification_keys,
            &positive_q_powers[power_index * party_count..(power_index + 1) * party_count],
        );
    }

    let mut scalar_block = vec![Scalar::ZERO; dimensions.total_weight];
    let mut negative_material = Vec::with_capacity(dimensions.negative_count);
    for power_index in 0..config.max_batch_size {
        fill_material_scalars(
            &mut scalar_block,
            &z_evaluations,
            &owner_by_virtual_point,
            &negative_q_powers[power_index * party_count..(power_index + 1) * party_count],
        );
        append_g1_block(&mut negative_material, &scalar_block);
    }

    let mut positive_material = Vec::with_capacity(dimensions.positive_count);
    for power_index in 0..config.max_batch_size.saturating_sub(1) {
        fill_material_scalars(
            &mut scalar_block,
            &z_evaluations,
            &owner_by_virtual_point,
            &positive_q_powers[power_index * party_count..(power_index + 1) * party_count],
        );
        append_g1_block(&mut positive_material, &scalar_block);
    }

    let encryption_element = Gt::generator() * z;
    let setup_id = setup_identifier(
        &encryption_element,
        config.max_batch_size,
        config.threshold_weight,
        &config.party_weights,
        &verification_keys[..party_count],
        &negative_material[..dimensions.total_weight],
    );
    for party_key in &mut party_keys {
        party_key.setup_id = setup_id;
    }

    let encryption_key = EncryptionKey {
        max_batch_size: config.max_batch_size,
        element: encryption_element,
        setup_id,
    };
    let decryption_key = PublicDecryptionKey {
        setup_id,
        max_batch_size: config.max_batch_size,
        threshold_weight: config.threshold_weight,
        total_weight: dimensions.total_weight,
        domain_size: dimensions.domain_size,
        party_weights: config.party_weights.into_boxed_slice(),
        party_offsets: party_offsets.into_boxed_slice(),
        evaluation_domain: evaluation_domain.into_boxed_slice(),
        verification_keys: verification_keys.into_boxed_slice(),
        negative_material: negative_material.into_boxed_slice(),
        positive_material: positive_material.into_boxed_slice(),
    };

    Ok(KeyMaterial {
        encryption_key,
        decryption_key,
        party_keys,
    })
}

impl EncryptionKey {
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub fn element(&self) -> Gt {
        self.element
    }

    pub fn setup_id(&self) -> [u8; 32] {
        self.setup_id
    }
}

impl PublicDecryptionKey {
    /// Stable identifier for this honestly generated setup context.
    pub fn setup_id(&self) -> [u8; 32] {
        self.setup_id
    }

    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
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

    pub(crate) fn party_range(&self, party_index: usize) -> Result<Range<usize>> {
        if party_index >= self.party_count() {
            return Err(Error::InvalidPartyIndex(party_index));
        }
        Ok(self.party_offsets[party_index]..self.party_offsets[party_index + 1])
    }

    pub(crate) fn verification_key(&self, party_index: usize, power: usize) -> Result<G1Affine> {
        if party_index >= self.party_count() {
            return Err(Error::InvalidPartyIndex(party_index));
        }
        validate_power(power, self.max_batch_size)?;
        Ok(self.verification_keys[(power - 1) * self.party_count() + party_index])
    }

    pub(crate) fn verification_power(&self, power: usize) -> Result<&[G1Affine]> {
        validate_power(power, self.max_batch_size)?;
        let start = (power - 1) * self.party_count();
        Ok(&self.verification_keys[start..start + self.party_count()])
    }

    pub(crate) fn negative_power(&self, power: usize) -> Result<&[G1Affine]> {
        validate_power(power, self.max_batch_size)?;
        let start = (power - 1) * self.total_weight;
        Ok(&self.negative_material[start..start + self.total_weight])
    }

    pub(crate) fn positive_power(&self, power: usize) -> Result<&[G1Affine]> {
        let max_power = self.max_batch_size.saturating_sub(1);
        validate_power(power, max_power)?;
        let start = (power - 1) * self.total_weight;
        Ok(&self.positive_material[start..start + self.total_weight])
    }
}

impl PartySecretKey {
    pub fn setup_id(&self) -> [u8; 32] {
        self.setup_id
    }

    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub(crate) fn secret_scalar(&self) -> Scalar {
        self.q
    }
}

#[derive(Clone, Copy, Debug)]
struct ValidatedDimensions {
    total_weight: usize,
    domain_size: usize,
    verification_count: usize,
    negative_count: usize,
    positive_count: usize,
}

fn validate_config(config: &KeygenConfig) -> Result<ValidatedDimensions> {
    if config.max_batch_size == 0 {
        return Err(Error::BatchIsEmpty);
    }
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
        .try_fold(0usize, |total, weight| total.checked_add(*weight))
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

    let verification_count =
        checked_point_count(config.max_batch_size, config.party_weights.len())?;
    let negative_count = checked_point_count(config.max_batch_size, total_weight)?;
    let positive_count = checked_point_count(config.max_batch_size - 1, total_weight)?;
    checked_scalar_count(config.max_batch_size, config.party_weights.len())?;

    Ok(ValidatedDimensions {
        total_weight,
        domain_size,
        verification_count,
        negative_count,
        positive_count,
    })
}

fn checked_point_count(left: usize, right: usize) -> Result<usize> {
    let count = left
        .checked_mul(right)
        .ok_or(Error::ParameterSizeOverflow)?;
    let bytes = count
        .checked_mul(size_of::<G1Affine>())
        .ok_or(Error::ParameterSizeOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(Error::ParameterSizeOverflow);
    }
    Ok(count)
}

fn checked_scalar_count(left: usize, right: usize) -> Result<usize> {
    let count = left
        .checked_mul(right)
        .ok_or(Error::ParameterSizeOverflow)?;
    let bytes = count
        .checked_mul(size_of::<Scalar>())
        .ok_or(Error::ParameterSizeOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(Error::ParameterSizeOverflow);
    }
    Ok(count)
}

fn prefix_offsets(weights: &[usize]) -> Result<Vec<usize>> {
    let mut offsets = Vec::with_capacity(weights.len() + 1);
    offsets.push(0usize);
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
    let total_weight = offsets.last().copied().unwrap_or(0);
    let mut owners = vec![0usize; total_weight];
    for party_index in 0..offsets.len().saturating_sub(1) {
        owners[offsets[party_index]..offsets[party_index + 1]].fill(party_index);
    }
    owners
}

fn sample_nonzero<R: rand_core::RngCore + rand_core::CryptoRng>(rng: &mut R) -> Scalar {
    loop {
        let candidate = Scalar::random(&mut *rng);
        if !bool::from(candidate.is_zero()) {
            return candidate;
        }
    }
}

fn fill_material_scalars(
    output: &mut [Scalar],
    z_evaluations: &[Scalar],
    owners: &[usize],
    party_powers: &[Scalar],
) {
    debug_assert_eq!(output.len(), z_evaluations.len());
    debug_assert_eq!(output.len(), owners.len());
    output
        .par_iter_mut()
        .zip(z_evaluations.par_iter())
        .zip(owners.par_iter())
        .for_each(|((value, z_evaluation), owner)| {
            *value = *z_evaluation * party_powers[*owner];
        });
}

fn append_g1_block(output: &mut Vec<G1Affine>, scalars: &[Scalar]) {
    let projective = scalars
        .par_iter()
        .map(|scalar| G1Projective::generator() * scalar)
        .collect::<Vec<_>>();
    let old_len = output.len();
    output.resize(old_len + projective.len(), G1Affine::default());
    batch_normalize_g1(&projective, &mut output[old_len..]);
}

fn setup_identifier(
    encryption_element: &Gt,
    max_batch_size: usize,
    threshold_weight: usize,
    party_weights: &[usize],
    first_verification_power: &[G1Affine],
    first_negative_power: &[G1Affine],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SETUP_ID_DOMAIN);
    hasher.update((max_batch_size as u64).to_le_bytes());
    hasher.update((threshold_weight as u64).to_le_bytes());
    hasher.update((party_weights.len() as u64).to_le_bytes());
    for weight in party_weights {
        hasher.update((*weight as u64).to_le_bytes());
    }

    let mut target_bytes = Vec::with_capacity(289);
    append_gt(encryption_element, &mut target_bytes);
    hasher.update(target_bytes);
    for point in first_verification_power {
        hasher.update(point.to_compressed());
    }
    for point in first_negative_power {
        hasher.update(point.to_compressed());
    }
    hasher.finalize().into()
}

fn validate_power(power: usize, max_power: usize) -> Result<()> {
    if power == 0 || power > max_power {
        return Err(Error::InvalidPower { power, max_power });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use blstrs::{pairing, G2Affine};
    use group::{prime::PrimeCurveAffine, Curve};

    use super::*;

    #[test]
    fn heterogeneous_key_material_has_expected_shapes_and_layout() {
        let material = keygen(4, &[2, 1, 3], 3).unwrap();
        let dk = &material.decryption_key;

        assert_eq!(material.encryption_key.max_batch_size(), 4);
        assert_eq!(dk.max_batch_size(), 4);
        assert_eq!(dk.party_count(), 3);
        assert_eq!(dk.total_weight(), 6);
        assert_eq!(dk.threshold_weight(), 3);
        assert_eq!(dk.required_weight(), 4);
        assert_eq!(dk.party_weights(), &[2, 1, 3]);
        assert_eq!(dk.domain_size(), 8);
        assert_eq!(dk.verification_keys.len(), 12);
        assert_eq!(dk.negative_material.len(), 24);
        assert_eq!(dk.positive_material.len(), 18);
        assert_eq!(material.party_keys.len(), 3);
        assert_eq!(dk.party_range(0).unwrap(), 0..2);
        assert_eq!(dk.party_range(1).unwrap(), 2..3);
        assert_eq!(dk.party_range(2).unwrap(), 3..6);
        assert_eq!(
            &dk.domain_points()[dk.party_range(2).unwrap()],
            &dk.domain_points()[3..6]
        );

        for party_key in &material.party_keys {
            assert!(!bool::from(party_key.secret_scalar().is_zero()));
            let mut power = party_key.secret_scalar();
            for exponent in 1..=4 {
                assert_eq!(
                    dk.verification_key(party_key.party_index, exponent)
                        .unwrap(),
                    (G1Projective::generator() * power).to_affine()
                );
                power *= party_key.secret_scalar();
            }
        }
    }

    #[test]
    fn evaluation_points_are_distinct_nonzero_roots() {
        let material = keygen(2, &[2, 3, 1], 2).unwrap();
        let domain = material.decryption_key.domain_points();
        assert_eq!(domain.len(), 6);
        assert_eq!(domain[0], Scalar::ONE);
        assert!(domain.iter().all(|point| !bool::from(point.is_zero())));

        let encodings = domain
            .iter()
            .map(Scalar::to_bytes_le)
            .collect::<HashSet<_>>();
        assert_eq!(encodings.len(), domain.len());

        let singleton = keygen(1, &[1], 0).unwrap();
        assert_eq!(singleton.decryption_key.domain_points(), &[Scalar::ONE]);
        assert!(singleton.decryption_key.positive_material.is_empty());
    }

    #[test]
    fn negative_and_positive_material_use_the_owning_party_power() {
        let material = keygen(3, &[2, 1], 1).unwrap();
        let dk = &material.decryption_key;

        for party_key in &material.party_keys {
            let q = party_key.secret_scalar();
            let range = dk.party_range(party_key.party_index).unwrap();
            let baseline = dk.negative_power(1).unwrap()[range.clone()]
                .iter()
                .map(|point| G1Projective::from(*point) * q)
                .collect::<Vec<_>>();

            let mut q_power = q;
            for power in 1..=3 {
                let recovered = dk.negative_power(power).unwrap()[range.clone()]
                    .iter()
                    .map(|point| G1Projective::from(*point) * q_power)
                    .collect::<Vec<_>>();
                assert_eq!(recovered, baseline);
                q_power *= q;
            }

            let q_inverse = Option::<Scalar>::from(q.invert()).unwrap();
            let mut inverse_power = q_inverse;
            for power in 1..=2 {
                let recovered = dk.positive_power(power).unwrap()[range.clone()]
                    .iter()
                    .map(|point| G1Projective::from(*point) * inverse_power)
                    .collect::<Vec<_>>();
                assert_eq!(recovered, baseline);
                inverse_power *= q_inverse;
            }
        }
    }

    #[test]
    fn public_virtual_shares_reconstruct_the_encryption_secret() {
        let material = keygen(2, &[2, 1, 3], 2).unwrap();
        let dk = &material.decryption_key;
        let selected = [0usize, 2, 5];
        let points = selected
            .iter()
            .map(|index| dk.domain_points()[*index])
            .collect::<Vec<_>>();
        let coefficients = points
            .iter()
            .enumerate()
            .map(|(j, point_j)| {
                let (numerator, denominator) =
                    points.iter().enumerate().filter(|(k, _)| *k != j).fold(
                        (Scalar::ONE, Scalar::ONE),
                        |(numerator, denominator), (_, point_k)| {
                            (numerator * -*point_k, denominator * (*point_j - point_k))
                        },
                    );
                numerator * Option::<Scalar>::from(denominator.invert()).unwrap()
            })
            .collect::<Vec<_>>();

        let bases = selected
            .iter()
            .map(|virtual_index| {
                let party_index = material
                    .party_keys
                    .iter()
                    .position(|key| {
                        dk.party_range(key.party_index)
                            .unwrap()
                            .contains(virtual_index)
                    })
                    .unwrap();
                G1Projective::from(dk.negative_power(1).unwrap()[*virtual_index])
                    * material.party_keys[party_index].secret_scalar()
            })
            .collect::<Vec<_>>();
        let reconstructed = G1Projective::multi_exp(&bases, &coefficients).to_affine();

        assert_eq!(
            pairing(&reconstructed, &G2Affine::generator()),
            material.encryption_key.element()
        );
    }

    #[test]
    fn accessors_reject_invalid_party_and_power() {
        let material = keygen(3, &[1, 2], 1).unwrap();
        let dk = &material.decryption_key;
        assert!(matches!(
            dk.party_range(2),
            Err(Error::InvalidPartyIndex(2))
        ));
        assert!(matches!(
            dk.party_weight(2),
            Err(Error::InvalidPartyIndex(2))
        ));
        assert!(matches!(
            dk.verification_key(0, 0),
            Err(Error::InvalidPower { .. })
        ));
        assert!(matches!(
            dk.negative_power(4),
            Err(Error::InvalidPower { .. })
        ));
        assert!(matches!(
            dk.positive_power(3),
            Err(Error::InvalidPower { .. })
        ));
    }

    #[test]
    fn rejects_invalid_weighted_configurations() {
        assert!(matches!(keygen(0, &[1], 0), Err(Error::BatchIsEmpty)));
        assert!(matches!(keygen(1, &[], 0), Err(Error::EmptyCommittee)));
        assert!(matches!(
            keygen(1, &[1, 0], 0),
            Err(Error::InvalidPartyWeight {
                party_index: 1,
                weight: 0
            })
        ));
        assert!(matches!(
            keygen(1, &[2, 1], 3),
            Err(Error::InvalidCommittee { .. })
        ));
        assert!(matches!(
            keygen(1, &[usize::MAX, 1], 0),
            Err(Error::TotalWeightOverflow)
        ));
    }
}
