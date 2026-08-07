//! Universal powers-of-tau setup and weighted trusted-dealer key generation.
//!
//! The public parameters are reusable across weighted committees with the same
//! index-space size.  Key generation assigns one Shamir evaluation point to
//! every virtual unit of weight, while each real party keeps only the inverse
//! of its masking scalar.

use core::{mem::size_of, ops::Range};

use blstrs::{G1Affine, G1Projective, G2Affine, G2Projective, Scalar};
use ff::{BatchInvert, Field};
use group::Group;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    blst_utils::{batch_normalize_g1, batch_normalize_g2},
    error::{Error, Result},
    fft::Radix2Domain,
};

const CRS_ID_DOMAIN: &[u8] = b"WEIGHTED-INDEXED-BTE-CRS-ID-BLS12381-v1";
const SETUP_ID_DOMAIN: &[u8] = b"WEIGHTED-INDEXED-BTE-SETUP-ID-BLS12381-v1";

/// Universal powers-of-tau parameters for an index space of size `n`.
///
/// `g_powers` contains G1 powers `tau^1, ..., tau^n`. `h_powers` contains
/// G2 powers `tau^1, ..., tau^(2n+1)` with the single power `tau^(n+1)`
/// omitted, as required by Figure 2 of the paper.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicParameters {
    crs_id: [u8; 32],
    index_space_size: usize,
    g_powers: Box<[G1Affine]>,
    h_powers: Box<[G2Affine]>,
}

/// Public key material for one weighted committee.
///
/// The large `gamma` array is index-major: every index owns one contiguous
/// block of `total_weight` G2 points, and each real party owns a contiguous
/// sub-range within that block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MasterPublicKey {
    setup_id: [u8; 32],
    crs_id: [u8; 32],
    index_space_size: usize,
    threshold_weight: usize,
    total_weight: usize,
    domain_size: usize,
    party_weights: Box<[usize]>,
    /// Prefix offsets into every W-point gamma block and the evaluation domain.
    party_offsets: Box<[usize]>,
    /// First W points of an enclosing radix-2 root-of-unity domain.
    evaluation_domain: Box<[Scalar]>,
    /// `delta_i = [msk * tau^(i+1)]_1`, indexed by zero-based ciphertext index.
    deltas: Box<[G1Affine]>,
    /// `[sk_j^(-1)]_1`, indexed by zero-based real-party index.
    inverse_public_keys: Box<[G1Affine]>,
    /// `gamma[i, omega]`, laid out as `i * W + omega`.
    gamma: Box<[G2Affine]>,
}

/// One real party's secret `q_j = sk_j^(-1)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartySecretKey {
    pub party_index: usize,
    setup_id: [u8; 32],
    inverse_scalar: Scalar,
    public_inverse: G1Affine,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyMaterial {
    pub public_key: MasterPublicKey,
    pub party_keys: Vec<PartySecretKey>,
}

#[derive(Clone, Debug)]
pub struct KeygenConfig {
    pub party_weights: Vec<usize>,
    /// A real-party set is authorized exactly when its total weight is greater
    /// than this value.
    pub threshold_weight: usize,
}

impl KeygenConfig {
    pub fn new(party_weights: &[usize], threshold_weight: usize) -> Self {
        Self {
            party_weights: party_weights.to_vec(),
            threshold_weight,
        }
    }
}

/// Generate a reusable powers-of-tau CRS.
pub fn setup(index_space_size: usize) -> Result<PublicParameters> {
    let mut rng = rand_core::OsRng;
    setup_with_rng(index_space_size, &mut rng)
}

pub fn setup_with_rng<R: rand_core::RngCore + rand_core::CryptoRng>(
    index_space_size: usize,
    rng: &mut R,
) -> Result<PublicParameters> {
    validate_index_space(index_space_size)?;

    let h_count = index_space_size
        .checked_mul(2)
        .ok_or(Error::ParameterSizeOverflow)?;
    let maximum_exponent = h_count.checked_add(1).ok_or(Error::ParameterSizeOverflow)?;
    let power_count = maximum_exponent
        .checked_add(1)
        .ok_or(Error::ParameterSizeOverflow)?;
    checked_allocation::<G1Affine>(index_space_size)?;
    checked_allocation::<G2Affine>(h_count)?;
    checked_allocation::<Scalar>(power_count)?;

    let tau = sample_nonzero(rng);
    let mut tau_powers = Vec::with_capacity(power_count);
    tau_powers.push(Scalar::ONE);
    for exponent in 1..=maximum_exponent {
        tau_powers.push(tau_powers[exponent - 1] * tau);
    }

    let g_projective = tau_powers[1..=index_space_size]
        .par_iter()
        .map(|power| G1Projective::generator() * power)
        .collect::<Vec<_>>();
    let mut g_powers = vec![G1Affine::default(); g_projective.len()];
    batch_normalize_g1(&g_projective, &mut g_powers);

    let h_exponents = (1..=maximum_exponent)
        .filter(|exponent| *exponent != index_space_size + 1)
        .collect::<Vec<_>>();
    debug_assert_eq!(h_exponents.len(), h_count);
    let h_projective = h_exponents
        .par_iter()
        .map(|exponent| G2Projective::generator() * tau_powers[*exponent])
        .collect::<Vec<_>>();
    let mut h_powers = vec![G2Affine::default(); h_projective.len()];
    batch_normalize_g2(&h_projective, &mut h_powers);

    let crs_id = crs_identifier(index_space_size, &g_powers, &h_powers);
    Ok(PublicParameters {
        crs_id,
        index_space_size,
        g_powers: g_powers.into_boxed_slice(),
        h_powers: h_powers.into_boxed_slice(),
    })
}

pub fn keygen(
    parameters: &PublicParameters,
    party_weights: &[usize],
    threshold_weight: usize,
) -> Result<KeyMaterial> {
    let mut rng = rand_core::OsRng;
    keygen_with_rng(
        parameters,
        KeygenConfig::new(party_weights, threshold_weight),
        &mut rng,
    )
}

pub fn keygen_with_rng<R: rand_core::RngCore + rand_core::CryptoRng>(
    parameters: &PublicParameters,
    config: KeygenConfig,
    rng: &mut R,
) -> Result<KeyMaterial> {
    let dimensions = validate_keygen_config(parameters, &config)?;
    let party_count = config.party_weights.len();

    let domain =
        Radix2Domain::new(dimensions.domain_size).map_err(|_| Error::InvalidEvaluationDomain {
            total_weight: dimensions.total_weight,
        })?;

    // Evaluate one degree-t Shamir polynomial at all virtual points.  The
    // constant coefficient is sampled nonzero: a zero master secret would
    // make every ciphertext mask public.
    let master_secret = sample_nonzero(rng);
    let mut polynomial = vec![Scalar::ZERO; dimensions.domain_size];
    polynomial[0] = master_secret;
    polynomial[1..=config.threshold_weight]
        .iter_mut()
        .for_each(|coefficient| *coefficient = Scalar::random(&mut *rng));
    domain.fft(&mut polynomial);
    polynomial.truncate(dimensions.total_weight);
    let share_evaluations = polynomial;

    let mut evaluation_domain = vec![Scalar::ZERO; dimensions.domain_size];
    if dimensions.domain_size == 1 {
        evaluation_domain[0] = Scalar::ONE;
    } else {
        // FFT(X) yields 1, omega, omega^2, ... in the same order as the sharing
        // polynomial evaluations. Every assigned point is nonzero.
        evaluation_domain[1] = Scalar::ONE;
        domain.fft(&mut evaluation_domain);
    }
    evaluation_domain.truncate(dimensions.total_weight);

    let party_offsets = prefix_offsets(&config.party_weights)?;
    let owners = owners_from_offsets(&party_offsets);
    debug_assert_eq!(owners.len(), dimensions.total_weight);

    // Figure 2 samples sk_j from Z_p but subsequently inverts it.  Rejection
    // sampling from Z_p^* resolves that undefined zero case.
    let masking_scalars = (0..party_count)
        .map(|_| sample_nonzero(rng))
        .collect::<Vec<_>>();
    let mut inverse_scalars = masking_scalars.clone();
    inverse_scalars.iter_mut().batch_invert();

    let inverse_projective = inverse_scalars
        .par_iter()
        .map(|inverse| G1Projective::generator() * inverse)
        .collect::<Vec<_>>();
    let mut inverse_public_keys = vec![G1Affine::default(); party_count];
    batch_normalize_g1(&inverse_projective, &mut inverse_public_keys);

    let delta_projective = parameters
        .g_powers
        .par_iter()
        .map(|power| G1Projective::from(*power) * master_secret)
        .collect::<Vec<_>>();
    let mut deltas = vec![G1Affine::default(); parameters.index_space_size];
    batch_normalize_g1(&delta_projective, &mut deltas);

    // The per-virtual-share scalar is common to every index. Keep just this W
    // scalar block and one W-projective block live while appending each
    // normalized index block to the final nW-point affine array.
    let masked_share_scalars = share_evaluations
        .par_iter()
        .zip(owners.par_iter())
        .map(|(share, owner)| *share * masking_scalars[*owner])
        .collect::<Vec<_>>();
    let mut gamma = Vec::with_capacity(dimensions.gamma_count);
    for index in 0..parameters.index_space_size {
        // Mathematical indices are one-based. For code index i, gamma uses
        // h_(n+2+i), which is always on the high side of the missing power.
        let exponent = parameters
            .index_space_size
            .checked_add(2)
            .and_then(|first| first.checked_add(index))
            .ok_or(Error::ParameterSizeOverflow)?;
        let base = parameters.h_power(exponent)?;
        let projective = masked_share_scalars
            .par_iter()
            .map(|scalar| G2Projective::from(base) * scalar)
            .collect::<Vec<_>>();
        let old_len = gamma.len();
        gamma.resize(old_len + dimensions.total_weight, G2Affine::default());
        batch_normalize_g2(&projective, &mut gamma[old_len..]);
    }
    debug_assert_eq!(gamma.len(), dimensions.gamma_count);

    let setup_id = setup_identifier(
        parameters,
        config.threshold_weight,
        dimensions.domain_size,
        &config.party_weights,
        &party_offsets,
        &evaluation_domain,
        &deltas,
        &inverse_public_keys,
        &gamma,
    );

    let party_keys = inverse_scalars
        .into_iter()
        .zip(inverse_public_keys.iter().copied())
        .enumerate()
        .map(
            |(party_index, (inverse_scalar, public_inverse))| PartySecretKey {
                party_index,
                setup_id,
                inverse_scalar,
                public_inverse,
            },
        )
        .collect::<Vec<_>>();

    let public_key = MasterPublicKey {
        setup_id,
        crs_id: parameters.crs_id,
        index_space_size: parameters.index_space_size,
        threshold_weight: config.threshold_weight,
        total_weight: dimensions.total_weight,
        domain_size: dimensions.domain_size,
        party_weights: config.party_weights.into_boxed_slice(),
        party_offsets: party_offsets.into_boxed_slice(),
        evaluation_domain: evaluation_domain.into_boxed_slice(),
        deltas: deltas.into_boxed_slice(),
        inverse_public_keys: inverse_public_keys.into_boxed_slice(),
        gamma: gamma.into_boxed_slice(),
    };

    Ok(KeyMaterial {
        public_key,
        party_keys,
    })
}

impl PublicParameters {
    /// Hash of every serialized powers-of-tau point and the index-space size.
    pub fn crs_id(&self) -> [u8; 32] {
        self.crs_id
    }

    pub fn index_space_size(&self) -> usize {
        self.index_space_size
    }

    pub fn g1_point_count(&self) -> usize {
        self.g_powers.len()
    }

    pub fn g2_point_count(&self) -> usize {
        self.h_powers.len()
    }

    pub fn g1_serialized_size(&self) -> usize {
        self.g_powers.len() * G1Affine::default().to_compressed().len()
    }

    pub fn g2_serialized_size(&self) -> usize {
        self.h_powers.len() * G2Affine::default().to_compressed().len()
    }

    pub fn serialized_size(&self) -> usize {
        self.g1_serialized_size() + self.g2_serialized_size()
    }

    pub fn serialized_size_bytes(&self) -> usize {
        self.serialized_size()
    }

    /// Return `[tau^exponent]_1` for a one-based exponent in `1..=n`.
    pub fn g_power(&self, exponent: usize) -> Result<G1Affine> {
        if exponent == 0 || exponent > self.index_space_size {
            return Err(Error::InvalidBatchSize(exponent));
        }
        Ok(self.g_powers[exponent - 1])
    }

    /// Return `[tau^exponent]_2` for a one-based exponent, rejecting the
    /// deliberately absent exponent `n + 1`.
    pub fn h_power(&self, exponent: usize) -> Result<G2Affine> {
        let maximum = self
            .index_space_size
            .checked_mul(2)
            .and_then(|twice| twice.checked_add(1))
            .ok_or(Error::ParameterSizeOverflow)?;
        let missing = self
            .index_space_size
            .checked_add(1)
            .ok_or(Error::ParameterSizeOverflow)?;
        if exponent == 0 || exponent > maximum || exponent == missing {
            return Err(Error::InvalidBatchSize(exponent));
        }

        let storage_index = if exponent <= self.index_space_size {
            exponent - 1
        } else {
            // Skip the absent exponent n+1.
            exponent - 2
        };
        Ok(self.h_powers[storage_index])
    }

    /// Return `[tau^(n+1+offset)]_2` for `-n <= offset <= n`, excluding zero.
    pub fn centered_power(&self, offset: isize) -> Result<G2Affine> {
        if offset == 0 {
            let missing = self
                .index_space_size
                .checked_add(1)
                .ok_or(Error::ParameterSizeOverflow)?;
            return Err(Error::InvalidBatchSize(missing));
        }
        let center = isize::try_from(self.index_space_size)
            .ok()
            .and_then(|size| size.checked_add(1))
            .ok_or(Error::ParameterSizeOverflow)?;
        let exponent = center
            .checked_add(offset)
            .filter(|exponent| *exponent > 0)
            .and_then(|exponent| usize::try_from(exponent).ok())
            .ok_or(Error::InvalidBatchSize(0))?;
        self.h_power(exponent)
    }
}

impl MasterPublicKey {
    /// Hash of the CRS identifier and every piece of committee public material.
    pub fn setup_id(&self) -> [u8; 32] {
        self.setup_id
    }

    pub fn crs_id(&self) -> [u8; 32] {
        self.crs_id
    }

    pub fn index_space_size(&self) -> usize {
        self.index_space_size
    }

    pub fn threshold_weight(&self) -> usize {
        self.threshold_weight
    }

    pub fn required_weight(&self) -> usize {
        self.threshold_weight + 1
    }

    pub fn total_weight(&self) -> usize {
        self.total_weight
    }

    pub fn party_count(&self) -> usize {
        self.party_weights.len()
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

    pub fn party_range(&self, party_index: usize) -> Result<Range<usize>> {
        if party_index >= self.party_count() {
            return Err(Error::InvalidPartyIndex(party_index));
        }
        Ok(self.party_offsets[party_index]..self.party_offsets[party_index + 1])
    }

    pub fn domain_points(&self) -> &[Scalar] {
        &self.evaluation_domain
    }

    pub fn domain_size(&self) -> usize {
        self.domain_size
    }

    pub fn delta(&self, index: usize) -> Result<G1Affine> {
        self.deltas
            .get(index)
            .copied()
            .ok_or(Error::InvalidCiphertextIndex(index))
    }

    pub fn inverse_key(&self, party_index: usize) -> Result<G1Affine> {
        self.inverse_public_keys
            .get(party_index)
            .copied()
            .ok_or(Error::InvalidPartyIndex(party_index))
    }

    /// Return the complete W-point gamma block for a zero-based ciphertext index.
    pub fn gamma_block(&self, index: usize) -> Result<&[G2Affine]> {
        if index >= self.index_space_size {
            return Err(Error::InvalidCiphertextIndex(index));
        }
        let start = index * self.total_weight;
        Ok(&self.gamma[start..start + self.total_weight])
    }

    pub fn g1_point_count(&self) -> usize {
        self.deltas.len() + self.inverse_public_keys.len()
    }

    pub fn g2_point_count(&self) -> usize {
        self.gamma.len()
    }

    pub fn g1_serialized_size(&self) -> usize {
        self.g1_point_count() * G1Affine::default().to_compressed().len()
    }

    pub fn g2_serialized_size(&self) -> usize {
        self.g2_point_count() * G2Affine::default().to_compressed().len()
    }

    pub fn serialized_size(&self) -> usize {
        self.g1_serialized_size() + self.g2_serialized_size()
    }

    pub fn serialized_size_bytes(&self) -> usize {
        self.serialized_size()
    }
}

impl PartySecretKey {
    pub fn setup_id(&self) -> [u8; 32] {
        self.setup_id
    }

    pub(crate) fn inverse_scalar(&self) -> Scalar {
        self.inverse_scalar
    }

    pub fn public_inverse(&self) -> G1Affine {
        self.public_inverse
    }
}

#[derive(Clone, Copy, Debug)]
struct ValidatedDimensions {
    total_weight: usize,
    domain_size: usize,
    gamma_count: usize,
}

fn validate_index_space(index_space_size: usize) -> Result<()> {
    if index_space_size < 2 {
        return Err(Error::InvalidIndexSpace(index_space_size));
    }
    let h_count = index_space_size
        .checked_mul(2)
        .ok_or(Error::ParameterSizeOverflow)?;
    h_count.checked_add(1).ok_or(Error::ParameterSizeOverflow)?;
    Ok(())
}

fn validate_keygen_config(
    parameters: &PublicParameters,
    config: &KeygenConfig,
) -> Result<ValidatedDimensions> {
    validate_index_space(parameters.index_space_size)?;
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
    let gamma_count = parameters
        .index_space_size
        .checked_mul(total_weight)
        .ok_or(Error::ParameterSizeOverflow)?;

    checked_allocation::<G2Affine>(gamma_count)?;
    checked_allocation::<G1Affine>(parameters.index_space_size)?;
    checked_allocation::<G1Affine>(config.party_weights.len())?;
    checked_allocation::<Scalar>(domain_size)?;
    let offset_count = config
        .party_weights
        .len()
        .checked_add(1)
        .ok_or(Error::ParameterSizeOverflow)?;
    checked_allocation::<usize>(offset_count)?;

    Ok(ValidatedDimensions {
        total_weight,
        domain_size,
        gamma_count,
    })
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

fn crs_identifier(
    index_space_size: usize,
    g_powers: &[G1Affine],
    h_powers: &[G2Affine],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(CRS_ID_DOMAIN);
    hasher.update((index_space_size as u64).to_le_bytes());
    hasher.update((g_powers.len() as u64).to_le_bytes());
    for point in g_powers {
        hasher.update(point.to_compressed());
    }
    hasher.update((h_powers.len() as u64).to_le_bytes());
    for point in h_powers {
        hasher.update(point.to_compressed());
    }
    hasher.finalize().into()
}

#[allow(clippy::too_many_arguments)]
fn setup_identifier(
    parameters: &PublicParameters,
    threshold_weight: usize,
    domain_size: usize,
    party_weights: &[usize],
    party_offsets: &[usize],
    evaluation_domain: &[Scalar],
    deltas: &[G1Affine],
    inverse_public_keys: &[G1Affine],
    gamma: &[G2Affine],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SETUP_ID_DOMAIN);
    hasher.update(parameters.crs_id);
    hasher.update((parameters.index_space_size as u64).to_le_bytes());
    hasher.update((threshold_weight as u64).to_le_bytes());
    hasher.update((domain_size as u64).to_le_bytes());

    hasher.update((party_weights.len() as u64).to_le_bytes());
    for weight in party_weights {
        hasher.update((*weight as u64).to_le_bytes());
    }
    hasher.update((party_offsets.len() as u64).to_le_bytes());
    for offset in party_offsets {
        hasher.update((*offset as u64).to_le_bytes());
    }
    hasher.update((evaluation_domain.len() as u64).to_le_bytes());
    for point in evaluation_domain {
        hasher.update(point.to_bytes_le());
    }

    hasher.update((deltas.len() as u64).to_le_bytes());
    for point in deltas {
        hasher.update(point.to_compressed());
    }
    hasher.update((inverse_public_keys.len() as u64).to_le_bytes());
    for point in inverse_public_keys {
        hasher.update(point.to_compressed());
    }
    hasher.update((gamma.len() as u64).to_le_bytes());
    for point in gamma {
        hasher.update(point.to_compressed());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use blstrs::{pairing, Gt};
    use group::{Curve, Group};

    use super::*;
    use crate::interpolation::lagrange_at_zero;

    #[test]
    fn crs_has_exact_paper_layout_and_serialized_size() {
        let parameters = setup(8).unwrap();
        assert_eq!(parameters.index_space_size(), 8);
        assert_eq!(parameters.g1_point_count(), 8);
        assert_eq!(parameters.g2_point_count(), 16);
        assert_eq!(parameters.g1_serialized_size(), 8 * 48);
        assert_eq!(parameters.g2_serialized_size(), 16 * 96);
        assert_eq!(parameters.serialized_size(), 8 * 48 + 16 * 96);

        for exponent in 1..=8 {
            assert!(parameters.g_power(exponent).is_ok());
        }
        for exponent in 1..=17 {
            assert_eq!(parameters.h_power(exponent).is_ok(), exponent != 9);
        }
        for offset in -8isize..=8 {
            assert_eq!(parameters.centered_power(offset).is_ok(), offset != 0);
        }

        // e(g_2, h_1) = e(g_1, h_2) checks that both source-group arrays use
        // the same hidden tau without exposing it.
        assert_eq!(
            pairing(
                &parameters.g_power(2).unwrap(),
                &parameters.h_power(1).unwrap()
            ),
            pairing(
                &parameters.g_power(1).unwrap(),
                &parameters.h_power(2).unwrap()
            )
        );
    }

    #[test]
    fn setup_rejects_invalid_or_overflowing_index_spaces() {
        assert!(matches!(setup(0), Err(Error::InvalidIndexSpace(0))));
        assert!(matches!(setup(1), Err(Error::InvalidIndexSpace(1))));
        assert!(matches!(
            setup(usize::MAX),
            Err(Error::ParameterSizeOverflow)
        ));
    }

    #[test]
    fn independent_crses_have_distinct_full_material_identifiers() {
        let first = setup(4).unwrap();
        let second = setup(4).unwrap();
        assert_ne!(first.crs_id(), second.crs_id());

        let encoded = bincode::serialize(&first).unwrap();
        let decoded: PublicParameters = bincode::deserialize(&encoded).unwrap();
        assert_eq!(decoded.crs_id(), first.crs_id());
        assert_eq!(decoded.g_powers, first.g_powers);
        assert_eq!(decoded.h_powers, first.h_powers);
    }

    #[test]
    fn weighted_key_material_has_expected_shapes_layout_and_sizes() {
        let parameters = setup(4).unwrap();
        let material = keygen(&parameters, &[2, 1, 3], 3).unwrap();
        let public_key = &material.public_key;

        assert_eq!(public_key.crs_id(), parameters.crs_id());
        assert_eq!(public_key.index_space_size(), 4);
        assert_eq!(public_key.threshold_weight(), 3);
        assert_eq!(public_key.required_weight(), 4);
        assert_eq!(public_key.total_weight(), 6);
        assert_eq!(public_key.party_count(), 3);
        assert_eq!(public_key.party_weights(), &[2, 1, 3]);
        assert_eq!(public_key.party_range(0).unwrap(), 0..2);
        assert_eq!(public_key.party_range(1).unwrap(), 2..3);
        assert_eq!(public_key.party_range(2).unwrap(), 3..6);
        assert_eq!(public_key.domain_points().len(), 6);
        assert_eq!(public_key.domain_size(), 8);
        assert_eq!(public_key.g1_point_count(), 4 + 3);
        assert_eq!(public_key.g2_point_count(), 4 * 6);
        assert_eq!(public_key.g1_serialized_size(), 7 * 48);
        assert_eq!(public_key.g2_serialized_size(), 24 * 96);
        assert_eq!(public_key.serialized_size(), 7 * 48 + 24 * 96);
        assert_eq!(material.party_keys.len(), 3);

        for index in 0..4 {
            assert_eq!(public_key.gamma_block(index).unwrap().len(), 6);
        }
        assert!(public_key.gamma_block(4).is_err());
        assert!(public_key.delta(4).is_err());
        assert!(public_key.inverse_key(3).is_err());
        assert!(public_key.party_range(3).is_err());

        for party in &material.party_keys {
            assert_eq!(party.setup_id(), public_key.setup_id());
            assert!(!bool::from(party.inverse_scalar().is_zero()));
            assert_eq!(
                party.public_inverse(),
                public_key.inverse_key(party.party_index).unwrap()
            );
            assert_eq!(
                party.public_inverse(),
                (G1Projective::generator() * party.inverse_scalar()).to_affine()
            );
        }
    }

    #[test]
    fn gamma_interpolation_recovers_the_indexed_mask() {
        let parameters = setup(4).unwrap();
        let material = keygen(&parameters, &[2, 1, 3], 3).unwrap();
        let public_key = &material.public_key;

        // Parties 0 and 2 contribute weight five, which exceeds t=3.
        let parties = [0usize, 2];
        let selected = parties
            .iter()
            .flat_map(|party| public_key.party_range(*party).unwrap())
            .collect::<Vec<_>>();
        let coefficients = lagrange_at_zero(public_key.domain_points(), &selected).unwrap();

        for index in 0..public_key.index_space_size() {
            let mut alpha = Gt::identity();
            let mut coefficient_offset = 0usize;
            for party in parties {
                let range = public_key.party_range(party).unwrap();
                let count = range.len();
                let opening_key = public_key.gamma_block(index).unwrap()[range]
                    .iter()
                    .zip(&coefficients[coefficient_offset..coefficient_offset + count])
                    .fold(G2Projective::identity(), |sum, (point, coefficient)| {
                        sum + G2Projective::from(*point) * coefficient
                    });
                alpha += pairing(
                    &public_key.inverse_key(party).unwrap(),
                    &opening_key.to_affine(),
                );
                coefficient_offset += count;
            }

            let auxiliary_index = if index == 0 { 1 } else { 0 };
            let offset = index as isize - auxiliary_index as isize;
            let expected = pairing(
                &public_key.delta(auxiliary_index).unwrap(),
                &parameters.centered_power(offset).unwrap(),
            );
            assert_eq!(alpha, expected, "index {index}");
        }
    }

    #[test]
    fn setup_identifier_hashes_full_committee_material() {
        let parameters = setup(4).unwrap();
        let first = keygen(&parameters, &[2, 1, 3], 3).unwrap();
        let second = keygen(&parameters, &[2, 1, 3], 3).unwrap();
        assert_eq!(first.public_key.crs_id(), second.public_key.crs_id());
        assert_ne!(first.public_key.setup_id(), second.public_key.setup_id());

        let encoded = bincode::serialize(&first).unwrap();
        let decoded: KeyMaterial = bincode::deserialize(&encoded).unwrap();
        assert_eq!(decoded.public_key.setup_id(), first.public_key.setup_id());
        assert_eq!(decoded.public_key.deltas, first.public_key.deltas);
        assert_eq!(
            decoded.public_key.inverse_public_keys,
            first.public_key.inverse_public_keys
        );
        assert_eq!(decoded.public_key.gamma, first.public_key.gamma);
        assert_eq!(decoded.party_keys.len(), first.party_keys.len());
    }

    #[test]
    fn keygen_rejects_invalid_weighted_committees() {
        let parameters = setup(4).unwrap();
        assert!(matches!(
            keygen(&parameters, &[], 0),
            Err(Error::EmptyCommittee)
        ));
        assert!(matches!(
            keygen(&parameters, &[2, 0, 1], 1),
            Err(Error::InvalidPartyWeight {
                party_index: 1,
                weight: 0
            })
        ));
        assert!(matches!(
            keygen(&parameters, &[2, 1], 3),
            Err(Error::InvalidCommittee { .. })
        ));
        assert!(matches!(
            keygen(&parameters, &[usize::MAX, 1], 0),
            Err(Error::TotalWeightOverflow)
        ));
    }

    #[test]
    fn one_virtual_share_uses_the_nonzero_singleton_domain() {
        let parameters = setup(2).unwrap();
        let material = keygen(&parameters, &[1], 0).unwrap();

        assert_eq!(material.public_key.domain_size(), 1);
        assert_eq!(material.public_key.domain_points(), &[Scalar::ONE]);
        assert_eq!(material.public_key.required_weight(), 1);
        assert_eq!(material.public_key.gamma_block(0).unwrap().len(), 1);
    }
}
