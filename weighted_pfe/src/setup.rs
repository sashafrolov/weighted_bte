//! Trusted setup and weighted Shamir key generation for Construction 5.

use std::{collections::HashSet, mem::size_of, ops::Range, sync::OnceLock};

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

const SETUP_ID_DOMAIN: &[u8] = b"WEIGHTED-PFE-SETUP-BLS12381-v2";
const PARTY_KEY_FORMAT_TAG: [u8; 8] = *b"WPFE-SK2";

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
    /// `vk[j,i] = [g_{j,i}]_2`, laid out as `i * N + j`.
    verification_keys: Box<[G2Affine]>,
    /// `D[w,i] = [Z(w) g_{owner(w),i}]_2`, laid out as `i * W + w`.
    d1: Box<[G2Affine]>,
    /// `U[w,i] = [Z(w) u_{owner(w),i}]_2`, laid out as `i * W + w`.
    d2: Box<[G2Affine]>,
    /// `V = [z (beta_{-1} + beta_0)]_2`.
    global_key: G2Affine,
}

#[derive(Clone, Debug, Serialize)]
pub struct PartySecretKey {
    format_tag: [u8; 8],
    pub party_index: usize,
    setup_id: [u8; 32],
    batch_size: usize,
    /// Public descriptor for the setup's affine root-of-unity alpha domain.
    alpha_shift_small: u64,
    /// The canonical Construction 5 secret key `rho_j`.
    rho: Scalar,
    /// Derived `g_{j,i} = 1 / (rho_j + alpha_i)` values used by the hot path.
    /// They are reproducible from `rho` and the setup's fixed alpha domain, so
    /// serde intentionally omits them from the canonical persisted key.
    #[serde(skip)]
    cached_fractions: OnceLock<Box<[Scalar]>>,
}

#[derive(Deserialize)]
struct EncodedPartySecretKey {
    format_tag: [u8; 8],
    party_index: usize,
    setup_id: [u8; 32],
    batch_size: usize,
    alpha_shift_small: u64,
    rho: Scalar,
}

impl<'de> Deserialize<'de> for PartySecretKey {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let encoded = EncodedPartySecretKey::deserialize(deserializer)?;
        if encoded.format_tag != PARTY_KEY_FORMAT_TAG {
            return Err(serde::de::Error::custom(
                "unsupported weighted-PFE party-key format",
            ));
        }
        Ok(Self {
            format_tag: encoded.format_tag,
            party_index: encoded.party_index,
            setup_id: encoded.setup_id,
            batch_size: encoded.batch_size,
            alpha_shift_small: encoded.alpha_shift_small,
            rho: encoded.rho,
            cached_fractions: OnceLock::new(),
        })
    }
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
    keygen_with_rng_and_masks(config, rng, None)
}

fn keygen_with_rng_and_masks<R: rand_core::RngCore + rand_core::CryptoRng>(
    config: KeygenConfig,
    rng: &mut R,
    fixed_global_masks: Option<(Scalar, Scalar)>,
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

    // Construction 5 permits fixed distinct alpha_i values and describes a
    // multiplicative root-of-unity choice. This affine alternative likewise
    // preserves a cyclic alpha_k-alpha_i structure for FFT convolutions.
    let (alpha_points, alpha_shift_small) = shifted_alpha_points(batch_size)?;

    // Construction 5 replaces the correlated values p_{-1}(x), p_0(x) with
    // directly sampled global masks. The rejection conditions keep both the
    // encryption key and the global decryption term nondegenerate.
    let (beta_minus_one, beta_zero) =
        fixed_global_masks.unwrap_or_else(|| sample_global_masks(rng));
    assert!(
        global_masks_are_valid(beta_minus_one, beta_zero),
        "fixed Construction 5 global masks must satisfy the setup constraints"
    );

    // Rejection sampling concretely instantiates the paper's requirement that
    // every rho_j+alpha_i be globally distinct and every hidden denominator
    // be nonzero.  Collisions occur only with negligible probability, but the
    // explicit check keeps the implementation valid for deterministic RNGs.
    let mut occupied_labels = HashSet::with_capacity(dimensions.verification_count);
    let mut party_rhos = Vec::with_capacity(party_count);
    for _ in 0..party_count {
        loop {
            let rho = Scalar::random(&mut *rng);
            let labels = alpha_points
                .iter()
                .map(|alpha| rho + alpha)
                .collect::<Vec<_>>();
            if labels.iter().any(|label| bool::from(label.is_zero()))
                || labels
                    .iter()
                    .any(|label| occupied_labels.contains(&label.to_bytes_le()))
            {
                continue;
            }
            for label in labels {
                occupied_labels.insert(label.to_bytes_le());
            }
            party_rhos.push(rho);
            break;
        }
    }

    let mut g_values = Vec::with_capacity(dimensions.verification_count);
    for rho in &party_rhos {
        g_values.extend(alpha_points.iter().map(|alpha| rho + alpha));
    }
    debug_assert!(g_values.iter().all(|value| !bool::from(value.is_zero())));
    g_values.iter_mut().batch_invert();

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
        .zip(g_values.par_chunks(batch_size))
        .for_each(|(party_u, party_g)| {
            for slot in 0..batch_size {
                party_u[slot] = party_g[slot].square()
                    + beta_minus_one * alpha_plus_one_inverses[slot]
                    + beta_zero * alpha_inverses[slot];
            }
        });

    // Verification keys are small compared with D/U, but retaining the same
    // slot-major layout makes randomized all-party verification contiguous.
    let mut verification_keys = Vec::with_capacity(dimensions.verification_count);
    for slot in 0..batch_size {
        let scalars = (0..party_count)
            .map(|party| g_values[party * batch_size + slot])
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
                        *share * g_values[owners[virtual_index] * batch_size + slot]
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

    let global_key =
        (G2Projective::generator() * (master_secret * (beta_minus_one + beta_zero))).into();
    let encryption_element = Gt::generator() * (master_secret * beta_zero);
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

    let party_keys = party_rhos
        .into_iter()
        .zip(g_values.chunks(batch_size))
        .enumerate()
        .map(|(party_index, (rho, party_fractions))| PartySecretKey {
            format_tag: PARTY_KEY_FORMAT_TAG,
            party_index,
            setup_id,
            batch_size,
            alpha_shift_small,
            rho,
            cached_fractions: OnceLock::from(party_fractions.to_vec().into_boxed_slice()),
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

    /// Number of secret field elements in Construction 5's canonical key.
    pub fn scalar_count(&self) -> usize {
        1
    }

    pub(crate) fn fractions(&self) -> Result<&[Scalar]> {
        if self.cached_fractions.get().is_none() {
            let derived = derive_fractions(self.rho, self.batch_size, self.alpha_shift_small)?
                .into_boxed_slice();
            // Another thread may populate the cache between `get` and `set`.
            let _ = self.cached_fractions.set(derived);
        }
        Ok(self
            .cached_fractions
            .get()
            .expect("the fraction cache was initialized above"))
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

fn derive_fractions(rho: Scalar, batch_size: usize, alpha_shift_small: u64) -> Result<Vec<Scalar>> {
    let mut fractions = root_domain_prefix(batch_size, batch_size)?;
    let alpha_shift = Scalar::from(alpha_shift_small);
    fractions
        .iter_mut()
        .for_each(|root| *root += alpha_shift + rho);
    if fractions.iter().any(|value| bool::from(value.is_zero())) {
        return Err(Error::InvalidSecretKey);
    }
    fractions.iter_mut().batch_invert();
    Ok(fractions)
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

fn sample_global_masks<R: rand_core::RngCore + rand_core::CryptoRng>(
    rng: &mut R,
) -> (Scalar, Scalar) {
    loop {
        let beta_minus_one = Scalar::random(&mut *rng);
        let beta_zero = Scalar::random(&mut *rng);
        if global_masks_are_valid(beta_minus_one, beta_zero) {
            return (beta_minus_one, beta_zero);
        }
    }
}

fn global_masks_are_valid(beta_minus_one: Scalar, beta_zero: Scalar) -> bool {
    !bool::from(beta_zero.is_zero()) && !bool::from((beta_minus_one + beta_zero).is_zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use blstrs::G1Affine;
    use group::{prime::PrimeCurveAffine, Curve};

    use crate::{
        accept_decryption_shares, encrypt, open_batch, partial_decrypt, precompute_batch,
        prepare_decryption, validate_batch, verify_decryption_share, CauchyKernel,
    };

    #[test]
    fn key_material_has_the_paper_dimensions_and_one_scalar_party_keys() {
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
        assert!(material.party_keys.iter().all(|key| key.scalar_count() == 1
            && key
                .cached_fractions
                .get()
                .is_some_and(|cache| cache.len() == 8)));
        assert_eq!(key.d1_slot(7).unwrap().len(), 6);
        assert_eq!(key.d2_slot(7).unwrap().len(), 6);
        assert_eq!(key.verification_slot(7).unwrap().len(), 3);
    }

    #[test]
    fn party_shifts_and_verification_keys_match_derived_fractions() {
        let material = keygen(16, &[2, 1, 4], 3).unwrap();
        let key = &material.decryption_key;
        let mut seen = HashSet::new();
        for alpha in key.alpha_points() {
            assert!(!bool::from(alpha.is_zero()));
            assert_ne!(*alpha, -Scalar::ONE);
            assert!(seen.insert(alpha.to_bytes_le()));
        }
        let mut labels = HashSet::new();
        for party in &material.party_keys {
            let fractions = party.fractions().unwrap();
            for (slot, fraction) in fractions.iter().enumerate() {
                let label = party.rho + key.alpha_points()[slot];
                assert!(!bool::from(label.is_zero()));
                assert!(labels.insert(label.to_bytes_le()));
                assert_eq!(*fraction, label.invert().unwrap());
                assert_eq!(
                    key.verification_key(party.party_index, slot).unwrap(),
                    (G2Projective::generator() * fraction).to_affine()
                );
            }
        }
    }

    #[test]
    fn global_mask_constraints_match_construction_five() {
        assert!(global_masks_are_valid(Scalar::ZERO, Scalar::ONE));
        assert!(!global_masks_are_valid(Scalar::ONE, Scalar::ZERO));
        assert!(!global_masks_are_valid(Scalar::ONE, -Scalar::ONE));

        for _ in 0..32 {
            let (beta_minus_one, beta_zero) = sample_global_masks(&mut rand_core::OsRng);
            assert!(global_masks_are_valid(beta_minus_one, beta_zero));
        }
    }

    #[test]
    fn construction_five_round_trips_with_beta_minus_one_zero() {
        let config = KeygenConfig::new(4, &[1, 2, 3], 2);
        let material = keygen_with_rng_and_masks(
            config,
            &mut rand_core::OsRng,
            Some((Scalar::ZERO, Scalar::ONE)),
        )
        .unwrap();
        let messages = (1..=4)
            .map(|value| Gt::generator() * Scalar::from(value))
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
        assert_eq!(
            open_batch(
                &material.decryption_key,
                &kernel,
                &decryption,
                &accepted,
                &batch,
                &ciphertexts,
                &batch_precomputation,
            )
            .unwrap(),
            messages
        );
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
        assert!(decoded
            .party_keys
            .iter()
            .all(|party| party.cached_fractions.get().is_none()));

        let ciphertexts = (1..=4)
            .map(|value| {
                encrypt(
                    &decoded.encryption_key,
                    Gt::generator() * Scalar::from(value),
                )
            })
            .collect::<Vec<_>>();
        let batch = validate_batch(&decoded.encryption_key, &ciphertexts).unwrap();
        let warm_share = partial_decrypt(&first.party_keys[0], &batch).unwrap();
        let share = partial_decrypt(&decoded.party_keys[0], &batch).unwrap();
        assert_eq!(share.sigma, warm_share.sigma);
        assert!(verify_decryption_share(&decoded.decryption_key, &batch, &share).unwrap());
        assert!(decoded.party_keys[0].cached_fractions.get().is_some());
        assert!(decoded.party_keys[1..]
            .iter()
            .all(|party| party.cached_fractions.get().is_none()));

        for (decoded_party, original_party) in decoded.party_keys.iter().zip(&first.party_keys) {
            assert_eq!(decoded_party.rho, original_party.rho);
            assert_eq!(
                decoded_party.alpha_shift_small,
                original_party.alpha_shift_small
            );
            assert_eq!(
                decoded_party.fractions().unwrap(),
                original_party.fractions().unwrap()
            );
            assert!(decoded_party.cached_fractions.get().is_some());
        }

        let larger = keygen(16, &[1, 2, 3], 2).unwrap();
        assert_eq!(
            bincode::serialized_size(&first.party_keys[0]).unwrap(),
            bincode::serialized_size(&larger.party_keys[0]).unwrap()
        );

        #[derive(Serialize)]
        struct LegacyPartySecretKey<'a> {
            party_index: usize,
            setup_id: [u8; 32],
            batch_size: usize,
            fractions: &'a [Scalar],
        }
        let legacy = LegacyPartySecretKey {
            party_index: first.party_keys[0].party_index,
            setup_id: first.party_keys[0].setup_id,
            batch_size: first.party_keys[0].batch_size,
            fractions: first.party_keys[0].fractions().unwrap(),
        };
        let legacy_encoding = bincode::serialize(&legacy).unwrap();
        assert!(bincode::deserialize::<PartySecretKey>(&legacy_encoding).is_err());
    }

    #[test]
    fn malformed_party_key_with_fraction_pole_is_rejected() {
        let material = keygen(4, &[1, 2], 1).unwrap();
        let mut party = material.party_keys[0].clone();
        party.rho = -material.decryption_key.alpha_points()[0];
        party.cached_fractions = OnceLock::new();
        assert!(matches!(party.fractions(), Err(Error::InvalidSecretKey)));
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
