//! PFE validation, threshold pre-decryption, FFT convolutions, and opening.

use blstrs::{Bls12, G1Affine, G1Projective, G2Affine, G2Prepared, G2Projective, Gt, Scalar};
use ff::{BatchInvert, Field};
use group::{prime::PrimeCurveAffine, Curve, Group};
use pairing::{MillerLoopResult, MultiMillerLoop};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

use crate::{
    encryption::Ciphertext,
    error::{Error, Result},
    fft::{mul_gt_signed_small, Radix2Domain},
    proof,
    setup::{extra_index_data, index_points, EncryptionKey, PublicDecryptionKey, ServerSecretKey},
};

const BATCH_DIGEST_DOMAIN: &[u8] = b"PFE-ORDERED-BATCH-BLS12381-v1";

#[derive(Clone, Debug)]
pub struct ValidatedBatch {
    batch_size: usize,
    digest: [u8; 32],
    first_affine: Box<[G1Affine]>,
    first_projective: Box<[G1Projective]>,
    second_projective: Box<[G1Projective]>,
}

#[derive(Clone, Debug)]
pub struct PartialFractionKernel {
    batch_size: usize,
    domain: Radix2Domain,
    frequency_coefficients: Box<[i64]>,
    gamma_inverse_small: i64,
    /// `2 B * s1`; the `2 B` cancels the scaled public key in the pairing.
    scaled_s1: Box<[Scalar]>,
    /// `[p_i(x)/(2B)]_2`.
    scaled_fraction_keys: Box<[G2Prepared]>,
    /// `[s2_i]_2`.
    scaled_generators: Box<[G2Prepared]>,
    /// The fused, setup-only third G2 operand for each opening.
    final_mixes: Box<[G2Prepared]>,
}

#[derive(Clone, Debug)]
pub struct BatchPrecomputation {
    batch_size: usize,
    digest: [u8; 32],
    /// `2B * (z_i/gamma - 1)` times the unscaled cyclic G1 convolution.
    weighted_first: Box<[G1Projective]>,
    /// `(z_i/gamma - 1)` times the cyclic convolution of scaled pairings.
    weighted_pairings: Box<[Gt]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecryptionShare {
    pub server_index: usize,
    pub pre_decryption_key: G1Projective,
    pub batch_digest: [u8; 32],
}

pub fn validate_batch(
    encryption_key: &EncryptionKey,
    ciphertexts: &[Ciphertext],
) -> Result<ValidatedBatch> {
    ValidatedBatch::from_ciphertexts(encryption_key, ciphertexts, true)
}

impl ValidatedBatch {
    pub fn verify(encryption_key: &EncryptionKey, ciphertexts: &[Ciphertext]) -> Result<Self> {
        Self::from_ciphertexts(encryption_key, ciphertexts, true)
    }

    /// Construct the batch after proofs have been checked by an earlier stage.
    ///
    /// This is used to time the paper's proof checking and core preprocessing
    /// phases independently.
    pub fn proofs_preverified(
        encryption_key: &EncryptionKey,
        ciphertexts: &[Ciphertext],
    ) -> Result<Self> {
        Self::from_ciphertexts(encryption_key, ciphertexts, false)
    }

    fn from_ciphertexts(
        encryption_key: &EncryptionKey,
        ciphertexts: &[Ciphertext],
        verify_proofs: bool,
    ) -> Result<Self> {
        if ciphertexts.is_empty() {
            return Err(Error::BatchIsEmpty);
        }
        if ciphertexts.len() != encryption_key.batch_size {
            return Err(Error::MismatchedBatchSize {
                expected: encryption_key.batch_size,
                actual: ciphertexts.len(),
            });
        }

        let first_affine = ciphertexts
            .iter()
            .map(|ciphertext| ciphertext.first)
            .collect::<Vec<_>>();
        let first_projective = first_affine
            .iter()
            .map(|point| G1Projective::from(*point))
            .collect::<Vec<_>>();
        let second_projective = ciphertexts
            .iter()
            .map(|ciphertext| G1Projective::from(ciphertext.second))
            .collect::<Vec<_>>();

        if verify_proofs {
            let proofs = ciphertexts
                .iter()
                .map(|ciphertext| ciphertext.proof.clone())
                .collect::<Vec<_>>();
            let mut rng = rand_core::OsRng;
            if !proof::verify_batch(
                encryption_key.relation_base,
                &first_projective,
                &second_projective,
                &proofs,
                &mut rng,
            ) {
                return Err(Error::InvalidProof);
            }
        }

        Ok(Self {
            batch_size: ciphertexts.len(),
            digest: batch_digest(ciphertexts),
            first_affine: first_affine.into_boxed_slice(),
            first_projective: first_projective.into_boxed_slice(),
            second_projective: second_projective.into_boxed_slice(),
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

impl PartialFractionKernel {
    /// Perform all reusable domain and G2 preprocessing.
    pub fn new(decryption_key: &PublicDecryptionKey) -> Result<Self> {
        let batch_size = decryption_key.batch_size();
        let domain = Radix2Domain::new(batch_size)?;
        let (_, gamma_inverse_small_u64) = extra_index_data(batch_size)?;
        let gamma_inverse_small = i64::try_from(gamma_inverse_small_u64)
            .map_err(|_| Error::InvalidBatchSize(batch_size))?;
        let gamma_inverse = Scalar::from(gamma_inverse_small_u64);
        let gamma =
            Option::<Scalar>::from(gamma_inverse.invert()).expect("gamma inverse is nonzero");
        let points = index_points(batch_size)?;

        // If d[0]=0 and d[r]=(1-w^{-r})^{-1}, then
        // FFT(d)[k]=(B-1)/2-k. Multiplying by E[k]=B-1-2k and
        // using an unnormalized inverse transform gives 2B times the desired
        // convolution. The small signed E coefficients avoid B full-width
        // group scalar multiplications in every online convolution.
        let frequency_coefficients = (0..batch_size)
            .map(|index| batch_size as i64 - 1 - 2 * index as i64)
            .collect::<Vec<_>>();

        let half = Option::<Scalar>::from(Scalar::from(2u64).invert())
            .expect("two is invertible in the scalar field");
        let half_b_plus_one = Scalar::from((batch_size + 1) as u64) * half;
        let mut point_minus_gamma = points
            .iter()
            .map(|point| *point - gamma)
            .collect::<Vec<_>>();
        point_minus_gamma.iter_mut().batch_invert();
        let opening_coefficients = points
            .iter()
            .zip(point_minus_gamma.iter())
            .map(|(point, inverse_difference)| {
                let point_inverse =
                    Option::<Scalar>::from(point.invert()).expect("roots are nonzero");
                point_inverse * half_b_plus_one + inverse_difference
            })
            .collect::<Vec<_>>();

        let two_b = Scalar::from((2 * batch_size) as u64);
        let inverse_two_b =
            Option::<Scalar>::from(two_b.invert()).expect("2B is nonzero in the scalar field");
        let original_key_sum = G2Projective::from(decryption_key.fraction_key_sum());
        let scaled_key_projective = decryption_key
            .fraction_keys()
            .par_iter()
            .map(|key| G2Projective::from(*key) * inverse_two_b)
            .collect::<Vec<_>>();

        // With input DK/(2B), the unnormalized closed-form convolution is
        // exactly `(z/gamma - 1) * raw_convolution(DK)`.
        let weighted_keys = weighted_convolution_group(
            &domain,
            &frequency_coefficients,
            gamma_inverse_small,
            scaled_key_projective.clone(),
        );

        let mut scaled_s1 = Vec::with_capacity(batch_size);
        let mut scaled_generators_projective = Vec::with_capacity(batch_size);
        let mut final_mixes_projective = Vec::with_capacity(batch_size);
        for slot in 0..batch_size {
            let point = points[slot];
            let s2 = point * gamma_inverse;
            let q = s2 - Scalar::ONE;
            let s1 = point * q;
            let s1_bar = two_b * s1;
            scaled_s1.push(s1_bar);
            scaled_generators_projective.push(G2Projective::generator() * s2);

            // Absorb `opening_coefficients[slot] * ct1` into the static third
            // pairing operand. This saves one variable-base G1 multiplication
            // per ciphertext during opening.
            let proof_term = scaled_key_projective[slot] * (s1_bar * opening_coefficients[slot]);
            final_mixes_projective.push(weighted_keys[slot] - original_key_sum * s2 - proof_term);
        }

        let mut all_prepared_projective = Vec::with_capacity(3 * batch_size);
        all_prepared_projective.extend_from_slice(&scaled_key_projective);
        all_prepared_projective.extend_from_slice(&scaled_generators_projective);
        all_prepared_projective.extend_from_slice(&final_mixes_projective);
        let mut all_prepared_affine = vec![G2Affine::default(); all_prepared_projective.len()];
        G2Projective::batch_normalize(&all_prepared_projective, &mut all_prepared_affine);
        let mut prepared = all_prepared_affine
            .into_par_iter()
            .map(G2Prepared::from)
            .collect::<Vec<_>>();
        let final_mixes = prepared.split_off(2 * batch_size);
        let scaled_generators = prepared.split_off(batch_size);
        let scaled_fraction_keys = prepared;

        Ok(Self {
            batch_size,
            domain,
            frequency_coefficients: frequency_coefficients.into_boxed_slice(),
            gamma_inverse_small,
            scaled_s1: scaled_s1.into_boxed_slice(),
            scaled_fraction_keys: scaled_fraction_keys.into_boxed_slice(),
            scaled_generators: scaled_generators.into_boxed_slice(),
            final_mixes: final_mixes.into_boxed_slice(),
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

pub fn precompute_batch(
    kernel: &PartialFractionKernel,
    batch: &ValidatedBatch,
) -> Result<BatchPrecomputation> {
    ensure_kernel_batch(kernel, batch)?;

    let (weighted_first, weighted_pairings) = rayon::join(
        || {
            weighted_convolution_group(
                &kernel.domain,
                &kernel.frequency_coefficients,
                kernel.gamma_inverse_small,
                batch.first_projective.to_vec(),
            )
        },
        || {
            let pairings = batch
                .first_affine
                .par_iter()
                .zip(kernel.scaled_fraction_keys.par_iter())
                .map(|(first, key)| {
                    Bls12::multi_miller_loop(&[(first, key)]).final_exponentiation()
                })
                .collect::<Vec<_>>();
            weighted_convolution_gt(
                &kernel.domain,
                &kernel.frequency_coefficients,
                kernel.gamma_inverse_small,
                pairings,
            )
        },
    );

    Ok(BatchPrecomputation {
        batch_size: batch.batch_size,
        digest: batch.digest,
        weighted_first: weighted_first.into_boxed_slice(),
        weighted_pairings: weighted_pairings.into_boxed_slice(),
    })
}

pub fn partial_decrypt(
    server_key: &ServerSecretKey,
    batch: &ValidatedBatch,
) -> Result<DecryptionShare> {
    if server_key.batch_size() != batch.batch_size {
        return Err(Error::MismatchedBatchSize {
            expected: server_key.batch_size(),
            actual: batch.batch_size,
        });
    }

    Ok(DecryptionShare {
        server_index: server_key.server_index,
        pre_decryption_key: G1Projective::multi_exp(
            &batch.first_projective,
            server_key.fraction_shares(),
        ),
        batch_digest: batch.digest,
    })
}

pub fn combine_shares(
    decryption_key: &PublicDecryptionKey,
    batch: &ValidatedBatch,
    shares: &[DecryptionShare],
) -> Result<G1Projective> {
    validate_share_set(decryption_key, batch, shares)?;

    let points = shares
        .iter()
        .map(|share| decryption_key.server_domain()[share.server_index])
        .collect::<Vec<_>>();
    let coefficients = lagrange_at_zero(&points);
    let bases = shares
        .iter()
        .map(|share| share.pre_decryption_key)
        .collect::<Vec<_>>();
    Ok(G1Projective::multi_exp(&bases, &coefficients))
}

pub fn combine_shares_checked(
    decryption_key: &PublicDecryptionKey,
    batch: &ValidatedBatch,
    shares: &[DecryptionShare],
) -> Result<G1Projective> {
    let optimistic = combine_shares(decryption_key, batch, shares)?;
    if verify_combined_share(decryption_key, batch, optimistic)? {
        return Ok(optimistic);
    }
    if !decryption_key.has_share_commitments() {
        return Err(Error::InvalidShare);
    }

    let valid_shares = shares
        .par_iter()
        .map(|share| {
            verify_decryption_share(decryption_key, batch, share)
                .map(|valid| valid.then_some(share.clone()))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let recovered = combine_shares(decryption_key, batch, &valid_shares)?;
    if verify_combined_share(decryption_key, batch, recovered)? {
        Ok(recovered)
    } else {
        Err(Error::InvalidShare)
    }
}

pub fn verify_decryption_share(
    decryption_key: &PublicDecryptionKey,
    batch: &ValidatedBatch,
    share: &DecryptionShare,
) -> Result<bool> {
    if share.batch_digest != batch.digest {
        return Ok(false);
    }
    if share.server_index >= decryption_key.server_count() {
        return Err(Error::InvalidServerIndex(share.server_index));
    }

    let mut left = batch.first_affine.to_vec();
    left.push(G1Affine::from(-share.pre_decryption_key));
    let mut right = (0..batch.batch_size)
        .into_par_iter()
        .map(|slot| {
            decryption_key
                .share_commitment(share.server_index, slot)
                .map(G2Prepared::from)
        })
        .collect::<Result<Vec<_>>>()?;
    right.push(G2Prepared::from(G2Affine::generator()));
    Ok(pairing_product_is_identity(&left, &right))
}

/// BTX-style optimistic aggregate verification for PFE's reconstructed key.
pub fn verify_combined_share(
    decryption_key: &PublicDecryptionKey,
    batch: &ValidatedBatch,
    pre_decryption_key: G1Projective,
) -> Result<bool> {
    ensure_key_batch(decryption_key, batch)?;
    let mut left = batch.first_affine.to_vec();
    left.push(G1Affine::from(-pre_decryption_key));
    let mut right = decryption_key
        .fraction_keys()
        .par_iter()
        .copied()
        .map(G2Prepared::from)
        .collect::<Vec<_>>();
    right.push(G2Prepared::from(G2Affine::generator()));
    Ok(pairing_product_is_identity(&left, &right))
}

pub fn open_batch(
    kernel: &PartialFractionKernel,
    batch: &ValidatedBatch,
    ciphertexts: &[Ciphertext],
    precomputation: &BatchPrecomputation,
    pre_decryption_key: G1Projective,
) -> Result<Vec<Gt>> {
    ensure_kernel_batch(kernel, batch)?;
    if ciphertexts.len() != batch.batch_size {
        return Err(Error::MismatchedBatchSize {
            expected: batch.batch_size,
            actual: ciphertexts.len(),
        });
    }
    if precomputation.batch_size != batch.batch_size || precomputation.digest != batch.digest {
        return Err(Error::InvalidShare);
    }

    // Batch-normalize all online G1 pairing inputs together. The second base
    // is `W1 - 2B*s1*(ct2-sbk)`; the c_j*ct1 term has already been folded
    // into `final_mixes` during fixed preprocessing.
    let mut pairing_bases = Vec::with_capacity(3 * batch.batch_size);
    for slot in 0..batch.batch_size {
        pairing_bases.push(batch.second_projective[slot]);
        pairing_bases.push(
            precomputation.weighted_first[slot]
                - (batch.second_projective[slot] - pre_decryption_key) * kernel.scaled_s1[slot],
        );
        pairing_bases.push(batch.first_projective[slot]);
    }
    let mut pairing_bases_affine = vec![G1Affine::default(); pairing_bases.len()];
    G1Projective::batch_normalize(&pairing_bases, &mut pairing_bases_affine);

    let masks = pairing_bases_affine
        .par_chunks_exact(3)
        .enumerate()
        .map(|(slot, bases)| {
            let paired = Bls12::multi_miller_loop(&[
                (&bases[0], &kernel.scaled_generators[slot]),
                (&bases[1], &kernel.scaled_fraction_keys[slot]),
                (&bases[2], &kernel.final_mixes[slot]),
            ])
            .final_exponentiation();
            paired - precomputation.weighted_pairings[slot]
        })
        .collect::<Vec<_>>();

    Ok(ciphertexts
        .iter()
        .zip(masks)
        .map(|(ciphertext, mask)| ciphertext.third - mask)
        .collect())
}

fn weighted_convolution_group<G>(
    domain: &Radix2Domain,
    frequency_coefficients: &[i64],
    gamma_inverse_small: i64,
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
            *value =
                mul_group_signed_small(spectrum[(index + 1) % spectrum.len()], gamma_inverse_small)
                    - spectrum[index];
        });
    domain.ifft_unscaled(&mut remapped);
    remapped
}

fn weighted_convolution_gt(
    domain: &Radix2Domain,
    frequency_coefficients: &[i64],
    gamma_inverse_small: i64,
    mut values: Vec<Gt>,
) -> Vec<Gt> {
    debug_assert_eq!(values.len(), domain.size());
    domain.fft_gt(&mut values);
    values
        .par_iter_mut()
        .zip(frequency_coefficients.par_iter())
        .for_each(|(value, coefficient)| *value = mul_gt_signed_small(*value, *coefficient));

    let spectrum = values;
    let mut remapped = vec![Gt::identity(); spectrum.len()];
    remapped
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, value)| {
            *value =
                mul_gt_signed_small(spectrum[(index + 1) % spectrum.len()], gamma_inverse_small)
                    - spectrum[index];
        });
    domain.ifft_gt_unscaled(&mut remapped);
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

fn validate_share_set(
    decryption_key: &PublicDecryptionKey,
    batch: &ValidatedBatch,
    shares: &[DecryptionShare],
) -> Result<()> {
    ensure_key_batch(decryption_key, batch)?;
    if shares.len() < decryption_key.required_shares() {
        return Err(Error::InsufficientShares {
            supplied: shares.len(),
            required: decryption_key.required_shares(),
        });
    }

    let mut seen = HashSet::with_capacity(shares.len());
    for share in shares {
        if share.batch_digest != batch.digest {
            return Err(Error::InvalidShare);
        }
        if share.server_index >= decryption_key.server_count() {
            return Err(Error::InvalidServerIndex(share.server_index));
        }
        if !seen.insert(share.server_index) {
            return Err(Error::DuplicateServerIndex(share.server_index));
        }
    }
    Ok(())
}

fn lagrange_at_zero(points: &[Scalar]) -> Vec<Scalar> {
    let mut numerators = Vec::with_capacity(points.len());
    let mut denominators = Vec::with_capacity(points.len());

    for (j, point_j) in points.iter().enumerate() {
        let mut numerator = Scalar::ONE;
        let mut denominator = Scalar::ONE;
        for (k, point_k) in points.iter().enumerate() {
            if j != k {
                numerator *= -*point_k;
                denominator *= *point_j - point_k;
            }
        }
        numerators.push(numerator);
        denominators.push(denominator);
    }

    denominators.iter_mut().batch_invert();
    numerators
        .into_iter()
        .zip(denominators)
        .map(|(numerator, denominator_inverse)| numerator * denominator_inverse)
        .collect()
}

fn pairing_product_is_identity(left: &[G1Affine], right: &[G2Prepared]) -> bool {
    debug_assert_eq!(left.len(), right.len());
    let terms = left.iter().zip(right.iter()).collect::<Vec<_>>();
    Bls12::multi_miller_loop(&terms).final_exponentiation() == Gt::identity()
}

fn ensure_key_batch(decryption_key: &PublicDecryptionKey, batch: &ValidatedBatch) -> Result<()> {
    if decryption_key.batch_size() != batch.batch_size {
        return Err(Error::MismatchedBatchSize {
            expected: decryption_key.batch_size(),
            actual: batch.batch_size,
        });
    }
    Ok(())
}

fn ensure_kernel_batch(kernel: &PartialFractionKernel, batch: &ValidatedBatch) -> Result<()> {
    if kernel.batch_size != batch.batch_size {
        return Err(Error::MismatchedBatchSize {
            expected: kernel.batch_size,
            actual: batch.batch_size,
        });
    }
    Ok(())
}

fn batch_digest(ciphertexts: &[Ciphertext]) -> [u8; 32] {
    let mut transcript = Vec::with_capacity(8 + ciphertexts.len() * 900);
    transcript.extend_from_slice(&(ciphertexts.len() as u64).to_le_bytes());
    for ciphertext in ciphertexts {
        ciphertext.append_canonical(&mut transcript);
    }

    let mut hasher = Sha256::new();
    hasher.update(BATCH_DIGEST_DOMAIN);
    hasher.update(transcript);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{encrypt, keygen};

    fn sample_batch(
        batch_size: usize,
        server_count: usize,
        threshold: usize,
    ) -> (
        crate::KeyMaterial,
        Vec<Gt>,
        Vec<Ciphertext>,
        ValidatedBatch,
        PartialFractionKernel,
    ) {
        let material = keygen(batch_size, server_count, threshold).unwrap();
        let messages = (0..batch_size)
            .map(|index| Gt::generator() * Scalar::from((index + 11) as u64))
            .collect::<Vec<_>>();
        let ciphertexts = messages
            .iter()
            .map(|message| encrypt(&material.encryption_key, *message))
            .collect::<Vec<_>>();
        let batch = validate_batch(&material.encryption_key, &ciphertexts).unwrap();
        let kernel = PartialFractionKernel::new(&material.decryption_key).unwrap();
        (material, messages, ciphertexts, batch, kernel)
    }

    #[test]
    fn optimized_convolution_matches_literal_denominators() {
        for batch_size in [1usize, 2, 4, 8, 16] {
            let domain = Radix2Domain::new(batch_size).unwrap();
            let (gamma, gamma_inverse_small_u64) = extra_index_data(batch_size).unwrap();
            let gamma_inverse_small = gamma_inverse_small_u64 as i64;
            let coefficients = (0..batch_size)
                .map(|index| batch_size as i64 - 1 - 2 * index as i64)
                .collect::<Vec<_>>();
            let points = index_points(batch_size).unwrap();
            let values = (0..batch_size)
                .map(|index| G1Projective::generator() * Scalar::from((index + 3) as u64))
                .collect::<Vec<_>>();
            let actual = weighted_convolution_group(
                &domain,
                &coefficients,
                gamma_inverse_small,
                values.clone(),
            );
            let two_b = Scalar::from((2 * batch_size) as u64);

            for output in 0..batch_size {
                let mut raw = G1Projective::identity();
                for (input, value) in values.iter().enumerate() {
                    if output == input {
                        continue;
                    }
                    let denominator_inverse = Option::<Scalar>::from(
                        (Scalar::ONE
                            - crate::fft::scalar_pow(
                                domain.generator_inverse(),
                                ((output + batch_size - input) % batch_size) as u64,
                            ))
                        .invert(),
                    )
                    .unwrap();
                    raw += *value * denominator_inverse;
                }
                let q = points[output] * Scalar::from(gamma_inverse_small_u64) - Scalar::ONE;
                assert_eq!(actual[output], raw * (two_b * q));
                assert_eq!(
                    q,
                    (points[output] - gamma) * Scalar::from(gamma_inverse_small_u64)
                );
            }
        }
    }

    #[test]
    fn threshold_protocol_round_trip() {
        let (material, messages, ciphertexts, batch, kernel) = sample_batch(8, 5, 2);
        let precomputation = precompute_batch(&kernel, &batch).unwrap();
        let shares = material.server_keys[..3]
            .iter()
            .map(|key| partial_decrypt(key, &batch).unwrap())
            .collect::<Vec<_>>();
        let combined = combine_shares(&material.decryption_key, &batch, &shares).unwrap();
        assert!(verify_combined_share(&material.decryption_key, &batch, combined).unwrap());
        let opened = open_batch(&kernel, &batch, &ciphertexts, &precomputation, combined).unwrap();
        assert_eq!(opened, messages);
    }

    #[test]
    fn round_trips_all_small_power_of_two_sizes_and_identity_messages() {
        for batch_size in [1usize, 2, 4, 8, 16] {
            let material = keygen(batch_size, 3, 1).unwrap();
            let messages = (0..batch_size)
                .map(|index| {
                    if index % 2 == 0 {
                        Gt::identity()
                    } else {
                        Gt::generator() * Scalar::from((index + 1) as u64)
                    }
                })
                .collect::<Vec<_>>();
            let ciphertexts = messages
                .iter()
                .map(|message| encrypt(&material.encryption_key, *message))
                .collect::<Vec<_>>();
            let batch = validate_batch(&material.encryption_key, &ciphertexts).unwrap();
            let kernel = PartialFractionKernel::new(&material.decryption_key).unwrap();
            let precomputation = precompute_batch(&kernel, &batch).unwrap();
            let shares = material.server_keys[..2]
                .iter()
                .map(|key| partial_decrypt(key, &batch).unwrap())
                .collect::<Vec<_>>();
            let combined = combine_shares(&material.decryption_key, &batch, &shares).unwrap();
            let opened =
                open_batch(&kernel, &batch, &ciphertexts, &precomputation, combined).unwrap();
            assert_eq!(opened, messages, "round trip failed for B={batch_size}");
        }
    }

    #[test]
    fn fused_opening_matches_literal_construction_one() {
        let (material, _, ciphertexts, batch, kernel) = sample_batch(4, 5, 2);
        let precomputation = precompute_batch(&kernel, &batch).unwrap();
        let shares = [0usize, 2, 4]
            .iter()
            .map(|index| partial_decrypt(&material.server_keys[*index], &batch).unwrap())
            .collect::<Vec<_>>();
        let combined = combine_shares(&material.decryption_key, &batch, &shares).unwrap();
        let optimized =
            open_batch(&kernel, &batch, &ciphertexts, &precomputation, combined).unwrap();

        let points = index_points(batch.batch_size).unwrap();
        let (gamma, gamma_inverse_small) = extra_index_data(batch.batch_size).unwrap();
        let gamma_inverse = Scalar::from(gamma_inverse_small);
        let half = Option::<Scalar>::from(Scalar::from(2u64).invert()).unwrap();
        let half_b_plus_one = Scalar::from((batch.batch_size + 1) as u64) * half;
        let key_sum = material.decryption_key.fraction_key_sum();
        let generator_prepared = G2Prepared::from(G2Affine::generator());

        let tm = (0..batch.batch_size)
            .map(|slot| {
                let key = G2Prepared::from(material.decryption_key.fraction_keys()[slot]);
                Bls12::multi_miller_loop(&[(&batch.first_affine[slot], &key)])
                    .final_exponentiation()
            })
            .collect::<Vec<_>>();

        let literal = (0..batch.batch_size)
            .map(|output| {
                let mut first_sum = G1Projective::identity();
                let mut key_sum_weighted = G2Projective::identity();
                let mut target_sum = Gt::identity();
                for input in 0..batch.batch_size {
                    if input == output {
                        continue;
                    }
                    let inverse_difference =
                        Option::<Scalar>::from((points[output] - points[input]).invert()).unwrap();
                    first_sum += batch.first_projective[input] * inverse_difference;
                    key_sum_weighted +=
                        G2Projective::from(material.decryption_key.fraction_keys()[input])
                            * inverse_difference;
                    target_sum += tm[input] * inverse_difference;
                }

                let point_inverse = Option::<Scalar>::from(points[output].invert()).unwrap();
                let gamma_difference_inverse =
                    Option::<Scalar>::from((points[output] - gamma).invert()).unwrap();
                let coefficient = point_inverse * half_b_plus_one + gamma_difference_inverse;
                let u = batch.second_projective[output] - combined
                    + batch.first_projective[output] * coefficient
                    - first_sum;

                let u_affine = G1Affine::from(u);
                let key = G2Prepared::from(material.decryption_key.fraction_keys()[output]);
                let weighted_key = G2Prepared::from(G2Affine::from(key_sum_weighted));
                let l1 = Bls12::multi_miller_loop(&[
                    (&u_affine, &key),
                    (
                        &G1Affine::from(-batch.first_projective[output]),
                        &weighted_key,
                    ),
                ])
                .final_exponentiation()
                    + target_sum;

                let second_affine = G1Affine::from(batch.second_projective[output]);
                let negative_first = G1Affine::from(-batch.first_projective[output]);
                let key_sum_prepared = G2Prepared::from(key_sum);
                let l2 = Bls12::multi_miller_loop(&[
                    (&second_affine, &generator_prepared),
                    (&negative_first, &key_sum_prepared),
                ])
                .final_exponentiation();

                let s2 = points[output] * gamma_inverse;
                let s1 = s2 * (points[output] - gamma);
                let mask = l2 * s2 - l1 * s1;
                ciphertexts[output].third - mask
            })
            .collect::<Vec<_>>();

        assert_eq!(optimized, literal);
    }

    #[test]
    fn arbitrary_eight_of_sixteen_subset_reconstructs() {
        let (material, messages, ciphertexts, batch, kernel) = sample_batch(8, 16, 7);
        let selected = [15usize, 1, 13, 3, 11, 5, 9, 7];
        let shares = selected
            .iter()
            .map(|index| partial_decrypt(&material.server_keys[*index], &batch).unwrap())
            .collect::<Vec<_>>();
        let combined = combine_shares(&material.decryption_key, &batch, &shares).unwrap();
        let precomputation = precompute_batch(&kernel, &batch).unwrap();
        let opened = open_batch(&kernel, &batch, &ciphertexts, &precomputation, combined).unwrap();
        assert_eq!(opened, messages);
    }

    #[test]
    fn checked_combine_discards_a_bad_share() {
        let (material, messages, ciphertexts, batch, kernel) = sample_batch(4, 5, 2);
        let precomputation = precompute_batch(&kernel, &batch).unwrap();
        let mut shares = material.server_keys[..4]
            .iter()
            .map(|key| partial_decrypt(key, &batch).unwrap())
            .collect::<Vec<_>>();
        shares[1].pre_decryption_key += G1Projective::generator();

        let combined = combine_shares_checked(&material.decryption_key, &batch, &shares).unwrap();
        let opened = open_batch(&kernel, &batch, &ciphertexts, &precomputation, combined).unwrap();
        assert_eq!(opened, messages);
    }

    #[test]
    fn rejects_invalid_proof_and_share_sets() {
        let (material, _, mut ciphertexts, batch, _) = sample_batch(4, 4, 2);
        ciphertexts[0].proof.response += Scalar::ONE;
        assert!(matches!(
            validate_batch(&material.encryption_key, &ciphertexts),
            Err(Error::InvalidProof)
        ));

        let share = partial_decrypt(&material.server_keys[0], &batch).unwrap();
        assert!(matches!(
            combine_shares(&material.decryption_key, &batch, &[share.clone(), share]),
            Err(Error::InsufficientShares { .. })
        ));

        let share_zero = partial_decrypt(&material.server_keys[0], &batch).unwrap();
        let share_one = partial_decrypt(&material.server_keys[1], &batch).unwrap();
        assert!(matches!(
            combine_shares(
                &material.decryption_key,
                &batch,
                &[share_zero.clone(), share_zero, share_one]
            ),
            Err(Error::DuplicateServerIndex(0))
        ));
    }
}
