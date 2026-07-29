//! BTX batch validation, FFT middle product, threshold decryption, and opening.

use blstrs::{Bls12, G1Affine, G1Projective, G2Affine, G2Prepared, G2Projective, Gt, Scalar};
use ff::{BatchInvert, Field};
use group::{prime::PrimeCurveAffine, Group};
use pairing::{MillerLoopResult, MultiMillerLoop};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

use crate::{
    blst_utils::{batch_normalize_g1, batch_normalize_g2},
    encryption::Ciphertext,
    error::{Error, Result},
    fft::Radix2Domain,
    final_exponentiation::{
        batch_easy_final_exponentiation, easy_final_exponentiation, full_final_exponentiation,
        hard_final_exponentiation, PreparedG2Lines,
    },
    setup::{PublicDecryptionKey, ServerSecretKey},
};

const BATCH_DIGEST_DOMAIN: &[u8] = b"BTX-ORDERED-BATCH-v1";

#[derive(Clone, Debug)]
pub struct ValidatedBatch {
    batch_size: usize,
    digest: [u8; 32],
    valid: Box<[bool]>,
    first_projective: Box<[G1Projective]>,
}

#[derive(Clone, Debug)]
pub struct MiddleProductKernel {
    batch_size: usize,
    transform_size: usize,
    domain: Radix2Domain,
    transformed_kernel: Box<[PreparedG2Lines]>,
    opening_powers: Box<[PreparedG2Lines]>,
}

#[derive(Clone, Debug)]
pub struct BatchPrecomputation {
    batch_size: usize,
    digest: [u8; 32],
    beta: Box<[Gt]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecryptionShare {
    pub server_index: usize,
    pub sigma: G1Projective,
    pub batch_digest: [u8; 32],
}

pub fn validate_batch(ciphertexts: &[Ciphertext]) -> Result<ValidatedBatch> {
    ValidatedBatch::from_ciphertexts(ciphertexts, true)
}

impl ValidatedBatch {
    /// Build the ordered batch and verify every client proof.
    pub fn verify(ciphertexts: &[Ciphertext]) -> Result<Self> {
        Self::from_ciphertexts(ciphertexts, true)
    }

    /// Build a batch without rechecking proofs.
    ///
    /// This is useful when proof verification was already performed by a
    /// preceding pipeline stage and for benchmarking the paper's isolated
    /// `precompute(B)` phase.
    pub fn proofs_preverified(ciphertexts: &[Ciphertext]) -> Result<Self> {
        Self::from_ciphertexts(ciphertexts, false)
    }

    fn from_ciphertexts(ciphertexts: &[Ciphertext], verify_proofs: bool) -> Result<Self> {
        if ciphertexts.is_empty() {
            return Err(Error::BatchIsEmpty);
        }

        let valid = if verify_proofs {
            ciphertexts
                .par_iter()
                .map(Ciphertext::verify)
                .collect::<Vec<_>>()
        } else {
            vec![true; ciphertexts.len()]
        };

        // Invalid slots remain in place as group identities.  BTX slot powers
        // are positional, so filtering must never compact the ordered batch.
        let first_projective = ciphertexts
            .iter()
            .zip(valid.iter())
            .map(|(ciphertext, is_valid)| {
                if *is_valid {
                    G1Projective::from(ciphertext.first)
                } else {
                    G1Projective::identity()
                }
            })
            .collect::<Vec<_>>();

        Ok(Self {
            batch_size: ciphertexts.len(),
            digest: batch_digest(ciphertexts),
            valid: valid.into_boxed_slice(),
            first_projective: first_projective.into_boxed_slice(),
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
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

impl MiddleProductKernel {
    /// Perform the fixed, reusable G2 preprocessing for one actual batch size.
    pub fn new(decryption_key: &PublicDecryptionKey, batch_size: usize) -> Result<Self> {
        validate_batch_size(decryption_key, batch_size)?;
        let doubled = batch_size
            .checked_mul(2)
            .ok_or(Error::InvalidBatchSize(batch_size))?;
        let transform_size = doubled
            .checked_next_power_of_two()
            .ok_or(Error::InvalidBatchSize(batch_size))?;
        let domain = Radix2Domain::new(transform_size)?;

        // Centered cyclic kernel:
        // K[0] = 0, K[d] = h_{-d}, K[m-d] = h_d.
        let mut kernel = vec![G2Projective::identity(); transform_size];
        for distance in 1..batch_size {
            kernel[distance] = decryption_key.centered_power(-(distance as isize))?.into();
            kernel[transform_size - distance] =
                decryption_key.centered_power(distance as isize)?.into();
        }

        domain.fft(&mut kernel);

        // Move the inverse-transform 1/m factor into the static G2 kernel.
        // This avoids m expensive online scalar multiplications in GT.
        let inverse_size = domain.size_inverse();
        kernel
            .par_iter_mut()
            .for_each(|point| *point *= inverse_size);

        let mut kernel_affine = vec![G2Affine::default(); transform_size];
        batch_normalize_g2(&kernel, &mut kernel_affine);
        let transformed_kernel = kernel_affine
            .par_iter()
            .map(PreparedG2Lines::from_affine)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let opening_powers = (0..batch_size)
            .into_par_iter()
            .map(|slot| {
                decryption_key
                    .centered_power(-((slot + 1) as isize))
                    .map(|point| PreparedG2Lines::from_affine(&point))
            })
            .collect::<Result<Vec<_>>>()?
            .into_boxed_slice();

        Ok(Self {
            batch_size,
            transform_size,
            domain,
            transformed_kernel,
            opening_powers,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn transform_size(&self) -> usize {
        self.transform_size
    }
}

pub fn precompute_batch(
    kernel: &MiddleProductKernel,
    batch: &ValidatedBatch,
) -> Result<BatchPrecomputation> {
    ensure_kernel_batch(kernel, batch)?;

    let mut transformed_ciphertexts = vec![G1Projective::identity(); kernel.transform_size];
    transformed_ciphertexts[..batch.batch_size].copy_from_slice(&batch.first_projective);
    kernel.domain.fft(&mut transformed_ciphertexts);

    let mut transformed_affine = vec![G1Affine::default(); kernel.transform_size];
    batch_normalize_g1(&transformed_ciphertexts, &mut transformed_affine);

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

    // The kernel was scaled by 1/m during fixed preprocessing.
    kernel
        .domain
        .ifft_cyclotomic_prefix_unscaled(&mut convolution, batch.batch_size);

    let beta = convolution[..batch.batch_size]
        .par_iter()
        .copied()
        .map(hard_final_exponentiation)
        .collect::<Vec<_>>();

    Ok(BatchPrecomputation {
        batch_size: batch.batch_size,
        digest: batch.digest,
        beta: beta.into_boxed_slice(),
    })
}

pub fn partial_decrypt(
    server_key: &ServerSecretKey,
    batch: &ValidatedBatch,
) -> Result<DecryptionShare> {
    let scalars = server_key.shares_for_batch(batch.batch_size)?;
    if scalars.len() != batch.first_projective.len() {
        return Err(Error::MismatchedBatchSize {
            expected: batch.first_projective.len(),
            actual: scalars.len(),
        });
    }

    let sigma = G1Projective::multi_exp(&batch.first_projective, scalars);
    Ok(DecryptionShare {
        server_index: server_key.server_index,
        sigma,
        batch_digest: batch.digest,
    })
}

pub fn combine_shares(
    decryption_key: &PublicDecryptionKey,
    batch: &ValidatedBatch,
    shares: &[DecryptionShare],
) -> Result<G1Projective> {
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

    let points = shares
        .iter()
        .map(|share| decryption_key.server_domain()[share.server_index])
        .collect::<Vec<_>>();
    let coefficients = lagrange_at_zero(&points);
    let bases = shares.iter().map(|share| share.sigma).collect::<Vec<_>>();

    if bases.len() != coefficients.len() {
        return Err(Error::MismatchedBatchSize {
            expected: bases.len(),
            actual: coefficients.len(),
        });
    }
    Ok(G1Projective::multi_exp(&bases, &coefficients))
}

/// Reconstruct optimistically, then fall back to direct per-server checks if
/// the aggregate check fails.
///
/// The paper specifies both checks but leaves failure recovery implicit. This
/// implementation excludes malformed shares and reconstructs again when the
/// public per-server commitments are available.
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
        .iter()
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

    let valid_slots = batch
        .valid
        .iter()
        .enumerate()
        .filter_map(|(slot, valid)| valid.then_some(slot))
        .collect::<Vec<_>>();

    let mut left = valid_slots
        .iter()
        .map(|slot| G1Affine::from(batch.first_projective[*slot]))
        .collect::<Vec<_>>();
    left.push(G1Affine::from(-share.sigma));

    let mut right = valid_slots
        .iter()
        .map(|slot| {
            decryption_key
                .share_commitment(share.server_index, *slot)
                .map(G2Prepared::from)
        })
        .collect::<Result<Vec<_>>>()?;
    right.push(G2Prepared::from(G2Affine::generator()));

    Ok(pairing_product_is_identity(&left, &right))
}

/// Optimistic aggregate share check from Section 6.5.
pub fn verify_combined_share(
    decryption_key: &PublicDecryptionKey,
    batch: &ValidatedBatch,
    sigma: G1Projective,
) -> Result<bool> {
    validate_batch_size(decryption_key, batch.batch_size)?;

    let valid_slots = batch
        .valid
        .iter()
        .enumerate()
        .filter_map(|(slot, valid)| valid.then_some(slot))
        .collect::<Vec<_>>();

    let mut left = valid_slots
        .iter()
        .map(|slot| G1Affine::from(batch.first_projective[*slot]))
        .collect::<Vec<_>>();
    left.push(G1Affine::from(-sigma));

    let mut right = valid_slots
        .iter()
        .map(|slot| decryption_key.power(slot + 1).map(G2Prepared::from))
        .collect::<Result<Vec<_>>>()?;
    right.push(G2Prepared::from(G2Affine::generator()));

    Ok(pairing_product_is_identity(&left, &right))
}

pub fn open_batch(
    kernel: &MiddleProductKernel,
    batch: &ValidatedBatch,
    ciphertexts: &[Ciphertext],
    precomputation: &BatchPrecomputation,
    sigma: G1Projective,
) -> Result<Vec<Option<Gt>>> {
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

    let sigma_affine = G1Affine::from(sigma);
    if rayon::current_num_threads() == 1 {
        let miller_results = kernel
            .opening_powers
            .iter()
            .map(|power| power.miller_loop(&sigma_affine))
            .collect::<Vec<_>>();
        let easy = batch_easy_final_exponentiation(&miller_results);

        Ok((0..batch.batch_size)
            .map(|slot| {
                batch.valid[slot].then(|| {
                    let alpha = hard_final_exponentiation(easy[slot]);
                    let pad = alpha - precomputation.beta[slot];
                    ciphertexts[slot].second - pad
                })
            })
            .collect())
    } else {
        Ok((0..batch.batch_size)
            .into_par_iter()
            .map(|slot| {
                batch.valid[slot].then(|| {
                    let alpha = full_final_exponentiation(
                        kernel.opening_powers[slot].miller_loop(&sigma_affine),
                    );
                    let pad = alpha - precomputation.beta[slot];
                    ciphertexts[slot].second - pad
                })
            })
            .collect())
    }
}

fn pairing_product_is_identity(left: &[G1Affine], right: &[G2Prepared]) -> bool {
    debug_assert_eq!(left.len(), right.len());
    let terms = left.iter().zip(right.iter()).collect::<Vec<_>>();
    Bls12::multi_miller_loop(&terms).final_exponentiation() == Gt::identity()
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

fn ensure_kernel_batch(kernel: &MiddleProductKernel, batch: &ValidatedBatch) -> Result<()> {
    if kernel.batch_size != batch.batch_size {
        return Err(Error::MismatchedBatchSize {
            expected: kernel.batch_size,
            actual: batch.batch_size,
        });
    }
    Ok(())
}

fn batch_digest(ciphertexts: &[Ciphertext]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(ciphertexts.len() * 420);
    bytes.extend_from_slice(BATCH_DIGEST_DOMAIN);
    bytes.extend_from_slice(&(ciphertexts.len() as u64).to_le_bytes());
    for ciphertext in ciphertexts {
        ciphertext.append_canonical(&mut bytes);
    }
    Sha256::digest(bytes).into()
}

#[cfg(test)]
fn naive_beta(decryption_key: &PublicDecryptionKey, batch: &ValidatedBatch) -> Vec<Gt> {
    use pairing::Engine;

    (0..batch.batch_size)
        .map(|output_slot| {
            (0..batch.batch_size)
                .filter(|input_slot| *input_slot != output_slot && batch.valid[*input_slot])
                .map(|input_slot| {
                    let offset = input_slot as isize - output_slot as isize;
                    let right = decryption_key.centered_power(offset).unwrap();
                    Bls12::pairing(&G1Affine::from(batch.first_projective[input_slot]), &right)
                })
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{encrypt, setup::keygen};

    fn fixture(
        max_batch_size: usize,
        batch_size: usize,
    ) -> (
        crate::setup::KeyMaterial,
        Vec<Gt>,
        Vec<Ciphertext>,
        ValidatedBatch,
        MiddleProductKernel,
    ) {
        let material = keygen(max_batch_size, 5, 2).unwrap();
        let messages = (0..batch_size)
            .map(|index| Gt::generator() * Scalar::from((index + 17) as u64))
            .collect::<Vec<_>>();
        let ciphertexts = messages
            .iter()
            .map(|message| encrypt(&material.encryption_key, *message))
            .collect::<Vec<_>>();
        let batch = ValidatedBatch::verify(&ciphertexts).unwrap();
        let kernel = MiddleProductKernel::new(&material.decryption_key, batch_size).unwrap();
        (material, messages, ciphertexts, batch, kernel)
    }

    #[test]
    fn fft_middle_product_matches_quadratic_formula() {
        for batch_size in [1, 2, 3, 7, 8, 15] {
            let (material, _, _, batch, kernel) = fixture(16, batch_size);
            let fast = precompute_batch(&kernel, &batch).unwrap();
            let slow = naive_beta(&material.decryption_key, &batch);
            assert_eq!(&*fast.beta, slow.as_slice(), "batch size {batch_size}");
        }
    }

    #[test]
    fn threshold_end_to_end() {
        let (material, messages, ciphertexts, batch, kernel) = fixture(16, 8);
        let precomputation = precompute_batch(&kernel, &batch).unwrap();
        let shares = material.server_keys[..3]
            .iter()
            .map(|key| partial_decrypt(key, &batch).unwrap())
            .collect::<Vec<_>>();

        for share in &shares {
            assert!(verify_decryption_share(&material.decryption_key, &batch, share).unwrap());
        }

        let sigma = combine_shares(&material.decryption_key, &batch, &shares).unwrap();
        assert!(verify_combined_share(&material.decryption_key, &batch, sigma).unwrap());

        let decrypted = open_batch(&kernel, &batch, &ciphertexts, &precomputation, sigma).unwrap();
        assert_eq!(
            decrypted,
            messages.into_iter().map(Some).collect::<Vec<_>>()
        );
    }

    #[test]
    fn invalid_proof_preserves_slot_and_returns_bottom() {
        let (material, messages, mut ciphertexts, _, kernel) = fixture(16, 8);
        ciphertexts[3].proof.response += Scalar::ONE;
        let batch = ValidatedBatch::verify(&ciphertexts).unwrap();
        assert!(!batch.valid[3]);

        let precomputation = precompute_batch(&kernel, &batch).unwrap();
        let shares = material.server_keys[..3]
            .iter()
            .map(|key| partial_decrypt(key, &batch).unwrap())
            .collect::<Vec<_>>();
        let sigma = combine_shares(&material.decryption_key, &batch, &shares).unwrap();
        let decrypted = open_batch(&kernel, &batch, &ciphertexts, &precomputation, sigma).unwrap();

        for slot in 0..8 {
            if slot == 3 {
                assert_eq!(decrypted[slot], None);
            } else {
                assert_eq!(decrypted[slot], Some(messages[slot]));
            }
        }
    }

    #[test]
    fn entirely_invalid_batch_returns_only_bottom() {
        let (material, _, mut ciphertexts, _, kernel) = fixture(8, 4);
        for ciphertext in &mut ciphertexts {
            ciphertext.proof.response += Scalar::ONE;
        }
        let batch = ValidatedBatch::verify(&ciphertexts).unwrap();
        assert_eq!(batch.valid_count(), 0);
        let precomputation = precompute_batch(&kernel, &batch).unwrap();
        let shares = material.server_keys[..3]
            .iter()
            .map(|key| partial_decrypt(key, &batch).unwrap())
            .collect::<Vec<_>>();
        let sigma = combine_shares(&material.decryption_key, &batch, &shares).unwrap();
        assert!(bool::from(sigma.is_identity()));
        assert!(verify_combined_share(&material.decryption_key, &batch, sigma).unwrap());
        assert_eq!(
            open_batch(&kernel, &batch, &ciphertexts, &precomputation, sigma).unwrap(),
            vec![None; 4]
        );
    }

    #[test]
    fn shuffled_server_subset_reconstructs() {
        let (material, messages, ciphertexts, batch, kernel) = fixture(8, 4);
        let precomputation = precompute_batch(&kernel, &batch).unwrap();
        let shares = [4usize, 1, 3]
            .iter()
            .map(|index| partial_decrypt(&material.server_keys[*index], &batch).unwrap())
            .collect::<Vec<_>>();
        let sigma = combine_shares(&material.decryption_key, &batch, &shares).unwrap();
        let decrypted = open_batch(&kernel, &batch, &ciphertexts, &precomputation, sigma).unwrap();
        assert_eq!(
            decrypted,
            messages.into_iter().map(Some).collect::<Vec<_>>()
        );
    }

    #[test]
    fn corrupted_share_fails_verification() {
        let (material, _, _, batch, _) = fixture(8, 4);
        let mut share = partial_decrypt(&material.server_keys[0], &batch).unwrap();
        share.sigma += G1Projective::generator();
        assert!(!verify_decryption_share(&material.decryption_key, &batch, &share).unwrap());
    }

    #[test]
    fn checked_combine_excludes_a_corrupted_share() {
        let (material, messages, ciphertexts, batch, kernel) = fixture(8, 4);
        let precomputation = precompute_batch(&kernel, &batch).unwrap();
        let mut shares = material.server_keys[..4]
            .iter()
            .map(|key| partial_decrypt(key, &batch).unwrap())
            .collect::<Vec<_>>();
        shares[0].sigma += G1Projective::generator();

        let sigma = combine_shares_checked(&material.decryption_key, &batch, &shares).unwrap();
        let decrypted = open_batch(&kernel, &batch, &ciphertexts, &precomputation, sigma).unwrap();
        assert_eq!(
            decrypted,
            messages.into_iter().map(Some).collect::<Vec<_>>()
        );
    }

    #[test]
    fn rejects_duplicate_and_insufficient_shares() {
        let (material, _, _, batch, _) = fixture(8, 4);
        let share = partial_decrypt(&material.server_keys[0], &batch).unwrap();
        assert!(matches!(
            combine_shares(
                &material.decryption_key,
                &batch,
                std::slice::from_ref(&share)
            ),
            Err(Error::InsufficientShares { .. })
        ));
        assert!(matches!(
            combine_shares(
                &material.decryption_key,
                &batch,
                &[share.clone(), share.clone(), share]
            ),
            Err(Error::DuplicateServerIndex(0))
        ));
    }
}
