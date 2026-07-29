//! Trusted-dealer setup and threshold sharing for PFE.

use blstrs::{G1Affine, G1Projective, G2Affine, G2Projective, Gt, Scalar};
use ff::{BatchInvert, Field};
use group::{Curve, Group};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    fft::{scalar_pow, Radix2Domain},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncryptionKey {
    pub batch_size: usize,
    /// `sum_a [1 / (x + a)]_1` over the two auxiliary indices and all slots.
    pub relation_base: G1Affine,
    /// `[1 / x]_T`, pre-paired during setup as suggested by Construction 1.
    pub masking_base: Gt,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicDecryptionKey {
    batch_size: usize,
    server_count: usize,
    threshold: usize,
    /// `[1 / (x + z_i)]_2` for the ordered root-of-unity slots.
    fraction_keys: Box<[G2Affine]>,
    fraction_key_sum: G2Affine,
    server_domain: Box<[Scalar]>,
    /// Optional robust-PFE commitments `[share_{j,i}]_2`.
    share_commitments: Option<Box<[Box<[G2Affine]>]>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerSecretKey {
    pub server_index: usize,
    pub domain_point: Scalar,
    fraction_shares: Box<[Scalar]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyMaterial {
    pub encryption_key: EncryptionKey,
    pub decryption_key: PublicDecryptionKey,
    pub server_keys: Vec<ServerSecretKey>,
}

#[derive(Clone, Copy, Debug)]
pub struct KeygenConfig {
    pub batch_size: usize,
    pub server_count: usize,
    /// Corruption threshold. Reconstruction requires `threshold + 1` shares.
    pub threshold: usize,
    pub publish_share_commitments: bool,
}

impl KeygenConfig {
    pub fn new(batch_size: usize, server_count: usize, threshold: usize) -> Self {
        Self {
            batch_size,
            server_count,
            threshold,
            publish_share_commitments: true,
        }
    }
}

pub fn keygen(batch_size: usize, server_count: usize, threshold: usize) -> Result<KeyMaterial> {
    let mut rng = rand_core::OsRng;
    keygen_with_rng(
        KeygenConfig::new(batch_size, server_count, threshold),
        &mut rng,
    )
}

pub fn keygen_with_rng<R: rand_core::RngCore + rand_core::CryptoRng>(
    config: KeygenConfig,
    rng: &mut R,
) -> Result<KeyMaterial> {
    validate_config(config)?;

    let index_points = index_points(config.batch_size)?;
    let (gamma, _) = extra_index_data(config.batch_size)?;

    // Construction 1 uses the two auxiliary labels 0 and gamma together with
    // the slot labels. Resample away from every pole before batch inversion.
    let (p_zero, slot_fractions, relation_sum) = loop {
        let x = Scalar::random(&mut *rng);
        let mut denominators = Vec::with_capacity(config.batch_size + 2);
        denominators.push(x);
        denominators.push(x + gamma);
        denominators.extend(index_points.iter().map(|point| x + point));
        if denominators
            .iter()
            .any(|denominator| bool::from(denominator.is_zero()))
        {
            continue;
        }
        denominators.iter_mut().batch_invert();
        let sum = denominators
            .iter()
            .copied()
            .fold(Scalar::ZERO, |accumulator, value| accumulator + value);
        break (denominators[0], denominators[2..].to_vec(), sum);
    };

    let relation_base = (G1Projective::generator() * relation_sum).to_affine();
    let encryption_key = EncryptionKey {
        batch_size: config.batch_size,
        relation_base,
        masking_base: Gt::generator() * p_zero,
    };

    let fraction_keys = scalars_to_g2_affine(&slot_fractions);
    let fraction_key_sum_projective = fraction_keys
        .iter()
        .fold(G2Projective::identity(), |sum, point| {
            sum + G2Projective::from(*point)
        });
    let fraction_key_sum = fraction_key_sum_projective.to_affine();

    let server_domain = (0..config.server_count)
        .map(|index| Scalar::from((index + 1) as u64))
        .collect::<Vec<_>>();

    // Each slot fraction is independently shared with a degree-threshold
    // polynomial, matching Construction 2 after translating its t-of-N
    // notation to BTX's corruption-threshold convention.
    let sharing_polynomials = slot_fractions
        .iter()
        .map(|constant| {
            let mut coefficients = Vec::with_capacity(config.threshold + 1);
            coefficients.push(*constant);
            coefficients.extend((0..config.threshold).map(|_| Scalar::random(&mut *rng)));
            coefficients
        })
        .collect::<Vec<_>>();

    let shares_by_slot = sharing_polynomials
        .par_iter()
        .map(|coefficients| {
            server_domain
                .iter()
                .map(|point| evaluate_polynomial(coefficients, *point))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let server_keys = (0..config.server_count)
        .map(|server_index| ServerSecretKey {
            server_index,
            domain_point: server_domain[server_index],
            fraction_shares: (0..config.batch_size)
                .map(|slot| shares_by_slot[slot][server_index])
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
        .collect::<Vec<_>>();

    let share_commitments = config.publish_share_commitments.then(|| {
        server_keys
            .par_iter()
            .map(|server_key| scalars_to_g2_affine(&server_key.fraction_shares).into_boxed_slice())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });

    let decryption_key = PublicDecryptionKey {
        batch_size: config.batch_size,
        server_count: config.server_count,
        threshold: config.threshold,
        fraction_keys: fraction_keys.into_boxed_slice(),
        fraction_key_sum,
        server_domain: server_domain.into_boxed_slice(),
        share_commitments,
    };

    Ok(KeyMaterial {
        encryption_key,
        decryption_key,
        server_keys,
    })
}

impl PublicDecryptionKey {
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn server_count(&self) -> usize {
        self.server_count
    }

    pub fn threshold(&self) -> usize {
        self.threshold
    }

    pub fn required_shares(&self) -> usize {
        self.threshold + 1
    }

    pub fn server_domain(&self) -> &[Scalar] {
        &self.server_domain
    }

    pub fn has_share_commitments(&self) -> bool {
        self.share_commitments.is_some()
    }

    pub(crate) fn fraction_keys(&self) -> &[G2Affine] {
        &self.fraction_keys
    }

    pub(crate) fn fraction_key_sum(&self) -> G2Affine {
        self.fraction_key_sum
    }

    pub(crate) fn share_commitment(&self, server_index: usize, slot: usize) -> Result<G2Affine> {
        if server_index >= self.server_count {
            return Err(Error::InvalidServerIndex(server_index));
        }
        if slot >= self.batch_size {
            return Err(Error::InvalidBatchSize(slot + 1));
        }
        self.share_commitments
            .as_ref()
            .map(|commitments| commitments[server_index][slot])
            .ok_or(Error::InvalidShare)
    }
}

impl ServerSecretKey {
    pub fn batch_size(&self) -> usize {
        self.fraction_shares.len()
    }

    pub(crate) fn fraction_shares(&self) -> &[Scalar] {
        &self.fraction_shares
    }
}

pub(crate) fn index_points(batch_size: usize) -> Result<Vec<Scalar>> {
    let domain = Radix2Domain::new(batch_size)?;
    let mut result = Vec::with_capacity(batch_size);
    let mut point = Scalar::ONE;
    for _ in 0..batch_size {
        result.push(point);
        point *= domain.generator();
    }
    Ok(result)
}

/// Select the first small nonzero field element outside the slot subgroup.
///
/// The printed PDF's primitive-2B-root convention collides with its `-1`
/// auxiliary index. The authors' released implementation repairs this by
/// using all B-th roots and an auxiliary `gamma` outside that subgroup.
pub(crate) fn extra_index_data(batch_size: usize) -> Result<(Scalar, u64)> {
    Radix2Domain::new(batch_size)?;
    for candidate in 2u64.. {
        let value = Scalar::from(candidate);
        if scalar_pow(value, batch_size as u64) != Scalar::ONE {
            let inverse = Option::<Scalar>::from(value.invert())
                .expect("a nonzero small integer is invertible");
            // Choosing gamma as the reciprocal of a small integer makes
            // gamma^{-1} cheap in the optimized convolution/opening formulas.
            return Ok((inverse, candidate));
        }
    }
    unreachable!("the scalar field has an element outside the slot subgroup")
}

fn validate_config(config: KeygenConfig) -> Result<()> {
    Radix2Domain::new(config.batch_size)?;
    if config.server_count == 0 || config.threshold >= config.server_count {
        return Err(Error::InvalidCommittee {
            server_count: config.server_count,
            threshold: config.threshold,
        });
    }
    if config.server_count > u64::MAX as usize {
        return Err(Error::InvalidCommittee {
            server_count: config.server_count,
            threshold: config.threshold,
        });
    }
    Ok(())
}

fn scalars_to_g2_affine(scalars: &[Scalar]) -> Vec<G2Affine> {
    let projective = scalars
        .par_iter()
        .map(|scalar| G2Projective::generator() * scalar)
        .collect::<Vec<_>>();
    let mut affine = vec![G2Affine::default(); projective.len()];
    G2Projective::batch_normalize(&projective, &mut affine);
    affine
}

fn evaluate_polynomial(coefficients: &[Scalar], point: Scalar) -> Scalar {
    coefficients
        .iter()
        .rev()
        .fold(Scalar::ZERO, |value, coefficient| {
            value * point + coefficient
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_material_has_expected_shapes() {
        let material = keygen(16, 5, 2).unwrap();
        assert_eq!(material.encryption_key.batch_size, 16);
        assert_eq!(material.decryption_key.fraction_keys.len(), 16);
        assert_eq!(material.server_keys.len(), 5);
        assert_eq!(material.server_keys[0].fraction_shares.len(), 16);
        assert_eq!(material.decryption_key.required_shares(), 3);
        assert!(material.decryption_key.has_share_commitments());
    }

    #[test]
    fn roots_and_extra_index_are_distinct() {
        for size in [1usize, 2, 4, 8, 16] {
            let roots = index_points(size).unwrap();
            let gamma = extra_index_data(size).unwrap().0;
            assert!(roots.iter().all(|root| *root != Scalar::ZERO));
            assert!(roots.iter().all(|root| *root != gamma));
        }
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(matches!(
            keygen(8, 3, 3),
            Err(Error::InvalidCommittee { .. })
        ));
        assert!(matches!(keygen(3, 3, 1), Err(Error::InvalidBatchSize(3))));
    }
}
