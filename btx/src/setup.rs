//! Trusted-dealer setup and Shamir sharing for BTX.

use blstrs::{G2Affine, G2Projective, Gt, Scalar};
use ff::Field;
use group::Group;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    blst_utils::batch_normalize_g2,
    error::{Error, Result},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncryptionKey {
    pub max_batch_size: usize,
    pub element: Gt,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicDecryptionKey {
    max_batch_size: usize,
    server_count: usize,
    threshold: usize,
    /// `tau^1` through `tau^B_max`, in G2.
    low_powers: Box<[G2Affine]>,
    /// `tau^(B_max+2)` through `tau^(2 B_max)`, in G2.
    high_powers: Box<[G2Affine]>,
    server_domain: Box<[Scalar]>,
    /// Optional robustness commitments `[share_j(tau^i)]_2`.
    share_commitments: Option<Box<[Box<[G2Affine]>]>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerSecretKey {
    pub server_index: usize,
    pub domain_point: Scalar,
    shares: Box<[Scalar]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyMaterial {
    pub encryption_key: EncryptionKey,
    pub decryption_key: PublicDecryptionKey,
    pub server_keys: Vec<ServerSecretKey>,
}

#[derive(Clone, Copy, Debug)]
pub struct KeygenConfig {
    pub max_batch_size: usize,
    pub server_count: usize,
    /// The scheme reconstructs from `threshold + 1` shares.
    pub threshold: usize,
    pub publish_share_commitments: bool,
}

impl KeygenConfig {
    pub fn new(max_batch_size: usize, server_count: usize, threshold: usize) -> Self {
        Self {
            max_batch_size,
            server_count,
            threshold,
            publish_share_commitments: true,
        }
    }
}

pub fn keygen(max_batch_size: usize, server_count: usize, threshold: usize) -> Result<KeyMaterial> {
    let mut rng = rand_core::OsRng;
    keygen_with_rng(
        KeygenConfig::new(max_batch_size, server_count, threshold),
        &mut rng,
    )
}

pub fn keygen_with_rng<R: rand_core::RngCore + rand_core::CryptoRng>(
    config: KeygenConfig,
    rng: &mut R,
) -> Result<KeyMaterial> {
    validate_config(config)?;

    let tau = loop {
        let candidate = Scalar::random(&mut *rng);
        if !bool::from(candidate.is_zero()) {
            break candidate;
        }
    };

    let mut tau_powers = Vec::with_capacity(2 * config.max_batch_size + 1);
    tau_powers.push(Scalar::ONE);
    for exponent in 1..=2 * config.max_batch_size {
        tau_powers.push(tau_powers[exponent - 1] * tau);
    }

    let encryption_key = EncryptionKey {
        max_batch_size: config.max_batch_size,
        element: Gt::generator() * tau_powers[config.max_batch_size + 1],
    };

    let low_powers = powers_to_affine(&tau_powers[1..=config.max_batch_size]);
    let high_powers =
        powers_to_affine(&tau_powers[config.max_batch_size + 2..2 * config.max_batch_size + 1]);

    let server_domain = (0..config.server_count)
        .map(|index| Scalar::from((index + 1) as u64))
        .collect::<Vec<_>>();

    // Each power is shared with an independent degree-threshold polynomial.
    let sharing_polynomials = (1..=config.max_batch_size)
        .map(|power| {
            let mut coefficients = Vec::with_capacity(config.threshold + 1);
            coefficients.push(tau_powers[power]);
            coefficients.extend((0..config.threshold).map(|_| Scalar::random(&mut *rng)));
            coefficients
        })
        .collect::<Vec<_>>();

    let shares_by_power = sharing_polynomials
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
            shares: (0..config.max_batch_size)
                .map(|power_index| shares_by_power[power_index][server_index])
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
        .collect::<Vec<_>>();

    let share_commitments = config.publish_share_commitments.then(|| {
        server_keys
            .par_iter()
            .map(|server_key| powers_to_affine(&server_key.shares).into_boxed_slice())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });

    let decryption_key = PublicDecryptionKey {
        max_batch_size: config.max_batch_size,
        server_count: config.server_count,
        threshold: config.threshold,
        low_powers: low_powers.into_boxed_slice(),
        high_powers: high_powers.into_boxed_slice(),
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
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
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

    pub(crate) fn power(&self, exponent: usize) -> Result<G2Affine> {
        if (1..=self.max_batch_size).contains(&exponent) {
            return Ok(self.low_powers[exponent - 1]);
        }
        if (self.max_batch_size + 2..=2 * self.max_batch_size).contains(&exponent) {
            return Ok(self.high_powers[exponent - self.max_batch_size - 2]);
        }
        Err(Error::InvalidBatchSize(exponent))
    }

    pub(crate) fn centered_power(&self, offset: isize) -> Result<G2Affine> {
        let exponent = self.max_batch_size as isize + 1 + offset;
        if exponent <= 0 {
            return Err(Error::InvalidBatchSize(exponent.max(0) as usize));
        }
        self.power(exponent as usize)
    }

    pub(crate) fn share_commitment(&self, server_index: usize, slot: usize) -> Result<G2Affine> {
        if server_index >= self.server_count {
            return Err(Error::InvalidServerIndex(server_index));
        }
        if slot >= self.max_batch_size {
            return Err(Error::BatchTooLarge {
                batch_size: slot + 1,
                max_batch_size: self.max_batch_size,
            });
        }
        self.share_commitments
            .as_ref()
            .map(|commitments| commitments[server_index][slot])
            .ok_or(Error::InvalidShare)
    }
}

impl ServerSecretKey {
    pub fn max_batch_size(&self) -> usize {
        self.shares.len()
    }

    pub(crate) fn shares_for_batch(&self, batch_size: usize) -> Result<&[Scalar]> {
        if batch_size > self.shares.len() {
            return Err(Error::BatchTooLarge {
                batch_size,
                max_batch_size: self.shares.len(),
            });
        }
        Ok(&self.shares[..batch_size])
    }
}

fn validate_config(config: KeygenConfig) -> Result<()> {
    if config.max_batch_size == 0 {
        return Err(Error::BatchIsEmpty);
    }
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

fn powers_to_affine(scalars: &[Scalar]) -> Vec<G2Affine> {
    let projective = scalars
        .par_iter()
        .map(|scalar| G2Projective::generator() * scalar)
        .collect::<Vec<_>>();
    let mut affine = vec![G2Affine::default(); projective.len()];
    batch_normalize_g2(&projective, &mut affine);
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
        assert_eq!(material.encryption_key.max_batch_size, 16);
        assert_eq!(material.decryption_key.low_powers.len(), 16);
        assert_eq!(material.decryption_key.high_powers.len(), 15);
        assert_eq!(material.server_keys.len(), 5);
        assert_eq!(material.server_keys[0].shares.len(), 16);
        assert!(material.decryption_key.has_share_commitments());
        assert!(material.decryption_key.power(17).is_err());
    }

    #[test]
    fn rejects_invalid_committee() {
        assert!(matches!(
            keygen(8, 3, 3),
            Err(Error::InvalidCommittee { .. })
        ));
    }

    #[test]
    fn minimum_batch_size_has_no_high_powers() {
        let material = keygen(1, 1, 0).unwrap();
        assert_eq!(material.decryption_key.low_powers.len(), 1);
        assert!(material.decryption_key.high_powers.is_empty());
        assert_eq!(material.server_keys[0].shares.len(), 1);
    }
}
