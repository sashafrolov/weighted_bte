//! Encryption for Construction 5's one-G1 partial-fraction ciphertext.

use blstrs::{G1Affine, G1Projective, Gt, Scalar};
use ff::Field;
use group::{Curve, Group};
use serde::{Deserialize, Serialize};

use crate::{encoding::append_gt, proof::SchnorrProof, setup::EncryptionKey};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Ciphertext {
    pub first: G1Affine,
    pub second: Gt,
    pub proof: SchnorrProof,
}

pub fn encrypt(encryption_key: &EncryptionKey, message: Gt) -> Ciphertext {
    let mut rng = rand_core::OsRng;
    encrypt_with_rng(encryption_key, message, &mut rng)
}

pub fn encrypt_with_rng<R: rand_core::RngCore + rand_core::CryptoRng>(
    encryption_key: &EncryptionKey,
    message: Gt,
    rng: &mut R,
) -> Ciphertext {
    let randomness = sample_nonzero(rng);
    let first = (G1Projective::generator() * randomness).to_affine();
    let second = message + encryption_key.element() * randomness;
    let proof = SchnorrProof::create(randomness, first, &second, encryption_key.setup_id(), rng);
    Ciphertext {
        first,
        second,
        proof,
    }
}

impl Ciphertext {
    pub fn verify(&self, encryption_key: &EncryptionKey) -> bool {
        self.proof
            .verify(self.first, &self.second, encryption_key.setup_id())
    }

    pub fn serialized_size_bytes(&self) -> usize {
        let mut encoded = Vec::new();
        self.append_canonical(&mut encoded);
        encoded.len()
    }

    pub(crate) fn append_canonical(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.first.to_compressed());
        append_gt(&self.second, output);
        self.proof.append_canonical(output);
    }
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
    use crate::setup::keygen;

    #[test]
    fn ciphertext_proof_binds_mask_and_setup() {
        let material = keygen(4, &[1, 3, 2], 3).unwrap();
        let message = Gt::generator() * Scalar::from(42u64);
        let ciphertext = encrypt(&material.encryption_key, message);
        assert!(ciphertext.verify(&material.encryption_key));

        let mut changed = ciphertext.clone();
        changed.second += Gt::generator();
        assert!(!changed.verify(&material.encryption_key));

        let other = keygen(4, &[1, 3, 2], 3).unwrap();
        assert!(!ciphertext.verify(&other.encryption_key));
        assert_eq!(ciphertext.serialized_size_bytes(), 48 + 289 + 80);
    }
}
