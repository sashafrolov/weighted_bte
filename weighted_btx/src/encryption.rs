//! Index-free weighted-BTX encryption.
//!
//! Ciphertexts carry no setup identifier, epoch, or batch slot. Their setup is
//! supplied as out-of-band proof context, and their position in an ordered
//! batch is chosen later by the decryptor. The proof transcript binds both
//! ciphertext components; see [`crate::proof`] for the paper discrepancy and
//! context-binding rationale.

use blstrs::{G1Affine, G1Projective, Gt, Scalar};
use ff::Field;
use group::{Curve, Group};
use serde::{Deserialize, Serialize};

use crate::{encoding::append_gt, proof::SchnorrProof, setup::EncryptionKey};

#[derive(Clone, Debug, Serialize, Deserialize)]
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
    let randomness = Scalar::random(&mut *rng);
    let first_projective = G1Projective::generator() * randomness;
    let second = message + encryption_key.element() * randomness;
    let proof = SchnorrProof::create(
        randomness,
        first_projective,
        &second,
        encryption_key.setup_id(),
        rng,
    );

    Ciphertext {
        first: first_projective.to_affine(),
        second,
        proof,
    }
}

impl Ciphertext {
    /// Verify against the setup context under which the ciphertext was made.
    pub fn verify(&self, encryption_key: &EncryptionKey) -> bool {
        self.verify_for_setup(encryption_key.setup_id())
    }

    pub(crate) fn verify_for_setup(&self, setup_id: [u8; 32]) -> bool {
        self.proof.verify(self.first.into(), &self.second, setup_id)
    }

    pub(crate) fn append_canonical(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.first.to_compressed());
        append_gt(&self.second, output);
        output.extend_from_slice(&G1Affine::from(self.proof.commitment).to_compressed());
        output.extend_from_slice(&self.proof.response.to_bytes_le());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::keygen;

    #[test]
    fn encryption_proof_verifies() {
        let material = keygen(8, &[1, 3, 2], 3).unwrap();
        let message = Gt::generator() * Scalar::from(42u64);
        let ciphertext = encrypt(&material.encryption_key, message);
        assert!(ciphertext.verify(&material.encryption_key));

        let other = keygen(8, &[1, 3, 2], 3).unwrap();
        assert!(!ciphertext.verify(&other.encryption_key));
    }
}
