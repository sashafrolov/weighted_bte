//! PFE encryption for Construction 1.

use blstrs::{G1Affine, G1Projective, Gt, Scalar};
use ff::Field;
use group::{Curve, Group};
use serde::{Deserialize, Serialize};

use crate::{encoding::append_gt, proof::DleqProof, setup::EncryptionKey};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ciphertext {
    pub first: G1Affine,
    pub second: G1Affine,
    pub third: Gt,
    pub proof: DleqProof,
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
    let first = (G1Projective::generator() * randomness).to_affine();
    let second = (G1Projective::from(encryption_key.relation_base) * randomness).to_affine();
    let third = message + encryption_key.masking_base * randomness;
    let proof = DleqProof::create(encryption_key.relation_base, randomness, first, second, rng);

    Ciphertext {
        first,
        second,
        third,
        proof,
    }
}

impl Ciphertext {
    pub fn verify(&self, encryption_key: &EncryptionKey) -> bool {
        self.proof
            .verify(encryption_key.relation_base, self.first, self.second)
    }

    pub(crate) fn append_canonical(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.first.to_compressed());
        output.extend_from_slice(&self.second.to_compressed());
        append_gt(&self.third, output);
        output.extend_from_slice(&self.proof.response.to_bytes_le());
        output.extend_from_slice(&self.proof.first_commitment.to_compressed());
        output.extend_from_slice(&self.proof.second_commitment.to_compressed());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::keygen;

    #[test]
    fn encryption_proof_verifies() {
        let material = keygen(8, 3, 1).unwrap();
        let message = Gt::generator() * Scalar::from(42u64);
        let ciphertext = encrypt(&material.encryption_key, message);
        assert!(ciphertext.verify(&material.encryption_key));
    }

    #[test]
    fn masked_component_is_not_part_of_the_paper_proof_statement() {
        let material = keygen(4, 3, 1).unwrap();
        let mut ciphertext = encrypt(&material.encryption_key, Gt::identity());
        ciphertext.third += Gt::generator();
        // Construction 4 proves only the relation between the two G1
        // components. This test makes that security-relevant paper choice
        // explicit rather than accidentally implying stronger binding.
        assert!(ciphertext.verify(&material.encryption_key));
    }
}
