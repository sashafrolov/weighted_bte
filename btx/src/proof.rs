//! Fiat–Shamir Schnorr proofs for ciphertext well-formedness.
//!
//! The paper requires a simulation-extractable NIZK.  Following its
//! implementation discussion, this module uses Fiat–Shamir Schnorr and binds
//! both ciphertext components into the transcript.  Its security claim is the
//! GGM/ROM route discussed in Section 7.3 of the paper, not a plain-ROM
//! straight-line extractability claim.

use blstrs::{G1Affine, G1Projective, Gt, Scalar};
use ff::Field;
use group::Group;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::encoding::append_gt;

const TRANSCRIPT_DOMAIN: &[u8] = b"BTX-SCHNORR-BLS12381-v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchnorrProof {
    pub commitment: G1Projective,
    pub response: Scalar,
}

impl SchnorrProof {
    pub fn create<R: rand_core::RngCore + rand_core::CryptoRng>(
        randomness: Scalar,
        ciphertext_first: G1Projective,
        ciphertext_second: &Gt,
        rng: &mut R,
    ) -> Self {
        let nonce = Scalar::random(rng);
        let commitment = G1Projective::generator() * nonce;
        let challenge = challenge_scalar(ciphertext_first, ciphertext_second, commitment);
        let response = nonce + challenge * randomness;

        Self {
            commitment,
            response,
        }
    }

    pub fn verify(&self, ciphertext_first: G1Projective, ciphertext_second: &Gt) -> bool {
        let challenge = challenge_scalar(ciphertext_first, ciphertext_second, self.commitment);
        G1Projective::generator() * self.response == self.commitment + ciphertext_first * challenge
    }
}

fn challenge_scalar(
    ciphertext_first: G1Projective,
    ciphertext_second: &Gt,
    commitment: G1Projective,
) -> Scalar {
    let generator_bytes = G1Affine::from(G1Projective::generator()).to_compressed();
    let ciphertext_bytes = G1Affine::from(ciphertext_first).to_compressed();
    let commitment_bytes = G1Affine::from(commitment).to_compressed();
    let mut target_bytes = Vec::with_capacity(289);
    append_gt(ciphertext_second, &mut target_bytes);

    // Rejection sampling makes the digest-to-field conversion unbiased.
    for counter in 0u32.. {
        let mut hasher = Sha256::new();
        hasher.update(TRANSCRIPT_DOMAIN);
        hasher.update(generator_bytes);
        hasher.update(ciphertext_bytes);
        hasher.update(&target_bytes);
        hasher.update(commitment_bytes);
        hasher.update(counter.to_le_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        if let Some(challenge) = Option::<Scalar>::from(Scalar::from_bytes_le(&digest)) {
            return challenge;
        }
    }

    unreachable!("rejection sampling terminates with overwhelming probability")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::OsRng;

    #[test]
    fn valid_proof_verifies_and_tampering_fails() {
        let mut rng = OsRng;
        let randomness = Scalar::random(&mut rng);
        let ct1 = G1Projective::generator() * randomness;
        let ct2 = Gt::generator() * Scalar::random(&mut rng);
        let proof = SchnorrProof::create(randomness, ct1, &ct2, &mut rng);

        assert!(proof.verify(ct1, &ct2));
        assert!(!proof.verify(ct1 + G1Projective::generator(), &ct2));
        assert!(!proof.verify(ct1, &(ct2 + Gt::generator())));
    }

    #[test]
    fn identity_target_element_is_transcript_safe() {
        let mut rng = OsRng;
        let randomness = Scalar::random(&mut rng);
        let ct1 = G1Projective::generator() * randomness;
        let ct2 = Gt::identity();
        let proof = SchnorrProof::create(randomness, ct1, &ct2, &mut rng);
        assert!(proof.verify(ct1, &ct2));
    }
}
