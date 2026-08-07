//! Fiat–Shamir Schnorr proofs for weighted-BTX ciphertext well-formedness.
//!
//! Construction 1 on page 6 writes the proof statement as only `ct[1]`.
//! However, the CCA proof on page 9 invokes extraction and simulation on the
//! pair `(ct[1], ct[2])`; its freshness argument also distinguishes changes to
//! the second component.  We therefore follow the security proof and bind
//! **both** ciphertext components into the Fiat–Shamir transcript.  In
//! particular, a proof cannot be replayed after changing the encrypted
//! payload.  This is an intentional correction to the construction's
//! pseudocode, not a wire-format optimization.
//!
//! The transcript also takes a setup-context digest derived from the generated
//! public material. The digest is supplied out of band at validation and is
//! not carried in the ciphertext, preserving the paper's ciphertext size and
//! index-free format while preventing accidental cross-setup use.
//!
//! As in the source BTX implementation, Fiat–Shamir Schnorr should only be
//! understood through the paper's GGM/ROM discussion.  It is not a generic
//! plain-ROM straight-line simulation-extractable NIZK.

use blstrs::{G1Affine, G1Projective, Gt, Scalar};
use ff::Field;
use group::Group;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::encoding::append_gt;

const TRANSCRIPT_DOMAIN: &[u8] = b"WEIGHTED-BTX-SCHNORR-BLS12381-v1";

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
        setup_id: [u8; 32],
        rng: &mut R,
    ) -> Self {
        let nonce = Scalar::random(rng);
        let commitment = G1Projective::generator() * nonce;
        let challenge = challenge_scalar(ciphertext_first, ciphertext_second, setup_id, commitment);
        let response = nonce + challenge * randomness;

        Self {
            commitment,
            response,
        }
    }

    pub fn verify(
        &self,
        ciphertext_first: G1Projective,
        ciphertext_second: &Gt,
        setup_id: [u8; 32],
    ) -> bool {
        let challenge = challenge_scalar(
            ciphertext_first,
            ciphertext_second,
            setup_id,
            self.commitment,
        );
        G1Projective::generator() * self.response == self.commitment + ciphertext_first * challenge
    }
}

fn challenge_scalar(
    ciphertext_first: G1Projective,
    ciphertext_second: &Gt,
    setup_id: [u8; 32],
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
        hasher.update(setup_id);
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
        let randomness = Scalar::from(7u64);
        let ct1 = G1Projective::generator() * randomness;
        let ct2 = Gt::generator() * Scalar::random(&mut rng);
        let setup_id = [11u8; 32];
        let proof = SchnorrProof::create(randomness, ct1, &ct2, setup_id, &mut rng);

        assert!(proof.verify(ct1, &ct2, setup_id));
        assert!(!proof.verify(ct1 + G1Projective::generator(), &ct2, setup_id));
        assert!(!proof.verify(ct1, &(ct2 + Gt::generator()), setup_id));
        assert!(!proof.verify(ct1, &ct2, [12u8; 32]));
    }

    #[test]
    fn identity_target_element_is_transcript_safe() {
        let mut rng = OsRng;
        let randomness = Scalar::random(&mut rng);
        let ct1 = G1Projective::generator() * randomness;
        let ct2 = Gt::identity();
        let setup_id = [13u8; 32];
        let proof = SchnorrProof::create(randomness, ct1, &ct2, setup_id, &mut rng);
        assert!(proof.verify(ct1, &ct2, setup_id));
    }
}
