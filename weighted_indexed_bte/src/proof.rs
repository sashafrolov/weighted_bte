//! Compact Fiat–Shamir Chaum–Pedersen proofs in G1.
//!
//! The proof establishes equality of the discrete logarithms in two pairs
//! `(base_1, target_1)` and `(base_2, target_2)`.  Following the compact
//! representation discussed in the paper, only the challenge and response
//! scalars are serialized.  A verifier reconstructs both nonce commitments
//! as `response * base - challenge * target`.
//!
//! The paper's client statement does not include the masked message.  That
//! omission permits changing the one-time-pad ciphertext without invalidating
//! its proof and gives a direct chosen-ciphertext attack.  Callers therefore
//! supply associated transcript context: encryption binds the masked payload
//! and index, while partial decryption binds the validated batch digest.

use blstrs::{G1Affine, G1Projective, Scalar};
use ff::Field;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::blst_utils::batch_normalize_g1;

const TRANSCRIPT_DOMAIN: &[u8] = b"WEIGHTED-INDEXED-BTE-DLEQ-BLS12381-v1";

/// Domain separation between client ciphertext and server-share proofs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProofPurpose {
    Client,
    Server,
}

impl ProofPurpose {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::Client => b"client-ciphertext",
            Self::Server => b"server-decryption-share",
        }
    }
}

/// A compact Chaum–Pedersen proof.
///
/// The two nonce commitments are reconstructed during verification, so the
/// wire representation is exactly two canonical field elements (64 bytes for
/// BLS12-381), as counted by Table 2 of the paper.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DleqProof {
    pub challenge: Scalar,
    pub response: Scalar,
}

impl DleqProof {
    pub const SERIALIZED_SIZE: usize = 2 * 32;

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create<R: rand_core::RngCore + rand_core::CryptoRng>(
        purpose: ProofPurpose,
        setup_id: [u8; 32],
        context: &[u8],
        first_base: G1Affine,
        first_target: G1Affine,
        second_base: G1Affine,
        second_target: G1Affine,
        witness: Scalar,
        rng: &mut R,
    ) -> Self {
        let nonce = Scalar::random(rng);
        let commitment_projective = [
            G1Projective::from(first_base) * nonce,
            G1Projective::from(second_base) * nonce,
        ];
        let mut commitments = [G1Affine::default(); 2];
        batch_normalize_g1(&commitment_projective, &mut commitments);
        let challenge = challenge_scalar(
            purpose,
            setup_id,
            context,
            first_base,
            first_target,
            second_base,
            second_target,
            commitments[0],
            commitments[1],
        );
        let response = nonce + challenge * witness;

        Self {
            challenge,
            response,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn verify(
        &self,
        purpose: ProofPurpose,
        setup_id: [u8; 32],
        context: &[u8],
        first_base: G1Affine,
        first_target: G1Affine,
        second_base: G1Affine,
        second_target: G1Affine,
    ) -> bool {
        let commitment_projective = [
            G1Projective::from(first_base) * self.response
                - G1Projective::from(first_target) * self.challenge,
            G1Projective::from(second_base) * self.response
                - G1Projective::from(second_target) * self.challenge,
        ];
        let mut commitments = [G1Affine::default(); 2];
        batch_normalize_g1(&commitment_projective, &mut commitments);
        let expected = challenge_scalar(
            purpose,
            setup_id,
            context,
            first_base,
            first_target,
            second_base,
            second_target,
            commitments[0],
            commitments[1],
        );

        expected == self.challenge
    }

    pub(crate) fn append_canonical(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.challenge.to_bytes_le());
        output.extend_from_slice(&self.response.to_bytes_le());
    }
}

#[allow(clippy::too_many_arguments)]
fn challenge_scalar(
    purpose: ProofPurpose,
    setup_id: [u8; 32],
    context: &[u8],
    first_base: G1Affine,
    first_target: G1Affine,
    second_base: G1Affine,
    second_target: G1Affine,
    first_commitment: G1Affine,
    second_commitment: G1Affine,
) -> Scalar {
    // Rejection sampling gives an unbiased scalar instead of reducing a hash
    // modulo the field order. Length-prefixing the variable context makes the
    // transcript unambiguous if more context fields are added later.
    let transcript = Sha256::new()
        .chain_update(TRANSCRIPT_DOMAIN)
        .chain_update((purpose.tag().len() as u64).to_le_bytes())
        .chain_update(purpose.tag())
        .chain_update(setup_id)
        .chain_update((context.len() as u64).to_le_bytes())
        .chain_update(context)
        .chain_update(first_base.to_compressed())
        .chain_update(first_target.to_compressed())
        .chain_update(second_base.to_compressed())
        .chain_update(second_target.to_compressed())
        .chain_update(first_commitment.to_compressed())
        .chain_update(second_commitment.to_compressed());
    for counter in 0u32.. {
        let digest: [u8; 32] = transcript
            .clone()
            .chain_update(counter.to_le_bytes())
            .finalize()
            .into();
        if let Some(challenge) = Option::<Scalar>::from(Scalar::from_bytes_le(&digest)) {
            return challenge;
        }
    }

    unreachable!("rejection sampling terminates with overwhelming probability")
}

#[cfg(test)]
mod tests {
    use super::*;
    use group::{prime::PrimeCurveAffine, Curve, Group};
    use rand_core::OsRng;

    fn statement(witness: Scalar) -> (G1Affine, G1Affine, G1Affine, G1Affine) {
        let first_base = G1Affine::generator();
        let second_base = (G1Projective::generator() * Scalar::from(19u64)).to_affine();
        let first_target = (G1Projective::from(first_base) * witness).to_affine();
        let second_target = (G1Projective::from(second_base) * witness).to_affine();
        (first_base, first_target, second_base, second_target)
    }

    #[test]
    fn valid_compact_proof_verifies() {
        let witness = Scalar::from(7u64);
        let (base_1, target_1, base_2, target_2) = statement(witness);
        let proof = DleqProof::create(
            ProofPurpose::Client,
            [3u8; 32],
            b"associated context",
            base_1,
            target_1,
            base_2,
            target_2,
            witness,
            &mut OsRng,
        );

        assert!(proof.verify(
            ProofPurpose::Client,
            [3u8; 32],
            b"associated context",
            base_1,
            target_1,
            base_2,
            target_2,
        ));
        assert_eq!(DleqProof::SERIALIZED_SIZE, 64);
    }

    #[test]
    fn statement_context_and_proof_tampering_fail() {
        let witness = Scalar::from(11u64);
        let (base_1, target_1, base_2, target_2) = statement(witness);
        let setup_id = [5u8; 32];
        let context = b"batch digest";
        let proof = DleqProof::create(
            ProofPurpose::Server,
            setup_id,
            context,
            base_1,
            target_1,
            base_2,
            target_2,
            witness,
            &mut OsRng,
        );
        let delta = G1Projective::generator().to_affine();

        assert!(!proof.verify(
            ProofPurpose::Server,
            [6u8; 32],
            context,
            base_1,
            target_1,
            base_2,
            target_2,
        ));
        assert!(!proof.verify(
            ProofPurpose::Server,
            setup_id,
            b"another batch",
            base_1,
            target_1,
            base_2,
            target_2,
        ));
        assert!(!proof.verify(
            ProofPurpose::Server,
            setup_id,
            context,
            base_1,
            (G1Projective::from(target_1) + delta).to_affine(),
            base_2,
            target_2,
        ));

        let mut bad_challenge = proof.clone();
        bad_challenge.challenge += Scalar::ONE;
        assert!(!bad_challenge.verify(
            ProofPurpose::Server,
            setup_id,
            context,
            base_1,
            target_1,
            base_2,
            target_2,
        ));

        let mut bad_response = proof;
        bad_response.response += Scalar::ONE;
        assert!(!bad_response.verify(
            ProofPurpose::Server,
            setup_id,
            context,
            base_1,
            target_1,
            base_2,
            target_2,
        ));
    }

    #[test]
    fn proof_purposes_are_not_interchangeable() {
        let witness = Scalar::from(13u64);
        let statement = statement(witness);
        let proof = DleqProof::create(
            ProofPurpose::Client,
            [9u8; 32],
            b"same bytes",
            statement.0,
            statement.1,
            statement.2,
            statement.3,
            witness,
            &mut OsRng,
        );

        assert!(!proof.verify(
            ProofPurpose::Server,
            [9u8; 32],
            b"same bytes",
            statement.0,
            statement.1,
            statement.2,
            statement.3,
        ));
    }
}
