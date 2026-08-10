//! Fiat--Shamir Schnorr proofs for ciphertext well-formedness.
//!
//! Construction 2 writes the proof statement as knowledge of the scalar in
//! `ct[1]`.  Its CCA game also needs the masked `ct[2]` component to be
//! immutable.  The transcript therefore binds both components and the setup
//! identifier while keeping the proof itself to one compressed G1 point and
//! one scalar.

use blstrs::{G1Affine, G1Projective, Gt, Scalar};
use ff::Field;
use group::{prime::PrimeCurveAffine, Curve, Group};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    blst_utils::{g1_multi_exp_affine_bytes, scalars_to_le_bytes},
    encoding::append_gt,
};

const TRANSCRIPT_DOMAIN: &[u8] = b"WEIGHTED-PFE-CIPHERTEXT-SCHNORR-BLS12381-v1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SchnorrProof {
    pub commitment: G1Affine,
    pub response: Scalar,
}

impl SchnorrProof {
    pub const SERIALIZED_SIZE: usize = 48 + 32;

    pub(crate) fn create<R: rand_core::RngCore + rand_core::CryptoRng>(
        randomness: Scalar,
        ciphertext_first: G1Affine,
        ciphertext_second: &Gt,
        setup_id: [u8; 32],
        rng: &mut R,
    ) -> Self {
        let nonce = Scalar::random(rng);
        let commitment = (G1Projective::generator() * nonce).to_affine();
        let challenge = challenge_scalar(ciphertext_first, ciphertext_second, setup_id, commitment);
        Self {
            commitment,
            response: nonce + challenge * randomness,
        }
    }

    pub(crate) fn verify(
        &self,
        ciphertext_first: G1Affine,
        ciphertext_second: &Gt,
        setup_id: [u8; 32],
    ) -> bool {
        let challenge = challenge_scalar(
            ciphertext_first,
            ciphertext_second,
            setup_id,
            self.commitment,
        );
        G1Projective::generator() * self.response
            == G1Projective::from(self.commitment)
                + G1Projective::from(ciphertext_first) * challenge
    }

    pub(crate) fn append_canonical(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.commitment.to_compressed());
        output.extend_from_slice(&self.response.to_bytes_le());
    }
}

/// Random-linearly aggregate all Schnorr equations into one affine BLST MSM.
///
/// The coefficients are verifier-generated after the proofs are fixed.  A
/// failed aggregate rejects the exact-size batch; callers do not need an
/// expensive per-ciphertext fallback merely to identify the bad slot.
pub(crate) fn verify_batch<R: rand_core::RngCore + rand_core::CryptoRng>(
    ciphertext_first: &[G1Affine],
    ciphertext_second: &[Gt],
    proofs: &[SchnorrProof],
    setup_id: [u8; 32],
    rng: &mut R,
) -> bool {
    if ciphertext_first.is_empty()
        || ciphertext_first.len() != ciphertext_second.len()
        || ciphertext_first.len() != proofs.len()
    {
        return false;
    }

    let challenges = ciphertext_first
        .iter()
        .zip(ciphertext_second)
        .zip(proofs)
        .map(|((first, second), proof)| {
            challenge_scalar(*first, second, setup_id, proof.commitment)
        })
        .collect::<Vec<_>>();
    let coefficients = (0..proofs.len())
        .map(|_| sample_nonzero(rng))
        .collect::<Vec<_>>();

    let generator_scalar = coefficients
        .iter()
        .zip(proofs)
        .fold(Scalar::ZERO, |sum, (coefficient, proof)| {
            sum + *coefficient * proof.response
        });
    let mut bases = Vec::with_capacity(1 + 2 * proofs.len());
    let mut scalars = Vec::with_capacity(1 + 2 * proofs.len());
    bases.push(G1Affine::generator());
    scalars.push(generator_scalar);
    for (proof, coefficient) in proofs.iter().zip(&coefficients) {
        bases.push(proof.commitment);
        scalars.push(-*coefficient);
    }
    for ((first, challenge), coefficient) in
        ciphertext_first.iter().zip(challenges).zip(coefficients)
    {
        bases.push(*first);
        scalars.push(-(coefficient * challenge));
    }

    g1_multi_exp_affine_bytes(&bases, &scalars_to_le_bytes(&scalars)) == G1Projective::identity()
}

fn challenge_scalar(
    ciphertext_first: G1Affine,
    ciphertext_second: &Gt,
    setup_id: [u8; 32],
    commitment: G1Affine,
) -> Scalar {
    let mut encoded_target = Vec::with_capacity(289);
    append_gt(ciphertext_second, &mut encoded_target);
    let transcript = Sha256::new()
        .chain_update(TRANSCRIPT_DOMAIN)
        .chain_update(setup_id)
        .chain_update(G1Affine::generator().to_compressed())
        .chain_update(ciphertext_first.to_compressed())
        .chain_update((encoded_target.len() as u64).to_le_bytes())
        .chain_update(encoded_target)
        .chain_update(commitment.to_compressed());

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
    use rand_core::OsRng;

    #[test]
    fn individual_and_batched_verification_bind_every_context_field() {
        let setup_id = [7u8; 32];
        let witnesses = [Scalar::from(3u64), Scalar::from(9u64)];
        let first = witnesses
            .iter()
            .map(|witness| (G1Projective::generator() * witness).to_affine())
            .collect::<Vec<_>>();
        let second = [Gt::generator(), Gt::generator() * Scalar::from(4u64)];
        let proofs = witnesses
            .iter()
            .zip(&first)
            .zip(&second)
            .map(|((witness, first), second)| {
                SchnorrProof::create(*witness, *first, second, setup_id, &mut OsRng)
            })
            .collect::<Vec<_>>();

        assert!(proofs[0].verify(first[0], &second[0], setup_id));
        assert!(verify_batch(&first, &second, &proofs, setup_id, &mut OsRng));

        let mut changed = second;
        changed[0] += Gt::generator();
        assert!(!proofs[0].verify(first[0], &changed[0], setup_id));
        assert!(!verify_batch(
            &first, &changed, &proofs, setup_id, &mut OsRng
        ));
        assert!(!proofs[0].verify(first[0], &second[0], [8u8; 32]));
        assert_eq!(SchnorrProof::SERIALIZED_SIZE, 80);
    }
}
