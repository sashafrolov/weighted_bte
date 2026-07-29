//! Construction 4's Fiat–Shamir equality-of-discrete-log proof.
//!
//! Faithfully following the paper, the statement contains the two G1
//! ciphertext components but not the masked GT component. See the crate README
//! for the resulting transcript-binding caveat.

use blstrs::{G1Affine, G1Projective, Scalar};
use ff::Field;
use group::{Curve, Group};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const TRANSCRIPT_DOMAIN: &[u8] = b"PFE-CONSTRUCTION4-BLS12381-v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DleqProof {
    pub response: Scalar,
    pub first_commitment: G1Affine,
    pub second_commitment: G1Affine,
}

impl DleqProof {
    pub fn create<R: rand_core::RngCore + rand_core::CryptoRng>(
        relation_base: G1Affine,
        randomness: Scalar,
        ciphertext_first: G1Affine,
        ciphertext_second: G1Affine,
        rng: &mut R,
    ) -> Self {
        let nonce = Scalar::random(rng);
        let first_commitment = (G1Projective::generator() * nonce).to_affine();
        let second_commitment = (G1Projective::from(relation_base) * nonce).to_affine();
        let challenge = challenge_scalar(
            relation_base,
            ciphertext_first,
            ciphertext_second,
            first_commitment,
            second_commitment,
        );

        Self {
            response: nonce + challenge * randomness,
            first_commitment,
            second_commitment,
        }
    }

    pub fn verify(
        &self,
        relation_base: G1Affine,
        ciphertext_first: G1Affine,
        ciphertext_second: G1Affine,
    ) -> bool {
        let challenge = challenge_scalar(
            relation_base,
            ciphertext_first,
            ciphertext_second,
            self.first_commitment,
            self.second_commitment,
        );
        let first_ok = G1Projective::from(self.first_commitment)
            + G1Projective::from(ciphertext_first) * challenge
            == G1Projective::generator() * self.response;
        let second_ok = G1Projective::from(self.second_commitment)
            + G1Projective::from(ciphertext_second) * challenge
            == G1Projective::from(relation_base) * self.response;
        first_ok && second_ok
    }
}

/// Randomized batch verification using two Pippenger MSM equations.
pub(crate) fn verify_batch<R: rand_core::RngCore + rand_core::CryptoRng>(
    relation_base: G1Affine,
    ciphertext_first: &[G1Projective],
    ciphertext_second: &[G1Projective],
    proofs: &[DleqProof],
    rng: &mut R,
) -> bool {
    if ciphertext_first.len() != ciphertext_second.len() || ciphertext_first.len() != proofs.len() {
        return false;
    }
    if proofs.is_empty() {
        return true;
    }

    let challenges = proofs
        .par_iter()
        .zip(ciphertext_first.par_iter())
        .zip(ciphertext_second.par_iter())
        .map(|((proof, first), second)| {
            challenge_scalar(
                relation_base,
                G1Affine::from(*first),
                G1Affine::from(*second),
                proof.first_commitment,
                proof.second_commitment,
            )
        })
        .collect::<Vec<_>>();

    let weights = (0..proofs.len())
        .map(|_| loop {
            let candidate = Scalar::random(&mut *rng);
            if !bool::from(candidate.is_zero()) {
                break candidate;
            }
        })
        .collect::<Vec<_>>();
    let weighted_challenges = weights
        .iter()
        .zip(challenges.iter())
        .map(|(weight, challenge)| *weight * challenge)
        .collect::<Vec<_>>();
    let weighted_response = weights
        .iter()
        .zip(proofs.iter())
        .fold(Scalar::ZERO, |sum, (weight, proof)| {
            sum + *weight * proof.response
        });

    let mut first_bases = Vec::with_capacity(2 * proofs.len());
    first_bases.extend(
        proofs
            .iter()
            .map(|proof| G1Projective::from(proof.first_commitment)),
    );
    first_bases.extend_from_slice(ciphertext_first);
    let mut scalars = Vec::with_capacity(2 * proofs.len());
    scalars.extend_from_slice(&weights);
    scalars.extend_from_slice(&weighted_challenges);
    let first_lhs = G1Projective::multi_exp(&first_bases, &scalars);
    if first_lhs != G1Projective::generator() * weighted_response {
        return false;
    }

    let mut second_bases = Vec::with_capacity(2 * proofs.len());
    second_bases.extend(
        proofs
            .iter()
            .map(|proof| G1Projective::from(proof.second_commitment)),
    );
    second_bases.extend_from_slice(ciphertext_second);
    let second_lhs = G1Projective::multi_exp(&second_bases, &scalars);
    second_lhs == G1Projective::from(relation_base) * weighted_response
}

fn challenge_scalar(
    relation_base: G1Affine,
    ciphertext_first: G1Affine,
    ciphertext_second: G1Affine,
    first_commitment: G1Affine,
    second_commitment: G1Affine,
) -> Scalar {
    let points = [
        relation_base,
        ciphertext_first,
        ciphertext_second,
        first_commitment,
        second_commitment,
    ];

    // Rejection sampling avoids the modular-reduction bias of truncating a
    // digest directly into the 255-bit BLS12-381 scalar field.
    for counter in 0u32.. {
        let mut hasher = Sha256::new();
        hasher.update(TRANSCRIPT_DOMAIN);
        for point in points {
            hasher.update(point.to_compressed());
        }
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
    fn proof_verifies_and_statement_tampering_fails() {
        let mut rng = OsRng;
        let base = (G1Projective::generator() * Scalar::random(&mut rng)).to_affine();
        let witness = Scalar::random(&mut rng);
        let first = (G1Projective::generator() * witness).to_affine();
        let second = (G1Projective::from(base) * witness).to_affine();
        let proof = DleqProof::create(base, witness, first, second, &mut rng);

        assert!(proof.verify(base, first, second));
        assert!(!proof.verify(
            base,
            (G1Projective::from(first) + G1Projective::generator()).to_affine(),
            second
        ));
        assert!(!proof.verify(
            base,
            first,
            (G1Projective::from(second) + G1Projective::generator()).to_affine()
        ));

        let mut first_commitment_tampered = proof.clone();
        first_commitment_tampered.first_commitment =
            (G1Projective::from(first_commitment_tampered.first_commitment)
                + G1Projective::generator())
            .to_affine();
        assert!(!first_commitment_tampered.verify(base, first, second));

        let mut second_commitment_tampered = proof;
        second_commitment_tampered.second_commitment =
            (G1Projective::from(second_commitment_tampered.second_commitment)
                + G1Projective::generator())
            .to_affine();
        assert!(!second_commitment_tampered.verify(base, first, second));
    }

    #[test]
    fn randomized_batch_verification_detects_tampering() {
        let mut rng = OsRng;
        let base = (G1Projective::generator() * Scalar::random(&mut rng)).to_affine();
        let witnesses = (0..8).map(|_| Scalar::random(&mut rng)).collect::<Vec<_>>();
        let first = witnesses
            .iter()
            .map(|witness| G1Projective::generator() * witness)
            .collect::<Vec<_>>();
        let second = witnesses
            .iter()
            .map(|witness| G1Projective::from(base) * witness)
            .collect::<Vec<_>>();
        let mut proofs = witnesses
            .iter()
            .zip(first.iter())
            .zip(second.iter())
            .map(|((witness, first), second)| {
                DleqProof::create(
                    base,
                    *witness,
                    G1Affine::from(*first),
                    G1Affine::from(*second),
                    &mut rng,
                )
            })
            .collect::<Vec<_>>();

        assert!(verify_batch(base, &first, &second, &proofs, &mut rng));
        proofs[3].response += Scalar::ONE;
        assert!(!verify_batch(base, &first, &second, &proofs, &mut rng));
    }
}
