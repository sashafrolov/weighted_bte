//! Encryption for the coordinated/indexed weighted-BTE construction.
//!
//! A ciphertext fixes its index when it is created.  The two G1 components
//! are accompanied by a compact Chaum–Pedersen proof that they use the same
//! scalar.  In addition to the paper's relation, the Fiat–Shamir transcript
//! binds the index and masked payload; see [`crate::proof`] for why this is
//! required for the intended chosen-ciphertext security.

use blstrs::{Bls12, G1Affine, G1Projective, Gt, Scalar};
use ff::Field;
use group::{prime::PrimeCurveAffine, Group};
use pairing::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    blst_utils::batch_normalize_g1,
    encoding::append_gt,
    error::{Error, Result},
    proof::{DleqProof, ProofPurpose},
    setup::{MasterPublicKey, PublicParameters},
};

const MASK_KDF_DOMAIN: &[u8] = b"WEIGHTED-INDEXED-BTE-MASK-KDF-BLS12381-v1";
const CLIENT_CONTEXT_DOMAIN: &[u8] = b"WEIGHTED-INDEXED-BTE-CLIENT-CONTEXT-v1";
const COMPRESSED_G1_SIZE: usize = 48;
const CANONICAL_INDEX_SIZE: usize = 8;
const CANONICAL_LENGTH_SIZE: usize = 8;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Ciphertext {
    pub first: G1Affine,
    pub punctured: G1Affine,
    pub index: usize,
    pub masked_message: Box<[u8]>,
    pub proof: DleqProof,
}

pub fn encrypt(
    public_parameters: &PublicParameters,
    public_key: &MasterPublicKey,
    message: &[u8],
    index: usize,
) -> Result<Ciphertext> {
    let mut rng = rand_core::OsRng;
    encrypt_with_rng(public_parameters, public_key, message, index, &mut rng)
}

pub fn encrypt_with_rng<R: rand_core::RngCore + rand_core::CryptoRng>(
    public_parameters: &PublicParameters,
    public_key: &MasterPublicKey,
    message: &[u8],
    index: usize,
    rng: &mut R,
) -> Result<Ciphertext> {
    if public_parameters.crs_id() != public_key.crs_id() {
        return Err(Error::MismatchedSetup);
    }
    if index >= public_parameters.index_space_size() || index >= public_key.index_space_size() {
        return Err(Error::InvalidCiphertextIndex(index));
    }

    // Figure 2 requires any u != i. Choosing it deterministically avoids
    // carrying another value in the ciphertext. Setup rejects n < 2.
    let auxiliary_index = if index == 0 { 1 } else { 0 };
    let offset = signed_difference(index, auxiliary_index)?;
    let centered_power = public_parameters.centered_power(offset)?;
    let auxiliary_delta = public_key.delta(auxiliary_index)?;
    let indexed_delta = public_key.delta(index)?;

    let randomness = random_nonzero_scalar(rng);
    let projective = [
        G1Projective::generator() * randomness,
        G1Projective::from(indexed_delta) * randomness,
        G1Projective::from(auxiliary_delta) * randomness,
    ];
    let mut affine = [G1Affine::identity(); 3];
    batch_normalize_g1(&projective, &mut affine);
    let [first, punctured, pairing_left] = affine;
    let mask = Bls12::pairing(&pairing_left, &centered_power);
    let masked_message = xor_with_mask(&mask, public_key.setup_id(), index, message);

    let context = client_proof_context(index, &masked_message);
    let proof = DleqProof::create(
        ProofPurpose::Client,
        public_key.setup_id(),
        &context,
        G1Affine::generator(),
        first,
        indexed_delta,
        punctured,
        randomness,
        rng,
    );

    Ok(Ciphertext {
        first,
        punctured,
        index,
        masked_message,
        proof,
    })
}

impl Ciphertext {
    /// Verify the ciphertext's index, setup context, and DLEq proof.
    pub fn verify(&self, public_key: &MasterPublicKey) -> bool {
        if self.index >= public_key.index_space_size() {
            return false;
        }
        let Ok(indexed_delta) = public_key.delta(self.index) else {
            return false;
        };
        let context = client_proof_context(self.index, &self.masked_message);
        self.proof.verify(
            ProofPurpose::Client,
            public_key.setup_id(),
            &context,
            G1Affine::generator(),
            self.first,
            indexed_delta,
            self.punctured,
        )
    }

    /// Append the canonical ciphertext representation used by batch digests.
    pub(crate) fn append_canonical(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.first.to_compressed());
        output.extend_from_slice(&self.punctured.to_compressed());
        output.extend_from_slice(&(self.index as u64).to_le_bytes());
        output.extend_from_slice(&(self.masked_message.len() as u64).to_le_bytes());
        output.extend_from_slice(&self.masked_message);
        self.proof.append_canonical(output);
    }

    /// Size of [`Self::append_canonical`], using compressed group points.
    pub fn serialized_size(&self) -> usize {
        2 * COMPRESSED_G1_SIZE
            + CANONICAL_INDEX_SIZE
            + CANONICAL_LENGTH_SIZE
            + self.masked_message.len()
            + DleqProof::SERIALIZED_SIZE
    }
}

/// XOR a SHA-256 counter-mode mask into an arbitrary-length byte string.
///
/// XOR is its own inverse, so decryption uses this same helper. The canonical
/// GT representation is identity-safe and the setup, index, and length are
/// included to prevent cross-context keystream reuse.
pub(crate) fn xor_with_mask(
    mask: &Gt,
    setup_id: [u8; 32],
    index: usize,
    input: &[u8],
) -> Box<[u8]> {
    let mut encoded_mask = Vec::with_capacity(289);
    append_gt(mask, &mut encoded_mask);

    let base = Sha256::new()
        .chain_update(MASK_KDF_DOMAIN)
        .chain_update(setup_id)
        .chain_update((index as u64).to_le_bytes())
        .chain_update((input.len() as u64).to_le_bytes())
        .chain_update((encoded_mask.len() as u64).to_le_bytes())
        .chain_update(encoded_mask);

    let mut output = input.to_vec();
    for (counter, chunk) in output.chunks_mut(32).enumerate() {
        let block = base
            .clone()
            .chain_update((counter as u64).to_le_bytes())
            .finalize();
        for (byte, mask_byte) in chunk.iter_mut().zip(block) {
            *byte ^= mask_byte;
        }
    }
    output.into_boxed_slice()
}

fn client_proof_context(index: usize, masked_message: &[u8]) -> Vec<u8> {
    let mut context = Vec::with_capacity(
        CLIENT_CONTEXT_DOMAIN.len() + 2 * core::mem::size_of::<u64>() + masked_message.len(),
    );
    context.extend_from_slice(CLIENT_CONTEXT_DOMAIN);
    context.extend_from_slice(&(index as u64).to_le_bytes());
    context.extend_from_slice(&(masked_message.len() as u64).to_le_bytes());
    context.extend_from_slice(masked_message);
    context
}

fn signed_difference(left: usize, right: usize) -> Result<isize> {
    if left >= right {
        isize::try_from(left - right).map_err(|_| Error::ParameterSizeOverflow)
    } else {
        isize::try_from(right - left)
            .map(|difference| -difference)
            .map_err(|_| Error::ParameterSizeOverflow)
    }
}

fn random_nonzero_scalar<R: rand_core::RngCore + rand_core::CryptoRng>(rng: &mut R) -> Scalar {
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
    use crate::setup::{keygen, setup};

    #[test]
    fn encryption_proof_and_mask_round_trip() {
        let public_parameters = setup(8).unwrap();
        let material = keygen(&public_parameters, &[1, 3, 2], 3).unwrap();
        let message = b"an indexed weighted-BTE message";
        let ciphertext = encrypt(&public_parameters, &material.public_key, message, 5).unwrap();

        assert!(ciphertext.verify(&material.public_key));
        assert_eq!(
            ciphertext.serialized_size(),
            2 * 48 + 2 * 8 + message.len() + 2 * 32
        );
        assert_eq!(
            bincode::serialize(&ciphertext).unwrap().len(),
            ciphertext.serialized_size()
        );

        // XOR masking is involutive for arbitrary-length messages, including
        // a final partial SHA-256 block.
        let mask = Gt::generator() * Scalar::from(91u64);
        let masked = xor_with_mask(
            &mask,
            material.public_key.setup_id(),
            ciphertext.index,
            message,
        );
        let plaintext = xor_with_mask(
            &mask,
            material.public_key.setup_id(),
            ciphertext.index,
            &masked,
        );
        assert_eq!(&*plaintext, message);
    }

    #[test]
    fn payload_index_and_setup_tampering_fail() {
        let public_parameters = setup(8).unwrap();
        let material = keygen(&public_parameters, &[1, 3, 2], 3).unwrap();
        let ciphertext = encrypt(
            &public_parameters,
            &material.public_key,
            b"bind every ciphertext component",
            2,
        )
        .unwrap();

        let mut changed_payload = ciphertext.clone();
        changed_payload.masked_message[0] ^= 1;
        assert!(!changed_payload.verify(&material.public_key));

        let mut changed_index = ciphertext.clone();
        changed_index.index = 3;
        assert!(!changed_index.verify(&material.public_key));

        let other_parameters = setup(8).unwrap();
        let other_material = keygen(&other_parameters, &[1, 3, 2], 3).unwrap();
        assert!(!ciphertext.verify(&other_material.public_key));
        assert!(matches!(
            encrypt(&other_parameters, &material.public_key, b"wrong setup", 2,),
            Err(Error::MismatchedSetup)
        ));
    }

    #[test]
    fn rejects_out_of_range_index() {
        let public_parameters = setup(4).unwrap();
        let material = keygen(&public_parameters, &[1, 1], 1).unwrap();
        assert!(matches!(
            encrypt(&public_parameters, &material.public_key, b"message", 4,),
            Err(Error::InvalidCiphertextIndex(4))
        ));
    }
}
