//! Experimental G1/G2-swapped weighted BTX over BLS12-381.
//!
//! This crate swaps the source groups in Construction 1 of “Weighted Batch Threshold
//! Encryption”, which virtualizes BTX secret-sharing points while keeping each
//! real party's batch decryption share to one G2 element.  Its public API
//! separates validation, per-party partial decryption, weighted share
//! acceptance, batch preparation, and opening.

mod blst_utils;
pub(crate) mod final_exponentiation;

pub mod decryption;
mod encoding;
pub mod encryption;
pub mod error;
pub mod fft;
mod interpolation;
pub mod proof;
pub mod setup;

pub use decryption::{
    accept_decryption_shares, decrypt, open_batch, partial_decrypt, precompute_batch,
    prepare_decryption, validate_batch, verify_decryption_share, AcceptedDecryptionShares,
    BatchPrecomputation, DecryptionPrecomputation, DecryptionShare, ValidatedBatch,
};
pub use encryption::{encrypt, encrypt_with_rng, Ciphertext};
pub use error::{Error, Result};
pub use setup::{
    keygen, keygen_with_rng, EncryptionKey, KeyMaterial, KeygenConfig, PartySecretKey,
    PublicDecryptionKey,
};
