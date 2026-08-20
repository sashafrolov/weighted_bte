//! Partial-fraction weighted batch threshold encryption over BLS12-381.
//!
//! This crate implements Construction 5 of the weighted-BTE paper.  It keeps
//! one constant-size response per real party while assigning a contiguous set
//! of Shamir evaluation points to that party according to its integer weight.

mod blst_utils;
pub mod decryption;
mod encoding;
pub mod encryption;
pub mod error;
mod fft;
mod final_exponentiation;
mod interpolation;
pub mod proof;
pub mod setup;

pub use decryption::{
    accept_decryption_shares, decrypt, open_batch, partial_decrypt, precompute_batch,
    prepare_decryption, validate_batch, verify_decryption_share, AcceptedDecryptionShares,
    BatchPrecomputation, CauchyKernel, DecryptionPrecomputation, DecryptionShare, ValidatedBatch,
};
pub use encryption::{encrypt, encrypt_with_rng, Ciphertext};
pub use error::{Error, Result};
pub use setup::{
    keygen, keygen_with_rng, EncryptionKey, KeyMaterial, KeygenConfig, PartySecretKey,
    PublicDecryptionKey,
};
