//! BTX batched threshold encryption over BLS12-381.
//!
//! This crate follows the construction in “BTX: Simple and Efficient Batch
//! Threshold Encryption”.  It exposes the paper's four decryption phases:
//! batch precomputation, partial decryption, share combination, and opening.

pub mod decryption;
mod encoding;
pub mod encryption;
pub mod error;
pub mod fft;
pub mod proof;
pub mod setup;

pub use decryption::{
    combine_shares, combine_shares_checked, open_batch, partial_decrypt, precompute_batch,
    validate_batch, verify_combined_share, verify_decryption_share, BatchPrecomputation,
    DecryptionShare, MiddleProductKernel, ValidatedBatch,
};
pub use encryption::{encrypt, encrypt_with_rng, Ciphertext};
pub use error::{Error, Result};
pub use setup::{
    keygen, keygen_with_rng, EncryptionKey, KeyMaterial, KeygenConfig, PublicDecryptionKey,
    ServerSecretKey,
};
