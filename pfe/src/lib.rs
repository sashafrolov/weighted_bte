//! Partial-fraction batch threshold encryption over BLS12-381.
//!
//! This crate implements Constructions 1, 2, 3, and 4 from “Efficient Batch
//! Threshold Encryption Using Partial Fraction Techniques”.  Its public API
//! follows the same phase decomposition as the sibling `btx` crate so that the
//! two constructions can be measured consistently.

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
    DecryptionShare, PartialFractionKernel, ValidatedBatch,
};
pub use encryption::{encrypt, encrypt_with_rng, Ciphertext};
pub use error::{Error, Result};
pub use setup::{
    keygen, keygen_with_rng, EncryptionKey, KeyMaterial, KeygenConfig, PublicDecryptionKey,
    ServerSecretKey,
};
