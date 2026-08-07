//! Indexed weighted batch threshold encryption over BLS12-381.
//!
//! This crate implements the main construction in Figure 2 of
//! `papers/weighted_bte_old_paper.pdf`. Ciphertexts choose their batch index
//! at encryption time. Weighted virtual Shamir shares keep each real party's
//! decryption response to one G1 point, while an adaptive direct/FFT middle
//! product evaluates all indexed cross terms.

mod blst_utils;
pub mod decryption;
mod encoding;
pub mod encryption;
pub mod error;
pub mod fft;
pub(crate) mod final_exponentiation;
mod interpolation;
pub mod proof;
pub mod setup;

pub use decryption::{
    accept_decryption_shares, decrypt, open_batch, partial_decrypt, precompute_batch,
    prepare_decryption, validate_batch, verify_decryption_share, AcceptedDecryptionShares,
    BatchPrecomputation, DecryptionPrecomputation, DecryptionShare, IndexedMiddleProductKernel,
    ValidatedBatch,
};
pub use encryption::{encrypt, encrypt_with_rng, Ciphertext};
pub use error::{Error, Result};
pub use setup::{
    keygen, keygen_with_rng, setup, setup_with_rng, KeyMaterial, KeygenConfig, MasterPublicKey,
    PartySecretKey, PublicParameters,
};
