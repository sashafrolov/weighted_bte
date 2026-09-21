//! Identical scalar draws must give equivalent protocol outputs in both
//! source-group orientations, including invalid ciphertext slots and shares.

use blstrs::{pairing, G1Affine, G1Projective, G2Affine, G2Projective, Gt, Scalar};
use ff::Field;
use group::{prime::PrimeCurveAffine, Curve, Group};
use rand_core::{CryptoRng, RngCore};
use sha2::{Digest, Sha256};
use weighted_btx as original;
use weighted_btx_swapped as swapped;

// Reproducible test-only stream; never used for production key generation.
struct TestRng(u64);

impl RngCore for TestRng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn next_u64(&mut self) -> u64 {
        let digest = Sha256::digest(self.0.to_le_bytes());
        self.0 += 1;
        u64::from_le_bytes(digest[..8].try_into().unwrap())
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        rand_core::impls::fill_bytes_via_next(self, dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for TestRng {}

#[test]
fn source_group_swap_preserves_scalar_algebra_and_opened_messages() {
    for batch_size in [1, 2, 3, 4, 7, 16] {
        for invalidate_slot in [false, true] {
            let mut original_rng = TestRng(123);
            let mut swapped_rng = TestRng(123);
            let weights = [1, 3, 2, 4, 1];
            let original_keys = original::keygen_with_rng(
                original::KeygenConfig::new(16, &weights, 5),
                &mut original_rng,
            )
            .unwrap();
            let swapped_keys = swapped::keygen_with_rng(
                swapped::KeygenConfig::new(16, &weights, 5),
                &mut swapped_rng,
            )
            .unwrap();
            assert_eq!(
                original_keys.encryption_key.element(),
                swapped_keys.encryption_key.element()
            );
            // Transcript domains and source-group encodings are intentionally distinct.
            assert_ne!(
                original_keys.encryption_key.setup_id(),
                swapped_keys.encryption_key.setup_id()
            );

            let messages: Vec<_> = (0..batch_size)
                .map(|slot| Gt::generator() * Scalar::from((slot + 1) as u64))
                .collect();
            let mut original_ct: Vec<_> = messages
                .iter()
                .map(|&message| {
                    original::encrypt_with_rng(
                        &original_keys.encryption_key,
                        message,
                        &mut original_rng,
                    )
                })
                .collect();
            let mut swapped_ct: Vec<_> = messages
                .iter()
                .map(|&message| {
                    swapped::encrypt_with_rng(
                        &swapped_keys.encryption_key,
                        message,
                        &mut swapped_rng,
                    )
                })
                .collect();
            for (original, swapped) in original_ct.iter().zip(&swapped_ct) {
                assert_eq!(original.second, swapped.second);
                assert_eq!(
                    pairing(&original.first, &G2Affine::generator()),
                    pairing(&G1Affine::generator(), &swapped.first),
                );
            }
            if invalidate_slot {
                original_ct[batch_size / 2].proof.response += Scalar::ONE;
                swapped_ct[batch_size / 2].proof.response += Scalar::ONE;
            }
            let original_batch =
                original::validate_batch(&original_keys.decryption_key, &original_ct).unwrap();
            let swapped_batch =
                swapped::validate_batch(&swapped_keys.decryption_key, &swapped_ct).unwrap();
            assert_eq!(original_batch.valid_mask(), swapped_batch.valid_mask());
            let mut original_shares: Vec<_> = [3, 2, 1]
                .iter()
                .map(|&party| {
                    original::partial_decrypt(&original_keys.party_keys[party], &original_batch)
                        .unwrap()
                })
                .collect();
            let mut swapped_shares: Vec<_> = [3, 2, 1]
                .iter()
                .map(|&party| {
                    swapped::partial_decrypt(&swapped_keys.party_keys[party], &swapped_batch)
                        .unwrap()
                })
                .collect();
            for (original, swapped) in original_shares.iter().zip(&swapped_shares) {
                assert_eq!(
                    pairing(&original.sigma.to_affine(), &G2Affine::generator()),
                    pairing(&G1Affine::generator(), &swapped.sigma.to_affine()),
                );
            }
            // Parties 1 and 3 still form an authorized set after rejecting 2.
            original_shares[1].sigma += G1Projective::generator();
            swapped_shares[1].sigma += G2Projective::generator();
            let original_accepted = original::accept_decryption_shares(
                &original_keys.decryption_key,
                &original_batch,
                &original_shares,
            )
            .unwrap();
            let swapped_accepted = swapped::accept_decryption_shares(
                &swapped_keys.decryption_key,
                &swapped_batch,
                &swapped_shares,
            )
            .unwrap();
            assert_eq!(original_accepted.rejected_parties(), &[2]);
            assert_eq!(swapped_accepted.rejected_parties(), &[2]);
            let original_fixed =
                original::prepare_decryption(&original_keys.decryption_key, &original_accepted)
                    .unwrap();
            let swapped_fixed =
                swapped::prepare_decryption(&swapped_keys.decryption_key, &swapped_accepted)
                    .unwrap();
            let original_cross =
                original::precompute_batch(&original_fixed, &original_batch).unwrap();
            let swapped_cross = swapped::precompute_batch(&swapped_fixed, &swapped_batch).unwrap();
            let original_opened = original::open_batch(
                &original_fixed,
                &original_accepted,
                &original_batch,
                &original_ct,
                &original_cross,
            )
            .unwrap();
            let swapped_opened = swapped::open_batch(
                &swapped_fixed,
                &swapped_accepted,
                &swapped_batch,
                &swapped_ct,
                &swapped_cross,
            )
            .unwrap();
            assert_eq!(original_opened, swapped_opened);
            for (slot, opened) in swapped_opened.iter().enumerate() {
                assert_eq!(
                    *opened,
                    (!invalidate_slot || slot != batch_size / 2).then_some(messages[slot])
                );
            }
        }
    }
}

#[test]
fn swapped_objects_round_trip_through_existing_serialization() {
    let keys = swapped::keygen(4, &[2, 1], 1).unwrap();
    let keys: swapped::KeyMaterial =
        bincode::deserialize(&bincode::serialize(&keys).unwrap()).unwrap();
    let message = Gt::generator() * Scalar::from(42);
    let ciphertext = swapped::encrypt(&keys.encryption_key, message);
    let ciphertext: swapped::Ciphertext =
        bincode::deserialize(&bincode::serialize(&ciphertext).unwrap()).unwrap();
    let batch = swapped::validate_batch(&keys.decryption_key, &[ciphertext.clone()]).unwrap();
    let share = swapped::partial_decrypt(&keys.party_keys[0], &batch).unwrap();
    let share: swapped::DecryptionShare =
        bincode::deserialize(&bincode::serialize(&share).unwrap()).unwrap();
    assert_eq!(
        swapped::decrypt(&keys.decryption_key, &[ciphertext], &[share]).unwrap(),
        vec![Some(message)]
    );
}
