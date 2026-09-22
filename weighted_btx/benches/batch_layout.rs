//! Compare one batch of 16 with four batches of 4 using identical ciphertexts,
//! setup, accepted validators, and thread budget. Timed combiner work includes
//! fresh share verification, committee preparation unless cached, cross terms,
//! and opening. Client proof validation, share generation, encryption, and
//! trusted setup are outside those timers and share generation is also measured
//! separately for one validator. Every workload is checked before measurement.

use std::{env, fs, path::PathBuf, time::Duration};

use blstrs::{Gt, Scalar};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use group::Group;
use rayon::prelude::*;
use serde::Deserialize;
use weighted_btx::{
    accept_decryption_shares, encrypt, keygen, open_batch, partial_decrypt, precompute_batch,
    prepare_decryption, validate_batch, Ciphertext, DecryptionPrecomputation, DecryptionShare,
    KeyMaterial, ValidatedBatch,
};

const TOTAL: usize = 16;

#[derive(Deserialize)]
struct Allocations {
    allocations: Vec<Profile>,
}

#[derive(Deserialize)]
struct Profile {
    error: String,
    weights: Vec<usize>,
    share_count: usize,
    reconstruction_threshold: usize,
}

struct Input {
    ciphertexts: Vec<Ciphertext>,
    batch: ValidatedBatch,
    shares: Vec<DecryptionShare>,
}

fn inputs(
    material: &KeyMaterial,
    ciphertexts: &[Ciphertext],
    size: usize,
    parties: &[usize],
) -> Vec<Input> {
    ciphertexts
        .chunks(size)
        .map(|chunk| {
            let batch = validate_batch(&material.decryption_key, chunk).unwrap();
            let shares = parties
                .iter()
                .map(|&party| partial_decrypt(&material.party_keys[party], &batch).unwrap())
                .collect();
            Input {
                ciphertexts: chunk.to_vec(),
                batch,
                shares,
            }
        })
        .collect()
}

fn prepare(material: &KeyMaterial, input: &Input) -> DecryptionPrecomputation {
    let accepted =
        accept_decryption_shares(&material.decryption_key, &input.batch, &input.shares).unwrap();
    prepare_decryption(&material.decryption_key, &accepted).unwrap()
}

fn combine(
    material: &KeyMaterial,
    inputs: &[Input],
    cached: Option<&DecryptionPrecomputation>,
    reuse: bool,
    parallel_chunks: bool,
) -> Vec<Vec<Option<Gt>>> {
    // Every chunk gets an independent randomized verification. Reuse only the
    // committee-dependent preparation, never a verification result or share.
    let accept = |input: &Input| {
        accept_decryption_shares(&material.decryption_key, &input.batch, &input.shares).unwrap()
    };
    let accepted: Vec<_> = if parallel_chunks {
        inputs.par_iter().map(accept).collect()
    } else {
        inputs.iter().map(accept).collect()
    };
    let shared = if reuse && cached.is_none() {
        Some(prepare_decryption(&material.decryption_key, &accepted[0]).unwrap())
    } else {
        None
    };
    let fixed = cached.or(shared.as_ref());
    let open = |index: usize| {
        let own;
        let prepared = match fixed {
            Some(prepared) => prepared,
            None => {
                own = prepare_decryption(&material.decryption_key, &accepted[index]).unwrap();
                &own
            }
        };
        let input = &inputs[index];
        let cross = precompute_batch(prepared, &input.batch).unwrap();
        // open_batch verifies that each chunk's accepted committee matches
        // the shared precomputation and that ciphertext/share digests match.
        open_batch(
            prepared,
            &accepted[index],
            &input.batch,
            &input.ciphertexts,
            &cross,
        )
        .unwrap()
    };
    if parallel_chunks {
        (0..inputs.len()).into_par_iter().map(open).collect()
    } else {
        (0..inputs.len()).map(open).collect()
    }
}

fn batch_layout(c: &mut Criterion) {
    let weights_path = env::var_os("BATCH_LAYOUT_WEIGHTS_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../scripts/data/solana_share_weights_2026-08-07T16-16-16Z.json")
        });
    let error = env::var("BATCH_LAYOUT_ERROR").unwrap_or_else(|_| "1/16".into());
    let profiles: Allocations = serde_json::from_slice(&fs::read(&weights_path).unwrap()).unwrap();
    let profile = profiles
        .allocations
        .into_iter()
        .find(|p| p.error == error)
        .expect("allocation profile");
    assert!(profile.weights.iter().all(|&w| w > 0));
    assert_eq!(profile.weights.iter().sum::<usize>(), profile.share_count);
    assert!((1..=profile.share_count).contains(&profile.reconstruction_threshold));
    let mut parties: Vec<_> = (0..profile.weights.len()).collect();
    parties.sort_by_key(|&party| (std::cmp::Reverse(profile.weights[party]), party));
    let mut weight = 0;
    let count = parties
        .iter()
        .position(|&party| {
            weight += profile.weights[party];
            weight >= profile.reconstruction_threshold
        })
        .unwrap()
        + 1;
    parties.truncate(count);

    eprintln!("LAYOUT_CONFIG total={TOTAL} W={} q={} parties={} accepted_parties={} accepted_weight={weight} threads={} error={error} weights={}",
        profile.share_count, profile.reconstruction_threshold, profile.weights.len(), count,
        rayon::current_num_threads(), weights_path.display());
    // Use one L=16 setup for both layouts: splitting does not require client
    // re-encryption. An L=4 setup would additionally reduce public-key storage.
    let material = keygen(
        TOTAL,
        &profile.weights,
        profile.reconstruction_threshold - 1,
    )
    .unwrap();
    let messages: Vec<_> = (1..=TOTAL)
        .map(|i| Gt::generator() * Scalar::from(i as u64))
        .collect();
    let ciphertexts: Vec<_> = messages
        .iter()
        .map(|&m| encrypt(&material.encryption_key, m))
        .collect();
    let single = inputs(&material, &ciphertexts, TOTAL, &parties);
    let split = inputs(&material, &ciphertexts, 4, &parties);
    let fixed_single = prepare(&material, &single[0]);
    let fixed_split = prepare(&material, &split[0]);

    let cases = [
        ("cold/1x16", &single, None, true, true),
        ("cold/4x4_reuse", &split, None, true, true),
        ("cold/4x4_no_reuse", &split, None, false, true),
        ("cold/4x4_reuse_serial", &split, None, true, false),
        ("cached/1x16", &single, Some(&fixed_single), true, true),
        ("cached/4x4", &split, Some(&fixed_split), true, true),
    ];
    let mut group = c.benchmark_group("batch_layout");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for (name, inputs, cached, reuse, parallel_chunks) in cases {
        let actual = combine(&material, inputs, cached, reuse, parallel_chunks);
        assert_eq!(
            actual.into_iter().flatten().collect::<Vec<_>>(),
            messages.iter().copied().map(Some).collect::<Vec<_>>(),
            "{name}"
        );
        group.bench_function(name, |b| {
            b.iter(|| {
                combine(
                    black_box(&material),
                    black_box(inputs),
                    black_box(cached),
                    reuse,
                    parallel_chunks,
                )
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("validator_shares");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for (name, inputs) in [("1x16", &single), ("4x4", &split)] {
        group.bench_function(name, |b| {
            b.iter(|| {
                // Serial within one validator: both layouts have the same local
                // core budget, and 4x4 includes all four returned shares.
                inputs
                    .iter()
                    .map(|input| {
                        partial_decrypt(
                            black_box(&material.party_keys[parties[0]]),
                            black_box(&input.batch),
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, batch_layout);
criterion_main!(benches);
