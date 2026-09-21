//! Single-threaded native BLST anchor for the MCL arithmetic experiments.
//!
//! This measures synthetic operation shapes, not complete decryption. Affine
//! point rows are prestored as in the native protocol. Opening and verification
//! include output normalization; positive MSMs do not, matching the MCL rows.
//! Coefficient encoding is outside committee timers; validator powers and their
//! encoding are inside the partial-decryption timer. Raw helpers retain native
//! Pippenger scratch allocations. No Rayon or BLST Rust worker pool is used.
//!
//! Run with no arguments for B=4,16, or supply those batch sizes positionally.
//! `--check-only` performs differential checks without recording timings.
//! `BATCH_LAYOUT_WEIGHTS_FILE` and `BATCH_LAYOUT_ERROR` select the same profile
//! and heaviest authorized committee as the protocol layout benchmark.

use std::{
    collections::BTreeSet,
    env,
    error::Error,
    fs,
    hint::black_box,
    path::PathBuf,
    time::{Duration, Instant},
};

use blstrs::{G1Projective, G2Projective, Scalar};
use ff::Field;
use group::Group;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[allow(dead_code)]
#[path = "../src/blst_utils.rs"]
mod blst_utils;

const SAMPLES: usize = 5;
const MIN_SAMPLE_TIME: Duration = Duration::from_millis(50);

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

struct Committee {
    weights: Vec<usize>,
    accepted_weight: usize,
}

fn committee() -> Result<Committee, Box<dyn Error>> {
    let weights_path = env::var_os("BATCH_LAYOUT_WEIGHTS_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../scripts/data/solana_share_weights_2026-08-07T16-16-16Z.json")
        });
    let error = env::var("BATCH_LAYOUT_ERROR").unwrap_or_else(|_| "1/16".into());
    let profiles: Allocations = serde_json::from_slice(&fs::read(&weights_path)?)?;
    let profile = profiles
        .allocations
        .into_iter()
        .find(|profile| profile.error == error)
        .ok_or("allocation profile not found")?;
    if profile.weights.is_empty()
        || profile.weights.contains(&0)
        || profile.weights.iter().sum::<usize>() != profile.share_count
        || !(1..=profile.share_count).contains(&profile.reconstruction_threshold)
    {
        return Err("invalid allocation counts, weights, or threshold".into());
    }
    let mut parties: Vec<_> = (0..profile.weights.len()).collect();
    parties.sort_by_key(|&party| (std::cmp::Reverse(profile.weights[party]), party));
    let mut accepted_weight = 0;
    let mut weights = Vec::new();
    for party in parties {
        let weight = profile.weights[party];
        weights.push(weight);
        accepted_weight += weight;
        if accepted_weight >= profile.reconstruction_threshold {
            break;
        }
    }
    eprintln!(
        "NATIVE_KERNEL_CONFIG backend=BLST curve=BLS12-381 threads=1 W={} q={} parties={} accepted_parties={} accepted_weight={} error={} weights={} synthetic_affine_rows=true samples={} min_sample_ms={}",
        profile.share_count,
        profile.reconstruction_threshold,
        profile.weights.len(),
        weights.len(),
        accepted_weight,
        error,
        weights_path.display(),
        SAMPLES,
        MIN_SAMPLE_TIME.as_millis(),
    );
    eprintln!("NATIVE_KERNEL_SELECTED_WEIGHTS {weights:?}");
    Ok(Committee {
        weights,
        accepted_weight,
    })
}

/// Public deterministic rejection-sampled full-field values, never secret data.
fn scalar(index: usize, label: &[u8]) -> Scalar {
    for counter in 0u32.. {
        let mut hash = Sha256::new();
        hash.update(b"WEIGHTED-BTX-NATIVE-KERNEL-v1");
        hash.update(label);
        hash.update((index as u64).to_le_bytes());
        hash.update(counter.to_le_bytes());
        let bytes: [u8; 32] = hash.finalize().into();
        if let Some(value) = Option::<Scalar>::from(Scalar::from_bytes_le(&bytes)) {
            if !bool::from(value.is_zero()) {
                return value;
            }
        }
    }
    unreachable!("field rejection sampling terminates")
}

/// Calibration and correctness checking happen before the five timed samples.
/// Each sample is extended if needed so its actual elapsed time reaches 50ms.
fn measure(
    phase: &str,
    group: &str,
    batch: usize,
    committee: &Committee,
    msm_count: usize,
    input_points: usize,
    mut work: impl FnMut(),
) {
    work();
    let mut repetitions = 1usize;
    loop {
        let start = Instant::now();
        for _ in 0..repetitions {
            work();
        }
        if start.elapsed() >= MIN_SAMPLE_TIME {
            break;
        }
        repetitions = repetitions.checked_mul(2).expect("calibration overflow");
    }

    let mut samples = Vec::with_capacity(SAMPLES);
    let mut min_elapsed = Duration::MAX;
    for _ in 0..SAMPLES {
        let start = Instant::now();
        let mut count = 0usize;
        loop {
            for _ in 0..repetitions {
                work();
            }
            count += repetitions;
            if start.elapsed() >= MIN_SAMPLE_TIME {
                break;
            }
        }
        let elapsed = start.elapsed();
        min_elapsed = min_elapsed.min(elapsed);
        samples.push(elapsed.as_secs_f64() * 1_000_000.0 / count as f64);
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "{phase},{group},{batch},{:.4},{msm_count},{input_points},{},{},{SAMPLES},{:.3}",
        samples[SAMPLES / 2],
        committee.accepted_weight,
        committee.weights.len(),
        min_elapsed.as_secs_f64() * 1_000.0,
    );
}

fn check_msm<P, A>(points: &[A], scalars: &[Scalar], msm: fn(&[A], &[u8]) -> P)
where
    P: Group<Scalar = Scalar> + From<A>,
    A: Copy,
{
    assert_eq!(points.len(), scalars.len());
    let expected = points
        .iter()
        .zip(scalars)
        .fold(P::identity(), |sum, (&point, &scalar)| {
            sum + P::from(point) * scalar
        });
    assert_eq!(
        msm(points, &blst_utils::scalars_to_le_bytes(scalars)),
        expected,
        "native MSM differs from independent scalar-multiply-and-add sum",
    );
}

fn group_kernels<P, A>(
    name: &str,
    batch: usize,
    committee: &Committee,
    normalize: fn(&[P], &mut [A]),
    msm: fn(&[A], &[u8]) -> P,
    check_only: bool,
) where
    P: Group<Scalar = Scalar> + From<A>,
    A: Copy + Default,
{
    let weight = committee.accepted_weight;
    let parties = committee.weights.len();
    let width = weight.max(2 * batch).max(parties);
    // As in the MCL harness, each slot gets its own row in one synthetic point
    // sequence. Generate and batch-normalize outside all measured intervals.
    let step = P::generator() * scalar(0, b"point-step");
    let mut point = P::generator() * scalar(0, b"point-start");
    let projective: Vec<_> = (0..width * 2 * batch)
        .map(|_| {
            let output = point;
            point += step;
            output
        })
        .collect();
    let mut points = vec![A::default(); projective.len()];
    normalize(&projective, &mut points);
    drop(projective);
    let coefficients: Vec<_> = (0..width)
        .map(|index| scalar(index, b"coefficient"))
        .collect();
    let coefficient_bytes = blst_utils::scalars_to_le_bytes(&coefficients);
    let q = scalar(0, b"validator-secret");
    let mut power = q;
    let powers: Vec<_> = (0..batch)
        .map(|_| {
            let output = power;
            power *= q;
            output
        })
        .collect();

    // Check each distinct real MSM width, including the large SIMD-sized case,
    // against independent scalar multiplications before measuring either group.
    let mut sizes: BTreeSet<_> = committee.weights.iter().copied().collect();
    sizes.extend([weight, parties]);
    for size in sizes {
        check_msm(&points[..size], &coefficients[..size], msm);
    }
    check_msm(&points[..batch], &powers, msm);
    // Exercise actual offset ranges used by the opening and positive phases.
    let mut offset = 0;
    for &party_weight in &committee.weights {
        check_msm(
            &points[(batch - 1) * width + offset..(batch - 1) * width + offset + party_weight],
            &coefficients[offset..offset + party_weight],
            msm,
        );
        offset += party_weight;
    }
    check_msm(
        &points[(2 * batch - 2) * width..(2 * batch - 2) * width + weight],
        &coefficients[..weight],
        msm,
    );
    eprintln!("NATIVE_KERNEL_CHECK group={name} batch={batch} direct_sum_checks=passed");
    if check_only {
        return;
    }

    measure(
        "partial_decrypt_one_msm",
        name,
        batch,
        committee,
        1,
        batch,
        || {
            let mut bytes = Vec::with_capacity(batch * 32);
            let mut power = black_box(q);
            for _ in 0..batch {
                bytes.extend_from_slice(&power.to_bytes_le());
                power *= q;
            }
            black_box(msm(black_box(&points[..batch]), black_box(&bytes)));
        },
    );

    let mut verification = vec![P::identity(); batch];
    let mut verification_affine = vec![A::default(); batch];
    measure(
        "verification_key_msms",
        name,
        batch,
        committee,
        batch,
        batch * parties,
        || {
            for (slot, output) in verification.iter_mut().enumerate() {
                *output = msm(
                    black_box(&points[slot * width..slot * width + parties]),
                    black_box(&coefficient_bytes[..parties * 32]),
                );
            }
            normalize(&verification, &mut verification_affine);
            black_box(&verification_affine);
        },
    );

    let mut opening = vec![P::identity(); batch * parties];
    let mut opening_affine = vec![A::default(); batch * parties];
    measure(
        "committee_opening_msms",
        name,
        batch,
        committee,
        batch * parties,
        batch * weight,
        || {
            for slot in 0..batch {
                let mut offset = 0;
                for (party, &party_weight) in committee.weights.iter().enumerate() {
                    opening[slot * parties + party] = msm(
                        black_box(
                            &points[slot * width + offset..slot * width + offset + party_weight],
                        ),
                        black_box(&coefficient_bytes[offset * 32..(offset + party_weight) * 32]),
                    );
                    offset += party_weight;
                }
            }
            normalize(&opening, &mut opening_affine);
            black_box(&opening_affine);
        },
    );

    let mut positive = vec![P::identity(); batch - 1];
    measure(
        "committee_positive_msms",
        name,
        batch,
        committee,
        batch - 1,
        (batch - 1) * weight,
        || {
            for (slot, output) in positive.iter_mut().enumerate() {
                *output = msm(
                    black_box(&points[(batch + slot) * width..(batch + slot) * width + weight]),
                    black_box(&coefficient_bytes[..weight * 32]),
                );
            }
            black_box(&positive);
        },
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut batches = Vec::new();
    let mut check_only = false;
    for argument in env::args().skip(1) {
        if argument == "--check-only" {
            check_only = true;
        } else {
            let batch: usize = argument.parse()?;
            if ![4, 16].contains(&batch) {
                return Err("supported batch sizes are 4 and 16".into());
            }
            batches.push(batch);
        }
    }
    if batches.is_empty() {
        batches.extend([4, 16]);
    }
    let committee = committee()?;
    if !check_only {
        println!("phase,group,batch,median_us,msm_count,input_points,accepted_weight,parties,samples,min_sample_ms");
    }
    for batch in batches {
        group_kernels::<G1Projective, _>(
            "G1",
            batch,
            &committee,
            blst_utils::batch_normalize_g1,
            blst_utils::g1_multi_exp_affine_bytes,
            check_only,
        );
        group_kernels::<G2Projective, _>(
            "G2",
            batch,
            &committee,
            blst_utils::batch_normalize_g2,
            blst_utils::g2_multi_exp_affine_bytes,
            check_only,
        );
    }
    Ok(())
}
