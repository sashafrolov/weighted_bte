//! Fresh-ciphertext end-to-end benchmark on one local machine.
//!
//! One global Rayon pool executes encryption/proofs, client validation, all
//! selected validators' partial decryptions, fresh share acceptance, committee
//! preparation, cross terms, and opening. Network latency is excluded. The
//! aggregate validator phase is local simulation, not distributed latency.
//! "cold" means committee preparation is inside the interval; "cached" reuses
//! only setup/size/committee material. Both modes generate fresh ciphertexts,
//! proofs, validator shares, and acceptance results on every iteration.
//!
//! Layouts: 1x16, 4x4, and 8x2, each with cold and cached preparation.
//! Default: profile 1/16, L=16, 12 threads, 11 samples after 2 warmups per case.
//! CLI: --threads N --samples N --warmup N --format json|csv --weights-file PATH
//!      --approximation-error ERROR. Environment defaults: E2E_THREADS,
//! E2E_SAMPLES, E2E_WARMUP, E2E_FORMAT, BATCH_LAYOUT_WEIGHTS_FILE,
//! BATCH_LAYOUT_ERROR. Message construction and trusted setup are untimed.

use std::{collections::BTreeMap, env, error::Error, fs, path::PathBuf, time::Instant};

use blstrs::{Gt, Scalar};
use group::Group;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use weighted_btx_swapped::{
    accept_decryption_shares, encrypt, keygen, open_batch, partial_decrypt, precompute_batch,
    prepare_decryption, validate_batch, DecryptionPrecomputation, KeyMaterial,
};

const TOTAL: usize = 16;
const USAGE: &str = "Usage: end_to_end [--threads N] [--samples N] [--warmup N] \
    [--format json|csv] [--weights-file PATH] [--approximation-error ERROR]";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Json,
    Csv,
}

#[derive(Debug)]
struct Config {
    threads: usize,
    samples: usize,
    warmup: usize,
    format: Format,
    weights_path: PathBuf,
    error: String,
}

fn parse_usize(label: &str, value: &str, positive: bool) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{label} must be a nonnegative integer, got {value:?}"))?;
    if positive && parsed == 0 {
        return Err(format!("{label} must be positive"));
    }
    Ok(parsed)
}

fn parse_format(value: &str) -> Result<Format, String> {
    match value {
        "json" => Ok(Format::Json),
        "csv" => Ok(Format::Csv),
        _ => Err(format!("format must be json or csv, got {value:?}")),
    }
}

fn parse_config(
    arguments: &[String],
    get_env: impl Fn(&str) -> Option<String>,
) -> Result<Config, String> {
    let mut config = Config {
        threads: parse_usize(
            "E2E_THREADS",
            &get_env("E2E_THREADS").unwrap_or_else(|| "12".into()),
            true,
        )?,
        samples: parse_usize(
            "E2E_SAMPLES",
            &get_env("E2E_SAMPLES").unwrap_or_else(|| "11".into()),
            true,
        )?,
        warmup: parse_usize(
            "E2E_WARMUP",
            &get_env("E2E_WARMUP").unwrap_or_else(|| "2".into()),
            false,
        )?,
        format: parse_format(&get_env("E2E_FORMAT").unwrap_or_else(|| "json".into()))?,
        weights_path: get_env("BATCH_LAYOUT_WEIGHTS_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../scripts/data/solana_share_weights_2026-08-07T16-16-16Z.json")
            }),
        error: get_env("BATCH_LAYOUT_ERROR").unwrap_or_else(|| "1/16".into()),
    };
    let mut args = arguments.iter();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--threads" => config.threads = parse_usize(flag, value, true)?,
            "--samples" => config.samples = parse_usize(flag, value, true)?,
            "--warmup" => config.warmup = parse_usize(flag, value, false)?,
            "--format" => config.format = parse_format(value)?,
            "--weights-file" => config.weights_path = PathBuf::from(value),
            "--approximation-error" => config.error.clone_from(value),
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    Ok(config)
}

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

#[derive(Serialize)]
struct ProfileContext {
    file: String,
    sha256: String,
    approximation_error: String,
    party_count: usize,
    total_weight: usize,
    required_weight: usize,
    threshold_weight: usize,
    selected_parties: Vec<usize>,
    selected_weights: Vec<usize>,
    accepted_weight: usize,
}

fn load_profile(config: &Config) -> Result<(Profile, ProfileContext), Box<dyn Error>> {
    let bytes = fs::read(&config.weights_path)?;
    let document: Allocations = serde_json::from_slice(&bytes)?;
    let mut matching = document
        .allocations
        .into_iter()
        .filter(|profile| profile.error == config.error);
    let profile = matching.next().ok_or("allocation profile not found")?;
    if matching.next().is_some() {
        return Err("duplicate allocation profile".into());
    }
    let sum = profile
        .weights
        .iter()
        .try_fold(0usize, |total, &weight| total.checked_add(weight))
        .ok_or("allocation weight sum overflows usize")?;
    if profile.weights.is_empty()
        || profile.weights.contains(&0)
        || sum != profile.share_count
        || !(1..=profile.share_count).contains(&profile.reconstruction_threshold)
    {
        return Err("inconsistent allocation weights, share count, or threshold".into());
    }
    let mut selected_parties: Vec<_> = (0..profile.weights.len()).collect();
    selected_parties.sort_by_key(|&party| (std::cmp::Reverse(profile.weights[party]), party));
    let mut accepted_weight = 0;
    let selected_count = selected_parties
        .iter()
        .position(|&party| {
            accepted_weight += profile.weights[party];
            accepted_weight >= profile.reconstruction_threshold
        })
        .ok_or("committee cannot reach threshold")?
        + 1;
    selected_parties.truncate(selected_count);
    let selected_weights = selected_parties
        .iter()
        .map(|&party| profile.weights[party])
        .collect();
    let context = ProfileContext {
        file: fs::canonicalize(&config.weights_path)?
            .display()
            .to_string(),
        sha256: Sha256::digest(&bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
        approximation_error: config.error.clone(),
        party_count: profile.weights.len(),
        total_weight: profile.share_count,
        required_weight: profile.reconstruction_threshold,
        threshold_weight: profile.reconstruction_threshold - 1,
        selected_parties,
        selected_weights,
        accepted_weight,
    };
    Ok((profile, context))
}

fn messages(sequence: u64, count: usize) -> Vec<Gt> {
    let first = sequence
        .checked_mul(TOTAL as u64)
        .and_then(|offset| offset.checked_add(1))
        .expect("message sequence overflow");
    (0..count)
        .map(|slot| Gt::generator() * Scalar::from(first + slot as u64))
        .collect()
}

#[derive(Serialize)]
struct CachePreparation {
    batch_size: usize,
    preparation_us: f64,
}

/// A fixture creates only the committee-dependent cache. No ciphertext, proof,
/// decryption share, or acceptance result escapes into measured iterations.
fn prepare_cache(
    material: &KeyMaterial,
    context: &ProfileContext,
    batch_size: usize,
) -> (DecryptionPrecomputation, CachePreparation) {
    let ciphertexts: Vec<_> = messages(0, batch_size)
        .into_par_iter()
        .map(|message| encrypt(&material.encryption_key, message))
        .collect();
    let batch = validate_batch(&material.decryption_key, &ciphertexts).unwrap();
    let shares: Vec<_> = context
        .selected_parties
        .par_iter()
        .map(|&party| partial_decrypt(&material.party_keys[party], &batch).unwrap())
        .collect();
    let accepted = accept_decryption_shares(&material.decryption_key, &batch, &shares).unwrap();
    let start = Instant::now();
    let prepared = prepare_decryption(&material.decryption_key, &accepted).unwrap();
    let preparation_us = start.elapsed().as_secs_f64() * 1_000_000.0;
    (
        prepared,
        CachePreparation {
            batch_size,
            preparation_us,
        },
    )
}

#[derive(Clone, Debug, Serialize)]
struct SampleTimes {
    end_to_end_us: f64,
    combiner_us: f64,
    encryption_us: f64,
    validation_us: f64,
    share_generation_us: f64,
    acceptance_us: f64,
    preparation_us: f64,
    precompute_us: f64,
    opening_us: f64,
}

/// Returns elapsed wall times only after all 16 outputs are verified equal to
/// their intended plaintexts. Assertions run after the ending timestamp.
fn iteration(
    material: &KeyMaterial,
    context: &ProfileContext,
    batch_size: usize,
    cached: Option<&DecryptionPrecomputation>,
    plaintexts: &[Gt],
) -> SampleTimes {
    assert_eq!(plaintexts.len(), TOTAL);
    let total_start = Instant::now();
    let phase_start = Instant::now();
    let ciphertexts: Vec<_> = plaintexts
        .par_iter()
        .map(|&message| encrypt(&material.encryption_key, message))
        .collect();
    let encryption_and_proofs = phase_start.elapsed();

    let phase_start = Instant::now();
    let batches: Vec<_> = ciphertexts
        .par_chunks(batch_size)
        .map(|chunk| validate_batch(&material.decryption_key, chunk).unwrap())
        .collect();
    let client_validation = phase_start.elapsed();

    let phase_start = Instant::now();
    let shares: Vec<Vec<_>> = batches
        .par_iter()
        .map(|batch| {
            context
                .selected_parties
                .par_iter()
                .map(|&party| partial_decrypt(&material.party_keys[party], batch).unwrap())
                .collect()
        })
        .collect();
    let all_validator_shares = phase_start.elapsed();

    // Combiner timing starts after all validators' local shares exist and
    // includes fresh acceptance, optional shared preparation, cross terms,
    // and opening of every chunk. It excludes message transport.
    let combiner_start = Instant::now();
    let phase_start = Instant::now();
    let accepted: Vec<_> = batches
        .par_iter()
        .zip(shares.par_iter())
        .map(|(batch, shares)| {
            accept_decryption_shares(&material.decryption_key, batch, shares).unwrap()
        })
        .collect();
    let share_acceptance = phase_start.elapsed();

    let owned_preparation;
    let committee_preparation;
    let prepared = if let Some(cached) = cached {
        committee_preparation = std::time::Duration::ZERO;
        cached
    } else {
        let phase_start = Instant::now();
        // Every chunk has the same size and selected honest committee. Reuse
        // this single preparation across all chunks; open_batch checks context.
        owned_preparation = prepare_decryption(&material.decryption_key, &accepted[0]).unwrap();
        committee_preparation = phase_start.elapsed();
        &owned_preparation
    };

    let phase_start = Instant::now();
    let cross_terms: Vec<_> = batches
        .par_iter()
        .map(|batch| precompute_batch(prepared, batch).unwrap())
        .collect();
    let cross_terms_time = phase_start.elapsed();

    let phase_start = Instant::now();
    let opened: Vec<Vec<_>> = ciphertexts
        .par_chunks(batch_size)
        .zip(batches.par_iter())
        .zip(accepted.par_iter())
        .zip(cross_terms.par_iter())
        .map(|(((ciphertexts, batch), accepted), cross_terms)| {
            open_batch(prepared, accepted, batch, ciphertexts, cross_terms).unwrap()
        })
        .collect();
    let opening = phase_start.elapsed();
    let combiner = combiner_start.elapsed();
    let total = total_start.elapsed();

    assert_eq!(batches.len(), TOTAL / batch_size);
    assert!(batches
        .iter()
        .all(|batch| batch.valid_count() == batch_size));
    assert!(accepted.iter().all(|set| {
        set.party_count() == context.selected_parties.len()
            && set.accepted_weight() == context.accepted_weight
            && set.rejected_parties().is_empty()
    }));
    assert_eq!(opened.iter().map(Vec::len).sum::<usize>(), TOTAL);
    for (actual, expected) in opened.iter().flatten().zip(plaintexts) {
        assert_eq!(*actual, Some(*expected), "end-to-end plaintext mismatch");
    }
    SampleTimes {
        end_to_end_us: total.as_secs_f64() * 1_000_000.0,
        combiner_us: combiner.as_secs_f64() * 1_000_000.0,
        encryption_us: encryption_and_proofs.as_secs_f64() * 1_000_000.0,
        validation_us: client_validation.as_secs_f64() * 1_000_000.0,
        share_generation_us: all_validator_shares.as_secs_f64() * 1_000_000.0,
        acceptance_us: share_acceptance.as_secs_f64() * 1_000_000.0,
        preparation_us: committee_preparation.as_secs_f64() * 1_000_000.0,
        precompute_us: cross_terms_time.as_secs_f64() * 1_000_000.0,
        opening_us: opening.as_secs_f64() * 1_000_000.0,
    }
}

#[derive(Serialize)]
struct Distribution {
    median_us: f64,
    p10_us: f64,
    p90_us: f64,
}

fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    assert!(!sorted.is_empty());
    let index = fraction * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * index.fract()
}

fn metrics() -> [(&'static str, fn(&SampleTimes) -> f64); 9] {
    [
        ("end_to_end", |sample| sample.end_to_end_us),
        ("combiner", |sample| sample.combiner_us),
        ("encryption", |sample| sample.encryption_us),
        ("validation", |sample| sample.validation_us),
        ("share_generation", |sample| sample.share_generation_us),
        ("acceptance", |sample| sample.acceptance_us),
        ("preparation", |sample| sample.preparation_us),
        ("precompute", |sample| sample.precompute_us),
        ("opening", |sample| sample.opening_us),
    ]
}

fn summarize(samples: &[SampleTimes]) -> BTreeMap<String, Distribution> {
    metrics()
        .into_iter()
        .map(|(name, read)| {
            let mut values: Vec<_> = samples.iter().map(read).collect();
            values.sort_by(f64::total_cmp);
            (
                name.into(),
                Distribution {
                    median_us: quantile(&values, 0.5),
                    p10_us: quantile(&values, 0.1),
                    p90_us: quantile(&values, 0.9),
                },
            )
        })
        .collect()
}

#[derive(Serialize)]
struct CaseResult {
    name: String,
    batch_size: usize,
    chunks: usize,
    committee_cached: bool,
    all_plaintexts_verified: bool,
    summary: BTreeMap<String, Distribution>,
    samples: Vec<SampleTimes>,
}

#[derive(Serialize)]
struct Report {
    implementation: String,
    scope: String,
    combiner_scope: String,
    scheduling: String,
    quantile_method: String,
    threads: usize,
    samples_per_case: usize,
    warmup_per_case: usize,
    total_ciphertexts: usize,
    setup_max_batch_size: usize,
    keygen_us: f64,
    profile: ProfileContext,
    cached_committee_preparations: Vec<CachePreparation>,
    cases: Vec<CaseResult>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = env::args().skip(1).collect();
    if arguments.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let config = parse_config(&arguments, |name| env::var(name).ok())
        .map_err(|error| format!("{error}\n{USAGE}"))?;
    let (profile, context) = load_profile(&config)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build_global()?;
    eprintln!(
        "E2E_CONFIG implementation={} threads={} samples={} warmup={} total={} L={} W={} q={} parties={} accepted_parties={} accepted_weight={}",
        env!("CARGO_PKG_NAME"), config.threads, config.samples, config.warmup,
        TOTAL, TOTAL, context.total_weight, context.required_weight,
        context.party_count, context.selected_parties.len(), context.accepted_weight,
    );
    let start = Instant::now();
    let material = keygen(TOTAL, &profile.weights, context.threshold_weight)?;
    let keygen_us = start.elapsed().as_secs_f64() * 1_000_000.0;
    eprintln!("E2E_SETUP keygen_us={keygen_us:.3}");
    let (cached16, preparation16) = prepare_cache(&material, &context, 16);
    let (cached4, preparation4) = prepare_cache(&material, &context, 4);
    let (cached2, preparation2) = prepare_cache(&material, &context, 2);
    let workloads = [
        ("cold/1x16", 16, None),
        ("cold/4x4", 4, None),
        ("cold/8x2", 2, None),
        ("cached/1x16", 16, Some(&cached16)),
        ("cached/4x4", 4, Some(&cached4)),
        ("cached/8x2", 2, Some(&cached2)),
    ];
    let mut cases = Vec::new();
    let mut sequence = 1u64;
    for (name, batch_size, cached) in workloads {
        eprintln!("E2E_CASE {name}");
        for _ in 0..config.warmup {
            let plaintexts = messages(sequence, TOTAL);
            sequence += 1;
            iteration(&material, &context, batch_size, cached, &plaintexts);
        }
        let mut samples = Vec::with_capacity(config.samples);
        for _ in 0..config.samples {
            let plaintexts = messages(sequence, TOTAL);
            sequence += 1;
            samples.push(iteration(
                &material,
                &context,
                batch_size,
                cached,
                &plaintexts,
            ));
        }
        cases.push(CaseResult {
            name: name.into(),
            batch_size,
            chunks: TOTAL / batch_size,
            committee_cached: cached.is_some(),
            all_plaintexts_verified: true,
            summary: summarize(&samples),
            samples,
        });
    }
    let report = Report {
        implementation: env!("CARGO_PKG_NAME").into(),
        scope: "Fresh encryption and proofs through verified opening of 16 messages; all selected validators simulated locally; network excluded; message generation and trusted setup excluded".into(),
        combiner_scope: "After all local validator shares exist: fresh share acceptance, one committee preparation unless cached, cross terms, and all openings; network excluded".into(),
        scheduling: "One global Rayon pool; phases execute sequentially; chunks and selected validators within each phase run in parallel; cold split layouts prepare the committee once and reuse it across all four or eight chunks".into(),
        quantile_method: "Linear interpolation at (n-1)*p; descriptive p10/p90, not confidence intervals".into(),
        threads: config.threads,
        samples_per_case: config.samples,
        warmup_per_case: config.warmup,
        total_ciphertexts: TOTAL,
        setup_max_batch_size: TOTAL,
        keygen_us,
        profile: context,
        cached_committee_preparations: vec![preparation16, preparation4, preparation2],
        cases,
    };
    match config.format {
        Format::Json => println!("{}", serde_json::to_string_pretty(&report)?),
        Format::Csv => {
            println!("implementation,case,threads,samples,warmup,total_ciphertexts,batch_size,chunks,accepted_parties,accepted_weight,metric,median_us,p10_us,p90_us");
            for case in &report.cases {
                for (metric, distribution) in &case.summary {
                    println!(
                        "{},{},{},{},{},{},{},{},{},{},{},{:.4},{:.4},{:.4}",
                        report.implementation,
                        case.name,
                        report.threads,
                        report.samples_per_case,
                        report.warmup_per_case,
                        report.total_ciphertexts,
                        case.batch_size,
                        case.chunks,
                        report.profile.selected_parties.len(),
                        report.profile.accepted_weight,
                        metric,
                        distribution.median_us,
                        distribution.p10_us,
                        distribution.p90_us,
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).into()).collect()
    }

    #[test]
    fn defaults_and_cli_overrides_are_explicit() {
        let defaults = parse_config(&[], |_| None).unwrap();
        assert_eq!(
            (defaults.threads, defaults.samples, defaults.warmup),
            (12, 11, 2)
        );
        assert_eq!(defaults.format, Format::Json);
        let override_config = parse_config(
            &arguments(&[
                "--threads",
                "3",
                "--samples",
                "2",
                "--warmup",
                "0",
                "--format",
                "csv",
            ]),
            |_| None,
        )
        .unwrap();
        assert_eq!(
            (
                override_config.threads,
                override_config.samples,
                override_config.warmup
            ),
            (3, 2, 0)
        );
        assert_eq!(override_config.format, Format::Csv);
    }

    #[test]
    fn invalid_environment_values_are_rejected() {
        for (name, value) in [
            ("E2E_THREADS", "0"),
            ("E2E_THREADS", "nope"),
            ("E2E_SAMPLES", "0"),
            ("E2E_WARMUP", "-1"),
            ("E2E_FORMAT", "yaml"),
        ] {
            assert!(parse_config(&[], |key| (key == name).then(|| value.into())).is_err());
        }
    }

    #[test]
    fn malformed_cli_is_rejected() {
        for args in [
            vec!["--threads"],
            vec!["--threads", "0"],
            vec!["--samples", "-1"],
            vec!["--unknown", "4"],
        ] {
            assert!(parse_config(&arguments(&args), |_| None).is_err());
        }
    }

    #[test]
    fn descriptive_quantiles_interpolate_and_handle_one_sample() {
        assert_eq!(quantile(&[7.0], 0.1), 7.0);
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 0.5), 2.5);
        assert!((quantile(&[1.0, 2.0, 3.0, 4.0], 0.1) - 1.3).abs() < 1e-12);
        assert!((quantile(&[1.0, 2.0, 3.0, 4.0], 0.9) - 3.7).abs() < 1e-12);
    }
}
