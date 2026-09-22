//! Parameterized fresh-ciphertext benchmark for paper-table comparisons.
//!
//! One invocation processes M messages in chunks of B with setup capacity L,
//! once with cold preparation and once with a cached committee. Both modes
//! generate fresh ciphertexts, proofs, shares, and acceptance results on every
//! iteration. Cold mode prepares once and reuses that work across all chunks.
//!
//! decryption_total starts after client validation and includes all validators'
//! shares, acceptance, preparation, cross terms, and opening. end_to_end also
//! includes fresh encryption/proofs and client validation. All validators run
//! locally in the same Rayon pool; these are not distributed network timings.
//!
//! Run from the repository root. Defaults: profile 1/16, M=B=L=16, 12 configured
//! threads, 11 samples and 2 warmups. If B is omitted it defaults to M; if L is
//! omitted it defaults to B. JSON records inputs, settings and raw samples;
//! setup and message construction are outside the online interval. The CLI
//! preserves the supplied profile path in metadata without canonicalizing it.

use std::{collections::BTreeMap, env, error::Error, fs, path::PathBuf, time::Instant};

use blstrs::{G1Affine, G2Affine, Gt, Scalar};
use group::Group;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use weighted_btx_swapped::{
    accept_decryption_shares, encrypt, keygen, open_batch, partial_decrypt, precompute_batch,
    prepare_decryption, validate_batch, DecryptionPrecomputation, KeyMaterial,
};

const USAGE: &str = "Usage: paper_tables [--weights-file PATH] [--error ERROR] \
    [--total M] [--batch B] [--setup L] [--threads N] [--samples N] [--warmup N]";

#[derive(Debug)]
struct Config {
    weights_path: PathBuf,
    error: String,
    total: usize,
    batch: usize,
    setup: usize,
    threads: usize,
    samples: usize,
    warmup: usize,
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

fn parse_config(arguments: &[String]) -> Result<Config, String> {
    let mut weights_path =
        PathBuf::from("scripts/data/solana_share_weights_2026-08-07T16-16-16Z.json");
    let mut error = String::from("1/16");
    let mut total = 16;
    let mut batch = None;
    let mut setup = None;
    let mut threads = 12;
    let mut samples = 11;
    let mut warmup = 2;
    let mut arguments = arguments.iter();
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--weights-file" => weights_path = PathBuf::from(value),
            "--error" => error.clone_from(value),
            "--total" => total = parse_usize(flag, value, true)?,
            "--batch" => batch = Some(parse_usize(flag, value, true)?),
            "--setup" => setup = Some(parse_usize(flag, value, true)?),
            "--threads" => threads = parse_usize(flag, value, true)?,
            "--samples" => samples = parse_usize(flag, value, true)?,
            "--warmup" => warmup = parse_usize(flag, value, false)?,
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    let batch = batch.unwrap_or(total);
    let setup = setup.unwrap_or(batch);
    if total % batch != 0 {
        return Err("--batch B must divide --total M".into());
    }
    if setup < batch {
        return Err("--setup L must be at least --batch B".into());
    }
    samples
        .checked_add(warmup)
        .and_then(|count| count.checked_mul(2))
        .ok_or("sample/warmup count overflows usize")?;
    Ok(Config {
        weights_path,
        error,
        total,
        batch,
        setup,
        threads,
        samples,
        warmup,
    })
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
struct Inputs {
    #[serde(rename = "N")]
    n: usize,
    #[serde(rename = "W")]
    w: usize,
    q: usize,
    tau: usize,
    threshold_weight: usize,
    accepted_weight: usize,
    total: usize,
    batch: usize,
    setup: usize,
    error: String,
    weights_file: String,
    sha256: String,
    selected_parties: Vec<usize>,
    selected_weights: Vec<usize>,
}

fn load_profile(config: &Config) -> Result<(Profile, Inputs), Box<dyn Error>> {
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
    let context = Inputs {
        n: profile.weights.len(),
        w: profile.share_count,
        q: profile.reconstruction_threshold,
        tau: selected_parties.len(),
        threshold_weight: profile.reconstruction_threshold - 1,
        accepted_weight,
        total: config.total,
        batch: config.batch,
        setup: config.setup,
        error: config.error.clone(),
        // Preserve the caller's supplied relative path in exported metadata.
        weights_file: config.weights_path.display().to_string(),
        sha256: Sha256::digest(&bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
        selected_parties,
        selected_weights,
    };
    Ok((profile, context))
}

fn messages(sequence: u64, count: usize) -> Vec<Gt> {
    let first = sequence
        .checked_mul(count as u64)
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
    context: &Inputs,
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
    decryption_total_us: f64,
    encryption_us: f64,
    validation_us: f64,
    share_generation_us: f64,
    acceptance_us: f64,
    preparation_us: f64,
    precompute_us: f64,
    opening_us: f64,
}

/// Returns elapsed wall times only after all M outputs are verified equal to
/// their intended plaintexts. Assertions run after the ending timestamp.
fn iteration(
    material: &KeyMaterial,
    context: &Inputs,
    batch_size: usize,
    cached: Option<&DecryptionPrecomputation>,
    plaintexts: &[Gt],
) -> SampleTimes {
    assert_eq!(plaintexts.len(), context.total);
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

    // Decryption starts after ciphertext validation, before any validator share.
    let decryption_start = Instant::now();
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
    let finished = Instant::now();
    let opening = finished.duration_since(phase_start);
    let combiner = finished.duration_since(combiner_start);
    let decryption_total = finished.duration_since(decryption_start);
    let total = finished.duration_since(total_start);

    assert_eq!(batches.len(), context.total / batch_size);
    assert!(batches
        .iter()
        .all(|batch| batch.valid_count() == batch_size));
    assert!(accepted.iter().all(|set| {
        set.party_count() == context.selected_parties.len()
            && set.accepted_weight() == context.accepted_weight
            && set.rejected_parties().is_empty()
    }));
    assert_eq!(opened.iter().map(Vec::len).sum::<usize>(), context.total);
    for (actual, expected) in opened.iter().flatten().zip(plaintexts) {
        assert_eq!(*actual, Some(*expected), "end-to-end plaintext mismatch");
    }
    SampleTimes {
        end_to_end_us: total.as_secs_f64() * 1_000_000.0,
        combiner_us: combiner.as_secs_f64() * 1_000_000.0,
        decryption_total_us: decryption_total.as_secs_f64() * 1_000_000.0,
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
    mean_us: f64,
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

fn metrics() -> [(&'static str, fn(&SampleTimes) -> f64); 10] {
    [
        ("end_to_end", |sample| sample.end_to_end_us),
        ("combiner", |sample| sample.combiner_us),
        ("decryption_total", |sample| sample.decryption_total_us),
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
                    mean_us: values.iter().sum::<f64>() / values.len() as f64,
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
struct Settings {
    threads: usize,
    samples: usize,
    warmup: usize,
}

#[derive(Serialize)]
struct GroupEncodingBytes {
    g1_compressed_bytes: usize,
    g2_compressed_bytes: usize,
    scalar_bytes: usize,
}

#[derive(Serialize)]
struct Report {
    schema_version: usize,
    benchmark: String,
    implementation: String,
    backend: String,
    curve: String,
    orientation: String,
    inputs: Inputs,
    settings: Settings,
    group_encoding_bytes: GroupEncodingBytes,
    scope: String,
    decryption_total_scope: String,
    combiner_scope: String,
    scheduling: String,
    quantile_method: String,
    keygen_us: f64,
    cached_committee_preparation: CachePreparation,
    cases: Vec<CaseResult>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = env::args().skip(1).collect();
    if arguments.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let config = parse_config(&arguments).map_err(|error| format!("{error}\n{USAGE}"))?;
    let (profile, context) = load_profile(&config)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build_global()?;
    eprintln!(
        "PAPER_CONFIG implementation={} threads={} samples={} warmup={} total={} batch={} setup={} N={} W={} q={} tau={} accepted_weight={} error={}",
        env!("CARGO_PKG_NAME"), config.threads, config.samples, config.warmup,
        config.total, config.batch, config.setup, context.n, context.w, context.q,
        context.tau, context.accepted_weight, config.error,
    );
    let start = Instant::now();
    let material = keygen(config.setup, &profile.weights, context.threshold_weight)?;
    let keygen_us = start.elapsed().as_secs_f64() * 1_000_000.0;
    eprintln!("PAPER_SETUP keygen_us={keygen_us:.3}");
    let (cached_preparation, cache_timing) = prepare_cache(&material, &context, config.batch);
    let mut cases = Vec::new();
    let mut sequence = 1u64;
    for (name, cached) in [("cold", None), ("cached", Some(&cached_preparation))] {
        eprintln!(
            "PAPER_CASE {name} chunks={} batch={}",
            config.total / config.batch,
            config.batch
        );
        for _ in 0..config.warmup {
            let plaintexts = messages(sequence, config.total);
            sequence += 1;
            iteration(&material, &context, config.batch, cached, &plaintexts);
        }
        let mut samples = Vec::with_capacity(config.samples);
        for _ in 0..config.samples {
            let plaintexts = messages(sequence, config.total);
            sequence += 1;
            samples.push(iteration(
                &material,
                &context,
                config.batch,
                cached,
                &plaintexts,
            ));
        }
        cases.push(CaseResult {
            name: name.into(),
            batch_size: config.batch,
            chunks: config.total / config.batch,
            committee_cached: cached.is_some(),
            all_plaintexts_verified: true,
            summary: summarize(&samples),
            samples,
        });
    }
    let report = Report {
        schema_version: 1,
        benchmark: "paper_tables".into(),
        implementation: env!("CARGO_PKG_NAME").into(),
        backend: "blst".into(),
        curve: "bls12_381".into(),
        orientation: if env!("CARGO_PKG_NAME").ends_with("_swapped") { "swapped" } else { "normal" }.into(),
        inputs: context,
        settings: Settings { threads: config.threads, samples: config.samples, warmup: config.warmup },
        group_encoding_bytes: GroupEncodingBytes {
            g1_compressed_bytes: G1Affine::default().to_compressed().len(),
            g2_compressed_bytes: G2Affine::default().to_compressed().len(),
            scalar_bytes: Scalar::from(0u64).to_bytes_le().len(),
        },
        scope: "Fresh encryption and proofs through opening of all M messages; all selected validators simulated locally; network, message generation, trusted setup, and output-equality checks excluded".into(),
        decryption_total_scope: "Starts after ciphertext validation, before all selected validators generate shares; includes share generation, fresh acceptance, one committee preparation unless cached, cross terms, and all openings; network excluded".into(),
        combiner_scope: "After all local shares exist: fresh acceptance, one committee preparation unless cached, cross terms, and all openings; network excluded".into(),
        scheduling: "One global Rayon pool at the configured thread count (default 12); phases execute sequentially; chunks and selected validators within phases run in parallel; cold mode prepares once and reuses that committee material across all chunks".into(),
        quantile_method: "Mean is arithmetic; median and p10/p90 use linear interpolation at (n-1)*p; descriptive quantiles, not confidence intervals".into(),
        keygen_us,
        cached_committee_preparation: cache_timing,
        cases,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).into()).collect()
    }

    #[test]
    fn dimensions_default_to_one_batch_and_cli_can_select_split_layouts() {
        let defaults = parse_config(&[]).unwrap();
        assert_eq!(
            (defaults.total, defaults.batch, defaults.setup),
            (16, 16, 16)
        );
        assert_eq!(
            (defaults.threads, defaults.samples, defaults.warmup),
            (12, 11, 2)
        );
        let larger = parse_config(&arguments(&["--total", "256"])).unwrap();
        assert_eq!((larger.total, larger.batch, larger.setup), (256, 256, 256));
        let split = parse_config(&arguments(&[
            "--weights-file",
            "profiles.json",
            "--error",
            "1/64",
            "--total",
            "32",
            "--batch",
            "2",
            "--setup",
            "32",
            "--threads",
            "3",
            "--samples",
            "2",
            "--warmup",
            "0",
        ]))
        .unwrap();
        assert_eq!((split.total, split.batch, split.setup), (32, 2, 32));
        assert_eq!((split.threads, split.samples, split.warmup), (3, 2, 0));
        assert_eq!(split.error, "1/64");
        assert_eq!(split.weights_path, PathBuf::from("profiles.json"));
        let irregular = parse_config(&arguments(&[
            "--total", "6", "--batch", "3", "--setup", "4",
        ]))
        .unwrap();
        assert_eq!(
            (irregular.total, irregular.batch, irregular.setup),
            (6, 3, 4)
        );
    }

    #[test]
    fn invalid_dimensions_counts_and_arguments_are_rejected() {
        for args in [
            vec!["--total", "0"],
            vec!["--batch", "0"],
            vec!["--setup", "0"],
            vec!["--total", "16", "--batch", "3"],
            vec!["--total", "16", "--batch", "32"],
            vec!["--batch", "4", "--setup", "2"],
            vec!["--threads", "0"],
            vec!["--samples", "0"],
            vec!["--warmup", "-1"],
            vec!["--total", "NaN"],
            vec!["--total"],
            vec!["--unknown", "4"],
        ] {
            assert!(parse_config(&arguments(&args)).is_err(), "{args:?}");
        }
    }

    #[test]
    fn descriptive_quantiles_handle_one_sample_and_interpolate() {
        assert_eq!(quantile(&[7.0], 0.1), 7.0);
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 0.5), 2.5);
        assert!((quantile(&[1.0, 2.0, 3.0, 4.0], 0.1) - 1.3).abs() < 1e-12);
        assert!((quantile(&[1.0, 2.0, 3.0, 4.0], 0.9) - 3.7).abs() < 1e-12);
    }
}
