//! End-to-end weighted-BTX run using a generated Solana weight distribution.

use std::{
    env,
    error::Error,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

use blstrs::{Gt, Scalar};
use group::Group;
use rayon::prelude::*;
use serde::Deserialize;
use weighted_btx::{
    accept_decryption_shares, encrypt, keygen, open_batch, partial_decrypt, precompute_batch,
    prepare_decryption, validate_batch,
};

/// Symmetric approximation-error profile selected from the distribution.
/// Change this to another generated profile such as `"1/32"` or `"1/64"`.
const APPROXIMATION_ERROR: &str = "1/128";
const DEFAULT_BATCH_SIZE: usize = 8;
const DEFAULT_THREADS: usize = 1;
const MAX_WEIGHTS_FILE_BYTES: u64 = 16 * 1024 * 1024;
const WEIGHTS_PREFIX: &str = "solana_share_weights_";
const WEIGHTS_SUFFIX: &str = ".json";

#[derive(Clone, Copy)]
struct Parameter {
    value: usize,
    overridden: bool,
}

#[derive(Clone, Copy, Default)]
struct Timings {
    keygen: Duration,
    encryption: Duration,
    proof_validation: Duration,
    partial_decryption: Duration,
    share_acceptance: Duration,
    committee_preparation: Duration,
    cross_term_precomputation: Duration,
    opening: Duration,
}

#[derive(Debug, Deserialize)]
struct AllocationDocument {
    method: String,
    target_reconstruction_ratio: String,
    allocations: Vec<AllocationProfile>,
}

#[derive(Debug, Deserialize)]
struct AllocationProfile {
    error: String,
    lower_stake_ratio: String,
    upper_stake_ratio: String,
    selected_resolution_m: usize,
    share_count: usize,
    reconstruction_threshold: usize,
    effective_reconstruction_stake_ratio: String,
    positive_validator_count: usize,
    weights: Vec<usize>,
}

#[derive(Debug)]
struct SelectedAllocation {
    method: String,
    target_reconstruction_ratio: String,
    profile: AllocationProfile,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let batch_size = env_usize("WEIGHTED_BTX_BATCH_SIZE", DEFAULT_BATCH_SIZE)?;
    require_positive("WEIGHTED_BTX_BATCH_SIZE", batch_size.value)?;
    let threads = env_usize("WEIGHTED_BTX_THREADS", DEFAULT_THREADS)?;
    require_positive("WEIGHTED_BTX_THREADS", threads.value)?;

    let (weights_path, path_overridden) = weights_path()?;
    let allocation = load_allocation(&weights_path, APPROXIMATION_ERROR)?;
    let profile = &allocation.profile;
    let total_weight = profile.share_count;
    let reconstruction_threshold = profile.reconstruction_threshold;
    // The distribution's q is the minimum reconstructing weight, whereas
    // weighted BTX authorizes strictly greater than t.
    let threshold_weight = reconstruction_threshold
        .checked_sub(1)
        .ok_or_else(|| invalid_data("reconstruction threshold q must be positive"))?;
    let weights = &profile.weights;
    let party_count = weights.len();
    let minimum_weight = weights
        .iter()
        .copied()
        .min()
        .ok_or_else(|| invalid_data("the selected allocation is empty"))?;
    let maximum_weight = weights
        .iter()
        .copied()
        .max()
        .ok_or_else(|| invalid_data("the selected allocation is empty"))?;

    let (selected_parties, selected_weight) =
        minimal_authorized_parties(weights, threshold_weight)?;
    let selected_party_count = selected_parties.len();

    let core_points = batch_size
        .value
        .checked_mul(2)
        .and_then(|twice| twice.checked_sub(1))
        .and_then(|offset_count| offset_count.checked_mul(total_weight))
        .ok_or_else(|| invalid_input("core public-key point count overflows usize"))?;
    let verification_points = party_count
        .checked_mul(batch_size.value)
        .ok_or_else(|| invalid_input("verification-key point count overflows usize"))?;
    let public_key_points = core_points
        .checked_add(verification_points)
        .ok_or_else(|| invalid_input("total public-key point count overflows usize"))?;

    println!("Weighted BTX Solana paper reproduction");
    println!("Input allocation: {}", weights_path.display());
    println!(
        "Input selection: {}",
        if path_overridden {
            "WEIGHTED_BTX_WEIGHTS_FILE override"
        } else {
            "lexicographically newest solana_share_weights_*.json"
        }
    );
    println!("Distribution method: {}", allocation.method);
    println!();
    println!("Selected approximation profile");
    println!(
        "error e={}: top-level APPROXIMATION_ERROR configuration",
        profile.error
    );
    println!(
        "target stake ratio={}, interval=[{}, {}]",
        allocation.target_reconstruction_ratio,
        profile.lower_stake_ratio,
        profile.upper_stake_ratio
    );
    println!(
        "nominal resolution M={}, actual cryptographic share count W={total_weight}",
        profile.selected_resolution_m
    );
    println!(
        "minimum reconstruction weight q={reconstruction_threshold}; weighted-BTX threshold t={threshold_weight} because authorization is weight > t"
    );
    println!(
        "effective guaranteed reconstruction stake ratio={}",
        profile.effective_reconstruction_stake_ratio
    );
    println!();
    println!("Runtime choices");
    println!(
        "B_max=B={}: {}",
        batch_size.value,
        choice_label(
            batch_size.overridden,
            "environment override; the paper gives no batch parameter (implementation default 8)",
            "implementation default; the weighted paper gives no batch parameter",
        )
    );
    println!(
        "Rayon threads={}: {}",
        threads.value,
        choice_label(
            threads.overridden,
            "environment override; the paper gives no thread count (implementation default 1)",
            "implementation default; the paper gives no thread count",
        )
    );
    println!();
    println!("Allocation summary");
    println!(
        "real parties N={party_count}, total virtual weight W={total_weight}, min/max party weight={minimum_weight}/{maximum_weight}"
    );
    println!(
        "minimal-party authorized set: {selected_party_count} parties, weight {selected_weight} >= q={reconstruction_threshold} (largest weights first, index tie-break)"
    );
    println!();
    println!("Estimated public G2 material (excluding the NIZK CRS)");
    println!("core D_(omega,d): {core_points} points = (2B_max-1)W");
    println!("party verification keys: {verification_points} points = N*B_max");
    println!("public decryption key total: {public_key_points} points");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.value)
        .build()?;

    let mut timings = Timings::default();

    let start = Instant::now();
    let material = pool.install(|| keygen(batch_size.value, weights, threshold_weight))?;
    timings.keygen = start.elapsed();

    let messages = (0..batch_size.value)
        .map(|index| Gt::generator() * Scalar::from((index as u64).wrapping_add(1)))
        .collect::<Vec<_>>();

    let start = Instant::now();
    let ciphertexts = pool.install(|| {
        messages
            .par_iter()
            .map(|message| encrypt(&material.encryption_key, *message))
            .collect::<Vec<_>>()
    });
    timings.encryption = start.elapsed();

    let start = Instant::now();
    let batch = pool.install(|| validate_batch(&material.decryption_key, &ciphertexts))?;
    timings.proof_validation = start.elapsed();
    assert_eq!(
        batch.valid_count(),
        batch_size.value,
        "honest ciphertext proofs must all verify"
    );

    let start = Instant::now();
    let shares = pool.install(|| {
        selected_parties
            .par_iter()
            .map(|party_index| partial_decrypt(&material.party_keys[*party_index], &batch))
            .collect::<weighted_btx::Result<Vec<_>>>()
    })?;
    timings.partial_decryption = start.elapsed();

    let start = Instant::now();
    let accepted =
        pool.install(|| accept_decryption_shares(&material.decryption_key, &batch, &shares))?;
    timings.share_acceptance = start.elapsed();
    assert_eq!(
        accepted.party_count(),
        selected_party_count,
        "all honest selected parties must be accepted"
    );
    assert_eq!(
        accepted.accepted_weight(),
        selected_weight,
        "accepted weight must equal the selected committee weight"
    );

    let start = Instant::now();
    let decryption_precomputation =
        pool.install(|| prepare_decryption(&material.decryption_key, &accepted))?;
    timings.committee_preparation = start.elapsed();

    let start = Instant::now();
    let batch_precomputation =
        pool.install(|| precompute_batch(&decryption_precomputation, &batch))?;
    timings.cross_term_precomputation = start.elapsed();

    let start = Instant::now();
    let decrypted = pool.install(|| {
        open_batch(
            &decryption_precomputation,
            &accepted,
            &batch,
            &ciphertexts,
            &batch_precomputation,
        )
    })?;
    timings.opening = start.elapsed();

    for (actual, expected) in decrypted.iter().zip(&messages) {
        assert_eq!(
            actual.as_ref(),
            Some(expected),
            "decrypted message mismatch"
        );
    }

    println!();
    println!("Measured phase durations");
    report("trusted key generation", timings.keygen, None);
    report("encrypt batch", timings.encryption, Some(batch_size.value));
    report(
        "client proof validation",
        timings.proof_validation,
        Some(batch_size.value),
    );
    report(
        "partial decryptions, all selected",
        timings.partial_decryption,
        Some(selected_party_count),
    );
    report("batched share acceptance", timings.share_acceptance, None);
    report(
        "committee / FFT preparation",
        timings.committee_preparation,
        None,
    );
    report(
        "cross-term precompute",
        timings.cross_term_precomputation,
        Some(batch_size.value),
    );
    report("opening", timings.opening, Some(batch_size.value));

    let decryption_total = timings.partial_decryption
        + timings.share_acceptance
        + timings.committee_preparation
        + timings.cross_term_precomputation
        + timings.opening;
    let end_to_end =
        timings.keygen + timings.encryption + timings.proof_validation + decryption_total;
    report("decryption pipeline total", decryption_total, None);
    report("measured end-to-end total", end_to_end, None);
    println!("Decryption successful.");

    Ok(())
}

fn weights_path() -> io::Result<(PathBuf, bool)> {
    if let Some(path) = env::var_os("WEIGHTED_BTX_WEIGHTS_FILE") {
        if path.is_empty() {
            return Err(invalid_input("WEIGHTED_BTX_WEIGHTS_FILE must not be empty"));
        }
        return Ok((PathBuf::from(path), true));
    }

    let data_directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("scripts")
        .join("data");
    let mut candidates = Vec::new();
    let entries = fs::read_dir(&data_directory).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to read Solana data directory {}: {error}",
                data_directory.display()
            ),
        )
    })?;
    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if file_name.starts_with(WEIGHTS_PREFIX) && file_name.ends_with(WEIGHTS_SUFFIX) {
            candidates.push(entry.path());
        }
    }
    candidates.sort();
    candidates.pop().map(|path| (path, false)).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "no {WEIGHTS_PREFIX}*{WEIGHTS_SUFFIX} file found in {}",
                data_directory.display()
            ),
        )
    })
}

fn load_allocation(path: &Path, requested_error: &str) -> io::Result<SelectedAllocation> {
    let file = fs::File::open(path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to read {}: {error}", path.display()),
        )
    })?;
    let mut json = String::new();
    file.take(MAX_WEIGHTS_FILE_BYTES + 1)
        .read_to_string(&mut json)
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("failed to read {}: {error}", path.display()),
            )
        })?;
    if json.len() as u64 > MAX_WEIGHTS_FILE_BYTES {
        return Err(invalid_data(format!(
            "allocation JSON {} exceeds the {}-byte input limit",
            path.display(),
            MAX_WEIGHTS_FILE_BYTES
        )));
    }
    parse_allocation_document(&json, path, requested_error)
}

fn parse_allocation_document(
    json: &str,
    path: &Path,
    requested_error: &str,
) -> io::Result<SelectedAllocation> {
    if requested_error.is_empty() {
        return Err(invalid_input("APPROXIMATION_ERROR must not be empty"));
    }

    let document: AllocationDocument = serde_json::from_str(json).map_err(|error| {
        invalid_data(format!(
            "failed to parse allocation JSON {}: {error}",
            path.display()
        ))
    })?;
    let AllocationDocument {
        method,
        target_reconstruction_ratio,
        allocations,
    } = document;
    if method.is_empty() || target_reconstruction_ratio.is_empty() {
        return Err(invalid_data(format!(
            "allocation document {} has empty method or target reconstruction ratio",
            path.display()
        )));
    }

    let available_errors = allocations
        .iter()
        .map(|allocation| allocation.error.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let mut matching = allocations
        .into_iter()
        .filter(|allocation| allocation.error == requested_error);
    let profile = matching.next().ok_or_else(|| {
        invalid_data(format!(
            "{} has no allocation for error {requested_error}; available errors: {available_errors}",
            path.display(),
        ))
    })?;
    if matching.next().is_some() {
        return Err(invalid_data(format!(
            "{} contains more than one allocation for error {requested_error}",
            path.display()
        )));
    }

    if profile.selected_resolution_m == 0 {
        return Err(invalid_data(format!(
            "allocation {requested_error} in {} has zero nominal resolution M",
            path.display()
        )));
    }
    if profile.share_count == 0 || profile.weights.is_empty() {
        return Err(invalid_data(format!(
            "allocation {requested_error} in {} is empty",
            path.display()
        )));
    }
    if profile.reconstruction_threshold == 0
        || profile.reconstruction_threshold > profile.share_count
    {
        return Err(invalid_data(format!(
            "allocation {requested_error} in {} has reconstruction threshold {} outside 1..={}",
            path.display(),
            profile.reconstruction_threshold,
            profile.share_count
        )));
    }
    if profile.positive_validator_count != profile.weights.len() {
        return Err(invalid_data(format!(
            "allocation {requested_error} in {} reports {} positive validators but contains {} weights",
            path.display(),
            profile.positive_validator_count,
            profile.weights.len()
        )));
    }
    if profile.lower_stake_ratio.is_empty()
        || profile.upper_stake_ratio.is_empty()
        || profile.effective_reconstruction_stake_ratio.is_empty()
    {
        return Err(invalid_data(format!(
            "allocation {requested_error} in {} has empty ratio metadata",
            path.display()
        )));
    }
    if let Some((party_index, _)) = profile
        .weights
        .iter()
        .enumerate()
        .find(|(_, weight)| **weight == 0)
    {
        return Err(invalid_data(format!(
            "allocation {requested_error} in {} contains zero weight at party index {party_index}",
            path.display()
        )));
    }
    let actual_total = profile.weights.iter().try_fold(0usize, |total, weight| {
        total.checked_add(*weight).ok_or_else(|| {
            invalid_data(format!(
                "allocation {requested_error} in {} overflows usize",
                path.display()
            ))
        })
    })?;
    if actual_total != profile.share_count {
        return Err(invalid_data(format!(
            "allocation {requested_error} in {} reports W={} but weights sum to {actual_total}",
            path.display(),
            profile.share_count
        )));
    }

    Ok(SelectedAllocation {
        method,
        target_reconstruction_ratio,
        profile,
    })
}

fn minimal_authorized_parties(
    weights: &[usize],
    threshold_weight: usize,
) -> io::Result<(Vec<usize>, usize)> {
    let mut ranked = weights.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|(left_index, left_weight), (right_index, right_weight)| {
        right_weight
            .cmp(left_weight)
            .then_with(|| left_index.cmp(right_index))
    });

    let mut selected = Vec::new();
    let mut selected_weight = 0usize;
    for (party_index, weight) in ranked {
        selected_weight = selected_weight
            .checked_add(weight)
            .ok_or_else(|| invalid_data("selected committee weight overflows usize"))?;
        selected.push(party_index);
        if selected_weight > threshold_weight {
            return Ok((selected, selected_weight));
        }
    }

    Err(invalid_input(format!(
        "total allocation weight {selected_weight} does not exceed threshold {threshold_weight}"
    )))
}

fn env_usize(name: &str, default: usize) -> io::Result<Parameter> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map(|value| Parameter {
                value,
                overridden: true,
            })
            .map_err(|error| invalid_input(format!("{name} must be a usize: {error}"))),
        Err(env::VarError::NotPresent) => Ok(Parameter {
            value: default,
            overridden: false,
        }),
        Err(env::VarError::NotUnicode(_)) => {
            Err(invalid_input(format!("{name} is not valid Unicode")))
        }
    }
}

fn require_positive(name: &str, value: usize) -> io::Result<()> {
    if value == 0 {
        Err(invalid_input(format!("{name} must be at least 1")))
    } else {
        Ok(())
    }
}

fn choice_label<'a>(overridden: bool, override_label: &'a str, default_label: &'a str) -> &'a str {
    if overridden {
        override_label
    } else {
        default_label
    }
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn report(label: &str, duration: Duration, items: Option<usize>) {
    match items {
        Some(items) => println!(
            "{label:38} {:>10.3} ms total  {:>8.3} ms/item",
            duration.as_secs_f64() * 1_000.0,
            duration.as_secs_f64() * 1_000.0 / items as f64
        ),
        None => println!("{label:38} {:>10.3} ms", duration.as_secs_f64() * 1_000.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn profile() -> Value {
        json!({
            "error": "1/16",
            "lower_stake_ratio": "7/16",
            "upper_stake_ratio": "9/16",
            "selected_resolution_m": 7,
            "share_count": 6,
            "reconstruction_threshold": 4,
            "effective_reconstruction_stake_ratio": "0.7",
            "positive_validator_count": 3,
            "weights": [3, 2, 1]
        })
    }

    fn document(allocations: Vec<Value>) -> String {
        json!({
            "method": "aptos_dkg_nearest_rounding_q64_64",
            "target_reconstruction_ratio": "1/2",
            "allocations": allocations
        })
        .to_string()
    }

    #[test]
    fn selects_profile_by_error_and_uses_actual_share_count() {
        let selected = parse_allocation_document(
            &document(vec![profile()]),
            Path::new("fixture.json"),
            "1/16",
        )
        .unwrap();

        assert_eq!(selected.profile.error, APPROXIMATION_ERROR);
        assert_eq!(selected.profile.selected_resolution_m, 7);
        assert_eq!(selected.profile.share_count, 6);
        assert_eq!(selected.profile.weights, [3, 2, 1]);
        assert_eq!(selected.profile.reconstruction_threshold - 1, 3);
    }

    #[test]
    fn rejects_missing_and_duplicate_error_profiles() {
        let missing = parse_allocation_document(
            &document(vec![profile()]),
            Path::new("fixture.json"),
            "1/32",
        )
        .unwrap_err();
        assert!(missing.to_string().contains("available errors: 1/16"));

        let duplicate_profile = profile();
        let duplicate = parse_allocation_document(
            &document(vec![duplicate_profile.clone(), duplicate_profile]),
            Path::new("fixture.json"),
            "1/16",
        )
        .unwrap_err();
        assert!(duplicate.to_string().contains("more than one allocation"));
    }

    #[test]
    fn rejects_inconsistent_profile_counts_and_thresholds() {
        let mut cases = Vec::new();

        let mut wrong_party_count = profile();
        wrong_party_count["positive_validator_count"] = json!(2);
        cases.push(wrong_party_count);

        let mut wrong_share_count = profile();
        wrong_share_count["share_count"] = json!(7);
        cases.push(wrong_share_count);

        let mut zero_weight = profile();
        zero_weight["weights"] = json!([3, 0, 3]);
        cases.push(zero_weight);

        let mut zero_threshold = profile();
        zero_threshold["reconstruction_threshold"] = json!(0);
        cases.push(zero_threshold);

        for malformed in cases {
            assert!(parse_allocation_document(
                &document(vec![malformed]),
                Path::new("fixture.json"),
                "1/16"
            )
            .is_err());
        }
    }
}
