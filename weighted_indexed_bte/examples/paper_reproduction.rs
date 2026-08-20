//! End-to-end indexed weighted-BTE run using a generated Solana weight distribution.

use std::{
    env,
    error::Error,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

use blstrs::{G1Affine, G2Affine, Scalar};
use ff::Field;
use rayon::prelude::*;
use serde::Deserialize;
use weighted_indexed_bte::{
    accept_decryption_shares, encrypt, keygen, open_batch, partial_decrypt, precompute_batch,
    prepare_decryption, setup, validate_batch, IndexedMiddleProductKernel,
};

const DEFAULT_BATCH_SIZE: usize = 8;
const DEFAULT_THREADS: usize = 1;
const MAX_WEIGHTS_FILE_BYTES: u64 = 16 * 1024 * 1024;
const WEIGHTS_PREFIX: &str = "solana_share_weights_";
const WEIGHTS_SUFFIX: &str = ".json";

#[derive(Clone, Copy, Default)]
struct Timings {
    crs_setup: Duration,
    keygen: Duration,
    fixed_kernel: Duration,
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

#[derive(Debug, Eq, PartialEq)]
enum Command {
    Run { approximation_error: String },
    Help,
}

const USAGE: &str =
    "Usage: paper_reproduction --approximation-error <ERROR>\n\nExample: --approximation-error 1/16";

fn main() -> ExitCode {
    let command = match parse_command(env::args().skip(1)) {
        Ok(command) => command,
        Err(error) => {
            eprintln!("error: {error}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    let approximation_error = match command {
        Command::Run {
            approximation_error,
        } => approximation_error,
        Command::Help => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
    };

    match run(&approximation_error) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(approximation_error: &str) -> Result<(), Box<dyn Error>> {
    let batch_size = env_usize("WEIGHTED_INDEXED_BTE_BATCH_SIZE", DEFAULT_BATCH_SIZE)?;
    if batch_size < 2 {
        return Err(invalid_input(
            "WEIGHTED_INDEXED_BTE_BATCH_SIZE must be at least 2 because encryption needs a distinct helper index",
        )
        .into());
    }
    let threads = env_usize("WEIGHTED_INDEXED_BTE_THREADS", DEFAULT_THREADS)?;
    require_positive("WEIGHTED_INDEXED_BTE_THREADS", threads)?;

    let (weights_path, path_overridden) = weights_path()?;
    let allocation = load_allocation(&weights_path, approximation_error)?;
    let profile = &allocation.profile;
    let total_weight = profile.share_count;
    let reconstruction_threshold = profile.reconstruction_threshold;
    // The distribution's q is the minimum reconstructing weight, whereas
    // indexed weighted BTE authorizes strictly greater than t.
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

    let crs_g1_points = batch_size;
    let crs_g2_points = batch_size
        .checked_mul(2)
        .ok_or_else(|| invalid_input("powers-of-tau G2 point count overflows usize"))?;
    let delta_points = batch_size;
    let gamma_points = batch_size
        .checked_mul(total_weight)
        .ok_or_else(|| invalid_input("gamma point count overflows usize"))?;
    let verification_points = party_count;
    let master_key_g1_points = delta_points
        .checked_add(verification_points)
        .ok_or_else(|| invalid_input("master-public-key G1 point count overflows usize"))?;
    let total_g1_points = crs_g1_points
        .checked_add(master_key_g1_points)
        .ok_or_else(|| invalid_input("total public G1 point count overflows usize"))?;
    let total_g2_points = crs_g2_points
        .checked_add(gamma_points)
        .ok_or_else(|| invalid_input("total public G2 point count overflows usize"))?;
    let bytes_per_g1_point = G1Affine::default().to_compressed().len();
    let bytes_per_g2_point = G2Affine::default().to_compressed().len();
    let bytes_per_scalar = Scalar::ZERO.to_bytes_le().len();
    let crs_g1_bytes = crs_g1_points
        .checked_mul(bytes_per_g1_point)
        .ok_or_else(|| invalid_input("powers-of-tau G1 byte count overflows usize"))?;
    let crs_g2_bytes = crs_g2_points
        .checked_mul(bytes_per_g2_point)
        .ok_or_else(|| invalid_input("powers-of-tau G2 byte count overflows usize"))?;
    let crs_bytes = crs_g1_bytes
        .checked_add(crs_g2_bytes)
        .ok_or_else(|| invalid_input("powers-of-tau CRS byte count overflows usize"))?;
    let delta_bytes = delta_points
        .checked_mul(bytes_per_g1_point)
        .ok_or_else(|| invalid_input("delta byte count overflows usize"))?;
    let verification_bytes = verification_points
        .checked_mul(bytes_per_g1_point)
        .ok_or_else(|| invalid_input("verification-key byte count overflows usize"))?;
    let gamma_bytes = gamma_points
        .checked_mul(bytes_per_g2_point)
        .ok_or_else(|| invalid_input("gamma byte count overflows usize"))?;
    let master_key_bytes = delta_bytes
        .checked_add(verification_bytes)
        .and_then(|bytes| bytes.checked_add(gamma_bytes))
        .ok_or_else(|| invalid_input("master-public-key byte count overflows usize"))?;
    let total_public_bytes = crs_bytes
        .checked_add(master_key_bytes)
        .ok_or_else(|| invalid_input("total public byte count overflows usize"))?;
    let ciphertext_overhead_bytes = 2usize
        .checked_mul(bytes_per_g1_point)
        .and_then(|bytes| bytes.checked_add(2 * bytes_per_scalar))
        .ok_or_else(|| invalid_input("ciphertext overhead byte count overflows usize"))?;
    let response_bytes = bytes_per_g1_point
        .checked_add(2 * bytes_per_scalar)
        .ok_or_else(|| invalid_input("server response byte count overflows usize"))?;
    let selected_response_bytes = response_bytes
        .checked_mul(selected_party_count)
        .ok_or_else(|| invalid_input("selected response byte count overflows usize"))?;
    let all_response_bytes = response_bytes
        .checked_mul(party_count)
        .ok_or_else(|| invalid_input("all-party response byte count overflows usize"))?;
    let transform_size = batch_size
        .checked_mul(2)
        .and_then(usize::checked_next_power_of_two)
        .ok_or_else(|| invalid_input("FFT transform size overflows usize"))?;
    let committee_msm_count = batch_size
        .checked_mul(selected_party_count)
        .ok_or_else(|| invalid_input("committee MSM count overflows usize"))?;
    let committee_msm_terms = batch_size
        .checked_mul(selected_weight)
        .ok_or_else(|| invalid_input("committee MSM term count overflows usize"))?;
    let opening_pair_count = batch_size
        .checked_mul(selected_party_count)
        .ok_or_else(|| invalid_input("opening pairing-input count overflows usize"))?;
    let direct_cross_term_count = batch_size
        .checked_mul(batch_size - 1)
        .ok_or_else(|| invalid_input("direct cross-term count overflows usize"))?;

    println!("Indexed weighted BTE Solana paper reproduction");
    println!("Input allocation: {}", weights_path.display());
    println!(
        "Input selection: {}",
        if path_overridden {
            "WEIGHTED_INDEXED_BTE_WEIGHTS_FILE override"
        } else {
            "lexicographically newest solana_share_weights_*.json"
        }
    );
    println!("Distribution method: {}", allocation.method);
    println!();
    println!("Selected approximation profile");
    println!("error e={}: --approximation-error", profile.error);
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
        "minimum reconstruction weight q={reconstruction_threshold}; indexed weighted-BTE threshold t={threshold_weight} because authorization is weight > t"
    );
    println!(
        "effective guaranteed reconstruction stake ratio={}",
        profile.effective_reconstruction_stake_ratio
    );
    println!();
    println!("Runtime choices");
    println!("B_max=B={batch_size}");
    println!(
        "index space n={batch_size}; ciphertext indices are fixed to 0..={}",
        batch_size - 1
    );
    println!("Rayon threads={threads}");
    println!();
    println!("Allocation summary");
    println!(
        "real parties N={party_count}, total virtual weight W={total_weight}, min/max party weight={minimum_weight}/{maximum_weight}"
    );
    println!(
        "minimal-party authorized set: {selected_party_count} parties, weight {selected_weight} >= q={reconstruction_threshold} (largest weights first, index tie-break)"
    );
    println!();
    let robustness_limit = total_weight / 2;
    println!(
        "old-paper robustness theorem requires t < floor(W/2)={robustness_limit}: {}",
        if threshold_weight < robustness_limit {
            "satisfied"
        } else {
            "NOT satisfied by this exact-half corruption threshold"
        }
    );
    println!();
    println!("Serialized public material");
    println!(
        "accounting: paper group elements with compressed encodings; derived identifiers, integer metadata, and container framing excluded"
    );
    println!(
        "compressed encodings: G1={bytes_per_g1_point}, G2={bytes_per_g2_point}, scalar={bytes_per_scalar} bytes"
    );
    println!(
        "powers-of-tau CRS pp: {crs_g1_points} G1 + {crs_g2_points} G2 points, {crs_bytes} bytes"
    );
    println!("  g_i powers: {crs_g1_points} G1 points, {crs_g1_bytes} bytes = n");
    println!(
        "  h_i powers: {crs_g2_points} G2 points, {crs_g2_bytes} bytes = 2n (missing h_(n+1))"
    );
    println!(
        "weighted master public key: {master_key_g1_points} G1 + {gamma_points} G2 points, {master_key_bytes} bytes"
    );
    println!("  delta_i: {delta_points} G1 points, {delta_bytes} bytes = n");
    println!(
        "  party inverse verification keys: {verification_points} G1 points, {verification_bytes} bytes = N"
    );
    println!("  gamma_(i,j,omega): {gamma_points} G2 points, {gamma_bytes} bytes = nW");
    println!(
        "combined pp + master public key: {total_g1_points} G1 + {total_g2_points} G2 points, {total_public_bytes} bytes"
    );
    println!("additional structured DLEq proof CRS: 0 bytes (Fiat-Shamir)");
    println!(
        "ciphertext cryptographic overhead: {ciphertext_overhead_bytes} bytes = 2 compressed G1 + 2 scalar proof elements (explicit index/framing and payload excluded)"
    );
    println!("one party cryptographic response: {response_bytes} bytes");
    println!("Table-2 all-party response download: {all_response_bytes} bytes = N * response size");
    println!(
        "selected V retained for opening: {selected_response_bytes} bytes for N'={selected_party_count} parties"
    );
    println!();
    println!("Optimized decryption work dimensions");
    if direct_cross_term_count <= transform_size / 2 {
        println!("cross terms: adaptive direct path with {direct_cross_term_count} pairing inputs");
    } else {
        println!(
            "middle product: transform size m={transform_size}, one prepared-G2/input pairing per transform point"
        );
    }
    println!(
        "committee preparation: {committee_msm_count} G2 MSMs with {committee_msm_terms} total scalar-point terms = B*N' MSMs, B*W_T terms"
    );
    println!(
        "opening: {batch_size} multi-pairings of arity {selected_party_count}, {opening_pair_count} total pairing inputs"
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;

    let mut timings = Timings::default();

    let start = Instant::now();
    let parameters = pool.install(|| setup(batch_size))?;
    timings.crs_setup = start.elapsed();

    let start = Instant::now();
    let material = pool.install(|| keygen(&parameters, weights, threshold_weight))?;
    timings.keygen = start.elapsed();
    assert_eq!(parameters.g1_point_count(), crs_g1_points);
    assert_eq!(parameters.g2_point_count(), crs_g2_points);
    assert_eq!(parameters.serialized_size_bytes(), crs_bytes);
    assert_eq!(material.public_key.g1_point_count(), master_key_g1_points);
    assert_eq!(material.public_key.g2_point_count(), gamma_points);
    assert_eq!(
        material.public_key.serialized_size_bytes(),
        master_key_bytes
    );

    let messages = (0..batch_size)
        .map(|index| {
            let mut message = vec![0u8; 32];
            message[..8].copy_from_slice(&(index as u64 + 1).to_le_bytes());
            message
        })
        .collect::<Vec<_>>();

    let start = Instant::now();
    let ciphertexts = pool.install(|| {
        messages
            .par_iter()
            .enumerate()
            .map(|(index, message)| encrypt(&parameters, &material.public_key, message, index))
            .collect::<weighted_indexed_bte::Result<Vec<_>>>()
    })?;
    timings.encryption = start.elapsed();

    let start = Instant::now();
    let batch = pool.install(|| validate_batch(&material.public_key, &ciphertexts))?;
    timings.proof_validation = start.elapsed();
    assert_eq!(
        batch.valid_count(),
        batch_size,
        "honest ciphertext proofs must all verify"
    );

    let start = Instant::now();
    let shares = pool.install(|| {
        material
            .party_keys
            .par_iter()
            .map(|party_key| partial_decrypt(party_key, &batch))
            .collect::<weighted_indexed_bte::Result<Vec<_>>>()
    })?;
    timings.partial_decryption = start.elapsed();

    let start = Instant::now();
    let accepted =
        pool.install(|| accept_decryption_shares(&material.public_key, &batch, &shares))?;
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
        pool.install(|| prepare_decryption(&material.public_key, &accepted))?;
    timings.committee_preparation = start.elapsed();

    let start = Instant::now();
    let fixed_kernel = pool.install(|| IndexedMiddleProductKernel::new(&parameters))?;
    timings.fixed_kernel = start.elapsed();
    assert_eq!(fixed_kernel.transform_size(), transform_size);

    let start = Instant::now();
    let batch_precomputation = pool.install(|| precompute_batch(&fixed_kernel, &batch))?;
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
            actual.as_deref(),
            Some(expected.as_slice()),
            "decrypted message mismatch"
        );
    }

    println!();
    println!("Measured phase durations");
    report("powers-of-tau CRS setup", timings.crs_setup, None);
    report("weighted master key generation", timings.keygen, None);
    report("fixed G2 FFT kernel", timings.fixed_kernel, None);
    report("encrypt batch", timings.encryption, Some(batch_size));
    report(
        "client proof validation",
        timings.proof_validation,
        Some(batch_size),
    );
    report(
        "partial decryptions, all N parties",
        timings.partial_decryption,
        Some(party_count),
    );
    report(
        "all-N server proofs / select V",
        timings.share_acceptance,
        None,
    );
    report(
        "committee interpolation / G2 MSMs",
        timings.committee_preparation,
        None,
    );
    report(
        "cross-term precompute",
        timings.cross_term_precomputation,
        Some(batch_size),
    );
    report(
        "weighted PRF opening / unmask",
        timings.opening,
        Some(batch_size),
    );

    let post_validation_total = timings.partial_decryption
        + timings.share_acceptance
        + timings.committee_preparation
        + timings.cross_term_precomputation
        + timings.opening;
    report(
        "post-validation cold-committee path",
        post_validation_total,
        None,
    );
    let post_validation_cached_total = post_validation_total - timings.committee_preparation;
    report(
        "post-validation cached-committee path",
        post_validation_cached_total,
        None,
    );
    report(
        "full cold decryption incl client checks",
        timings.proof_validation + post_validation_total,
        None,
    );
    report(
        "full cached decryption incl client checks",
        timings.proof_validation + post_validation_cached_total,
        None,
    );
    println!("Decryption successful.");

    Ok(())
}

fn parse_command(args: impl IntoIterator<Item = String>) -> io::Result<Command> {
    let mut args = args.into_iter();
    let mut approximation_error = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--approximation-error" => {
                let value = args.next().ok_or_else(|| {
                    invalid_input("--approximation-error requires a value such as 1/16")
                })?;
                set_approximation_error(&mut approximation_error, value)?;
            }
            _ => {
                if let Some(value) = argument.strip_prefix("--approximation-error=") {
                    set_approximation_error(&mut approximation_error, value.to_owned())?;
                } else {
                    return Err(invalid_input(format!("unknown argument {argument:?}")));
                }
            }
        }
    }

    approximation_error
        .map(|approximation_error| Command::Run {
            approximation_error,
        })
        .ok_or_else(|| invalid_input("missing required --approximation-error <ERROR> argument"))
}

fn set_approximation_error(slot: &mut Option<String>, value: String) -> io::Result<()> {
    if value.is_empty() {
        return Err(invalid_input("--approximation-error must not be empty"));
    }
    if slot.replace(value).is_some() {
        return Err(invalid_input(
            "--approximation-error may only be specified once",
        ));
    }
    Ok(())
}

fn weights_path() -> io::Result<(PathBuf, bool)> {
    if let Some(path) = env::var_os("WEIGHTED_INDEXED_BTE_WEIGHTS_FILE") {
        if path.is_empty() {
            return Err(invalid_input(
                "WEIGHTED_INDEXED_BTE_WEIGHTS_FILE must not be empty",
            ));
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
        return Err(invalid_input("--approximation-error must not be empty"));
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

fn env_usize(name: &str, default: usize) -> io::Result<usize> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|error| invalid_input(format!("{name} must be a usize: {error}"))),
        Err(env::VarError::NotPresent) => Ok(default),
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
    fn parses_approximation_error_argument() {
        assert_eq!(
            parse_command(["--approximation-error", "1/16"].map(str::to_owned)).unwrap(),
            Command::Run {
                approximation_error: "1/16".to_owned()
            }
        );
        assert_eq!(
            parse_command(["--approximation-error=1/32"].map(str::to_owned)).unwrap(),
            Command::Run {
                approximation_error: "1/32".to_owned()
            }
        );
    }

    #[test]
    fn rejects_invalid_command_lines() {
        assert!(parse_command(Vec::<String>::new()).is_err());
        assert!(parse_command(
            [
                "--approximation-error",
                "1/16",
                "--approximation-error=1/32"
            ]
            .map(str::to_owned)
        )
        .is_err());
        assert!(parse_command(["--approximation-error="].map(str::to_owned)).is_err());
        assert!(parse_command(["--unknown"].map(str::to_owned)).is_err());
        assert_eq!(
            parse_command(["--help"].map(str::to_owned)).unwrap(),
            Command::Help
        );
    }

    #[test]
    fn selects_profile_by_error_and_uses_actual_share_count() {
        let selected = parse_allocation_document(
            &document(vec![profile()]),
            Path::new("fixture.json"),
            "1/16",
        )
        .unwrap();

        assert_eq!(selected.profile.error, "1/16");
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
