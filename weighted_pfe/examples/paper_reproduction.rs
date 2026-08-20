//! End-to-end weighted partial-fraction BTE run on a generated Solana allocation.

use std::{
    env,
    error::Error,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

use blstrs::{G1Affine, G2Affine, Gt, Scalar};
use ff::Field;
use group::Group;
use rayon::prelude::*;
use serde::Deserialize;
use weighted_pfe::{
    accept_decryption_shares, encrypt, keygen, open_batch, partial_decrypt, precompute_batch,
    prepare_decryption, validate_batch, CauchyKernel, Ciphertext, KeyMaterial,
};

const DEFAULT_BATCH_SIZE: usize = 32;
const DEFAULT_THREADS: usize = 12;
const DEFAULT_REPETITIONS: usize = 1;
const MAX_WEIGHTS_FILE_BYTES: u64 = 16 * 1024 * 1024;
const WEIGHTS_PREFIX: &str = "solana_share_weights_";
const WEIGHTS_SUFFIX: &str = ".json";

// Table 3 counts a compressed BLS12-381 GT element as six base-field
// elements. The implementation prefixes that torus encoding with an
// identity/non-identity tag so its canonical encoding is total and fixed-width.
const PAPER_GT_BYTES: usize = 6 * 48;

#[derive(Clone, Copy, Default)]
struct OneTimeTimings {
    keygen: Duration,
    encryption: Duration,
    fixed_kernel: Duration,
}

#[derive(Clone, Copy, Default)]
struct OnlineTimings {
    proof_validation: Duration,
    partial_decryption: Duration,
    share_acceptance: Duration,
    committee_preparation: Duration,
    cauchy_precomputation: Duration,
    opening: Duration,
}

impl OnlineTimings {
    fn add_assign(&mut self, other: Self) {
        self.proof_validation += other.proof_validation;
        self.partial_decryption += other.partial_decryption;
        self.share_acceptance += other.share_acceptance;
        self.committee_preparation += other.committee_preparation;
        self.cauchy_precomputation += other.cauchy_precomputation;
        self.opening += other.opening;
    }

    fn averaged(self, repetitions: usize) -> Self {
        Self {
            proof_validation: div_duration(self.proof_validation, repetitions),
            partial_decryption: div_duration(self.partial_decryption, repetitions),
            share_acceptance: div_duration(self.share_acceptance, repetitions),
            committee_preparation: div_duration(self.committee_preparation, repetitions),
            cauchy_precomputation: div_duration(self.cauchy_precomputation, repetitions),
            opening: div_duration(self.opening, repetitions),
        }
    }

    fn cold_committee_total(self) -> Duration {
        self.proof_validation
            + self.partial_decryption
            + self.share_acceptance
            + self.committee_preparation
            + self.cauchy_precomputation
            + self.opening
    }

    fn cached_committee_total(self) -> Duration {
        self.proof_validation
            + self.partial_decryption
            + self.share_acceptance
            + self.cauchy_precomputation
            + self.opening
    }
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

#[derive(Debug, Eq, PartialEq)]
struct SerializedSizes {
    g1_bytes: usize,
    g2_bytes: usize,
    scalar_bytes: usize,
    canonical_gt_bytes: usize,
    verification_points: usize,
    verification_bytes: usize,
    core_points: usize,
    core_bytes: usize,
    public_decryption_points: usize,
    public_decryption_bytes: usize,
    paper_public_bytes: usize,
    canonical_public_bytes: usize,
    party_secret_key_bytes: usize,
    all_secret_key_bytes: usize,
    proof_bytes: usize,
    paper_ciphertext_without_proof_bytes: usize,
    canonical_ciphertext_without_proof_bytes: usize,
    paper_ciphertext_with_proof_bytes: usize,
    canonical_ciphertext_with_proof_bytes: usize,
    paper_batch_without_proofs_bytes: usize,
    canonical_batch_without_proofs_bytes: usize,
    paper_batch_with_proofs_bytes: usize,
    canonical_batch_with_proofs_bytes: usize,
    party_response_bytes: usize,
    selected_response_bytes: usize,
}

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
    let batch_size = env_usize("WEIGHTED_PFE_BATCH_SIZE", DEFAULT_BATCH_SIZE)?;
    if batch_size == 0 || !batch_size.is_power_of_two() {
        return Err(invalid_input("WEIGHTED_PFE_BATCH_SIZE must be a nonzero power of two").into());
    }
    let threads = env_usize("WEIGHTED_PFE_THREADS", DEFAULT_THREADS)?;
    require_positive("WEIGHTED_PFE_THREADS", threads)?;
    let repetitions = env_usize("WEIGHTED_PFE_REPETITIONS", DEFAULT_REPETITIONS)?;
    require_positive("WEIGHTED_PFE_REPETITIONS", repetitions)?;

    let (weights_path, path_overridden) = weights_path()?;
    let allocation = load_allocation(&weights_path, approximation_error)?;
    let profile = &allocation.profile;
    let total_weight = profile.share_count;
    let reconstruction_threshold = profile.reconstruction_threshold;
    // The allocation's q is the minimum reconstructing weight. Construction 5
    // authorizes weight strictly greater than t, so its polynomial degree is q-1.
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

    let sizes = serialized_sizes(batch_size, party_count, total_weight, selected_party_count)?;
    let twice_batch = checked_mul(2, batch_size, "twice-batch count")?;
    let proof_msm_bases = twice_batch
        .checked_add(1)
        .ok_or_else(|| invalid_input("proof MSM base count overflows usize"))?;
    let partial_msm_terms = checked_mul(
        selected_party_count,
        batch_size,
        "partial-decryption MSM term count",
    )?;
    let acceptance_g2_msms = batch_size;
    let acceptance_g2_terms = checked_mul(
        batch_size,
        selected_party_count,
        "acceptance G2 MSM term count",
    )?;
    let committee_party_msms = checked_mul(
        batch_size,
        selected_party_count,
        "committee party-block MSM count",
    )?;
    let committee_one_table_terms = checked_mul(
        batch_size,
        selected_weight,
        "committee one-table MSM term count",
    )?;
    let committee_all_terms = checked_mul(
        2,
        committee_one_table_terms,
        "committee total MSM term count",
    )?;
    let committee_party_additions = committee_party_msms
        .checked_sub(batch_size)
        .expect("an authorized committee contains at least one party");
    let committee_normalized_points = committee_party_msms
        .checked_add(twice_batch)
        .ok_or_else(|| invalid_input("committee normalization point count overflows usize"))?;
    let acceptance_pairing_arity = batch_size
        .checked_add(1)
        .ok_or_else(|| invalid_input("acceptance pairing arity overflows usize"))?;
    let opening_multi_pair_inputs = checked_mul(
        batch_size,
        selected_party_count,
        "opening multi-pairing input count",
    )?;

    println!("Weighted PFE Solana paper reproduction (Construction 5)");
    println!("Input allocation: {}", weights_path.display());
    println!(
        "Input selection: {}",
        if path_overridden {
            "WEIGHTED_PFE_WEIGHTS_FILE override"
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
        "minimum reconstruction weight q={reconstruction_threshold}; weighted-PFE threshold t={threshold_weight} because authorization is weight > t"
    );
    println!(
        "effective guaranteed reconstruction stake ratio={}",
        profile.effective_reconstruction_stake_ratio
    );
    println!();
    println!("Runtime choices");
    println!("B_max=B={batch_size}");
    println!("Rayon threads={threads}");
    println!("measured online repetitions={repetitions}, plus one unmeasured online warm-up");
    println!();
    println!("Allocation summary");
    println!(
        "real parties N={party_count}, total virtual weight W={total_weight}, min/max party weight={minimum_weight}/{maximum_weight}"
    );
    println!(
        "minimal-party authorized set: tau={selected_party_count} parties, W_T={selected_weight} >= q={reconstruction_threshold} (largest weights first, index tie-break)"
    );
    println!();
    print_serialized_sizes(
        &sizes,
        batch_size,
        party_count,
        total_weight,
        selected_party_count,
    );
    println!();
    println!("Optimized online work dimensions");
    println!("ctxtCheck(B): one randomized G1 MSM with {proof_msm_bases}=2B+1 bases");
    println!(
        "selected PreDec: {selected_party_count} G1 MSMs of {batch_size} terms ({partial_msm_terms} scalar-point terms)"
    );
    println!(
        "batched acceptance: one {selected_party_count}-term G1 MSM, {acceptance_g2_msms} G2 MSMs of {selected_party_count} terms ({acceptance_g2_terms} terms), and one pairing product of arity {}",
        acceptance_pairing_arity
    );
    println!(
        "committee preparation: {committee_party_msms} party-block G2 MSMs plus {batch_size} whole-committee G2 MSMs, {committee_all_terms}=2B*W_T total scalar-point terms"
    );
    println!(
        "committee finishing: {batch_size} size-2 G2 mask combinations, {} party-key additions, normalization of {} points, and {} prepared line tables",
        committee_party_additions,
        committee_normalized_points,
        twice_batch,
    );
    println!(
        "Cauchy precompute: forward+inverse size-{batch_size} G1 transforms, {batch_size} prepared-G2 pairings, and forward+inverse cyclotomic transforms"
    );
    println!(
        "opening: {batch_size} multi-pairings of arity {selected_party_count} ({opening_multi_pair_inputs} inputs) plus {batch_size} prepared-G2 mask pairings; all target terms fuse before {batch_size} hard final exponentiations"
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    let mut one_time = OneTimeTimings::default();

    let start = Instant::now();
    let material = pool.install(|| keygen(batch_size, weights, threshold_weight))?;
    one_time.keygen = start.elapsed();
    assert_eq!(material.decryption_key.party_count(), party_count);
    assert_eq!(material.decryption_key.total_weight(), total_weight);
    assert_eq!(
        material.decryption_key.verification_key_count(),
        sizes.verification_points
    );
    assert_eq!(
        material.decryption_key.core_g2_point_count(),
        sizes.core_points
    );
    assert_eq!(
        material.decryption_key.g2_point_count(),
        sizes.public_decryption_points
    );
    assert_eq!(
        material.decryption_key.serialized_size_bytes(),
        sizes.public_decryption_bytes
    );
    assert_eq!(
        material.encryption_key.serialized_size_bytes(),
        sizes.canonical_gt_bytes
    );
    assert!(material
        .party_keys
        .iter()
        .all(|party_key| party_key.scalar_count() == 1));

    let messages = (0..batch_size)
        .map(|index| Gt::generator() * Scalar::from((index as u64).wrapping_add(1)))
        .collect::<Vec<_>>();

    let start = Instant::now();
    let ciphertexts = pool.install(|| {
        messages
            .par_iter()
            .map(|message| encrypt(&material.encryption_key, *message))
            .collect::<Vec<_>>()
    });
    one_time.encryption = start.elapsed();
    assert!(ciphertexts.iter().all(|ciphertext| {
        ciphertext.serialized_size_bytes() == sizes.canonical_ciphertext_with_proof_bytes
    }));

    let start = Instant::now();
    let kernel = pool.install(|| CauchyKernel::new(&material.decryption_key))?;
    one_time.fixed_kernel = start.elapsed();
    assert_eq!(kernel.batch_size(), batch_size);

    // Warm all phase-specific allocation, normalization, pairing, and Rayon paths
    // without charging that first execution to the reported averages.
    run_online(
        &pool,
        &material,
        &kernel,
        &ciphertexts,
        &messages,
        &selected_parties,
        selected_weight,
    )?;

    let mut total = OnlineTimings::default();
    for _ in 0..repetitions {
        total.add_assign(run_online(
            &pool,
            &material,
            &kernel,
            &ciphertexts,
            &messages,
            &selected_parties,
            selected_weight,
        )?);
    }
    let average = total.averaged(repetitions);

    println!();
    println!("One-time setup and input preparation");
    report("trusted weighted key generation", one_time.keygen, None);
    report("encrypt batch", one_time.encryption, Some(batch_size));
    report("fixed Cauchy FFT kernel", one_time.fixed_kernel, None);
    println!();
    println!("Average over {repetitions} measured online repetition(s)");
    report(
        "ctxtCheck(B), batched client proofs",
        average.proof_validation,
        Some(batch_size),
    );
    report(
        "PreDec(B), all selected parties",
        average.partial_decryption,
        Some(selected_party_count),
    );
    report(
        "batched response acceptance",
        average.share_acceptance,
        Some(selected_party_count),
    );
    report(
        "committee interpolation / G2 MSMs",
        average.committee_preparation,
        Some(batch_size),
    );
    report(
        "ciphertext-dependent Cauchy precompute",
        average.cauchy_precomputation,
        Some(batch_size),
    );
    report(
        "weighted opening / unmask",
        average.opening,
        Some(batch_size),
    );
    report(
        "full cold-committee path incl checks",
        average.cold_committee_total(),
        None,
    );
    report(
        "full cached-committee path incl checks",
        average.cached_committee_total(),
        None,
    );
    println!("Decryption successful in the warm-up and every measured repetition.");

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

fn run_online(
    pool: &rayon::ThreadPool,
    material: &KeyMaterial,
    kernel: &CauchyKernel,
    ciphertexts: &[Ciphertext],
    messages: &[Gt],
    selected_parties: &[usize],
    selected_weight: usize,
) -> Result<OnlineTimings, Box<dyn Error>> {
    let start = Instant::now();
    let batch = pool.install(|| validate_batch(&material.encryption_key, ciphertexts))?;
    let proof_validation = start.elapsed();
    assert_eq!(batch.batch_size(), ciphertexts.len());

    let start = Instant::now();
    let shares = pool.install(|| {
        selected_parties
            .par_iter()
            .map(|party_index| partial_decrypt(&material.party_keys[*party_index], &batch))
            .collect::<weighted_pfe::Result<Vec<_>>>()
    })?;
    let partial_decryption = start.elapsed();

    let start = Instant::now();
    let accepted =
        pool.install(|| accept_decryption_shares(&material.decryption_key, &batch, &shares))?;
    let share_acceptance = start.elapsed();
    assert_eq!(
        accepted.party_count(),
        selected_parties.len(),
        "all honest selected parties must be accepted"
    );
    assert_eq!(
        accepted.accepted_weight(),
        selected_weight,
        "accepted weight must equal the selected committee weight"
    );
    assert!(
        accepted.rejected_parties().is_empty(),
        "honest responses must not be rejected"
    );

    let start = Instant::now();
    let decryption = pool.install(|| prepare_decryption(&material.decryption_key, &accepted))?;
    let committee_preparation = start.elapsed();
    assert_eq!(decryption.party_count(), selected_parties.len());
    assert_eq!(decryption.accepted_weight(), selected_weight);

    let start = Instant::now();
    let batch_precomputation = pool.install(|| precompute_batch(kernel, &decryption, &batch))?;
    let cauchy_precomputation = start.elapsed();

    let start = Instant::now();
    let decrypted = pool.install(|| {
        open_batch(
            &material.decryption_key,
            kernel,
            &decryption,
            &accepted,
            &batch,
            ciphertexts,
            &batch_precomputation,
        )
    })?;
    let opening = start.elapsed();
    assert_eq!(decrypted, messages, "decrypted message mismatch");

    Ok(OnlineTimings {
        proof_validation,
        partial_decryption,
        share_acceptance,
        committee_preparation,
        cauchy_precomputation,
        opening,
    })
}

fn serialized_sizes(
    batch_size: usize,
    party_count: usize,
    total_weight: usize,
    selected_party_count: usize,
) -> io::Result<SerializedSizes> {
    let g1_bytes = G1Affine::default().to_compressed().len();
    let g2_bytes = G2Affine::default().to_compressed().len();
    let scalar_bytes = Scalar::ZERO.to_bytes_le().len();
    let canonical_gt_bytes = PAPER_GT_BYTES
        .checked_add(1)
        .ok_or_else(|| invalid_input("canonical GT byte count overflows usize"))?;

    let verification_points = checked_mul(party_count, batch_size, "verification-key point count")?;
    let twice_weight = checked_mul(2, total_weight, "twice-weight count")?;
    let core_points = checked_mul(twice_weight, batch_size, "core G2 point count")?
        .checked_add(1)
        .ok_or_else(|| invalid_input("core G2 point count overflows usize"))?;
    let public_decryption_points = core_points
        .checked_add(verification_points)
        .ok_or_else(|| invalid_input("public decryption-key point count overflows usize"))?;
    let verification_bytes =
        checked_mul(verification_points, g2_bytes, "verification-key byte count")?;
    let core_bytes = checked_mul(core_points, g2_bytes, "core G2 byte count")?;
    let public_decryption_bytes = checked_mul(
        public_decryption_points,
        g2_bytes,
        "public decryption-key byte count",
    )?;
    let paper_public_bytes = public_decryption_bytes
        .checked_add(PAPER_GT_BYTES)
        .ok_or_else(|| invalid_input("paper public-material byte count overflows usize"))?;
    let canonical_public_bytes = public_decryption_bytes
        .checked_add(canonical_gt_bytes)
        .ok_or_else(|| invalid_input("canonical public-material byte count overflows usize"))?;

    let party_secret_key_bytes = scalar_bytes;
    let all_secret_key_bytes = checked_mul(
        party_count,
        party_secret_key_bytes,
        "all secret-key byte count",
    )?;
    let proof_bytes = g1_bytes
        .checked_add(scalar_bytes)
        .ok_or_else(|| invalid_input("proof byte count overflows usize"))?;
    let paper_ciphertext_without_proof_bytes = g1_bytes
        .checked_add(PAPER_GT_BYTES)
        .ok_or_else(|| invalid_input("paper ciphertext byte count overflows usize"))?;
    let canonical_ciphertext_without_proof_bytes = g1_bytes
        .checked_add(canonical_gt_bytes)
        .ok_or_else(|| invalid_input("canonical ciphertext byte count overflows usize"))?;
    let paper_ciphertext_with_proof_bytes = paper_ciphertext_without_proof_bytes
        .checked_add(proof_bytes)
        .ok_or_else(|| invalid_input("paper ciphertext-with-proof byte count overflows usize"))?;
    let canonical_ciphertext_with_proof_bytes = canonical_ciphertext_without_proof_bytes
        .checked_add(proof_bytes)
        .ok_or_else(|| {
            invalid_input("canonical ciphertext-with-proof byte count overflows usize")
        })?;
    let paper_batch_without_proofs_bytes = checked_mul(
        batch_size,
        paper_ciphertext_without_proof_bytes,
        "paper proof-free ciphertext-batch byte count",
    )?;
    let canonical_batch_without_proofs_bytes = checked_mul(
        batch_size,
        canonical_ciphertext_without_proof_bytes,
        "canonical proof-free ciphertext-batch byte count",
    )?;
    let paper_batch_with_proofs_bytes = checked_mul(
        batch_size,
        paper_ciphertext_with_proof_bytes,
        "paper ciphertext-batch-with-proofs byte count",
    )?;
    let canonical_batch_with_proofs_bytes = checked_mul(
        batch_size,
        canonical_ciphertext_with_proof_bytes,
        "canonical ciphertext-batch-with-proofs byte count",
    )?;
    let party_response_bytes = g1_bytes;
    let selected_response_bytes = checked_mul(
        selected_party_count,
        party_response_bytes,
        "selected response byte count",
    )?;

    Ok(SerializedSizes {
        g1_bytes,
        g2_bytes,
        scalar_bytes,
        canonical_gt_bytes,
        verification_points,
        verification_bytes,
        core_points,
        core_bytes,
        public_decryption_points,
        public_decryption_bytes,
        paper_public_bytes,
        canonical_public_bytes,
        party_secret_key_bytes,
        all_secret_key_bytes,
        proof_bytes,
        paper_ciphertext_without_proof_bytes,
        canonical_ciphertext_without_proof_bytes,
        paper_ciphertext_with_proof_bytes,
        canonical_ciphertext_with_proof_bytes,
        paper_batch_without_proofs_bytes,
        canonical_batch_without_proofs_bytes,
        paper_batch_with_proofs_bytes,
        canonical_batch_with_proofs_bytes,
        party_response_bytes,
        selected_response_bytes,
    })
}

fn print_serialized_sizes(
    sizes: &SerializedSizes,
    batch_size: usize,
    party_count: usize,
    total_weight: usize,
    selected_party_count: usize,
) {
    println!("Table 3 serialized material");
    println!(
        "accounting: compressed cryptographic elements; indices, setup IDs, batch digests, integer metadata, and container framing excluded"
    );
    println!(
        "compressed encodings: G1={} bytes, G2={} bytes, scalar={} bytes, paper GT={} bytes, canonical identity-tagged GT={} bytes",
        sizes.g1_bytes, sizes.g2_bytes, sizes.scalar_bytes, PAPER_GT_BYTES, sizes.canonical_gt_bytes
    );
    println!(
        "party verification keys: {} G2 points, {} bytes ({:.3} kB) = N*B",
        sizes.verification_points,
        sizes.verification_bytes,
        sizes.verification_bytes as f64 / 1024.0
    );
    println!(
        "core D1/D2/global material: {} G2 points, {} bytes ({:.3} kB) = 2*W*B+1",
        sizes.core_points,
        sizes.core_bytes,
        sizes.core_bytes as f64 / 1024.0
    );
    println!(
        "public decryption key: {} G2 points, {} bytes ({:.3} kB) = (2W+N)B+1",
        sizes.public_decryption_points,
        sizes.public_decryption_bytes,
        sizes.public_decryption_bytes as f64 / 1024.0
    );
    println!(
        "encryption key: one GT, {} bytes ({:.3} kB) in Table 3, {} bytes ({:.3} kB) canonical",
        PAPER_GT_BYTES,
        PAPER_GT_BYTES as f64 / 1024.0,
        sizes.canonical_gt_bytes,
        sizes.canonical_gt_bytes as f64 / 1024.0
    );
    println!(
        "total public material, paper encoding: {} bytes ({:.3} kB) = public decryption key + one GT encryption key",
        sizes.paper_public_bytes,
        sizes.paper_public_bytes as f64 / 1024.0
    );
    println!(
        "total public material, canonical cryptographic-element encoding: {} bytes ({:.3} kB)",
        sizes.canonical_public_bytes,
        sizes.canonical_public_bytes as f64 / 1024.0
    );
    println!(
        "implementation structured proof CRS: 0 bytes (Fiat-Shamir Schnorr); paper abstract Pi_DL CRS: unspecified and excluded by Table 3"
    );
    println!(
        "one party secret key: {} bytes ({:.3} kB) = one rho_j scalar (B={} derived fractions may be cached); all N={} secret keys: {} bytes ({:.3} kB)",
        sizes.party_secret_key_bytes,
        sizes.party_secret_key_bytes as f64 / 1024.0,
        batch_size,
        party_count,
        sizes.all_secret_key_bytes,
        sizes.all_secret_key_bytes as f64 / 1024.0
    );
    println!(
        "one ciphertext proof: {} bytes = one compressed G1 commitment + one scalar response",
        sizes.proof_bytes
    );
    println!(
        "one ciphertext excluding its proof (Table 3): {} bytes with paper GT, {} bytes with canonical tagged GT",
        sizes.paper_ciphertext_without_proof_bytes,
        sizes.canonical_ciphertext_without_proof_bytes,
    );
    println!(
        "one ciphertext including this implementation's proof: {} bytes with paper GT, {} bytes canonical",
        sizes.paper_ciphertext_with_proof_bytes,
        sizes.canonical_ciphertext_with_proof_bytes,
    );
    println!(
        "B={} ciphertexts excluding proofs: {}/{} bytes (paper/canonical); including proofs: {}/{} bytes",
        batch_size,
        sizes.paper_batch_without_proofs_bytes,
        sizes.canonical_batch_without_proofs_bytes,
        sizes.paper_batch_with_proofs_bytes,
        sizes.canonical_batch_with_proofs_bytes,
    );
    println!(
        "one party cryptographic response: {} bytes = one compressed G1; selected tau={} responses: {} bytes",
        sizes.party_response_bytes, selected_party_count, sizes.selected_response_bytes
    );
    println!("formula parameters used above: N={party_count}, W={total_weight}, B={batch_size}");
}

fn weights_path() -> io::Result<(PathBuf, bool)> {
    if let Some(path) = env::var_os("WEIGHTED_PFE_WEIGHTS_FILE") {
        if path.is_empty() {
            return Err(invalid_input("WEIGHTED_PFE_WEIGHTS_FILE must not be empty"));
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
            path.display()
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

fn checked_mul(left: usize, right: usize, description: &str) -> io::Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_input(format!("{description} overflows usize")))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn div_duration(duration: Duration, divisor: usize) -> Duration {
    Duration::from_secs_f64(duration.as_secs_f64() / divisor as f64)
}

fn report(label: &str, duration: Duration, items: Option<usize>) {
    match items {
        Some(items) => println!(
            "{label:42} {:>10.3} ms total  {:>8.3} ms/item",
            duration.as_secs_f64() * 1_000.0,
            duration.as_secs_f64() * 1_000.0 / items as f64
        ),
        None => println!("{label:42} {:>10.3} ms", duration.as_secs_f64() * 1_000.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn profile() -> Value {
        json!({
            "error": "1/64",
            "lower_stake_ratio": "31/64",
            "upper_stake_ratio": "33/64",
            "selected_resolution_m": 7,
            "share_count": 6,
            "reconstruction_threshold": 4,
            "effective_reconstruction_stake_ratio": "0.515",
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
            "1/64",
        )
        .unwrap();

        assert_eq!(selected.profile.error, "1/64");
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
        assert!(missing.to_string().contains("available errors: 1/64"));

        let duplicate_profile = profile();
        let duplicate = parse_allocation_document(
            &document(vec![duplicate_profile.clone(), duplicate_profile]),
            Path::new("fixture.json"),
            "1/64",
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
                "1/64"
            )
            .is_err());
        }
    }

    #[test]
    fn minimal_authorized_set_is_largest_first_with_index_ties() {
        let (parties, weight) = minimal_authorized_parties(&[4, 4, 3, 1], 6).unwrap();
        assert_eq!(parties, [0, 1]);
        assert_eq!(weight, 8);
    }

    #[test]
    fn table_three_size_formulas_use_compressed_encodings() {
        let sizes = serialized_sizes(4, 3, 6, 2).unwrap();
        assert_eq!(sizes.g1_bytes, 48);
        assert_eq!(sizes.g2_bytes, 96);
        assert_eq!(sizes.scalar_bytes, 32);
        assert_eq!(sizes.canonical_gt_bytes, 289);
        assert_eq!(sizes.verification_points, 12);
        assert_eq!(sizes.core_points, 49);
        assert_eq!(sizes.public_decryption_points, 61);
        assert_eq!(sizes.public_decryption_bytes, 61 * 96);
        assert_eq!(sizes.proof_bytes, 80);
        assert_eq!(sizes.paper_ciphertext_without_proof_bytes, 336);
        assert_eq!(sizes.canonical_ciphertext_without_proof_bytes, 337);
        assert_eq!(sizes.paper_ciphertext_with_proof_bytes, 416);
        assert_eq!(sizes.canonical_ciphertext_with_proof_bytes, 417);
        assert_eq!(sizes.paper_batch_without_proofs_bytes, 4 * 336);
        assert_eq!(sizes.canonical_batch_without_proofs_bytes, 4 * 337);
        assert_eq!(sizes.paper_batch_with_proofs_bytes, 4 * 416);
        assert_eq!(sizes.canonical_batch_with_proofs_bytes, 4 * 417);
        assert_eq!(sizes.party_secret_key_bytes, 32);
        assert_eq!(sizes.all_secret_key_bytes, 3 * 32);
        assert_eq!(sizes.party_response_bytes, 48);
        assert_eq!(sizes.selected_response_bytes, 2 * 48);
    }
}
