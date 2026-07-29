//! End-to-end PFE run at the same parameters as the BTX reproduction example.

use std::{env, time::Duration, time::Instant};

use blstrs::{Gt, Scalar};
use group::Group;
use pfe::{
    combine_shares, encrypt, keygen, open_batch, partial_decrypt, precompute_batch, validate_batch,
    verify_combined_share, PartialFractionKernel, ValidatedBatch,
};
use rayon::prelude::*;

#[derive(Clone, Copy, Default)]
struct Timing {
    setup: Duration,
    encryption: Duration,
    kernel: Duration,
    proof: Duration,
    precompute: Duration,
    partial_per_server: Duration,
    combine: Duration,
    server_check: Duration,
    open: Duration,
}

impl Timing {
    fn add_assign(&mut self, other: Self) {
        self.setup += other.setup;
        self.encryption += other.encryption;
        self.kernel += other.kernel;
        self.proof += other.proof;
        self.precompute += other.precompute;
        self.partial_per_server += other.partial_per_server;
        self.combine += other.combine;
        self.server_check += other.server_check;
        self.open += other.open;
    }

    fn averaged(self, repetitions: usize) -> Self {
        Self {
            setup: div_duration(self.setup, repetitions),
            encryption: div_duration(self.encryption, repetitions),
            kernel: div_duration(self.kernel, repetitions),
            proof: div_duration(self.proof, repetitions),
            precompute: div_duration(self.precompute, repetitions),
            partial_per_server: div_duration(self.partial_per_server, repetitions),
            combine: div_duration(self.combine, repetitions),
            server_check: div_duration(self.server_check, repetitions),
            open: div_duration(self.open, repetitions),
        }
    }

    fn sequential_core(self) -> Duration {
        self.precompute + self.partial_per_server + self.combine + self.open
    }

    fn robust_sequential(self) -> Duration {
        self.sequential_core() + self.proof + self.server_check
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = env_usize("PFE_BATCH_SIZE", 512);
    let server_count = env_usize("PFE_SERVERS", 16);
    // Match BTX's convention: t is the corruption/polynomial-degree
    // threshold, so reconstruction consumes t + 1 shares.
    let threshold = env_usize("PFE_THRESHOLD", 7);
    let threads = env_usize("PFE_THREADS", 1);
    let repetitions = env_usize("PFE_REPETITIONS", 1);

    if threshold >= server_count {
        return Err("PFE_THRESHOLD must be smaller than PFE_SERVERS".into());
    }
    if batch_size == 0 || !batch_size.is_power_of_two() {
        return Err("PFE_BATCH_SIZE must be a nonzero power of two".into());
    }
    if repetitions == 0 {
        return Err("PFE_REPETITIONS must be nonzero".into());
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;

    println!("PFE paper-parameter reproduction");
    println!(
        "B_max={batch_size}, B={batch_size}, N={server_count}, t={threshold}, shares={}, Rayon threads={threads}",
        threshold + 1
    );
    println!("measured repetitions={repetitions}, plus one unmeasured full warm-up");

    run_once(&pool, batch_size, server_count, threshold)?;

    let mut total = Timing::default();
    for _ in 0..repetitions {
        total.add_assign(run_once(&pool, batch_size, server_count, threshold)?);
    }
    let average = total.averaged(repetitions);

    println!();
    println!("Average over {repetitions} measured repetition(s)");
    println!();
    println!("One-time / input preparation");
    report("trusted key generation", average.setup, None);
    report("encrypt batch", average.encryption, Some(batch_size));
    report("fixed PFE FFT/G2 kernel", average.kernel, None);

    println!();
    println!("Paper phase decomposition");
    report("ctxtCheck(B)", average.proof, Some(batch_size));
    report("precompute(B)", average.precompute, Some(batch_size));
    report(
        "partialDecrypt(B), one server",
        average.partial_per_server,
        Some(batch_size),
    );
    report("combine(n)", average.combine, Some(threshold + 1));
    report("serverCheck(B,n)", average.server_check, Some(batch_size));
    report("open(B), fused 3-pair path", average.open, Some(batch_size));
    report("core sequential total", average.sequential_core(), None);
    report("robust sequential total", average.robust_sequential(), None);

    println!();
    println!(
        "BTX paper's PFE target at B=512, single core: ~1197 ms core total; \
         1.596/0.019/0.723 ms per item for precompute/partial/open."
    );
    println!(
        "This implementation additionally uses the corrected B-point circulant \
         indexing and fused three-pair opening from the PFE authors' released code."
    );
    println!("Decryption successful in every warm-up and measured repetition.");
    Ok(())
}

fn run_once(
    pool: &rayon::ThreadPool,
    batch_size: usize,
    server_count: usize,
    threshold: usize,
) -> Result<Timing, Box<dyn std::error::Error>> {
    let setup_start = Instant::now();
    let material = pool.install(|| keygen(batch_size, server_count, threshold))?;
    let setup_time = setup_start.elapsed();

    let messages = (0..batch_size)
        .map(|index| Gt::generator() * Scalar::from((index as u64).wrapping_add(1)))
        .collect::<Vec<_>>();

    let encryption_start = Instant::now();
    let ciphertexts = pool.install(|| {
        messages
            .par_iter()
            .map(|message| encrypt(&material.encryption_key, *message))
            .collect::<Vec<_>>()
    });
    let encryption_time = encryption_start.elapsed();

    // This is fixed domain/G2 preprocessing and is excluded from the paper's
    // online per-batch phase measurements.
    let kernel_start = Instant::now();
    let kernel = pool.install(|| PartialFractionKernel::new(&material.decryption_key))?;
    let kernel_time = kernel_start.elapsed();

    // PFE Construction 2 rejects the whole batch if any client proof fails.
    // The implementation batches the two proof equations into two MSM checks.
    let proof_start = Instant::now();
    let checked_batch = pool.install(|| validate_batch(&material.encryption_key, &ciphertexts))?;
    let proof_time = proof_start.elapsed();
    assert_eq!(checked_batch.batch_size(), batch_size);

    // Proof checking is reported separately, matching the BTX example.
    let batch = ValidatedBatch::proofs_preverified(&material.encryption_key, &ciphertexts)?;

    let precompute_start = Instant::now();
    let precomputation = pool.install(|| precompute_batch(&kernel, &batch))?;
    let precompute_time = precompute_start.elapsed();

    let share_count = threshold + 1;
    let partial_start = Instant::now();
    let shares = pool.install(|| {
        material.server_keys[..share_count]
            .iter()
            .map(|server_key| partial_decrypt(server_key, &batch))
            .collect::<pfe::Result<Vec<_>>>()
    })?;
    let all_partial_time = partial_start.elapsed();
    let partial_per_server = div_duration(all_partial_time, share_count);

    let combine_start = Instant::now();
    let pre_decryption_key =
        pool.install(|| combine_shares(&material.decryption_key, &batch, &shares))?;
    let combine_time = combine_start.elapsed();

    let server_check_start = Instant::now();
    let combined_share_valid = pool
        .install(|| verify_combined_share(&material.decryption_key, &batch, pre_decryption_key))?;
    let server_check_time = server_check_start.elapsed();
    assert!(
        combined_share_valid,
        "combined pre-decryption key failed verification"
    );

    let open_start = Instant::now();
    let decrypted = pool.install(|| {
        open_batch(
            &kernel,
            &batch,
            &ciphertexts,
            &precomputation,
            pre_decryption_key,
        )
    })?;
    let open_time = open_start.elapsed();

    assert_eq!(decrypted, messages);

    Ok(Timing {
        setup: setup_time,
        encryption: encryption_time,
        kernel: kernel_time,
        proof: proof_time,
        precompute: precompute_time,
        partial_per_server,
        combine: combine_time,
        server_check: server_check_time,
        open: open_time,
    })
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn div_duration(duration: Duration, divisor: usize) -> Duration {
    Duration::from_secs_f64(duration.as_secs_f64() / divisor as f64)
}

fn report(label: &str, duration: Duration, items: Option<usize>) {
    match items {
        Some(items) => println!(
            "{label:34} {:>10.3} ms total  {:>8.3} ms/item",
            duration.as_secs_f64() * 1_000.0,
            duration.as_secs_f64() * 1_000.0 / items as f64
        ),
        None => println!("{label:34} {:>10.3} ms", duration.as_secs_f64() * 1_000.0),
    }
}
