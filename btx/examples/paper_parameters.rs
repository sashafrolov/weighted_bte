//! End-to-end BTX run at the largest batch size evaluated in the paper.
//!
//! The paper tests B up to 512 but does not state one headline `(N, t)` tuple.
//! This example therefore uses B_max = B = 512 and an 8-of-16 committee
//! (`t = 7`). Override `BTX_BATCH_SIZE`, `BTX_SERVERS`, `BTX_THRESHOLD`, or
//! `BTX_THREADS` in the environment for exploratory runs.

use std::{env, time::Duration, time::Instant};

use blstrs::{Gt, Scalar};
use btx::{
    combine_shares, encrypt, keygen, open_batch, partial_decrypt, precompute_batch, validate_batch,
    verify_combined_share, MiddleProductKernel, ValidatedBatch,
};
use group::Group;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = env_usize("BTX_BATCH_SIZE", 512);
    let server_count = env_usize("BTX_SERVERS", 16);
    let threshold = env_usize("BTX_THRESHOLD", 7);
    let threads = env_usize("BTX_THREADS", 1);

    if threshold >= server_count {
        return Err("BTX_THRESHOLD must be smaller than BTX_SERVERS".into());
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;

    println!("BTX paper-parameter reproduction");
    println!(
        "B_max={batch_size}, B={batch_size}, N={server_count}, t={threshold}, shares={}, Rayon threads={threads}",
        threshold + 1
    );
    println!(
        "The paper specifies B but not a headline N/t; this example uses the values shown above."
    );

    let setup_start = Instant::now();
    let material = pool.install(|| keygen(batch_size, server_count, threshold))?;
    let setup_time = setup_start.elapsed();

    let messages = (0..batch_size)
        .map(|index| Gt::generator() * Scalar::from((index as u64).wrapping_add(1)))
        .collect::<Vec<_>>();

    let encryption_start = Instant::now();
    let ciphertexts = pool.install(|| {
        messages
            .iter()
            .map(|message| encrypt(&material.encryption_key, *message))
            .collect::<Vec<_>>()
    });
    let encryption_time = encryption_start.elapsed();

    // This is fixed G2 preprocessing and is excluded from the paper's
    // per-batch precompute(B) measurements.
    let kernel_start = Instant::now();
    let kernel = pool.install(|| MiddleProductKernel::new(&material.decryption_key, batch_size))?;
    let kernel_time = kernel_start.elapsed();

    let proof_start = Instant::now();
    let checked_batch = pool.install(|| validate_batch(&ciphertexts))?;
    let proof_time = proof_start.elapsed();
    assert_eq!(checked_batch.valid_count(), batch_size);

    // The paper reports ciphertext checks separately, so build the already-
    // validated view before timing its core precompute phase.
    let batch = ValidatedBatch::proofs_preverified(&ciphertexts)?;

    let precompute_start = Instant::now();
    let precomputation = pool.install(|| precompute_batch(&kernel, &batch))?;
    let precompute_time = precompute_start.elapsed();

    let share_count = threshold + 1;
    let partial_start = Instant::now();
    let shares = pool.install(|| {
        material.server_keys[..share_count]
            .iter()
            .map(|server_key| partial_decrypt(server_key, &batch))
            .collect::<btx::Result<Vec<_>>>()
    })?;
    let all_partial_time = partial_start.elapsed();
    let partial_per_server = div_duration(all_partial_time, share_count);

    let combine_start = Instant::now();
    let sigma = pool.install(|| combine_shares(&material.decryption_key, &batch, &shares))?;
    let combine_time = combine_start.elapsed();

    let server_check_start = Instant::now();
    let combined_share_valid =
        pool.install(|| verify_combined_share(&material.decryption_key, &batch, sigma))?;
    let server_check_time = server_check_start.elapsed();
    assert!(combined_share_valid, "combined share failed verification");

    let open_start = Instant::now();
    let decrypted =
        pool.install(|| open_batch(&kernel, &batch, &ciphertexts, &precomputation, sigma))?;
    let open_time = open_start.elapsed();

    for (actual, expected) in decrypted.iter().zip(messages.iter()) {
        assert_eq!(actual.as_ref(), Some(expected));
    }

    let sequential_core = precompute_time + partial_per_server + combine_time + open_time;
    let robust_sequential = sequential_core + proof_time + server_check_time;

    println!();
    println!("One-time / input preparation");
    report("trusted key generation", setup_time, None);
    report("encrypt batch", encryption_time, Some(batch_size));
    report("fixed G2 FFT kernel", kernel_time, None);

    println!();
    println!("Paper phase decomposition");
    report("ctxtCheck(B)", proof_time, Some(batch_size));
    report("precompute(B)", precompute_time, Some(batch_size));
    report(
        "partialDecrypt(B), one server",
        partial_per_server,
        Some(batch_size),
    );
    report("combine(n)", combine_time, Some(share_count));
    report("serverCheck(B,n)", server_check_time, Some(batch_size));
    report("open(B)", open_time, Some(batch_size));
    report("core sequential total", sequential_core, None);
    report("robust sequential total", robust_sequential, None);

    println!();
    println!(
        "Paper target at B=512, single core: ~598 ms core total; \
         0.959/0.019/0.171 ms per item for precompute/partial/open."
    );
    println!("Decryption successful.");
    Ok(())
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
