//! End-to-end Simple BTE run at the same parameters as the BTX and PFE
//! reproduction examples.

use std::{env, time::Duration, time::Instant};

use ark_bls12_381::Bls12_381;
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    PrimeGroup,
};
use ark_std::test_rng;
use simple_batched_threshold_encryption::bte::{
    crs::setup,
    decryption::{
        combine, finalize_decrypt, partial_decrypt_preverified, predecrypt_fft,
        verify_ciphertext_batch,
    },
    encryption::encrypt,
    Ciphertext, DecryptionKey,
};

type E = Bls12_381;
type Fr = <E as Pairing>::ScalarField;

#[derive(Clone, Copy)]
struct Config {
    batch_size: usize,
    server_count: usize,
    share_count: usize,
}

#[derive(Default)]
struct Timings {
    setup: Duration,
    encryption: Duration,
    proof: Duration,
    precompute: Duration,
    partial_per_server: Duration,
    combine: Duration,
    server_check: Duration,
    open: Duration,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = env_usize("SIMPLE_BTE_BATCH_SIZE", 512);
    let server_count = env_usize("SIMPLE_BTE_SERVERS", 16);
    // Match the BTX/PFE convention: t is the corruption/polynomial-degree
    // threshold, so reconstruction consumes t + 1 shares.
    let threshold = env_usize("SIMPLE_BTE_THRESHOLD", 7);
    let threads = env_usize("SIMPLE_BTE_THREADS", 1);
    let repetitions = env_usize("SIMPLE_BTE_REPETITIONS", 1);

    if batch_size == 0 {
        return Err("SIMPLE_BTE_BATCH_SIZE must be nonzero".into());
    }
    if threshold >= server_count {
        return Err("SIMPLE_BTE_THRESHOLD must be smaller than SIMPLE_BTE_SERVERS".into());
    }
    if threads != 1 {
        return Err(
            "this Simple BTE crate does not enable Arkworks parallelism; \
             SIMPLE_BTE_THREADS must be 1"
                .into(),
        );
    }
    if repetitions == 0 {
        return Err("SIMPLE_BTE_REPETITIONS must be nonzero".into());
    }

    let share_count = threshold + 1;
    let config = Config {
        batch_size,
        server_count,
        share_count,
    };

    println!("Simple BTE paper-parameter reproduction");
    println!(
        "B_max={batch_size}, B={batch_size}, N={server_count}, t={threshold}, \
         shares={share_count}, threads=1 (serial Arkworks)"
    );
    println!("warm-up runs=1, measured repetitions={repetitions}");

    println!("Running warm-up...");
    run_once(config);

    let mut aggregate = Timings::default();
    for _ in 0..repetitions {
        aggregate.add_assign(run_once(config));
    }
    let timings = aggregate.div(repetitions);

    let sequential_core =
        timings.precompute + timings.partial_per_server + timings.combine + timings.open;
    let robust_sequential = sequential_core + timings.proof + timings.server_check;

    println!();
    println!("Average over {repetitions} measured repetition(s)");
    println!();
    println!("One-time / input preparation");
    report("trusted setup + fixed FFT/G2 kernel", timings.setup, None);
    report("encrypt batch", timings.encryption, Some(batch_size));

    println!();
    println!("Paper phase decomposition");
    report("ctxtCheck(B)", timings.proof, Some(batch_size));
    report("precompute(B)", timings.precompute, Some(batch_size));
    report(
        "partialDecrypt(B), one server",
        timings.partial_per_server,
        Some(batch_size),
    );
    report("combine(n)", timings.combine, Some(share_count));
    report(
        "serverCheck(B,n), aggregate",
        timings.server_check,
        Some(batch_size),
    );
    report("open(B)", timings.open, Some(batch_size));
    report("core sequential total", sequential_core, None);
    report("robust sequential total", robust_sequential, None);

    println!();
    println!(
        "BTX paper target at B=512, single core: ~598 ms core total; \
         0.959/0.019/0.171 ms per item for precompute/partial/open."
    );
    println!(
        "Simple BTE setup calls its reconstruction parameter `threshold`; \
         this run passes {share_count} to reproduce the paper convention t={threshold}."
    );
    println!("Decryption successful in the warm-up and every measured repetition.");
    Ok(())
}

fn run_once(config: Config) -> Timings {
    let mut rng = test_rng();

    // Unlike the BTX/PFE crates, upstream Simple BTE constructs its fixed FFT
    // domain and prepared G2 kernel inside setup, so they cannot be timed
    // independently without changing the library API.
    let setup_start = Instant::now();
    let (encryption_key, decryption_key, server_keys) = setup::<E>(
        config.batch_size,
        config.server_count,
        config.share_count,
        &mut rng,
    );
    let setup_time = setup_start.elapsed();

    let messages = (0..config.batch_size)
        .map(|index| PairingOutput::<E>::generator() * Fr::from((index as u64).wrapping_add(1)))
        .collect::<Vec<_>>();

    let encryption_start = Instant::now();
    let ciphertexts = messages
        .iter()
        .map(|message| encrypt(&encryption_key, message, &mut rng))
        .collect::<Vec<_>>();
    let encryption_time = encryption_start.elapsed();

    let proof_start = Instant::now();
    let ciphertexts_valid = verify_ciphertext_batch(&ciphertexts, &mut rng);
    let proof_time = proof_start.elapsed();
    assert!(ciphertexts_valid, "ciphertext proof verification failed");

    // This is the paper's pipelineable precompute(B) phase: it depends only
    // on the checked ciphertext batch and the fixed public FFT kernel.
    let precompute_start = Instant::now();
    let cross_terms = predecrypt_fft(&decryption_key, &ciphertexts);
    let precompute_time = precompute_start.elapsed();

    // Upstream partial_decrypt repeats verify_ciphertext_batch for every
    // server. The comparison examples report that work once as ctxtCheck(B),
    // so time the exact post-verification MSM body here instead.
    let partial_start = Instant::now();
    let shares = server_keys[..config.share_count]
        .iter()
        .map(|server_key| partial_decrypt_preverified(server_key, &ciphertexts))
        .collect::<Vec<_>>();
    let all_partial_time = partial_start.elapsed();
    let partial_per_server = div_duration(all_partial_time, config.share_count);

    let combine_start = Instant::now();
    let combined_share = combine::<E>(&shares);
    let combine_time = combine_start.elapsed();

    // Upstream exposes per-share verification. For a phase comparison with
    // BTX/PFE, check the reconstructed share once against the public powers,
    // which is the optimistic aggregate serverCheck(B,n) from the BTX paper.
    let server_check_start = Instant::now();
    let combined_share_valid =
        verify_combined_share(&decryption_key, &combined_share, &ciphertexts);
    let server_check_time = server_check_start.elapsed();
    assert!(combined_share_valid, "combined share failed verification");

    let open_start = Instant::now();
    let decrypted = finalize_decrypt(&decryption_key, &combined_share, &ciphertexts, &cross_terms);
    let open_time = open_start.elapsed();

    assert_eq!(decrypted, messages);

    Timings {
        setup: setup_time,
        encryption: encryption_time,
        proof: proof_time,
        precompute: precompute_time,
        partial_per_server,
        combine: combine_time,
        server_check: server_check_time,
        open: open_time,
    }
}

impl Timings {
    fn add_assign(&mut self, other: Self) {
        self.setup += other.setup;
        self.encryption += other.encryption;
        self.proof += other.proof;
        self.precompute += other.precompute;
        self.partial_per_server += other.partial_per_server;
        self.combine += other.combine;
        self.server_check += other.server_check;
        self.open += other.open;
    }

    fn div(self, divisor: usize) -> Self {
        Self {
            setup: div_duration(self.setup, divisor),
            encryption: div_duration(self.encryption, divisor),
            proof: div_duration(self.proof, divisor),
            precompute: div_duration(self.precompute, divisor),
            partial_per_server: div_duration(self.partial_per_server, divisor),
            combine: div_duration(self.combine, divisor),
            server_check: div_duration(self.server_check, divisor),
            open: div_duration(self.open, divisor),
        }
    }
}

/// Verify the reconstructed batch secret with one aggregate pairing equation.
fn verify_combined_share(
    decryption_key: &DecryptionKey<E>,
    combined_share: &<E as Pairing>::G1,
    ciphertexts: &[Ciphertext<E>],
) -> bool {
    assert_eq!(ciphertexts.len(), decryption_key.batch_size);

    let lhs = E::pairing(*combined_share, <E as Pairing>::G2::generator());
    let rhs = E::multi_pairing(
        ciphertexts.iter().map(|ciphertext| ciphertext.ct1),
        decryption_key.powers_of_h[1..=ciphertexts.len()]
            .iter()
            .cloned(),
    );
    lhs == rhs
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
