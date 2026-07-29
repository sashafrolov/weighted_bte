use blstrs::{Gt, Scalar};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use group::Group;
use pfe::{
    combine_shares, encrypt, keygen, open_batch, partial_decrypt, precompute_batch, validate_batch,
    verify_combined_share, PartialFractionKernel, ValidatedBatch,
};

const SERVER_COUNT: usize = 16;
const THRESHOLD: usize = 1;

fn phase_benchmarks(criterion: &mut Criterion) {
    for batch_size in [32usize, 64, 128, 256, 512] {
        // PFE setup is exact-size: the root-of-unity slot set and encryption
        // key both depend on B, so each benchmark size gets its own setup.
        let material = keygen(batch_size, SERVER_COUNT, THRESHOLD).expect("keygen");
        let messages = (0..batch_size)
            .map(|index| Gt::generator() * Scalar::from((index + 1) as u64))
            .collect::<Vec<_>>();
        let ciphertexts = messages
            .iter()
            .map(|message| encrypt(&material.encryption_key, *message))
            .collect::<Vec<_>>();
        let batch = ValidatedBatch::proofs_preverified(&material.encryption_key, &ciphertexts)
            .expect("batch");
        let kernel = PartialFractionKernel::new(&material.decryption_key).expect("kernel");
        let precomputation = precompute_batch(&kernel, &batch).expect("precompute");
        let shares = material
            .server_keys
            .iter()
            .map(|key| partial_decrypt(key, &batch).expect("share"))
            .collect::<Vec<_>>();
        let pre_decryption_key =
            combine_shares(&material.decryption_key, &batch, &shares[..2]).expect("combine");

        let mut group = criterion.benchmark_group("precompute");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    precompute_batch(black_box(&kernel), black_box(&batch)).expect("precompute")
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("partial_decrypt");
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    partial_decrypt(black_box(&material.server_keys[0]), black_box(&batch))
                        .expect("partial decrypt")
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("ciphertext_check");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    validate_batch(black_box(&material.encryption_key), black_box(&ciphertexts))
                        .expect("ciphertext check")
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("server_check");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    verify_combined_share(
                        black_box(&material.decryption_key),
                        black_box(&batch),
                        black_box(pre_decryption_key),
                    )
                    .expect("server check")
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("open");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    open_batch(
                        black_box(&kernel),
                        black_box(&batch),
                        black_box(&ciphertexts),
                        black_box(&precomputation),
                        black_box(pre_decryption_key),
                    )
                    .expect("open")
                })
            },
        );
        group.finish();

        if batch_size == 512 {
            let mut group = criterion.benchmark_group("combine");
            for share_count in [2usize, 4, 8, 16] {
                group.bench_with_input(
                    BenchmarkId::from_parameter(share_count),
                    &share_count,
                    |bencher, &share_count| {
                        bencher.iter(|| {
                            combine_shares(
                                black_box(&material.decryption_key),
                                black_box(&batch),
                                black_box(&shares[..share_count]),
                            )
                            .expect("combine")
                        })
                    },
                );
            }
            group.finish();
        }
    }
}

criterion_group!(benches, phase_benchmarks);
criterion_main!(benches);
