use blstrs::{Gt, Scalar};
use btx::{
    combine_shares, encrypt, keygen, open_batch, partial_decrypt, precompute_batch, validate_batch,
    verify_combined_share, MiddleProductKernel, ValidatedBatch,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use group::Group;

const MAX_BATCH_SIZE: usize = 512;
const SERVER_COUNT: usize = 16;
const THRESHOLD: usize = 1;

fn phase_benchmarks(criterion: &mut Criterion) {
    let material = keygen(MAX_BATCH_SIZE, SERVER_COUNT, THRESHOLD).expect("keygen");

    for batch_size in [32usize, 64, 128, 256, 512] {
        let messages = (0..batch_size)
            .map(|index| Gt::generator() * Scalar::from((index + 1) as u64))
            .collect::<Vec<_>>();
        let ciphertexts = messages
            .iter()
            .map(|message| encrypt(&material.encryption_key, *message))
            .collect::<Vec<_>>();
        let batch = ValidatedBatch::proofs_preverified(&ciphertexts).expect("batch");
        let kernel =
            MiddleProductKernel::new(&material.decryption_key, batch_size).expect("kernel");
        let precomputation = precompute_batch(&kernel, &batch).expect("precompute");
        let shares = material.server_keys[..SERVER_COUNT]
            .iter()
            .map(|key| partial_decrypt(key, &batch).expect("share"))
            .collect::<Vec<_>>();
        let sigma =
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
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| validate_batch(black_box(&ciphertexts)).expect("ciphertext check"))
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
                        black_box(sigma),
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
                        black_box(sigma),
                    )
                    .expect("open")
                })
            },
        );
        group.finish();

        if batch_size == MAX_BATCH_SIZE {
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
