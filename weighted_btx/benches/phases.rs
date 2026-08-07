use blstrs::{Gt, Scalar};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use group::Group;
use weighted_btx::{
    accept_decryption_shares, encrypt, keygen, open_batch, partial_decrypt, precompute_batch,
    prepare_decryption, validate_batch,
};

const MAX_BATCH_SIZE: usize = 512;
const PARTY_WEIGHTS: &[usize] = &[16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
const THRESHOLD_WEIGHT: usize = 90;

fn phase_benchmarks(criterion: &mut Criterion) {
    let material = keygen(MAX_BATCH_SIZE, PARTY_WEIGHTS, THRESHOLD_WEIGHT).expect("keygen");

    for batch_size in [32usize, 64, 128, 256, 512] {
        let messages = (0..batch_size)
            .map(|index| Gt::generator() * Scalar::from((index + 1) as u64))
            .collect::<Vec<_>>();
        let ciphertexts = messages
            .iter()
            .map(|message| encrypt(&material.encryption_key, *message))
            .collect::<Vec<_>>();
        let batch = validate_batch(&material.decryption_key, &ciphertexts).expect("batch");
        let shares = material
            .party_keys
            .iter()
            .map(|key| partial_decrypt(key, &batch).expect("share"))
            .collect::<Vec<_>>();
        let accepted = accept_decryption_shares(&material.decryption_key, &batch, &shares)
            .expect("accepted shares");
        let fixed =
            prepare_decryption(&material.decryption_key, &accepted).expect("committee preparation");
        let cross_terms = precompute_batch(&fixed, &batch).expect("cross terms");

        let mut group = criterion.benchmark_group("ciphertext_check");
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    validate_batch(black_box(&material.decryption_key), black_box(&ciphertexts))
                        .expect("ciphertext check")
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
                    partial_decrypt(black_box(&material.party_keys[0]), black_box(&batch))
                        .expect("partial decrypt")
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("accept_shares_batched");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    accept_decryption_shares(
                        black_box(&material.decryption_key),
                        black_box(&batch),
                        black_box(&shares),
                    )
                    .expect("accept shares")
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("prepare_committee");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    prepare_decryption(black_box(&material.decryption_key), black_box(&accepted))
                        .expect("prepare committee")
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("precompute_cross_terms");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    precompute_batch(black_box(&fixed), black_box(&batch))
                        .expect("precompute cross terms")
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
                        black_box(&fixed),
                        black_box(&accepted),
                        black_box(&batch),
                        black_box(&ciphertexts),
                        black_box(&cross_terms),
                    )
                    .expect("open")
                })
            },
        );
        group.finish();
    }
}

criterion_group!(benches, phase_benchmarks);
criterion_main!(benches);
