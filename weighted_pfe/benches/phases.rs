use blstrs::{Gt, Scalar};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use group::Group;
use weighted_pfe::{
    accept_decryption_shares, encrypt, keygen, open_batch, partial_decrypt, precompute_batch,
    prepare_decryption, validate_batch, CauchyKernel,
};

const PARTY_WEIGHTS: &[usize] = &[16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
const THRESHOLD_WEIGHT: usize = 67;

fn phase_benchmarks(criterion: &mut Criterion) {
    for batch_size in [32usize, 64, 128, 256, 512] {
        // Construction 2 is exact-size: alpha_i, every party secret key, and
        // both W-by-B public tables depend on the padded batch size.
        let material = keygen(batch_size, PARTY_WEIGHTS, THRESHOLD_WEIGHT).expect("keygen");
        let messages = (0..batch_size)
            .map(|slot| Gt::generator() * Scalar::from((slot + 1) as u64))
            .collect::<Vec<_>>();
        let ciphertexts = messages
            .iter()
            .map(|message| encrypt(&material.encryption_key, *message))
            .collect::<Vec<_>>();
        let batch = validate_batch(&material.encryption_key, &ciphertexts).expect("checked batch");
        let shares = material
            .party_keys
            .iter()
            .map(|party| partial_decrypt(party, &batch).expect("share"))
            .collect::<Vec<_>>();
        let accepted = accept_decryption_shares(&material.decryption_key, &batch, &shares)
            .expect("accepted shares");
        let kernel = CauchyKernel::new(&material.decryption_key).expect("Cauchy kernel");
        let decryption =
            prepare_decryption(&material.decryption_key, &accepted).expect("committee preparation");
        let batch_precomputation =
            precompute_batch(&kernel, &decryption, &batch).expect("batch precomputation");

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

        let mut group = criterion.benchmark_group("partial_decrypt_one_party");
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

        let mut group = criterion.benchmark_group("precompute_cauchy");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    precompute_batch(
                        black_box(&kernel),
                        black_box(&decryption),
                        black_box(&batch),
                    )
                    .expect("Cauchy precomputation")
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
                        black_box(&material.decryption_key),
                        black_box(&kernel),
                        black_box(&decryption),
                        black_box(&accepted),
                        black_box(&batch),
                        black_box(&ciphertexts),
                        black_box(&batch_precomputation),
                    )
                    .expect("open")
                })
            },
        );
        group.finish();

        // Keep the validated representation live across every phase benchmark.
        assert_eq!(batch.batch_size(), batch_size);
    }

    let mut group = criterion.benchmark_group("keygen_B32");
    group.sample_size(10);
    group.bench_function("W136_N16", |bencher| {
        bencher.iter(|| {
            keygen(
                black_box(32),
                black_box(PARTY_WEIGHTS),
                black_box(THRESHOLD_WEIGHT),
            )
            .expect("keygen")
        })
    });
    group.finish();
}

criterion_group!(benches, phase_benchmarks);
criterion_main!(benches);
