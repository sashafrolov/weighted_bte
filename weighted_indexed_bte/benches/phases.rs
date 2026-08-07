use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;
use weighted_indexed_bte::{
    accept_decryption_shares, encrypt, keygen, open_batch, partial_decrypt, precompute_batch,
    prepare_decryption, setup, validate_batch, IndexedMiddleProductKernel,
};

const PARTY_WEIGHTS: &[usize] = &[16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
// W=136, so t=67 satisfies the old paper's strict t < floor(W/2) premise.
const THRESHOLD_WEIGHT: usize = 67;

fn phase_benchmarks(criterion: &mut Criterion) {
    for index_space in [32usize, 64, 128, 256, 512] {
        let parameters = setup(index_space).expect("powers-of-tau setup");
        let material = keygen(&parameters, PARTY_WEIGHTS, THRESHOLD_WEIGHT).expect("keygen");
        let messages = (0..index_space)
            .map(|index| (index as u64 + 1).to_le_bytes().repeat(4))
            .collect::<Vec<_>>();
        let ciphertexts = messages
            .par_iter()
            .enumerate()
            .map(|(index, message)| {
                encrypt(&parameters, &material.public_key, message, index).expect("encrypt")
            })
            .collect::<Vec<_>>();
        let batch = validate_batch(&material.public_key, &ciphertexts).expect("validate batch");
        let shares = material
            .party_keys
            .par_iter()
            .map(|key| partial_decrypt(key, &batch).expect("partial decrypt"))
            .collect::<Vec<_>>();
        let accepted =
            accept_decryption_shares(&material.public_key, &batch, &shares).expect("accept shares");
        let committee =
            prepare_decryption(&material.public_key, &accepted).expect("prepare committee");
        let kernel = IndexedMiddleProductKernel::new(&parameters).expect("fixed kernel");
        let cross_terms = precompute_batch(&kernel, &batch).expect("cross terms");
        let sparse_indices = (0..8usize)
            .map(|position| position * (index_space - 1) / 7)
            .collect::<Vec<_>>();
        let sparse_ciphertexts = sparse_indices
            .par_iter()
            .enumerate()
            .map(|(slot, index)| {
                encrypt(
                    &parameters,
                    &material.public_key,
                    &(slot as u64 + 1).to_le_bytes(),
                    *index,
                )
                .expect("sparse encrypt")
            })
            .collect::<Vec<_>>();
        let sparse_batch =
            validate_batch(&material.public_key, &sparse_ciphertexts).expect("sparse batch");

        let mut group = criterion.benchmark_group("encrypt_batch");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(index_space),
            &index_space,
            |bencher, _| {
                bencher.iter(|| {
                    messages
                        .par_iter()
                        .enumerate()
                        .map(|(index, message)| {
                            encrypt(
                                black_box(&parameters),
                                black_box(&material.public_key),
                                black_box(message),
                                index,
                            )
                            .expect("encrypt")
                        })
                        .collect::<Vec<_>>()
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("ciphertext_check");
        group.bench_with_input(
            BenchmarkId::from_parameter(index_space),
            &index_space,
            |bencher, _| {
                bencher.iter(|| {
                    validate_batch(black_box(&material.public_key), black_box(&ciphertexts))
                        .expect("ciphertext check")
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("partial_decrypt_one_party");
        group.bench_with_input(
            BenchmarkId::from_parameter(index_space),
            &index_space,
            |bencher, _| {
                bencher.iter(|| {
                    partial_decrypt(black_box(&material.party_keys[0]), black_box(&batch))
                        .expect("partial decrypt")
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("accept_shares");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(index_space),
            &index_space,
            |bencher, _| {
                bencher.iter(|| {
                    accept_decryption_shares(
                        black_box(&material.public_key),
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
            BenchmarkId::from_parameter(index_space),
            &index_space,
            |bencher, _| {
                bencher.iter(|| {
                    prepare_decryption(black_box(&material.public_key), black_box(&accepted))
                        .expect("prepare committee")
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("prepare_fixed_fft_kernel");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(index_space),
            &index_space,
            |bencher, _| {
                bencher.iter(|| {
                    IndexedMiddleProductKernel::new(black_box(&parameters)).expect("fixed kernel")
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("precompute_cross_terms");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(index_space),
            &index_space,
            |bencher, _| {
                bencher.iter(|| {
                    precompute_batch(black_box(&kernel), black_box(&batch))
                        .expect("precompute cross terms")
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("precompute_cross_terms_sparse_8");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(index_space),
            &index_space,
            |bencher, _| {
                bencher.iter(|| {
                    precompute_batch(black_box(&kernel), black_box(&sparse_batch))
                        .expect("precompute sparse cross terms")
                })
            },
        );
        group.finish();

        let mut group = criterion.benchmark_group("open");
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::from_parameter(index_space),
            &index_space,
            |bencher, _| {
                bencher.iter(|| {
                    open_batch(
                        black_box(&committee),
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
