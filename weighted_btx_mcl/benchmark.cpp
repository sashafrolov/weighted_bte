// Local aggregate online protocol benchmark. Setup and network are excluded.
#include "protocol.hpp"
#include "thread_pool.hpp"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>

using Clock = std::chrono::steady_clock;
static mcl::G1::MulVecOptiFunc original_msm = nullptr;
static mcl::G1::MulEachOptiFunc original_each = nullptr;
static std::atomic<uint64_t> msm_calls{0}, each_calls{0};
static void counted_msm(mcl::G1& out, mcl::G1* points, const mcl::Fr* scalars, size_t n, size_t b) {
    msm_calls.fetch_add(1, std::memory_order_relaxed);
    original_msm(out, points, scalars, n, b);
}
static void counted_each(mcl::G1* points, const mcl::Fr* scalars, size_t n) {
    each_calls.fetch_add(1, std::memory_order_relaxed);
    original_each(points, scalars, n);
}
static void check(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
struct Settings {
    std::string curve, simd, orientation, profile;
    size_t threads, samples, warmup;
};
struct Measurement {
    std::string phase;
    double ms;
    uint64_t msm, each;
};
struct Marker {
    Clock::time_point start = Clock::now();
    uint64_t msm = msm_calls.load(), each = each_calls.load();
    Measurement finish(const char* phase) const {
        const auto end = Clock::now();
        return {phase, std::chrono::duration<double, std::milli>(end - start).count(),
                msm_calls.load() - msm, each_calls.load() - each};
    }
};

template<class P> void run(const Settings& settings, ThreadPool& pool) {
    std::ifstream file(settings.profile);
    check(bool(file), "cannot open profile");
    size_t required_weight = 0, w;
    file >> required_weight;
    std::vector<size_t> weights;
    while (file >> w) { check(w > 0, "zero weight"); weights.push_back(w); }
    check(!weights.empty() && required_weight > 0, "invalid profile");
    check(required_weight <= std::accumulate(weights.begin(), weights.end(), size_t(0)), "threshold exceeds weight");
    std::vector<size_t> candidates(weights.size());
    std::iota(candidates.begin(), candidates.end(), 0);
    std::stable_sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) { return weights[a] > weights[b]; });
    std::vector<size_t> parties;
    size_t accepted_weight = 0;
    for (auto party : candidates) {
        parties.push_back(party);
        accepted_weight += weights[party];
        if (accepted_weight >= required_weight) break;
    }
    std::sort(parties.begin(), parties.end());
    Marker setup;
    const auto material = P::keygen(16, weights, required_weight - 1);
    std::cerr << "setup_ms=" << setup.finish("setup").ms << " parties=" << weights.size()
              << " total_weight=" << std::accumulate(weights.begin(), weights.end(), size_t(0))
              << " required_weight=" << required_weight << " selected_parties=" << parties.size()
              << " accepted_weight=" << accepted_weight << '\n';
    std::vector<mcl::Fp12> messages(16);
    // Plaintexts are public benchmark fixtures; encryption and proof randomness
    // are fresh CSPRNG draws in every timed iteration.
    for (size_t i = 0; i < messages.size(); ++i)
        mcl::Fp12::pow(messages[i], material.target_generator, mcl::Fr(int(i + 1)));

    using CT = typename P::Ciphertext;
    using Batch = typename P::ValidatedBatch;
    using Share = typename P::Share;
    using Accepted = typename P::Accepted;
    using Cross = typename P::CrossTerms;
    using Committee = typename P::Committee;

    std::cout << "curve,simd,orientation,layout,cache,threads,sample,phase,milliseconds,avx_msm,avx_each\n";
    for (const size_t batch_size : {size_t(16), size_t(4), size_t(2)}) {
        const size_t chunks = 16 / batch_size;
        for (const bool cached : {false, true}) {
            std::optional<Committee> cached_committee;
            if (cached) {
                std::vector<CT> fixture(batch_size);
                pool.parallel_for(batch_size, [&](size_t i) { fixture[i] = P::encrypt(material, messages[i]); });
                auto batch = P::validate(material, fixture);
                std::vector<Share> shares(parties.size());
                pool.parallel_for(parties.size(), [&](size_t j) { shares[j] = P::partial(material, parties[j], batch); });
                auto accepted = P::accept(material, batch, shares);
                cached_committee.emplace(P::prepare(material, accepted));
            }
            for (size_t iteration = 0; iteration < settings.warmup + settings.samples; ++iteration) {
                std::vector<Measurement> timings;
                timings.reserve(9);
                msm_calls = 0; each_calls = 0;
                Marker total;
                Marker encryption;
                std::vector<std::vector<CT>> ciphertexts(chunks, std::vector<CT>(batch_size));
                pool.parallel_for(16, [&](size_t i) {
                    ciphertexts[i / batch_size][i % batch_size] = P::encrypt(material, messages[i]);
                });
                timings.push_back(encryption.finish("encryption"));
                Marker validation;
                std::vector<Batch> batches(chunks);
                pool.parallel_for(chunks, [&](size_t c) { batches[c] = P::validate(material, ciphertexts[c]); });
                timings.push_back(validation.finish("validation"));
                Marker share_generation;
                std::vector<std::vector<Share>> shares(chunks, std::vector<Share>(parties.size()));
                pool.parallel_for(chunks * parties.size(), [&](size_t i) {
                    const size_t c = i / parties.size(), j = i % parties.size();
                    shares[c][j] = P::partial(material, parties[j], batches[c]);
                });
                timings.push_back(share_generation.finish("share_generation"));
                Marker combiner;
                Marker acceptance;
                std::vector<Accepted> accepted(chunks);
                pool.parallel_for(chunks, [&](size_t c) { accepted[c] = P::accept(material, batches[c], shares[c]); });
                timings.push_back(acceptance.finish("acceptance"));
                Marker preparation;
                std::optional<Committee> fresh_committee;
                if (!cached) fresh_committee.emplace(P::prepare(material, accepted[0]));
                const Committee& committee = cached ? *cached_committee : *fresh_committee;
                timings.push_back(preparation.finish("preparation"));
                Marker precompute;
                std::vector<Cross> cross(chunks);
                pool.parallel_for(chunks, [&](size_t c) { cross[c] = P::precompute(committee, batches[c]); });
                timings.push_back(precompute.finish("precompute"));
                Marker opening;
                std::vector<std::vector<std::optional<mcl::Fp12>>> opened(chunks);
                pool.parallel_for(chunks, [&](size_t c) {
                    opened[c] = P::open(committee, accepted[c], batches[c], ciphertexts[c], cross[c]);
                });
                timings.push_back(opening.finish("opening"));
                timings.push_back(combiner.finish("combiner"));
                timings.push_back(total.finish("end_to_end"));
                // All outputs are checked on every iteration, outside the timer.
                for (size_t c = 0; c < chunks; ++c) {
                    check(accepted[c].parties == parties, "unexpected accepted committee");
                    check(opened[c].size() == batch_size, "wrong number of plaintexts");
                    for (size_t i = 0; i < batch_size; ++i)
                        check(opened[c][i].has_value() && *opened[c][i] == messages[c * batch_size + i], "plaintext mismatch");
                }
                if (iteration >= settings.warmup) for (const auto& t : timings)
                    std::cout << settings.curve << ',' << settings.simd << ',' << settings.orientation << ','
                              << chunks << 'x' << batch_size << ',' << (cached ? "cached" : "cold") << ','
                              << settings.threads << ',' << iteration - settings.warmup << ',' << t.phase << ','
                              << std::fixed << std::setprecision(6) << t.ms << ',' << t.msm << ',' << t.each << '\n';
            }
            std::cout.flush();
            std::cerr << "correctness=passed layout=" << chunks << 'x' << batch_size
                      << " cache=" << (cached ? "cached" : "cold") << " iterations="
                      << settings.samples + settings.warmup << '\n';
        }
    }
}

int main(int argc, char** argv) try {
    check(argc == 8, "usage: benchmark bls12_381|bn254 off|avx512 normal|swapped profile.txt threads samples warmup");
    Settings settings{argv[1], argv[2], argv[3], argv[4], std::stoul(argv[5]), std::stoul(argv[6]), std::stoul(argv[7])};
    check(settings.samples > 0 && settings.threads > 0, "samples and threads must be positive");
    check(settings.curve == "bls12_381" || settings.curve == "bn254", "unknown curve");
    check(settings.simd == "off" || settings.simd == "avx512", "unknown SIMD mode");
    check(settings.orientation == "normal" || settings.orientation == "swapped", "unknown orientation");
    mcl::initPairing(settings.curve == "bls12_381" ? mcl::BLS12_381 : mcl::BN_SNARK1);
    original_msm = mcl::G1::mulVecOpti;
    original_each = mcl::G1::mulEachOpti;
    if (settings.simd == "avx512") {
        check(original_msm && original_each, "AVX512IFMA callbacks unavailable for this curve/CPU/build");
        mcl::G1::setMulVecOpti(counted_msm);
        mcl::G1::setMulEachOpti(counted_each);
    } else {
        mcl::G1::setMulVecOpti(nullptr);
        mcl::G1::setMulEachOpti(nullptr);
    }
    std::cerr << "curve=" << settings.curve << " mcl_curve=" << (settings.curve == "bn254" ? "BN_SNARK1" : "BLS12_381")
              << " simd=" << settings.simd << " orientation=" << settings.orientation << " threads=" << settings.threads
              << " samples=" << settings.samples << " warmup=" << settings.warmup << '\n';
    ThreadPool pool(settings.threads);
    wbtx::set_parallel_executor([&](size_t n, const std::function<void(size_t)>& fn) { pool.parallel_for(n, fn); });
    if (settings.orientation == "normal") run<wbtx::Normal>(settings, pool);
    else run<wbtx::Swapped>(settings, pool);
    wbtx::set_parallel_executor({});
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
}
