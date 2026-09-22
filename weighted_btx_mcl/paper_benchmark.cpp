// Configurable paper-table workload. Timings simulate every selected validator
// locally in one bounded pool; they do not measure distributed network latency.
#include "protocol.hpp"
#include "thread_pool.hpp"
#include <atomic>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
static void check(bool valid, const std::string& message) {
    if (!valid) throw std::runtime_error(message);
}
static size_t parse_size(const std::string& text, const char* label, bool allow_zero = false) {
    size_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    check(!text.empty() && parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() &&
          (allow_zero || value != 0), std::string(label) + " must be a " +
          (allow_zero ? "nonnegative" : "positive") + " integer");
    return value;
}
struct Settings {
    std::string curve, simd, orientation, profile;
    size_t total, batch, setup, threads, samples, warmup;
};
struct Profile {
    size_t required_weight = 0, total_weight = 0, accepted_weight = 0;
    std::vector<size_t> weights, parties;
};
static Profile read_profile(const std::string& path) {
    std::ifstream file(path);
    check(bool(file), "cannot open profile");
    Profile profile;
    std::string token;
    check(bool(file >> token), "profile is empty");
    profile.required_weight = parse_size(token, "profile reconstruction threshold");
    while (file >> token) {
        const size_t weight = parse_size(token, "profile weight");
        profile.weights.push_back(weight);
        profile.total_weight = wbtx::checked_add(profile.total_weight, weight);
    }
    check(file.eof() && !file.bad(), "cannot read profile");
    check(!profile.weights.empty(), "profile has no weights");
    check(profile.required_weight <= profile.total_weight, "threshold exceeds total weight");
    profile.parties.resize(profile.weights.size());
    std::iota(profile.parties.begin(), profile.parties.end(), 0);
    std::stable_sort(profile.parties.begin(), profile.parties.end(), [&](size_t a, size_t b) {
        return profile.weights[a] > profile.weights[b];
    });
    size_t count = 0;
    for (const size_t party : profile.parties) {
        profile.accepted_weight = wbtx::checked_add(profile.accepted_weight, profile.weights[party]);
        ++count;
        if (profile.accepted_weight >= profile.required_weight) break;
    }
    profile.parties.resize(count);
    std::sort(profile.parties.begin(), profile.parties.end());
    return profile;
}
struct Measurement {
    const char* phase;
    double ms;
    uint64_t msm, each;
};
struct Marker {
    Clock::time_point start = Clock::now();
    uint64_t msm = msm_calls.load(), each = each_calls.load();
    Measurement finish(const char* phase, Clock::time_point end = Clock::now()) const {
        return {phase, std::chrono::duration<double, std::milli>(end - start).count(),
                msm_calls.load() - msm, each_calls.load() - each};
    }
};
template<class T> static size_t serialized_size(const T& value) {
    std::array<uint8_t, 1024> bytes{};
    const size_t size = value.serialize(bytes.data(), bytes.size());
    check(size > 0, "element serialization failed");
    return size;
}

template<class P> static void run(const Settings& settings, const Profile& profile, ThreadPool& pool) {
    const size_t chunks = settings.total / settings.batch;
    const size_t tau = profile.parties.size();
    const size_t share_count = wbtx::checked_mul(chunks, tau);
    Marker setup;
    const auto material = P::keygen(settings.setup, profile.weights, profile.required_weight - 1);
    const auto setup_measurement = setup.finish("setup");
    const size_t cipher_bytes = serialized_size(material.cipher_generator);
    const size_t key_bytes = serialized_size(material.key_generator);
    const size_t scalar_bytes = serialized_size(mcl::Fr(1));
    const size_t target_bytes = serialized_size(material.target_generator);
    const size_t core_points = wbtx::checked_add(material.negative.size(), material.positive.size());
    const size_t public_points = wbtx::checked_add(core_points, material.verification.size());
    const size_t public_bytes = wbtx::checked_mul(public_points, key_bytes);
    // Fixed-width compressed source-group encodings are measured above. These
    // are cryptographic payload sizes, not a defined public wire serialization.
    std::cerr << std::setprecision(9)
              << "metadata curve=" << settings.curve << " mcl_curve="
              << (settings.curve == "bn254" ? "BN_SNARK1" : "BLS12_381")
              << " simd=" << settings.simd << " orientation=" << settings.orientation
              << " total=" << settings.total << " batch=" << settings.batch << " setup=" << settings.setup
              << " chunks=" << chunks << " threads=" << settings.threads
              << " samples=" << settings.samples << " warmup=" << settings.warmup
              << " N=" << profile.weights.size() << " W=" << profile.total_weight
              << " q=" << profile.required_weight << " t=" << profile.required_weight - 1
              << " tau=" << tau << " W_T=" << profile.accepted_weight << " share_count=" << share_count
              << " setup_ms=" << setup_measurement.ms << " setup_avx_msm=" << setup_measurement.msm
              << " setup_avx_each=" << setup_measurement.each << '\n'
              << "sizes g1_bytes=" << (P::swapped ? key_bytes : cipher_bytes)
              << " g2_bytes=" << (P::swapped ? cipher_bytes : key_bytes)
              << " scalar_bytes=" << scalar_bytes << " target_full_fp12_bytes=" << target_bytes
              << " cipher_group_bytes=" << cipher_bytes << " public_group_bytes=" << key_bytes
              << " core_points=" << core_points << " verification_points=" << material.verification.size()
              << " public_array_points=" << public_points << " public_array_compressed_bytes=" << public_bytes
              << " proof_payload_bytes=" << wbtx::checked_add(cipher_bytes, scalar_bytes)
              << " ciphertext_payload_bytes=" << wbtx::checked_add(wbtx::checked_mul(2, cipher_bytes),
                                                                     wbtx::checked_add(target_bytes, scalar_bytes))
              << " share_payload_bytes=" << cipher_bytes
              << " validator_share_payload_bytes=" << wbtx::checked_mul(chunks, cipher_bytes)
              << " all_share_payload_bytes=" << wbtx::checked_mul(share_count, cipher_bytes) << '\n';

    std::vector<mcl::Fp12> messages(settings.total);
    for (size_t i = 0; i < messages.size(); ++i)
        mcl::Fp12::pow(messages[i], material.target_generator, mcl::Fr(int64_t(i) + 1));
    using CT = typename P::Ciphertext;
    using Batch = typename P::ValidatedBatch;
    using Share = typename P::Share;
    using Accepted = typename P::Accepted;
    using Cross = typename P::CrossTerms;
    using Committee = typename P::Committee;

    std::cout << "curve,simd,orientation,total,batch,setup,cache,threads,parties,total_weight,required_weight,"
                 "selected_parties,accepted_weight,sample,phase,milliseconds,avx_msm,avx_each\n";
    for (const bool cached : {false, true}) {
        std::optional<Committee> cached_committee;
        if (cached) {
            std::vector<CT> fixture(settings.batch);
            pool.parallel_for(settings.batch, [&](size_t i) { fixture[i] = P::encrypt(material, messages[i]); });
            auto batch = P::validate(material, fixture);
            std::vector<Share> shares(tau);
            pool.parallel_for(tau, [&](size_t j) { shares[j] = P::partial(material, profile.parties[j], batch); });
            auto accepted = P::accept(material, batch, shares);
            check(accepted.parties == profile.parties, "unexpected cached fixture committee");
            cached_committee.emplace(P::prepare(material, accepted));
        }
        for (size_t iteration = 0; iteration < settings.warmup + settings.samples; ++iteration) {
            std::vector<Measurement> timings;
            timings.reserve(10);
            msm_calls = 0; each_calls = 0;
            Marker total;
            Marker encryption;
            std::vector<std::vector<CT>> ciphertexts(chunks, std::vector<CT>(settings.batch));
            pool.parallel_for(settings.total, [&](size_t i) {
                ciphertexts[i / settings.batch][i % settings.batch] = P::encrypt(material, messages[i]);
            });
            timings.push_back(encryption.finish("encryption"));
            Marker validation;
            std::vector<Batch> batches(chunks);
            pool.parallel_for(chunks, [&](size_t c) { batches[c] = P::validate(material, ciphertexts[c]); });
            timings.push_back(validation.finish("validation"));
            Marker decryption;
            Marker share_generation;
            std::vector<std::vector<Share>> shares(chunks, std::vector<Share>(tau));
            pool.parallel_for(share_count, [&](size_t i) {
                const size_t c = i / tau, j = i % tau;
                shares[c][j] = P::partial(material, profile.parties[j], batches[c]);
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
            const auto opening_end = Clock::now();
            timings.push_back(opening.finish("opening", opening_end));
            timings.push_back(combiner.finish("combiner", opening_end));
            timings.push_back(decryption.finish("decryption_total", opening_end));
            timings.push_back(total.finish("end_to_end", opening_end));
            for (size_t c = 0; c < chunks; ++c) {
                check(accepted[c].parties == profile.parties && accepted[c].accepted_weight == profile.accepted_weight,
                      "unexpected accepted committee");
                check(opened[c].size() == settings.batch, "wrong number of plaintexts");
                for (size_t i = 0; i < settings.batch; ++i)
                    check(batches[c].valid[i] && opened[c][i].has_value() &&
                          *opened[c][i] == messages[c * settings.batch + i], "plaintext mismatch");
            }
            if (iteration >= settings.warmup) for (const auto& timing : timings)
                std::cout << settings.curve << ',' << settings.simd << ',' << settings.orientation << ','
                          << settings.total << ',' << settings.batch << ',' << settings.setup << ','
                          << (cached ? "cached" : "cold") << ',' << settings.threads << ','
                          << profile.weights.size() << ',' << profile.total_weight << ',' << profile.required_weight << ','
                          << tau << ',' << profile.accepted_weight << ',' << iteration - settings.warmup << ','
                          << timing.phase << ',' << std::fixed << std::setprecision(6) << timing.ms << ','
                          << timing.msm << ',' << timing.each << '\n';
        }
        std::cout.flush();
        std::cerr << "correctness=passed layout=" << chunks << 'x' << settings.batch
                  << " cache=" << (cached ? "cached" : "cold")
                  << " iterations=" << settings.samples + settings.warmup << '\n';
    }
}

int main(int argc, char** argv) try {
    check(argc == 11, "usage: paper_benchmark bls12_381|bn254 off|avx512 normal|swapped "
                      "profile.txt total batch setup threads samples warmup");
    Settings settings{argv[1], argv[2], argv[3], argv[4],
                      parse_size(argv[5], "total"), parse_size(argv[6], "batch"), parse_size(argv[7], "setup"),
                      parse_size(argv[8], "threads"), parse_size(argv[9], "samples"), parse_size(argv[10], "warmup", true)};
    check(settings.curve == "bls12_381" || settings.curve == "bn254", "unknown curve");
    check(settings.simd == "off" || settings.simd == "avx512", "unknown SIMD mode");
    check(settings.orientation == "normal" || settings.orientation == "swapped", "unknown orientation");
    check(settings.total % settings.batch == 0, "batch must divide total ciphertext count");
    check(settings.setup >= settings.batch, "setup must support the chunk batch size");
    check(settings.total <= size_t(std::numeric_limits<int64_t>::max()), "too many benchmark messages");
    (void)wbtx::checked_add(settings.samples, settings.warmup);
    (void)wbtx::checked_mul(settings.threads, 4);
    const Profile profile = read_profile(settings.profile);
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
    ThreadPool pool(settings.threads);
    wbtx::set_parallel_executor([&](size_t n, const std::function<void(size_t)>& fn) { pool.parallel_for(n, fn); });
    if (settings.orientation == "normal") run<wbtx::Normal>(settings, profile, pool);
    else run<wbtx::Swapped>(settings, profile, pool);
    wbtx::set_parallel_executor({});
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
}
