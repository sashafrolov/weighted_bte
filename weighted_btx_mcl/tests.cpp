// Independent protocol checks. Keep reference interpolation and cross terms
// literal so they do not repeat the optimized implementation's algorithms.
#include "protocol.hpp"
#include "thread_pool.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using mcl::Fp12;
using mcl::Fr;
using mcl::G1;
using mcl::G2;
size_t checks = 0;

void check(bool condition, const std::string& label) {
    ++checks;
    if (!condition) throw std::runtime_error("TEST FAILED: " + label);
}

template<class F> void rejects(F&& action, const std::string& label) {
    bool rejected = false;
    try { action(); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, label);
}

template<class C, class K> Fp12 reference_pair(const C& ciphertext, const K& key) {
    Fp12 result;
    if constexpr (std::is_same_v<C, G1>) {
        mcl::pairing(result, ciphertext, key);
    } else {
        mcl::pairing(result, key, ciphertext);
    }
    return result;
}

void independent_math_checks() {
    G1 g1;
    G2 g2;
    mcl::hashAndMapToG1(g1, "test-g1", 7);
    mcl::hashAndMapToG2(g2, "test-g2", 7);
    const Fp12 target = reference_pair(g1, g2);
    for (size_t n : {1, 2, 8, 32, 128}) {
        const wbtx::Domain domain(n);
        std::vector<Fr> scalars(n);
        std::vector<G1> points1(n);
        std::vector<G2> points2(n);
        std::vector<Fp12> targets(n);
        for (size_t i = 0; i < n; ++i) {
            const auto label = std::string("independent-test-input:") + std::to_string(i);
            scalars[i].setHashOf(label);
            if (i == 0) scalars[i] = 0;
            G1::mul(points1[i], g1, scalars[i]);
            G2::mul(points2[i], g2, scalars[i]);
            Fp12 generic;
            Fp12::powGeneric(generic, target, scalars[i]);
            Fp12::pow(targets[i], target, scalars[i]);
            check(generic == targets[i], "GT GLV exponent equals generic exponent on subgroup");
        }
        const auto original_scalars = scalars;
        const auto original_points1 = points1;
        const auto original_points2 = points2;
        const auto original_targets = targets;
        domain.fft(scalars);
        domain.fft(points1);
        domain.fft(points2);
        domain.fft(targets);
        // Scalar DFT supplies a reference for all three group transforms.
        // Using powGeneric avoids sharing the implementation's GT GLV path.
        if (n <= 8) for (size_t k = 0; k < n; ++k) {
            Fr exponent(0);
            for (size_t j = 0; j < n; ++j)
                exponent += original_scalars[j] * domain.powers[(j * k) % n];
            G1 expected1;
            G2 expected2;
            Fp12 expected_target;
            G1::mul(expected1, g1, exponent);
            G2::mul(expected2, g2, exponent);
            Fp12::powGeneric(expected_target, target, exponent);
            check(scalars[k] == exponent, "scalar FFT equals direct DFT");
            check(points1[k] == expected1, "G1 FFT equals direct DFT");
            check(points2[k] == expected2, "G2 FFT equals direct DFT");
            check(targets[k] == expected_target, "GT FFT equals direct DFT");
        }
        domain.fft(scalars, true);
        domain.fft(points1, true);
        domain.fft(points2, true);
        domain.fft(targets, true);
        check(scalars == original_scalars, "scalar FFT inverse roundtrip");
        check(points1 == original_points1, "G1 FFT inverse roundtrip including vector-sized stage");
        check(points2 == original_points2, "G2 FFT inverse roundtrip");
        check(targets == original_targets, "GT FFT inverse roundtrip");
    }
    const wbtx::Domain domain(8);
    const std::vector<size_t> all{0, 1, 2, 3, 4, 5, 6, 7};
    const auto full_coefficients = wbtx::lagrange_at_zero(domain.powers, all);
    for (const auto& value : full_coefficients)
        check(value == Fr(1) / Fr(8), "full root domain interpolation shortcut");
    rejects([&] { wbtx::lagrange_at_zero(domain.powers, {1, 1}); }, "duplicate interpolation point rejected");
    rejects([&] { wbtx::lagrange_at_zero(domain.powers, {8}); }, "out-of-range interpolation point rejected");
    rejects([] { wbtx::batch_inverse({Fr(1), Fr(0)}); }, "zero inversion denominator rejected");
}

template<class P> struct Fixture {
    typename P::Material material;
    std::vector<Fp12> messages;
    std::vector<typename P::Ciphertext> ciphertexts;
    typename P::ValidatedBatch batch;
    std::vector<typename P::Share> shares;
    typename P::Accepted accepted;
    typename P::Committee committee;
    typename P::CrossTerms cross;
};

template<class P> std::vector<typename P::Share> shares_for(
    const typename P::Material& material,
    const typename P::ValidatedBatch& batch,
    const std::vector<size_t>& parties) {
    std::vector<typename P::Share> result;
    for (auto party : parties) result.push_back(P::partial(material, party, batch));
    return result;
}

template<class P> Fixture<P> fixture(
    size_t batch_size, const std::vector<size_t>& parties = {3, 1},
    size_t maximum = 16, const std::vector<size_t>& weights = {1, 3, 2, 4, 1},
    size_t threshold = 5) {
    Fixture<P> f;
    f.material = P::keygen(maximum, weights, threshold);
    for (size_t i = 0; i < batch_size; ++i) {
        Fp12 message;
        // Include the target-group identity as a real, valid plaintext.
        Fp12::pow(message, f.material.target_generator, Fr(int(i * i + i)));
        f.messages.push_back(message);
        f.ciphertexts.push_back(P::encrypt(f.material, message));
    }
    f.batch = P::validate(f.material, f.ciphertexts);
    f.shares = shares_for<P>(f.material, f.batch, parties);
    f.accepted = P::accept(f.material, f.batch, f.shares);
    f.committee = P::prepare(f.material, f.accepted);
    f.cross = P::precompute(f.committee, f.batch);
    return f;
}

template<class P> void opens_expected(const Fixture<P>& f, const std::string& label) {
    const auto opened = P::open(f.committee, f.accepted, f.batch, f.ciphertexts, f.cross);
    check(opened.size() == f.messages.size(), label + ": output count");
    for (size_t i = 0; i < opened.size(); ++i) {
        check(bool(opened[i]) == bool(f.batch.valid[i]), label + ": validity mask");
        if (opened[i]) check(*opened[i] == f.messages[i], label + ": plaintext");
    }
}

template<class P> void literal_interpolation_and_middle_product(const Fixture<P>& f) {
    using K = typename std::decay_t<decltype(f.material.negative)>::value_type;
    std::vector<size_t> selected;
    for (auto party : f.accepted.parties)
        for (size_t i = f.material.offsets[party]; i < f.material.offsets[party + 1]; ++i)
            selected.push_back(i);
    check(selected == f.committee.selected, "canonical selected virtual points");
    std::vector<Fr> lambda(selected.size());
    for (size_t i = 0; i < selected.size(); ++i) {
        Fr numerator(1), denominator(1);
        for (size_t j = 0; j < selected.size(); ++j) if (i != j) {
            numerator *= -f.material.domain[selected[j]];
            denominator *= f.material.domain[selected[i]] - f.material.domain[selected[j]];
        }
        lambda[i] = numerator / denominator;
    }
    check(lambda == f.committee.coefficients, "coefficients equal direct Lagrange products");
    for (size_t degree = 0; degree <= f.material.threshold; ++degree) {
        Fr interpolated(0), term;
        for (size_t i = 0; i < selected.size(); ++i) {
            Fr::pow(term, f.material.domain[selected[i]], int(degree));
            interpolated += lambda[i] * term;
        }
        check(interpolated == Fr(degree == 0 ? 1 : 0), "interpolation reproduces polynomial at zero");
    }
    for (size_t slot = 0; slot < f.batch.batch_size; ++slot) {
        size_t offset = 0;
        for (size_t j = 0; j < f.accepted.parties.size(); ++j) {
            const auto party = f.accepted.parties[j];
            K expected, term;
            expected.clear();
            for (size_t virtual_index = f.material.offsets[party];
                 virtual_index < f.material.offsets[party + 1]; ++virtual_index) {
                K::mul(term, f.material.negative[slot * f.material.total_weight + virtual_index], lambda[offset++]);
                K::add(expected, expected, term);
            }
            check(expected == f.committee.opening_keys[slot * f.accepted.parties.size() + j],
                  "opening key equals independent weighted sum");
        }
    }
    for (size_t i = 0; i < f.batch.batch_size; ++i) {
        Fp12 expected(1);
        if (f.batch.valid[i]) {
            for (size_t j = 0; j < f.batch.batch_size; ++j) {
                if (j == i || !f.batch.valid[j]) continue;
                const size_t distance = j > i ? j - i : i - j;
                const auto& material = j > i ? f.material.positive : f.material.negative;
                K aggregate, term;
                aggregate.clear();
                for (size_t k = 0; k < selected.size(); ++k) {
                    K::mul(term, material[(distance - 1) * f.material.total_weight + selected[k]], lambda[k]);
                    K::add(aggregate, aggregate, term);
                }
                expected *= reference_pair(f.batch.first[j], aggregate);
            }
        }
        check(expected == f.cross.beta[i], "FFT beta equals literal pairwise cross terms");
    }
}

template<class P> void malformed_and_weighted_shares() {
    auto f = fixture<P>(4);
    check(f.accepted.parties == std::vector<size_t>({1, 3}), "accepted parties sorted canonically");
    check(f.accepted.accepted_weight == 7, "authorization uses summed weight");
    auto insufficient = shares_for<P>(f.material, f.batch, {0, 2, 4});
    rejects([&] { P::accept(f.material, f.batch, insufficient); }, "three light parties cannot authorize");
    auto submitted = shares_for<P>(f.material, f.batch, {3, 0, 1});
    submitted[1].sigma += f.material.cipher_generator;
    const auto accepted = P::accept(f.material, f.batch, submitted);
    check(accepted.parties == std::vector<size_t>({1, 3}), "fallback preserves honest sufficient committee");
    check(accepted.rejected == std::vector<size_t>({0}), "fallback blames exactly malformed party");
    auto opened = P::open(f.committee, accepted, f.batch, f.ciphertexts, f.cross);
    for (size_t i = 0; i < opened.size(); ++i)
        check(opened[i] && *opened[i] == f.messages[i], "fallback committee recovers plaintext");
    auto malformed = f.shares;
    malformed[0].sigma += f.material.cipher_generator;
    rejects([&] { P::accept(f.material, f.batch, malformed); }, "malformed share can remove threshold weight");
    auto duplicate = f.shares;
    duplicate.push_back(f.shares[0]);
    rejects([&] { P::accept(f.material, f.batch, duplicate); }, "duplicate validator cannot count twice");
    auto out_of_range = f.shares;
    out_of_range[0].party = f.material.weights.size();
    rejects([&] { P::accept(f.material, f.batch, out_of_range); }, "out-of-range submitted validator rejected");
    rejects([&] { P::partial(f.material, f.material.weights.size(), f.batch); }, "out-of-range local validator rejected");
    rejects([&] { P::accept(f.material, f.batch, {}); }, "empty share set rejected");
}

template<class P> void proof_binding_and_invalid_slots() {
    auto f = fixture<P>(7);
    f.ciphertexts[2].proof.response += Fr(1);
    f.batch = P::validate(f.material, f.ciphertexts);
    check(!f.batch.valid[2] && f.batch.first[2].isZero(), "invalid client slot becomes identity");
    check(f.batch.batch_size == 7, "invalid slot is not compacted");
    f.shares = shares_for<P>(f.material, f.batch, {1, 3});
    f.accepted = P::accept(f.material, f.batch, f.shares);
    f.committee = P::prepare(f.material, f.accepted);
    f.cross = P::precompute(f.committee, f.batch);
    opens_expected(f, "one invalid slot");
    literal_interpolation_and_middle_product(f);

    auto first_changed = f.ciphertexts;
    first_changed[0].first += f.material.cipher_generator;
    check(!P::validate(f.material, first_changed).valid[0], "proof binds ciphertext first component");
    auto second_changed = f.ciphertexts;
    second_changed[0].second *= f.material.target_generator;
    check(!P::validate(f.material, second_changed).valid[0], "proof binds ciphertext second component");
    auto commitment_changed = f.ciphertexts;
    commitment_changed[0].proof.commitment += f.material.cipher_generator;
    check(!P::validate(f.material, commitment_changed).valid[0], "proof binds commitment");
    rejects([&] { P::open(f.committee, f.accepted, f.batch, second_changed, f.cross); },
            "opening rejects ciphertext mutation after validation");

    for (auto& ciphertext : f.ciphertexts) ciphertext.proof.response += Fr(1);
    f.batch = P::validate(f.material, f.ciphertexts);
    check(std::none_of(f.batch.valid.begin(), f.batch.valid.end(), [](auto v) { return bool(v); }),
          "entire invalid batch is masked");
    f.shares = shares_for<P>(f.material, f.batch, {1, 3});
    for (const auto& share : f.shares) check(share.sigma.isZero(), "all-invalid batch has identity shares");
    f.accepted = P::accept(f.material, f.batch, f.shares);
    f.committee = P::prepare(f.material, f.accepted);
    f.cross = P::precompute(f.committee, f.batch);
    opens_expected(f, "all invalid slots");
}

template<class P> void context_bindings_and_reuse() {
    auto f = fixture<P>(4);
    auto other = P::keygen(16, f.material.weights, f.material.threshold);
    const auto wrong_setup = P::validate(other, f.ciphertexts);
    check(std::none_of(wrong_setup.valid.begin(), wrong_setup.valid.end(), [](auto v) { return bool(v); }),
          "client proof binds expected setup");
    rejects([&] { P::partial(other, 1, f.batch); }, "partial share rejects different setup");
    rejects([&] { P::accept(other, f.batch, f.shares); }, "acceptance rejects different setup");
    rejects([&] { P::prepare(other, f.accepted); }, "preparation rejects different setup");
    rejects([&] { P::precompute(f.committee, wrong_setup); }, "cross terms reject different setup");

    auto fresh_ciphertexts = f.ciphertexts;
    for (size_t i = 0; i < f.messages.size(); ++i)
        fresh_ciphertexts[i] = P::encrypt(f.material, f.messages[i]);
    const auto fresh = P::validate(f.material, fresh_ciphertexts);
    rejects([&] { P::accept(f.material, fresh, f.shares); }, "shares bind ciphertext batch");
    auto fresh_shares = shares_for<P>(f.material, fresh, {1, 3});
    auto fresh_accepted = P::accept(f.material, fresh, fresh_shares);
    const auto fresh_cross = P::precompute(f.committee, fresh);
    auto fresh_opened = P::open(f.committee, fresh_accepted, fresh, fresh_ciphertexts, fresh_cross);
    for (size_t i = 0; i < fresh_opened.size(); ++i)
        check(fresh_opened[i] && *fresh_opened[i] == f.messages[i], "committee preparation reused for fresh batch");
    const auto independently_prepared = P::prepare(f.material, fresh_accepted);
    check(independently_prepared.context_id == f.committee.context_id, "committee context excludes batch digest");
    check(independently_prepared.opening_keys == f.committee.opening_keys, "reused opening keys equal fresh preparation");
    check(independently_prepared.kernel_fft == f.committee.kernel_fft, "reused FFT kernel equals fresh preparation");
    rejects([&] { P::open(f.committee, f.accepted, fresh, fresh_ciphertexts, fresh_cross); },
            "opening rejects shares accepted for another batch");
    rejects([&] { P::open(f.committee, fresh_accepted, fresh, fresh_ciphertexts, f.cross); },
            "cross terms bind exact batch");

    const auto alternate_shares = shares_for<P>(f.material, f.batch, {0, 2, 3});
    const auto alternate_accepted = P::accept(f.material, f.batch, alternate_shares);
    const auto alternate_committee = P::prepare(f.material, alternate_accepted);
    const auto alternate_cross = P::precompute(alternate_committee, f.batch);
    rejects([&] { P::open(f.committee, alternate_accepted, f.batch, f.ciphertexts, f.cross); },
            "opening rejects different accepted committee");
    rejects([&] { P::open(f.committee, f.accepted, f.batch, f.ciphertexts, alternate_cross); },
            "cross terms bind committee context");
    const auto alternate_opened = P::open(alternate_committee, alternate_accepted, f.batch, f.ciphertexts, alternate_cross);
    for (size_t i = 0; i < alternate_opened.size(); ++i)
        check(alternate_opened[i] && *alternate_opened[i] == f.messages[i], "different authorized committees agree");

    auto reordered = f.ciphertexts;
    std::reverse(reordered.begin(), reordered.end());
    const auto reordered_batch = P::validate(f.material, reordered);
    check(std::all_of(reordered_batch.valid.begin(), reordered_batch.valid.end(), [](auto v) { return bool(v); }),
          "encryption-time ciphertext is index-free");
    rejects([&] { P::accept(f.material, reordered_batch, f.shares); }, "shares bind order of ciphertexts");
    const auto reordered_shares = shares_for<P>(f.material, reordered_batch, {1, 3});
    const auto reordered_accepted = P::accept(f.material, reordered_batch, reordered_shares);
    const auto reordered_cross = P::precompute(f.committee, reordered_batch);
    const auto reordered_opened = P::open(f.committee, reordered_accepted, reordered_batch, reordered, reordered_cross);
    for (size_t i = 0; i < reordered_opened.size(); ++i)
        check(reordered_opened[i] && *reordered_opened[i] == f.messages[f.messages.size() - 1 - i],
              "index-free reordered ciphertext decrypts at new slot");

    auto shorter = f.ciphertexts;
    shorter.pop_back();
    auto shorter_batch = P::validate(f.material, shorter);
    rejects([&] { P::precompute(f.committee, shorter_batch); }, "prepared committee binds actual batch size");
    auto wrong_digest = f.shares;
    for (auto& share : wrong_digest) share.batch_digest[0] ^= 1;
    rejects([&] { P::accept(f.material, f.batch, wrong_digest); }, "share metadata binds batch digest");
    auto wrong_share_setup = f.shares;
    for (auto& share : wrong_share_setup) share.setup_id[0] ^= 1;
    rejects([&] { P::accept(f.material, f.batch, wrong_share_setup); }, "share metadata binds setup digest");
    rejects([&] { P::validate(f.material, {}); }, "empty batch rejected");
    auto oversized = f.ciphertexts;
    oversized.resize(f.material.max_batch + 1, f.ciphertexts[0]);
    rejects([&] { P::validate(f.material, oversized); }, "oversized batch rejected");
}

template<class P> void setup_boundaries() {
    rejects([] { P::keygen(0, {1}, 0); }, "zero maximum batch rejected");
    rejects([] { P::keygen(1, {}, 0); }, "empty validator set rejected");
    rejects([] { P::keygen(1, {1, 0}, 0); }, "zero validator weight rejected");
    rejects([] { P::keygen(1, {2, 1}, 3); }, "unreachable threshold rejected");
    auto singleton = fixture<P>(1, {0}, 1, {1}, 0);
    opens_expected(singleton, "singleton threshold-zero boundary");
    literal_interpolation_and_middle_product(singleton);
}

template<class P> void identity_boundaries() {
    auto f = fixture<P>(4);
    for (size_t i = 0; i < f.ciphertexts.size(); ++i) {
        const Fr randomness(i % 2 == 0 ? 0 : int(i + 1));
        // Force one second component to be the identity while retaining a
        // nonidentity first component. Both are valid ciphertext boundaries.
        if (i == 1) {
            Fp12 mask;
            Fp12::powGeneric(mask, f.material.encryption_element, randomness);
            Fp12::inv(f.messages[i], mask);
        }
        f.ciphertexts[i] = P::encrypt_with_randomness(f.material, f.messages[i], randomness);
        if (i % 2 == 0) check(f.ciphertexts[i].first.isZero(), "zero randomness gives identity first component");
        if (i == 1) check(f.ciphertexts[i].second == Fp12(1), "valid ciphertext can have identity second component");
        // A zero Schnorr nonce must also work; construct its response using
        // the stated public challenge instead of the random prover helper.
        if (i == 3) {
            f.ciphertexts[i].proof.commitment.clear();
            f.ciphertexts[i].proof.response = randomness * P::proof_challenge(
                f.material, f.ciphertexts[i].first, f.ciphertexts[i].second,
                f.ciphertexts[i].proof.commitment);
        }
    }
    f.batch = P::validate(f.material, f.ciphertexts);
    check(std::all_of(f.batch.valid.begin(), f.batch.valid.end(), [](auto v) { return bool(v); }),
          "valid identity ciphertext and proof components accepted");
    f.shares = shares_for<P>(f.material, f.batch, {1, 3});
    f.accepted = P::accept(f.material, f.batch, f.shares);
    f.committee = P::prepare(f.material, f.accepted);
    f.cross = P::precompute(f.committee, f.batch);
    opens_expected(f, "mixed zero randomness");
    literal_interpolation_and_middle_product(f);

    for (size_t i = 0; i < f.ciphertexts.size(); ++i)
        f.ciphertexts[i] = P::encrypt_with_randomness(f.material, f.messages[i], Fr(0));
    f.batch = P::validate(f.material, f.ciphertexts);
    check(std::all_of(f.batch.valid.begin(), f.batch.valid.end(), [](auto v) { return bool(v); }),
          "all zero randomness remains a valid batch");
    f.shares = shares_for<P>(f.material, f.batch, {1, 3});
    for (const auto& share : f.shares) check(share.sigma.isZero(), "valid all-identity inputs give identity share");
    f.accepted = P::accept(f.material, f.batch, f.shares);
    f.cross = P::precompute(f.committee, f.batch);
    opens_expected(f, "all zero randomness");
    literal_interpolation_and_middle_product(f);

    rejects([&] { P::encrypt(f.material, Fp12(0)); }, "zero Fp12 is not a GT plaintext");
    auto zero_target = f.ciphertexts;
    zero_target[0].second = 0;
    check(!P::validate(f.material, zero_target).valid[0], "zero Fp12 ciphertext target is masked");
}

template<class P> void stale_curve_guards(const mcl::CurveParam& original, const mcl::CurveParam& other) {
    const auto f = fixture<P>(1);
    mcl::initPairing(other);
    rejects([&] { P::encrypt(f.material, f.messages[0]); }, "encrypt rejects stale global curve");
    rejects([&] { P::validate(f.material, f.ciphertexts); }, "validate rejects stale global curve");
    rejects([&] { P::partial(f.material, 1, f.batch); }, "partial rejects stale global curve");
    rejects([&] { P::accept(f.material, f.batch, f.shares); }, "accept rejects stale global curve");
    rejects([&] { P::prepare(f.material, f.accepted); }, "prepare rejects stale global curve");
    rejects([&] { P::precompute(f.committee, f.batch); }, "cross terms reject stale global curve");
    rejects([&] { P::open(f.committee, f.accepted, f.batch, f.ciphertexts, f.cross); },
            "opening rejects stale global curve");
    mcl::initPairing(original);
    opens_expected(f, "original global curve restored");
}

template<class P> void protocol_suite(const std::string& label) {
    std::cerr << "Testing " << label << '\n';
    for (size_t batch : {1, 2, 3, 4, 7, 16}) {
        const auto f = fixture<P>(batch);
        opens_expected(f, label + " B=" + std::to_string(batch));
        literal_interpolation_and_middle_product(f);
    }
    setup_boundaries<P>();
    identity_boundaries<P>();
    malformed_and_weighted_shares<P>();
    proof_binding_and_invalid_slots<P>();
    context_bindings_and_reuse<P>();
    // W_T=150 reaches MCL's >=128-point vector-MSM dispatch when available.
    const auto large = fixture<P>(4, {0, 1, 2}, 4, {60, 50, 40}, 100);
    opens_expected(large, label + " large committee");
    literal_interpolation_and_middle_product(large);
    std::cerr << "Passed " << label << '\n';
}

void nested_pool_checks(ThreadPool& pool) {
    std::vector<size_t> results(96 * 37);
    pool.parallel_for(96, [&](size_t i) {
        pool.parallel_for(37, [&](size_t j) { results[i * 37 + j] = (i + 1) * (j + 3); });
    });
    for (size_t i = 0; i < 96; ++i)
        for (size_t j = 0; j < 37; ++j)
            check(results[i * 37 + j] == (i + 1) * (j + 3), "nested pool joins every job");
    rejects([&] {
        pool.parallel_for(17, [&](size_t i) {
            pool.parallel_for(19, [&](size_t j) {
                if (i == 8 && j == 7) throw std::runtime_error("expected nested failure");
            });
        });
    }, "nested pool propagates worker exceptions");
    std::vector<size_t> after_error(43);
    pool.parallel_for(after_error.size(), [&](size_t i) { after_error[i] = i + 1; });
    for (size_t i = 0; i < after_error.size(); ++i)
        check(after_error[i] == i + 1, "pool remains usable after exception");
}
} // namespace

int main() try {
    ThreadPool pool(4);
    nested_pool_checks(pool);
    for (bool threaded : {false, true}) {
        if (threaded) wbtx::set_parallel_executor([&](size_t n, const std::function<void(size_t)>& fn) {
            pool.parallel_for(n, fn);
        });
        else wbtx::set_parallel_executor({});
        for (const auto& curve : std::vector<std::pair<std::string, mcl::CurveParam>>{
                 {"BLS12_381", mcl::BLS12_381}, {"BN_SNARK1", mcl::BN_SNARK1}}) {
            mcl::initPairing(curve.second);
            const auto mode = threaded ? " 4workers" : " serial";
            independent_math_checks();
            protocol_suite<wbtx::Normal>(curve.first + " normal" + mode);
            protocol_suite<wbtx::Swapped>(curve.first + " swapped" + mode);
            const auto& other = curve.first == "BLS12_381" ? mcl::BN_SNARK1 : mcl::BLS12_381;
            stale_curve_guards<wbtx::Normal>(curve.second, other);
            stale_curve_guards<wbtx::Swapped>(curve.second, other);
        }
    }
    wbtx::set_parallel_executor({});
    std::cout << "Protocol tests passed; independent assertions=" << checks << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
