// Weighted-BTX operation-shape benchmark; this is not a protocol implementation.
// Public, deterministic synthetic points and coefficients only.
#include <mcl/bn.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mcl;
using Clock = std::chrono::steady_clock;
static G1::MulVecOptiFunc avx_msm = nullptr;
static G1::MulEachOptiFunc avx_each = nullptr;
static uint64_t msm_dispatches = 0, each_dispatches = 0;
static volatile uint64_t sink = 0;

void counted_msm(G1& out, G1* points, const Fr* scalars, size_t n, size_t b) {
    ++msm_dispatches;
    avx_msm(out, points, scalars, n, b);
}
void counted_each(G1* points, const Fr* scalars, size_t n) {
    ++each_dispatches;
    avx_each(points, scalars, n);
}
void require(bool yes, const std::string& why) {
    if (!yes) throw std::runtime_error(why);
}
template<class T> void consume(const T& value) {
    unsigned char buf[1024];
    const auto n = value.serialize(buf, sizeof(buf));
    require(n != 0, "serialization failed");
    uint64_t digest = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) digest = (digest ^ buf[i]) * 1099511628211ULL;
    sink = digest;
}
Fr scalar(size_t index, const char* label) {
    const auto text = std::string(label) + ":" + std::to_string(index);
    Fr out;
    out.setHashOf(text);
    if (out.isZero()) out = 1;
    return out;
}
template<class G> std::vector<G> point_sequence(const G& generator, size_t n) {
    std::vector<G> points(n);
    G step, cur;
    G::mul(step, generator, scalar(0, "point-step"));
    G::mul(cur, generator, scalar(0, "point-start"));
    for (auto& point : points) {
        point = cur;
        G::add(cur, cur, step);
    }
    G::normalizeVec(points.data(), points.data(), points.size());
    return points;
}

struct Domain {
    size_t n;
    Fr root, inverse_root, inverse_n;
    std::vector<Fr> powers, inverse_powers;
    explicit Domain(size_t size) : n(size), powers(size), inverse_powers(size) {
        require(n >= 2 && (n & (n - 1)) == 0, "FFT size must be a power of two");
        auto exponent = Fr::getOp().mp - 1;
        require(exponent % int(n) == 0, "scalar field does not support FFT size");
        exponent /= int(n);
        for (int candidate = 2;; ++candidate) {
            Fr::pow(root, Fr(candidate), exponent);
            Fr test;
            Fr::pow(test, root, int(n / 2));
            if (!test.isOne()) break;
        }
        Fr test;
        Fr::pow(test, root, int(n));
        require(test.isOne(), "root order check failed");
        Fr::inv(inverse_root, root);
        Fr::inv(inverse_n, Fr(int(n)));
        powers[0] = inverse_powers[0] = 1;
        for (size_t i = 1; i < n; ++i) {
            Fr::mul(powers[i], powers[i - 1], root);
            Fr::mul(inverse_powers[i], inverse_powers[i - 1], inverse_root);
        }
    }
    // Same transform algorithm and scratch allocations with SIMD enabled or off.
    // Collecting all nontrivial twiddle multiplies in a stage makes mulEach useful.
    template<class G> void fft(std::vector<G>& points, bool inverse = false) const {
        require(points.size() == n, "wrong FFT input size");
        for (size_t i = 1, j = 0; i < n; ++i) {
            size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(points[i], points[j]);
        }
        const auto& ws = inverse ? inverse_powers : powers;
        std::vector<G> products;
        std::vector<Fr> twiddles;
        products.reserve(n / 2);
        twiddles.reserve(n / 2);
        for (size_t width = 2; width <= n; width *= 2) {
            const auto half = width / 2, stride = n / width;
            products.clear();
            twiddles.clear();
            for (size_t base = 0; base < n; base += width)
                for (size_t j = 1; j < half; ++j) {
                    products.push_back(points[base + half + j]);
                    twiddles.push_back(ws[j * stride]);
                }
            G::mulEach(products.data(), twiddles.data(), products.size());
            size_t k = 0;
            for (size_t base = 0; base < n; base += width)
                for (size_t j = 0; j < half; ++j) {
                    const G even = points[base + j];
                    const G odd = j == 0 ? points[base + half] : products[k++];
                    G::add(points[base + j], even, odd);
                    G::sub(points[base + half + j], even, odd);
                }
        }
        if (inverse) {
            twiddles.assign(n, inverse_n);
            G::mulEach(points.data(), twiddles.data(), n);
        }
    }
};

struct Settings {
    std::string curve, mode;
    size_t batch, weight, parties, samples;
    double sample_ms;
    std::vector<size_t> weights;
};

void measure(const Settings& s, const char* phase, const char* group,
             const std::function<void()>& work, const std::function<void()>& finish) {
    work();
    finish();
    size_t repetitions = 1;
    for (;;) {
        const auto start = Clock::now();
        for (size_t i = 0; i < repetitions; ++i) work();
        const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        finish();
        if (elapsed >= s.sample_ms || repetitions >= (1U << 20)) break;
        repetitions *= std::min<size_t>(8, std::max<size_t>(2, size_t(s.sample_ms / std::max(0.001, elapsed))));
    }
    std::vector<double> samples;
    uint64_t total_msm = 0, total_each = 0;
    for (size_t sample = 0; sample < s.samples; ++sample) {
        msm_dispatches = each_dispatches = 0;
        const auto start = Clock::now();
        for (size_t i = 0; i < repetitions; ++i) work();
        samples.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / repetitions);
        total_msm += msm_dispatches;
        total_each += each_dispatches;
        finish();
    }
    std::sort(samples.begin(), samples.end());
    auto quantile = [&](double q) { return samples[size_t(std::floor(q * (samples.size() - 1)))]; };
    std::cout << s.curve << ',' << s.mode << ',' << phase << ',' << group << ','
              << s.batch << ',' << s.weight << ',' << s.parties << ',' << s.samples << ',' << repetitions << ','
              << std::fixed << std::setprecision(4) << quantile(0.5) << ',' << quantile(0.1) << ',' << quantile(0.9) << ','
              << double(total_msm) / (repetitions * s.samples) << ','
              << double(total_each) / (repetitions * s.samples) << '\n';
}

template<class G> void check_msm(std::vector<G>& points, const std::vector<Fr>& scalars,
                                size_t n) {
    G reference, term, actual;
    reference.clear();
    for (size_t i = 0; i < n; ++i) {
        G::mul(term, points[i], scalars[i]);
        G::add(reference, reference, term);
    }
    G::mulVec(actual, points.data(), scalars.data(), n);
    require(actual == reference, "MSM disagrees with direct scalar multiplication and addition");
}

template<class G> void group_kernels(const Settings& s, const char* name, const G& generator) {
    const size_t width = std::max({s.weight, s.batch * 2, s.parties});
    auto points = point_sequence(generator, width * (2 * s.batch));
    std::vector<Fr> coefficients(width), powers(s.batch);
    for (size_t i = 0; i < width; ++i) coefficients[i] = scalar(i, "coefficient");
    const Fr q = scalar(0, "validator-secret");
    Fr power = q;
    for (auto& entry : powers) { entry = power; power *= q; }
    // Cover both the scalar MSM path and the actual SIMD-sized path.
    check_msm(points, coefficients, s.weight);
    check_msm(points, coefficients, s.parties);
    check_msm(points, powers, s.batch);
    for (size_t w : s.weights) check_msm(points, coefficients, w);
    Domain domain(2 * s.batch), small(8);
    std::vector<G> input(points.begin(), points.begin() + 2 * s.batch), transformed = input;
    domain.fft(transformed);
    domain.fft(transformed, true);
    require(transformed == input, "group FFT inverse roundtrip failed");
    std::vector<G> small_in(points.begin(), points.begin() + 8), small_out = small_in;
    small.fft(small_out);
    for (size_t k = 0; k < 8; ++k) {
        G sum, term;
        sum.clear();
        for (size_t j = 0; j < 8; ++j) {
            G::mul(term, small_in[j], small.powers[(j * k) % 8]);
            G::add(sum, sum, term);
        }
        require(sum == small_out[k], "group FFT disagrees with direct DFT");
    }
    G out;
    out.clear();
    std::vector<G> output_keys(s.batch * s.parties), output_positive(s.batch - 1), output_verify(s.batch);
    measure(s, "partial_decrypt_one_msm", name, [&] {
        // Includes computing q^1,...,q^B as in the native implementation.
        Fr p = q;
        for (auto& entry : powers) { entry = p; p *= q; }
        G::mulVec(out, points.data(), powers.data(), s.batch);
    }, [&] { consume(out); });
    measure(s, "verification_key_msms", name, [&] {
        for (size_t slot = 0; slot < s.batch; ++slot)
            G::mulVec(output_verify[slot], points.data() + slot * width, coefficients.data(), s.parties);
        G::normalizeVec(output_verify.data(), output_verify.data(), output_verify.size());
    }, [&] { consume(output_verify.back()); });
    measure(s, "committee_opening_msms", name, [&] {
        for (size_t slot = 0; slot < s.batch; ++slot) {
            size_t offset = 0;
            for (size_t party = 0; party < s.parties; ++party) {
                G::mulVec(output_keys[slot * s.parties + party], points.data() + slot * width + offset,
                          coefficients.data() + offset, s.weights[party]);
                offset += s.weights[party];
            }
        }
        G::normalizeVec(output_keys.data(), output_keys.data(), output_keys.size());
    }, [&] { consume(output_keys.back()); });
    measure(s, "committee_positive_msms", name, [&] {
        for (size_t slot = 0; slot + 1 < s.batch; ++slot)
            G::mulVec(output_positive[slot], points.data() + (s.batch + slot) * width,
                      coefficients.data(), s.weight);
    }, [&] { consume(output_positive.back()); });
    measure(s, "group_fft", name, [&] {
        transformed = input;
        domain.fft(transformed);
        G::normalizeVec(transformed.data(), transformed.data(), transformed.size());
    }, [&] { consume(transformed.back()); });
}

void pairing_kernels(const Settings& s, const G1& g1, const G2& g2) {
    auto p = point_sequence(g1, std::max(s.parties * s.batch, 2 * s.batch));
    auto q = point_sequence(g2, std::max(s.parties * s.batch, 2 * s.batch));
    Fp12 out, reference, term;
    reference = 1;
    for (size_t j = 0; j < s.parties; ++j) {
        pairing(term, p[j], q[j]);
        Fp12::mul(reference, reference, term);
    }
    millerLoopVec(out, p.data(), q.data(), s.parties);
    finalExp(out, out);
    require(out == reference, "multi-pairing disagrees with product of pairings");
    std::vector<Fp12> opened(s.batch), middle(2 * s.batch);
    std::vector<std::vector<Fp6>> lines(2 * s.batch);
    for (size_t i = 0; i < lines.size(); ++i) precomputeG2(lines[i], q[i]);
    precomputedMillerLoop(out, p[0], lines[0]);
    finalExp(out, out);
    pairing(reference, p[0], q[0]);
    require(out == reference, "prepared pairing disagrees with ordinary pairing");
    measure(s, "opening_multi_pairings", "G1xG2", [&] {
        for (size_t slot = 0; slot < s.batch; ++slot) {
            millerLoopVec(opened[slot], p.data() + slot * s.parties, q.data() + slot * s.parties, s.parties);
            finalExp(opened[slot], opened[slot]);
        }
    }, [&] { consume(opened.back()); });
    measure(s, "g2_line_preparation", "G2", [&] {
        for (size_t i = 0; i < lines.size(); ++i) precomputeG2(lines[i], q[i]);
    }, [&] { consume(lines.back().back()); });
    measure(s, "middle_prepared_miller_loops", "G1xG2", [&] {
        for (size_t i = 0; i < middle.size(); ++i) precomputedMillerLoop(middle[i], p[i], lines[i]);
    }, [&] { consume(middle.back()); });
    const auto miller_values = middle;
    measure(s, "middle_final_exponents", "GT", [&] {
        for (size_t i = 0; i < middle.size(); ++i) finalExp(middle[i], miller_values[i]);
    }, [&] { consume(middle.back()); });
    // The previous phase is a primitive-only cost probe. It is deliberately
    // not added to an end-to-end estimate (native uses split final exponents).
}

int main(int argc, char** argv) try {
    if (argc != 7) throw std::runtime_error("usage: kernel_bench bls12_381|bn254 off|avx512 B weights.txt samples sample_ms");
    Settings s;
    s.curve = argv[1]; s.mode = argv[2]; s.batch = std::stoul(argv[3]);
    s.samples = std::stoul(argv[5]); s.sample_ms = std::stod(argv[6]);
    require(s.batch >= 4 && s.batch <= 4096 && (s.batch & (s.batch - 1)) == 0, "invalid batch size");
    require(s.samples >= 3 && s.sample_ms > 0, "invalid sample settings");
    std::ifstream weights(argv[4]);
    require(bool(weights), "cannot open weights file");
    size_t weight;
    while (weights >> weight) { require(weight > 0, "zero weight"); s.weights.push_back(weight); }
    require(!s.weights.empty(), "empty weights");
    s.parties = s.weights.size();
    s.weight = std::accumulate(s.weights.begin(), s.weights.end(), size_t(0));
    require(s.curve == "bls12_381" || s.curve == "bn254", "unknown curve");
    initPairing(s.curve == "bls12_381" ? BLS12_381 : BN_SNARK1);
    avx_msm = G1::mulVecOpti;
    avx_each = G1::mulEachOpti;
    require(s.mode == "off" || s.mode == "avx512", "unknown SIMD mode");
    if (s.mode == "avx512") {
        require(avx_msm && avx_each, "AVX512 callbacks unavailable: require BLS12-381 and AVX512IFMA CPU/build");
        G1::setMulVecOpti(counted_msm);
        G1::setMulEachOpti(counted_each);
    } else {
        G1::setMulVecOpti(nullptr);
        G1::setMulEachOpti(nullptr);
    }
    std::cerr << "curve=" << s.curve << " mcl_curve=" << (s.curve == "bn254" ? "BN_SNARK1" : "BLS12_381")
              << " scalar_modulus=" << Fr::getModulo() << " simd=" << s.mode
              << " avx_msm_available=" << bool(avx_msm) << " avx_each_available=" << bool(avx_each)
              << " B=" << s.batch << " T=" << s.parties << " W_T=" << s.weight << '\n';
    G1 g1; G2 g2;
    hashAndMapToG1(g1, "weighted-btx-kernel-generator", 29);
    hashAndMapToG2(g2, "weighted-btx-kernel-generator", 29);
    std::cout << "curve,simd,phase,group,batch,accepted_weight,parties,samples,repetitions,median_us,p10_us,p90_us,avx_msm_calls_per_rep,avx_each_calls_per_rep\n";
    group_kernels(s, "G1", g1);
    group_kernels(s, "G2", g2);
    pairing_kernels(s, g1, g2);
    std::cerr << "correctness_checks=passed sink=" << sink << '\n';
} catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
}
