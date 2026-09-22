#pragma once
#include <mcl/bn.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace wbtx {
using mcl::Fr;
using mcl::G1;
using mcl::G2;
using mcl::Fp12;
using Digest = std::array<uint8_t, 32>;
using ParallelExecutor = std::function<void(size_t, const std::function<void(size_t)>&)>;
inline ParallelExecutor parallel_executor;
// Set only between operations; the executor must join every submitted operation.
inline void set_parallel_executor(ParallelExecutor executor) { parallel_executor = std::move(executor); }
template<class F> inline void parallel_for(size_t n, F&& fn) {
    if (n == 0) return;
    if (parallel_executor && n > 1) parallel_executor(n, std::function<void(size_t)>(std::forward<F>(fn)));
    else for (size_t i = 0; i < n; ++i) fn(i);
}
inline void require(bool condition, const char* why) { if (!condition) throw std::runtime_error(why); }
inline size_t checked_add(size_t a, size_t b) {
    require(b <= std::numeric_limits<size_t>::max() - a, "size overflow"); return a + b;
}
inline size_t checked_mul(size_t a, size_t b) {
    require(a == 0 || b <= std::numeric_limits<size_t>::max() / a, "size overflow"); return a * b;
}
inline size_t next_power_two(size_t n) {
    require(n > 0, "zero domain size");
    size_t result = 1;
    while (result < n) result = checked_mul(result, 2);
    return result;
}
inline Fr random_scalar(bool nonzero = false) {
    Fr result;
    do { result.setByCSPRNG(); } while (nonzero && result.isZero());
    return result;
}
inline void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) bytes.push_back(uint8_t(value >> (i * 8)));
}
inline void append_text(std::vector<uint8_t>& bytes, const std::string& value) {
    append_u64(bytes, value.size()); bytes.insert(bytes.end(), value.begin(), value.end());
}
inline void append_digest(std::vector<uint8_t>& bytes, const Digest& digest) {
    bytes.insert(bytes.end(), digest.begin(), digest.end());
}
template<class T> inline void append_element(std::vector<uint8_t>& bytes, const T& element) {
    std::array<uint8_t, 1024> encoded{};
    const size_t n = element.serialize(encoded.data(), encoded.size());
    require(n != 0, "canonical serialization failed");
    append_u64(bytes, n);
    bytes.insert(bytes.end(), encoded.begin(), encoded.begin() + n);
}
inline Digest sha256(const std::vector<uint8_t>& bytes) {
    require(bytes.size() <= std::numeric_limits<uint32_t>::max(), "hash input too large");
    Digest digest{};
    require(mcl::fp::sha256(digest.data(), digest.size(), bytes.data(), uint32_t(bytes.size())) == digest.size(),
            "SHA256 failed");
    return digest;
}
inline Digest current_curve_id() {
    std::vector<uint8_t> bytes;
    append_text(bytes, "WEIGHTED-BTX-MCL-CURVE-v1");
    append_text(bytes, mcl::Fp::getModulo());
    append_text(bytes, Fr::getModulo());
    return sha256(bytes);
}
inline Fr challenge_scalar(std::vector<uint8_t> transcript) {
    const size_t prefix = transcript.size();
    for (uint64_t counter = 0;; ++counter) {
        transcript.resize(prefix); append_u64(transcript, counter);
        const Digest digest = sha256(transcript);
        Fr result;
        // Scalar deserialize rejects integers >= r. No reduction bias.
        if (result.deserialize(digest.data(), digest.size()) == digest.size()) return result;
        require(counter != std::numeric_limits<uint64_t>::max(), "challenge rejection limit");
    }
}
inline bool valid_target(const Fp12& point) {
    if (point.isZero()) return false;
    Fp12 order_check;
    Fp12::powGeneric(order_check, point, Fr::getOp().mp);
    return order_check.isOne();
}
template<class G> inline bool valid_point(const G& point) {
    // isValid already performs the subgroup test when MCL's verification flag
    // is enabled. Preserve the explicit check if a caller disabled that flag,
    // without doing the potentially expensive order multiplication twice.
    return point.isValid() && (G::verifyOrder_ || point.isValidOrder());
}

template<class T> struct AdditiveOps {
    static T zero() { T result; result.clear(); return result; }
    static void add(T& out, const T& a, const T& b) { T::add(out, a, b); }
    static void sub(T& out, const T& a, const T& b) { T::sub(out, a, b); }
    static void mul(T& out, const T& a, const Fr& scalar) { T::mul(out, a, scalar); }
    static void many(T* values, const Fr* scalars, size_t n) { T::mulEach(values, scalars, n); }
};
template<> struct AdditiveOps<Fr> {
    static Fr zero() { return Fr(0); }
    static void add(Fr& out, const Fr& a, const Fr& b) { Fr::add(out, a, b); }
    static void sub(Fr& out, const Fr& a, const Fr& b) { Fr::sub(out, a, b); }
    static void mul(Fr& out, const Fr& a, const Fr& scalar) { Fr::mul(out, a, scalar); }
    static void many(Fr* values, const Fr* scalars, size_t n) {
        for (size_t i = 0; i < n; ++i) Fr::mul(values[i], values[i], scalars[i]);
    }
};
template<> struct AdditiveOps<Fp12> {
    static Fp12 zero() { return Fp12(1); }
    static void add(Fp12& out, const Fp12& a, const Fp12& b) { Fp12::mul(out, a, b); }
    static void sub(Fp12& out, const Fp12& a, const Fp12& b) {
        Fp12 inverse; Fp12::unitaryInv(inverse, b); Fp12::mul(out, a, inverse);
    }
    // Every use is on r-torsion GT, after full finalExp (or another GT operation).
    static void mul(Fp12& out, const Fp12& a, const Fr& scalar) { Fp12::pow(out, a, scalar); }
    static void many(Fp12* values, const Fr* scalars, size_t n) {
        parallel_for(n, [&](size_t i) { Fp12::pow(values[i], values[i], scalars[i]); });
    }
};

struct Domain {
    size_t n = 1;
    Fr root = Fr(1), inverse_n = Fr(1);
    std::vector<Fr> powers{Fr(1)}, inverse_powers{Fr(1)};
    Domain() = default;
    explicit Domain(size_t size) : n(size), powers(size), inverse_powers(size) {
        require(n > 0 && (n & (n - 1)) == 0, "invalid radix-two domain");
        require(n <= size_t(std::numeric_limits<int>::max()), "FFT domain too large");
        auto exponent = Fr::getOp().mp - 1;
        require(exponent % int(n) == 0, "curve scalar field lacks requested roots");
        if (n == 1) { powers[0] = inverse_powers[0] = root = inverse_n = 1; return; }
        exponent /= int(n);
        for (int candidate = 2;; ++candidate) {
            Fr::pow(root, Fr(candidate), exponent);
            Fr test; Fr::pow(test, root, int(n / 2));
            if (!test.isOne()) break;
        }
        Fr test; Fr::pow(test, root, int(n));
        require(test.isOne(), "invalid FFT root");
        Fr inverse_root; Fr::inv(inverse_root, root);
        Fr::inv(inverse_n, Fr(int(n)));
        powers[0] = inverse_powers[0] = 1;
        for (size_t i = 1; i < n; ++i) {
            Fr::mul(powers[i], powers[i - 1], root);
            Fr::mul(inverse_powers[i], inverse_powers[i - 1], inverse_root);
        }
    }
    template<class T> void fft(std::vector<T>& values, bool inverse = false, bool normalize = true) const {
        require(values.size() == n, "FFT input length mismatch");
        for (size_t i = 1, j = 0; i < n; ++i) {
            size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(values[i], values[j]);
        }
        using Op = AdditiveOps<T>;
        const auto& ws = inverse ? inverse_powers : powers;
        std::vector<T> products;
        std::vector<Fr> twiddles;
        products.reserve(n / 2); twiddles.reserve(n / 2);
        for (size_t width = 2; width <= n; width *= 2) {
            const size_t half = width / 2, stride = n / width;
            products.clear(); twiddles.clear();
            for (size_t base = 0; base < n; base += width)
                for (size_t j = 1; j < half; ++j) {
                    products.push_back(values[base + half + j]);
                    twiddles.push_back(ws[j * stride]);
                }
            Op::many(products.data(), twiddles.data(), products.size());
            size_t k = 0;
            for (size_t base = 0; base < n; base += width)
                for (size_t j = 0; j < half; ++j) {
                    const T even = values[base + j];
                    const T odd = j == 0 ? values[base + half] : products[k++];
                    Op::add(values[base + j], even, odd);
                    Op::sub(values[base + half + j], even, odd);
                }
            if (width == n) break;
        }
        if (inverse && normalize) {
            twiddles.assign(n, inverse_n);
            Op::many(values.data(), twiddles.data(), n);
        }
    }
};

inline std::vector<Fr> batch_inverse(const std::vector<Fr>& values) {
    std::vector<Fr> prefix(values.size());
    Fr product = 1;
    for (size_t i = 0; i < values.size(); ++i) {
        require(!values[i].isZero(), "zero interpolation denominator");
        prefix[i] = product; product *= values[i];
    }
    Fr::inv(product, product);
    std::vector<Fr> result(values.size());
    for (size_t i = values.size(); i-- > 0;) {
        Fr::mul(result[i], product, prefix[i]); product *= values[i];
    }
    return result;
}
inline std::vector<Fr> polynomial_multiply(const std::vector<Fr>& left, const std::vector<Fr>& right) {
    require(!left.empty() && !right.empty(), "empty polynomial");
    const size_t count = checked_add(left.size(), right.size()) - 1;
    if (std::min(left.size(), right.size()) <= 32) {
        std::vector<Fr> result(count, Fr(0));
        for (size_t i = 0; i < left.size(); ++i)
            for (size_t j = 0; j < right.size(); ++j) result[i + j] += left[i] * right[j];
        return result;
    }
    Domain domain(next_power_two(count));
    std::vector<Fr> a(domain.n, Fr(0)), b(domain.n, Fr(0));
    std::copy(left.begin(), left.end(), a.begin()); std::copy(right.begin(), right.end(), b.begin());
    domain.fft(a); domain.fft(b);
    for (size_t i = 0; i < domain.n; ++i) a[i] *= b[i];
    domain.fft(a, true); a.resize(count); return a;
}
inline std::vector<Fr> lagrange_at_zero(const std::vector<Fr>& points, const std::vector<size_t>& selected) {
    require(!points.empty() && !selected.empty(), "empty interpolation selection");
    std::vector<uint8_t> seen(points.size(), 0);
    for (size_t i : selected) { require(i < points.size() && !seen[i], "invalid interpolation index"); seen[i] = 1; }
    if (selected.size() == points.size() && (points.size() & (points.size() - 1)) == 0) {
        Fr inverse; Fr::inv(inverse, Fr(int(points.size()))); return std::vector<Fr>(selected.size(), inverse);
    }
    std::vector<Fr> numerators(selected.size(), Fr(1)), denominators(selected.size(), Fr(1));
    if (selected.size() <= 64) {
        for (size_t i = 0; i < selected.size(); ++i)
            for (size_t j = 0; j < selected.size(); ++j) if (j != i) {
                numerators[i] *= -points[selected[j]];
                denominators[i] *= points[selected[i]] - points[selected[j]];
            }
    } else {
        std::vector<std::vector<Fr>> level(selected.size());
        for (size_t i = 0; i < selected.size(); ++i) level[i] = {-points[selected[i]], Fr(1)};
        while (level.size() > 1) {
            std::vector<std::vector<Fr>> next((level.size() + 1) / 2);
            parallel_for(next.size(), [&](size_t i) {
                next[i] = 2 * i + 1 == level.size() ? level[2 * i] : polynomial_multiply(level[2 * i], level[2 * i + 1]);
            });
            level = std::move(next);
        }
        const auto& polynomial = level[0];
        Domain domain(next_power_two(points.size()));
        std::vector<Fr> derivative(domain.n, Fr(0));
        for (size_t i = 1; i < polynomial.size(); ++i) derivative[i - 1] = polynomial[i] * Fr(int(i));
        domain.fft(derivative);
        for (size_t i = 0; i < selected.size(); ++i) {
            numerators[i] = -polynomial[0];
            denominators[i] = points[selected[i]] * derivative[selected[i]];
        }
    }
    const auto inverse = batch_inverse(denominators);
    for (size_t i = 0; i < numerators.size(); ++i) numerators[i] *= inverse[i];
    return numerators;
}
template<class G> inline G msm(const G* points, const Fr* scalars, size_t n) {
    G result;
    if (n == 0) { result.clear(); return result; }
    // MCL accepts mutable points and can normalize them in place. Never mutate
    // shared material or validated batches from concurrent readers.
    std::vector<G> work(points, points + n);
    G::mulVec(result, work.data(), scalars, n);
    return result;
}
} // namespace wbtx
