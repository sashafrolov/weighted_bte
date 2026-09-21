#pragma once
#include "protocol_math.hpp"
#include <numeric>

// Experimental, in-memory weighted BTX implementation. Initialize MCL for
// BLS12_381 or BN_SNARK1 before constructing any objects. Curve initialization
// and SIMD/executor changes must not run concurrently with protocol operations.
// There is deliberately no untrusted wire decoder in this implementation.
namespace wbtx {

template<class CipherGroup, class KeyGroup> struct Protocol {
    using C = CipherGroup;
    using K = KeyGroup;
    static_assert((std::is_same_v<C, G1> && std::is_same_v<K, G2>) ||
                  (std::is_same_v<C, G2> && std::is_same_v<K, G1>), "pairing groups must be opposite");
    static constexpr bool swapped = std::is_same_v<C, G2>;

    struct Material {
        size_t max_batch = 0, threshold = 0, total_weight = 0, domain_size = 0;
        std::vector<size_t> weights, offsets;
        std::vector<Fr> domain, secrets;
        std::vector<K> verification, negative, positive;
        C cipher_generator;
        K key_generator;
        Fp12 target_generator, encryption_element;
        Digest setup_id{}, curve_id{};
    };
    struct Proof { C commitment; Fr response; };
    struct Ciphertext { C first; Fp12 second; Proof proof; };
    struct ValidatedBatch {
        Digest setup_id{}, digest{}, curve_id{};
        size_t batch_size = 0;
        std::vector<uint8_t> valid;
        std::vector<C> first;
    };
    struct Share {
        size_t party = 0;
        C sigma;
        Digest setup_id{}, batch_digest{};
    };
    struct Accepted {
        Digest setup_id{}, digest{}, curve_id{};
        size_t batch_size = 0, accepted_weight = 0;
        std::vector<Share> shares;
        std::vector<size_t> parties, rejected;
    };
    struct Committee {
        Digest setup_id{}, context_id{}, curve_id{};
        size_t batch_size = 0, accepted_weight = 0;
        std::vector<size_t> parties, selected;
        std::vector<Fr> coefficients;
        std::vector<K> opening_keys, kernel_fft;
        Domain domain;
        std::vector<std::vector<mcl::Fp6>> prepared_kernel;
    };
    struct CrossTerms {
        Digest context_id{}, digest{}, curve_id{};
        size_t batch_size = 0;
        std::vector<Fp12> beta;
    };

    static void ensure_curve(const Digest& id) { require(id == current_curve_id(), "MCL curve context changed"); }
    static void check_material(const Material& material) {
        ensure_curve(material.curve_id);
        require(material.max_batch > 0 && !material.weights.empty(), "invalid material dimensions");
        require(material.offsets.size() == material.weights.size() + 1 &&
                material.secrets.size() == material.weights.size() &&
                material.domain.size() == material.total_weight, "invalid material shape");
    }
    static Fp12 pairing(const C& ciphertext, const K& key) {
        Fp12 result;
        if constexpr (swapped) mcl::pairing(result, key, ciphertext);
        else mcl::pairing(result, ciphertext, key);
        return result;
    }
    static Fp12 multi_pairing(const C* ciphertext, const K* keys, size_t n) {
        if (n == 0) return Fp12(1);
        Fp12 result;
        if constexpr (swapped) mcl::millerLoopVec(result, keys, ciphertext, n);
        else mcl::millerLoopVec(result, ciphertext, keys, n);
        mcl::finalExp(result, result);
        return result;
    }
    static std::vector<K> generator_multiples(const K& generator, const std::vector<Fr>& scalars) {
        std::vector<K> points(scalars.size());
        parallel_for(points.size(), [&](size_t i) { K::mul(points[i], generator, scalars[i]); });
        K::normalizeVec(points.data(), points.data(), points.size());
        return points;
    }
    static Material keygen(size_t max_batch, const std::vector<size_t>& weights, size_t threshold_weight) {
        require(max_batch > 0 && !weights.empty(), "empty setup dimensions");
        Material material;
        material.curve_id = current_curve_id();
        material.max_batch = max_batch; material.threshold = threshold_weight; material.weights = weights;
        material.offsets.push_back(0);
        for (size_t weight : weights) {
            require(weight > 0, "validator weight must be positive");
            material.total_weight = checked_add(material.total_weight, weight);
            material.offsets.push_back(material.total_weight);
        }
        require(threshold_weight < material.total_weight, "threshold must be below total weight");
        material.domain_size = next_power_two(material.total_weight);
        require(material.total_weight <= size_t(std::numeric_limits<int>::max()), "too many virtual shares");
        const size_t parties = weights.size(), total = material.total_weight;
        const size_t verification_count = checked_mul(max_batch, parties);
        const size_t negative_count = checked_mul(max_batch, total);
        const size_t positive_count = checked_mul(max_batch - 1, total);
        require(negative_count <= std::vector<K>().max_size() && positive_count <= std::vector<K>().max_size(), "setup allocation too large");
        G1 generator1; G2 generator2;
        const std::string generator_domain = "WEIGHTED-BTX-MCL-GENERATOR-v1";
        mcl::hashAndMapToG1(generator1, generator_domain.data(), generator_domain.size());
        mcl::hashAndMapToG2(generator2, generator_domain.data(), generator_domain.size());
        if constexpr (swapped) { material.cipher_generator = generator2; material.key_generator = generator1; }
        else { material.cipher_generator = generator1; material.key_generator = generator2; }
        material.cipher_generator.normalize(); material.key_generator.normalize();
        material.target_generator = pairing(material.cipher_generator, material.key_generator);
        Domain evaluation_domain(material.domain_size);
        std::vector<Fr> polynomial(material.domain_size, Fr(0));
        for (size_t i = 0; i <= threshold_weight; ++i) polynomial[i] = random_scalar();
        const Fr secret = polynomial[0];
        evaluation_domain.fft(polynomial);
        polynomial.resize(total);
        material.domain.assign(evaluation_domain.powers.begin(), evaluation_domain.powers.begin() + total);
        material.secrets.resize(parties);
        std::vector<Fr> positive_q(verification_count), negative_q(verification_count);
        std::vector<size_t> owner(total);
        for (size_t party = 0; party < parties; ++party) {
            material.secrets[party] = random_scalar(true);
            Fr inverse; Fr::inv(inverse, material.secrets[party]);
            Fr p = material.secrets[party], q = inverse;
            for (size_t power = 0; power < max_batch; ++power) {
                positive_q[power * parties + party] = p; negative_q[power * parties + party] = q;
                p *= material.secrets[party]; q *= inverse;
            }
            for (size_t i = material.offsets[party]; i < material.offsets[party + 1]; ++i) owner[i] = party;
        }
        material.verification = generator_multiples(material.key_generator, positive_q);
        material.negative.resize(negative_count); material.positive.resize(positive_count);
        parallel_for(negative_count, [&](size_t index) {
            const size_t power = index / total, virtual_index = index % total;
            const Fr scalar = polynomial[virtual_index] * negative_q[power * parties + owner[virtual_index]];
            K::mul(material.negative[index], material.key_generator, scalar);
        });
        parallel_for(positive_count, [&](size_t index) {
            const size_t power = index / total, virtual_index = index % total;
            const Fr scalar = polynomial[virtual_index] * positive_q[power * parties + owner[virtual_index]];
            K::mul(material.positive[index], material.key_generator, scalar);
        });
        K::normalizeVec(material.negative.data(), material.negative.data(), material.negative.size());
        if (!material.positive.empty()) K::normalizeVec(material.positive.data(), material.positive.data(), material.positive.size());
        Fp12::pow(material.encryption_element, material.target_generator, secret);
        std::vector<uint8_t> transcript;
        append_text(transcript, "WEIGHTED-BTX-MCL-SETUP-v1");
        append_digest(transcript, material.curve_id); append_u64(transcript, swapped);
        append_u64(transcript, max_batch); append_u64(transcript, threshold_weight); append_u64(transcript, parties);
        for (size_t weight : weights) append_u64(transcript, weight);
        append_element(transcript, material.cipher_generator); append_element(transcript, material.key_generator);
        append_element(transcript, material.encryption_element);
        for (const auto& point : material.verification) append_element(transcript, point);
        for (const auto& point : material.negative) append_element(transcript, point);
        for (const auto& point : material.positive) append_element(transcript, point);
        material.setup_id = sha256(transcript);
        return material;
    }
    static Fr proof_challenge(const Material& material, const C& first, const Fp12& second, const C& commitment) {
        std::vector<uint8_t> transcript;
        append_text(transcript, "WEIGHTED-BTX-MCL-SCHNORR-v1");
        append_digest(transcript, material.setup_id); append_digest(transcript, material.curve_id);
        append_u64(transcript, swapped);
        append_element(transcript, material.cipher_generator); append_element(transcript, first);
        append_element(transcript, second); append_element(transcript, commitment);
        return challenge_scalar(std::move(transcript));
    }
    static Proof prove(const Material& material, const Fr& randomness, const C& first, const Fp12& second) {
        check_material(material);
        Proof proof;
        const Fr nonce = random_scalar();
        C::mul(proof.commitment, material.cipher_generator, nonce);
        proof.response = nonce + proof_challenge(material, first, second, proof.commitment) * randomness;
        return proof;
    }
    static Ciphertext encrypt_with_randomness(const Material& material, const Fp12& message, const Fr& randomness) {
        check_material(material);
        require(valid_target(message), "message must belong to GT");
        Ciphertext ciphertext;
        C::mul(ciphertext.first, material.cipher_generator, randomness);
        ciphertext.first.normalize();
        Fp12::pow(ciphertext.second, material.encryption_element, randomness);
        ciphertext.second *= message;
        ciphertext.proof = prove(material, randomness, ciphertext.first, ciphertext.second);
        return ciphertext;
    }
    static Ciphertext encrypt(const Material& material, const Fp12& message) {
        return encrypt_with_randomness(material, message, random_scalar());
    }
    static bool verify_ciphertext(const Material& material, const Ciphertext& ciphertext) {
        check_material(material);
        if (!valid_point(ciphertext.first) || !valid_point(ciphertext.proof.commitment) || !valid_target(ciphertext.second)) return false;
        const Fr challenge = proof_challenge(material, ciphertext.first, ciphertext.second, ciphertext.proof.commitment);
        C left, right, challenged;
        C::mul(left, material.cipher_generator, ciphertext.proof.response);
        C::mul(challenged, ciphertext.first, challenge);
        C::add(right, ciphertext.proof.commitment, challenged);
        return left == right;
    }
    static Digest ciphertext_digest(const std::vector<Ciphertext>& ciphertexts) {
        std::vector<uint8_t> transcript;
        append_text(transcript, "WEIGHTED-BTX-MCL-ORDERED-BATCH-v1"); append_u64(transcript, ciphertexts.size());
        for (const auto& ciphertext : ciphertexts) {
            append_element(transcript, ciphertext.first); append_element(transcript, ciphertext.second);
            append_element(transcript, ciphertext.proof.commitment); append_element(transcript, ciphertext.proof.response);
        }
        return sha256(transcript);
    }
    static void check_batch(const Material& material, const ValidatedBatch& batch) {
        check_material(material); ensure_curve(batch.curve_id);
        require(batch.setup_id == material.setup_id, "batch belongs to another setup");
        require(batch.batch_size > 0 && batch.batch_size <= material.max_batch, "invalid batch size");
        require(batch.first.size() == batch.batch_size && batch.valid.size() == batch.batch_size, "invalid batch shape");
    }
    static ValidatedBatch validate(const Material& material, const std::vector<Ciphertext>& ciphertexts) {
        check_material(material);
        require(!ciphertexts.empty() && ciphertexts.size() <= material.max_batch, "invalid ciphertext batch size");
        ValidatedBatch batch;
        batch.setup_id = material.setup_id; batch.curve_id = material.curve_id;
        batch.digest = ciphertext_digest(ciphertexts); batch.batch_size = ciphertexts.size();
        batch.valid.resize(batch.batch_size); batch.first.resize(batch.batch_size);
        parallel_for(batch.batch_size, [&](size_t i) {
            batch.valid[i] = verify_ciphertext(material, ciphertexts[i]);
            if (batch.valid[i]) batch.first[i] = ciphertexts[i].first;
            else batch.first[i].clear();
        });
        C::normalizeVec(batch.first.data(), batch.first.data(), batch.first.size());
        return batch;
    }
    static Share partial(const Material& material, size_t party, const ValidatedBatch& batch) {
        check_batch(material, batch);
        require(party < material.weights.size(), "invalid validator index");
        std::vector<Fr> powers(batch.batch_size);
        Fr power = material.secrets[party];
        for (auto& scalar : powers) { scalar = power; power *= material.secrets[party]; }
        Share share;
        share.party = party; share.setup_id = material.setup_id; share.batch_digest = batch.digest;
        share.sigma = msm(batch.first.data(), powers.data(), batch.batch_size);
        return share;
    }
    static bool verify_share(const Material& material, const ValidatedBatch& batch, const Share& share) {
        check_batch(material, batch);
        require(share.party < material.weights.size(), "invalid validator index");
        if (share.setup_id != material.setup_id || share.batch_digest != batch.digest || !valid_point(share.sigma)) return false;
        std::vector<C> ciphertext;
        std::vector<K> keys;
        for (size_t slot = 0; slot < batch.batch_size; ++slot) if (batch.valid[slot]) {
            ciphertext.push_back(batch.first[slot]);
            keys.push_back(material.verification[slot * material.weights.size() + share.party]);
        }
        C negative; C::neg(negative, share.sigma); ciphertext.push_back(negative); keys.push_back(material.key_generator);
        C::normalizeVec(ciphertext.data(), ciphertext.data(), ciphertext.size());
        return multi_pairing(ciphertext.data(), keys.data(), keys.size()).isOne();
    }
    static bool verify_shares_batched(const Material& material, const ValidatedBatch& batch, const std::vector<Share>& shares) {
        if (shares.empty()) return true;
        std::vector<Fr> challenges(shares.size());
        std::vector<C> share_points(shares.size());
        for (size_t i = 0; i < shares.size(); ++i) {
            challenges[i] = random_scalar(true); share_points[i] = shares[i].sigma;
        }
        C aggregate = msm(share_points.data(), challenges.data(), shares.size());
        std::vector<size_t> valid_slots;
        for (size_t slot = 0; slot < batch.batch_size; ++slot) if (batch.valid[slot]) valid_slots.push_back(slot);
        std::vector<K> keys(valid_slots.size() + 1);
        parallel_for(valid_slots.size(), [&](size_t index) {
            std::vector<K> selected(shares.size());
            for (size_t j = 0; j < shares.size(); ++j)
                selected[j] = material.verification[valid_slots[index] * material.weights.size() + shares[j].party];
            keys[index] = msm(selected.data(), challenges.data(), shares.size());
        });
        keys.back() = material.key_generator;
        K::normalizeVec(keys.data(), keys.data(), keys.size());
        std::vector<C> ciphertext;
        for (size_t slot : valid_slots) ciphertext.push_back(batch.first[slot]);
        C::neg(aggregate, aggregate); aggregate.normalize(); ciphertext.push_back(aggregate);
        return multi_pairing(ciphertext.data(), keys.data(), keys.size()).isOne();
    }
    static Accepted accept(const Material& material, const ValidatedBatch& batch, const std::vector<Share>& shares) {
        check_batch(material, batch);
        require(shares.size() <= material.weights.size(), "too many submitted shares");
        Accepted accepted;
        accepted.setup_id = material.setup_id; accepted.curve_id = material.curve_id;
        accepted.digest = batch.digest; accepted.batch_size = batch.batch_size;
        std::vector<uint8_t> seen(material.weights.size(), 0);
        std::vector<Share> candidates;
        for (const auto& share : shares) {
            require(share.party < material.weights.size(), "invalid validator index");
            require(!seen[share.party], "duplicate validator share"); seen[share.party] = 1;
            if (share.setup_id == material.setup_id && share.batch_digest == batch.digest && valid_point(share.sigma)) candidates.push_back(share);
            else accepted.rejected.push_back(share.party);
        }
        if (verify_shares_batched(material, batch, candidates)) accepted.shares = std::move(candidates);
        else {
            std::vector<uint8_t> valid(candidates.size());
            parallel_for(candidates.size(), [&](size_t i) { valid[i] = verify_share(material, batch, candidates[i]); });
            for (size_t i = 0; i < candidates.size(); ++i) {
                if (valid[i]) accepted.shares.push_back(candidates[i]);
                else accepted.rejected.push_back(candidates[i].party);
            }
        }
        std::sort(accepted.shares.begin(), accepted.shares.end(), [](const Share& a, const Share& b) { return a.party < b.party; });
        std::sort(accepted.rejected.begin(), accepted.rejected.end());
        for (const auto& share : accepted.shares) {
            accepted.parties.push_back(share.party);
            accepted.accepted_weight = checked_add(accepted.accepted_weight, material.weights[share.party]);
        }
        require(accepted.accepted_weight > material.threshold, "insufficient accepted weight");
        return accepted;
    }
    static Digest context_id(const Digest& setup, size_t batch_size, const std::vector<size_t>& parties) {
        std::vector<uint8_t> transcript;
        append_text(transcript, "WEIGHTED-BTX-MCL-COMMITTEE-CONTEXT-v1");
        append_digest(transcript, setup); append_u64(transcript, batch_size); append_u64(transcript, parties.size());
        for (size_t party : parties) append_u64(transcript, party);
        return sha256(transcript);
    }
    static void check_accepted(const Material& material, const Accepted& accepted) {
        check_material(material); ensure_curve(accepted.curve_id);
        require(accepted.setup_id == material.setup_id, "accepted shares belong to another setup");
        require(accepted.batch_size > 0 && accepted.batch_size <= material.max_batch, "invalid accepted batch size");
        require(!accepted.parties.empty() && accepted.parties.size() == accepted.shares.size(), "invalid accepted shape");
        size_t weight = 0;
        for (size_t i = 0; i < accepted.parties.size(); ++i) {
            const size_t party = accepted.parties[i];
            require(party < material.weights.size() && (i == 0 || accepted.parties[i - 1] < party), "invalid accepted committee");
            require(accepted.shares[i].party == party && accepted.shares[i].setup_id == accepted.setup_id &&
                    accepted.shares[i].batch_digest == accepted.digest, "invalid accepted share context");
            weight = checked_add(weight, material.weights[party]);
        }
        require(weight == accepted.accepted_weight && weight > material.threshold, "insufficient accepted weight");
    }
    static Committee prepare(const Material& material, const Accepted& accepted) {
        check_accepted(material, accepted);
        Committee committee;
        committee.setup_id = material.setup_id; committee.curve_id = material.curve_id;
        committee.batch_size = accepted.batch_size; committee.accepted_weight = accepted.accepted_weight;
        committee.parties = accepted.parties;
        committee.context_id = context_id(material.setup_id, accepted.batch_size, accepted.parties);
        std::vector<size_t> coefficient_offsets{0};
        for (size_t party : accepted.parties) {
            for (size_t i = material.offsets[party]; i < material.offsets[party + 1]; ++i) committee.selected.push_back(i);
            coefficient_offsets.push_back(committee.selected.size());
        }
        committee.coefficients = lagrange_at_zero(material.domain, committee.selected);
        const size_t parties = accepted.parties.size(), batch_size = accepted.batch_size;
        committee.opening_keys.resize(checked_mul(batch_size, parties));
        parallel_for(committee.opening_keys.size(), [&](size_t flat) {
            const size_t slot = flat / parties, position = flat % parties, party = accepted.parties[position];
            committee.opening_keys[flat] = msm(material.negative.data() + slot * material.total_weight + material.offsets[party],
                                               committee.coefficients.data() + coefficient_offsets[position], material.weights[party]);
        });
        committee.domain = Domain(next_power_two(checked_mul(batch_size, 2)));
        committee.kernel_fft.assign(committee.domain.n, AdditiveOps<K>::zero());
        for (size_t distance = 1; distance < batch_size; ++distance) {
            K sum; sum.clear();
            for (size_t j = 0; j < parties; ++j) K::add(sum, sum, committee.opening_keys[(distance - 1) * parties + j]);
            committee.kernel_fft[distance] = sum;
        }
        parallel_for(batch_size - 1, [&](size_t index) {
            const size_t distance = index + 1;
            std::vector<K> selected(committee.selected.size());
            for (size_t j = 0; j < selected.size(); ++j)
                selected[j] = material.positive[(distance - 1) * material.total_weight + committee.selected[j]];
            committee.kernel_fft[committee.domain.n - distance] = msm(selected.data(), committee.coefficients.data(), selected.size());
        });
        committee.domain.fft(committee.kernel_fft);
        // Absorb inverse-transform normalization in the reusable source-group
        // kernel, avoiding n GT exponentiations for every subsequent batch.
        std::vector<Fr> inverse_sizes(committee.domain.n, committee.domain.inverse_n);
        K::mulEach(committee.kernel_fft.data(), inverse_sizes.data(), committee.kernel_fft.size());
        K::normalizeVec(committee.kernel_fft.data(), committee.kernel_fft.data(), committee.kernel_fft.size());
        K::normalizeVec(committee.opening_keys.data(), committee.opening_keys.data(), committee.opening_keys.size());
        if constexpr (!swapped) {
            committee.prepared_kernel.resize(committee.domain.n);
            parallel_for(committee.domain.n, [&](size_t i) { mcl::precomputeG2(committee.prepared_kernel[i], committee.kernel_fft[i]); });
        }
        return committee;
    }
    static void check_committee_batch(const Committee& committee, const ValidatedBatch& batch) {
        ensure_curve(committee.curve_id); ensure_curve(batch.curve_id);
        require(committee.setup_id == batch.setup_id, "committee/batch setup mismatch");
        require(committee.batch_size == batch.batch_size && batch.batch_size > 0, "committee/batch size mismatch");
        require(batch.first.size() == batch.batch_size && batch.valid.size() == batch.batch_size, "invalid validated batch shape");
        require(committee.context_id == context_id(committee.setup_id, committee.batch_size, committee.parties), "invalid committee context");
        require(committee.kernel_fft.size() == committee.domain.n && committee.domain.n >= 2 * batch.batch_size &&
                committee.opening_keys.size() == checked_mul(batch.batch_size, committee.parties.size()), "invalid committee material shape");
    }
    static CrossTerms precompute(const Committee& committee, const ValidatedBatch& batch) {
        check_committee_batch(committee, batch);
        std::vector<C> transformed(committee.domain.n, AdditiveOps<C>::zero());
        std::copy(batch.first.begin(), batch.first.end(), transformed.begin());
        committee.domain.fft(transformed);
        C::normalizeVec(transformed.data(), transformed.data(), transformed.size());
        std::vector<Fp12> convolution(committee.domain.n);
        if constexpr (!swapped) require(committee.prepared_kernel.size() == committee.domain.n, "missing prepared kernel");
        parallel_for(committee.domain.n, [&](size_t i) {
            if constexpr (swapped) mcl::millerLoop(convolution[i], committee.kernel_fft[i], transformed[i]);
            else mcl::precomputedMillerLoop(convolution[i], transformed[i], committee.prepared_kernel[i]);
            // Full curve-specific MCL projection first: subsequent powers are
            // legitimately GT powers for either supported curve. This avoids
            // importing BLS-specific cyclotomic decomposition into BN_SNARK1.
            mcl::finalExp(convolution[i], convolution[i]);
        });
        committee.domain.fft(convolution, true, false);
        CrossTerms cross;
        cross.context_id = committee.context_id; cross.curve_id = committee.curve_id;
        cross.digest = batch.digest; cross.batch_size = batch.batch_size;
        cross.beta.assign(convolution.begin(), convolution.begin() + batch.batch_size);
        for (size_t i = 0; i < batch.batch_size; ++i) if (!batch.valid[i]) cross.beta[i] = 1;
        return cross;
    }
    static std::vector<std::optional<Fp12>> open(const Committee& committee, const Accepted& accepted,
                                                const ValidatedBatch& batch, const std::vector<Ciphertext>& ciphertexts,
                                                const CrossTerms& cross) {
        check_committee_batch(committee, batch);
        ensure_curve(accepted.curve_id); ensure_curve(cross.curve_id);
        require(accepted.setup_id == committee.setup_id && accepted.parties == committee.parties, "opening committee mismatch");
        require(accepted.batch_size == batch.batch_size && cross.batch_size == batch.batch_size &&
                ciphertexts.size() == batch.batch_size, "opening batch size mismatch");
        require(accepted.digest == batch.digest && cross.digest == batch.digest &&
                ciphertext_digest(ciphertexts) == batch.digest, "opening batch digest mismatch");
        require(cross.context_id == committee.context_id && cross.beta.size() == batch.batch_size, "cross-term context mismatch");
        require(accepted.shares.size() == committee.parties.size() &&
                accepted.accepted_weight == committee.accepted_weight, "opening accepted weight mismatch");
        const size_t parties = accepted.shares.size();
        std::vector<C> shares(parties);
        for (size_t i = 0; i < parties; ++i) {
            require(accepted.shares[i].party == committee.parties[i] && accepted.shares[i].setup_id == batch.setup_id &&
                    accepted.shares[i].batch_digest == batch.digest, "opening accepted share context mismatch");
            shares[i] = accepted.shares[i].sigma;
        }
        C::normalizeVec(shares.data(), shares.data(), shares.size());
        std::vector<std::optional<Fp12>> messages(batch.batch_size);
        parallel_for(batch.batch_size, [&](size_t slot) {
            if (!batch.valid[slot]) return;
            Fp12 alpha = multi_pairing(shares.data(), committee.opening_keys.data() + slot * parties, parties);
            Fp12::unitaryInv(alpha, alpha);
            Fp12 message = ciphertexts[slot].second * alpha;
            message *= cross.beta[slot];
            messages[slot] = message;
        });
        return messages;
    }
};
using Normal = Protocol<G1, G2>;
using Swapped = Protocol<G2, G1>;
} // namespace wbtx
