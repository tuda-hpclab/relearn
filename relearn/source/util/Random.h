#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"

#include "util/OMPHelper.h"
#include "util/RelearnException.h"
#include "util/shuffle/shuffle.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#include <boost/container_hash/hash.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#pragma GCC diagnostic pop

#include <range/v3/algorithm/generate.hpp>
#include <range/v3/iterator/concepts.hpp>
#include <range/v3/range_fwd.hpp>
#include <range/v3/view/subrange.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <random>
#include <type_traits>
#include <vector>

template <typename T>
using uniform_int_distribution = boost::random::uniform_int_distribution<T>;
template <typename T>
using uniform_real_distribution = boost::random::uniform_real_distribution<T>;
template <typename T>
using normal_distribution = boost::random::normal_distribution<T>;

using mt19937 = std::mt19937;
using ranlux = std::ranlux24_base;
using lincong = std::minstd_rand;

/**
 * This enum allows a type safe differentiation between the types that require access to random numbers.
 */

#include "RandomHolderKey.h"

constexpr std::size_t NUMBER_RANDOM_HOLDER_KEYS = 10;

enum class RNGType : std::uint8_t {
    Mersenne,
    Fast,
};

/**
 * This type provides a static thread-safe interface for generating random numbers.
 * Each instance of RandomHolderKey and each thread in an OMP parallel region has its own random number generator.
 */
class RandomHolder {
public:
    /**
     * @brief Generates a random integer (uniformly distributed in [lower_inclusive, upper_inclusive]).
     *      Uses the RNG that is associated with the key.
     * @param key The type whose RNG shall be used
     * @param lower_inclusive The lower inclusive bound for the random integer
     * @param upper_inclusive The upper inclusive bound for the random integer
     * @exception Throws a RelearnException if lower_inclusive > upper_inclusive
     * @return A uniformly integer double in [lower_inclusive, upper_inclusive]
     */
    template <typename integer_type>
    static integer_type get_random_uniform_integer(const RandomHolderKey key, const integer_type lower_inclusive, const integer_type upper_inclusive) {
        RelearnException::check(lower_inclusive <= upper_inclusive,
                                "RandomHolder::get_random_uniform_integer: Random number from invalid interval [{}, {}] for key {}", lower_inclusive, upper_inclusive, static_cast<unsigned int>(key));

        const auto uid = uniform_int_distribution<integer_type>(lower_inclusive, upper_inclusive);
        auto& generator = get_generator(key);
        return uid(generator);
    }

    /**
     * @brief Returns the desired number of indices from a total number of elements, drawn uniformly.
     * @param key The type whose RNG shall be used
     * @param number_indices The number of indices, must be <= number_elements
     * @param number_elements The total number of elements
     * @exception Throws a RelearnException if number_indices > number_elements
     * @return The vector of indices in no particular order
     */
    static std::vector<std::size_t> get_random_uniform_indices(const RandomHolderKey key, const std::size_t number_indices, const std::size_t number_elements) {
        RelearnException::check(number_indices <= number_elements, "RandomHolder::get_uniform_indices: Cannot get more indices than elements");

        auto drawn_indices = std::vector<std::size_t>{};
        drawn_indices.reserve(number_indices);

        for (auto i = std::size_t{ 0 }; i < number_indices; i++) {
            auto random_number = get_random_uniform_integer(key, std::size_t{ 0 }, number_elements - 1);
            while (std::ranges::find(drawn_indices, random_number) != drawn_indices.end()) {
                random_number = get_random_uniform_integer(key, std::size_t{ 0 }, number_elements - 1);
            }

            drawn_indices.emplace_back(random_number);
        }

        return drawn_indices;
    }

    /**
     * @brief Generates a random floating point value (normally distributed in with specified mean and standard deviation).
     *      Uses the RNG that is associated with the key.
     * @param key The type whose RNG shall be used
     * @param mean The mean of the normal distribution
     * @param stddev The standard deviation of the normal distribution
     * @tparam floating_point_type The floating point type the value is drawn in and returned as
     * @exception Throws a RelearnException if stddev <= 0.0
     * @return A normally distributed value with specified mean and standard deviation
     */
    template <std::floating_point floating_point_type>
    static floating_point_type get_random_normal_double(const RandomHolderKey key, const floating_point_type mean, const floating_point_type stddev) {
        RelearnException::check(floating_point_type{ 0 } <= stddev, "RandomHolder::get_random_normal_double: Random number with invalid standard deviation {} for key {}", stddev, static_cast<int>(key));

        auto nd = normal_distribution<floating_point_type>(mean, stddev);
        auto& generator = get_generator(key);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
        const auto value = nd(generator);
#pragma GCC diagnostic pop
        return value;
    }

    /**
     * @brief Generates a random floating point value (uniformly distributed in [lower_inclusive, upper_exclusive)).
     *      Uses the RNG that is associated with the key.
     * @param key The type whose RNG shall be used
     * @param lower_inclusive The lower inclusive bound for the random value
     * @param upper_exclusive The upper exclusive bound for the random value, not inf
     * @tparam floating_point_type The floating point type the value is drawn in and returned as
     * @exception Throws a RelearnException if lower_inclusive >= upper_exclusive or if upper_exclusive is inf
     * @return A uniformly distributed value in [lower_inclusive, upper_exclusive)
     */
    template <std::floating_point floating_point_type>
    static floating_point_type get_random_uniform_double(const RandomHolderKey key, const floating_point_type lower_inclusive, const floating_point_type upper_exclusive) {
        RelearnException::check(lower_inclusive < upper_exclusive,
                                "RandomHolder::get_random_uniform_double: Random number from invalid interval [{}, {}) for key {}", lower_inclusive, upper_exclusive, static_cast<unsigned int>(key));
        RelearnException::check(upper_exclusive <= std::numeric_limits<floating_point_type>::max(), "RandomHolder::get_random_uniform_double: upper_exclusive was inf");

        const auto dist = uniform_real_distribution<floating_point_type>(lower_inclusive, upper_exclusive);
        auto& generator = get_generator(key);
        return dist(generator);
    }

    /**
     * @brief Fills all values in [begin, end) with uniformly distributed values from [lower_inclusive, upper_exclusive).
     *      Uses the RNG that is associated with the key. There should be a natural number n st. begin + n = end.
     * @param key The type whose RNG shall be used
     * @param begin The iterator that marks the inclusive begin
     * @param end the iterator that marks the exclusive end
     * @param lower_inclusive The lower inclusive bound for the random values
     * @param upper_exclusive The upper exclusive bound for the random values
     * @tparam IteratorType The iterator type that is used to iterate the elements.
     * @tparam floating_point_type The floating point type the values are drawn in
     * @exception Throws a RelearnException if lower_inclusive >= upper_exclusive.
     */
    template <typename IteratorType, std::floating_point floating_point_type>
        requires ranges::output_iterator<IteratorType, floating_point_type>
    static void fill(const RandomHolderKey key, const IteratorType begin, const IteratorType end, const floating_point_type lower_inclusive, const floating_point_type upper_exclusive) {
        RelearnException::check(lower_inclusive < upper_exclusive, "RandomHolder::fill: Random number from invalid interval [{}, {}) for key {}", lower_inclusive, upper_exclusive, static_cast<unsigned int>(key));

        auto urd = uniform_real_distribution<floating_point_type>(lower_inclusive, upper_exclusive);
        auto& generator = get_generator(key);

        ranges::generate(ranges::subrange{ begin, end }, [&generator, &urd]() { return urd(generator); });
    }

    /**
     * @brief Fills all values in range with uniformly distributed values from [lower_inclusive, upper_exclusive).
     *      Uses the RNG that is associated with the key. There should be a natural number n st. begin + n = end.
     * @param key The type whose RNG shall be used
     * @param range The range to fill
     * @param lower_inclusive The lower inclusive bound for the random values
     * @param upper_exclusive The upper exclusive bound for the random values
     * @tparam RangeType The range type that is used
     * @tparam floating_point_type The floating point type the values are drawn in
     * @exception Throws a RelearnException if lower_inclusive >= upper_exclusive.
     */
    template <typename RangeType, std::floating_point floating_point_type>
        requires ranges::output_range<RangeType, floating_point_type>
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    static void fill(const RandomHolderKey key, RangeType&& range, const floating_point_type lower_inclusive, const floating_point_type upper_exclusive) {
        fill(key, ranges::begin(range), ranges::end(range), lower_inclusive, upper_exclusive);
    }

    /**
     * @brief Shuffles all values in [begin, end) such that all permutations have equal probability.
     *      Uses the RNG that is associated with the key. There should be a natural number n st. begin + n = end.
     * @param key The type which's RNG shall be used
     * @param begin The iterator that marks the inclusive begin
     * @param end the iterator that marks the exclusive end
     * @tparam IteratorType The iterator type that is used to iterate the elements.
     */
    template <typename IteratorType>
    static void shuffle(const RandomHolderKey key, const IteratorType begin, const IteratorType end) {
        auto& generator = get_generator(key);
        ::shuffle(begin, end, generator);
    }

    /**
     * @brief Shuffles all values in [begin, end) such that all permutations have equal probability.
     *      Uses the RNG that is associated with the key. There should be a natural number n st. begin + n = end.
     * @param key The type which's RNG shall be used
     * @param range The range to shuffle
     * @tparam Range The range type
     */
    template <typename Range>
    static void shuffle(const RandomHolderKey key, Range& range) {
        ::shuffle(range, get_generator(key));
    }

    /**
     * @brief Returns an action closure to shuffle elements
     * @param key The type which's RNG shall be used
     * @return A shuffle action closure
     */
    [[nodiscard]] static auto shuffleAction(const RandomHolderKey key) {
        return actions::shuffle(get_generator(key));
    }

    /**
     * @brief Seeds the random number generators associated with the key.
     *      The seed used is boost::hash_combine(seed, omp_get_thread_num()).
     * @param key The type whose RNG shall be seeded
     * @param seed The base seed that should be used
     */
    static void seed(const RandomHolderKey key, const std::size_t seed) {
        // NOLINTNEXTLINE
#pragma omp parallel shared(key, seed)
        {
            const auto thread_id = omp_get_thread_num();
            auto& generator = get_generator(key);

            auto current_seed = seed;
            boost::hash_combine(current_seed, thread_id);
            generator.seed(static_cast<unsigned int>(current_seed));
        }
    }

    /**
     * @brief Seeds all number generators.
     *      The seed used is boost::hash_combine(seed, omp_get_thread_num()).
     * @param seed The base seed that should be used
     */
    static void seed_all(const std::size_t seed) {
        RandomHolder::seed(RandomHolderKey::Algorithm, seed);
        RandomHolder::seed(RandomHolderKey::Partition, seed);
        RandomHolder::seed(RandomHolderKey::Subdomain, seed);
        RandomHolder::seed(RandomHolderKey::PoissonModel, seed);
        RandomHolder::seed(RandomHolderKey::SynapseDeletion, seed);
        RandomHolder::seed(RandomHolderKey::SynapticElements, seed);
        RandomHolder::seed(RandomHolderKey::NeuronsExtraInformation, seed);
        RandomHolder::seed(RandomHolderKey::Connector, seed);
        RandomHolder::seed(RandomHolderKey::BackgroundActivity, seed);
    }

private:
    RandomHolder() = default;

    static mt19937& get_generator(const RandomHolderKey key) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        return random_number_generators[static_cast<std::underlying_type_t<RandomHolderKey>>(key)];
    }

    thread_local static inline std::array<mt19937, NUMBER_RANDOM_HOLDER_KEYS> random_number_generators{};
};
