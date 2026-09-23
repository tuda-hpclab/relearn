#pragma once

/*
 * This file is part of the CPP-Utility software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cpp-utility/Concepts.hpp"
#include "cpp-utility/Exception.hpp"

#include <range/v3/action/shuffle.hpp>
#include <range/v3/algorithm/shuffle.hpp>
#include <range/v3/range/access.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/range/traits.hpp>

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace utility {

/**
 * @brief Abstract source of uniformly distributed random numbers from [0, 1).
 *
 * Randomized algorithms draw their random numbers through this interface, so callers
 * (especially tests) can inject their own deterministic generator.
 */
class RandomNumberGenerator {
public:
    RandomNumberGenerator() = default;
    virtual ~RandomNumberGenerator() = default;

    RandomNumberGenerator(const RandomNumberGenerator&) = delete;
    RandomNumberGenerator& operator=(const RandomNumberGenerator&) = delete;
    RandomNumberGenerator(RandomNumberGenerator&&) = delete;
    RandomNumberGenerator& operator=(RandomNumberGenerator&&) = delete;

    /**
     * @brief Draws the next random number.
     * @return A uniformly distributed random number from [0, 1)
     */
    [[nodiscard]] virtual double draw() = 0;
};

/**
 * @brief The generator used whenever the caller does not supply their own: an Engine behind a
 *      uniform real distribution on [0, 1).
 * @tparam Engine The underlying uniform random bit generator (defaults to std::mt19937)
 */
template <typename Engine = std::mt19937>
class DefaultRandomNumberGenerator : public RandomNumberGenerator {
public:
    /**
     * @brief Constructs the generator.
     * @param seed The seed for the underlying engine
     */
    explicit DefaultRandomNumberGenerator(const typename Engine::result_type seed)
        : prng_(seed) {
    }

    [[nodiscard]] double draw() override {
        return distribution_(prng_);
    }

private:
    Engine prng_;
    std::uniform_real_distribution<double> distribution_{ 0.0, 1.0 };
};

namespace detail {

/**
 * @brief The integer type used to parameterize std::uniform_int_distribution for a given IntegerType.
 *
 * std::uniform_int_distribution is only specified for short/int/long/long long and their unsigned
 * counterparts, so narrow integer types (e.g. std::int8_t, char) are widened to int/unsigned int.
 * Wider types are used as-is.
 */
template <std::integral T>
using uniform_int_engine_type = std::conditional_t<
    (sizeof(T) < sizeof(short)),
    std::conditional_t<std::is_signed_v<T>, int, unsigned int>,
    T>;

} // namespace detail

/**
 * @brief A static, thread-safe interface for generating random numbers keyed by a caller-defined enum.
 *
 * Each enum value and each thread has its own random number generator: generators are stored per
 * thread (thread_local) and created lazily on first use, so draws never contend and each thread owns
 * an independent stream. The key enum is supplied by the consuming project; the library does not
 * define it. Consequently seed(key, seed) affects only the calling thread's generator for that key;
 * every thread that wants a reproducible stream must seed itself.
 *
 * @tparam KeyType The consumer-supplied enum type identifying a generator
 * @tparam Engine The underlying uniform random bit generator (defaults to std::mt19937)
 */
template <detail::Enum KeyType, typename Engine = std::mt19937>
class RandomHolder {
public:
    /**
     * @brief Generates a random integer, uniformly distributed in [lower_inclusive, upper_inclusive].
     *      Uses the generator associated with the key.
     * @tparam IntegerType The integer type of the bounds and the result (bool is not supported)
     * @param key The key whose generator shall be used
     * @param lower_inclusive The lower inclusive bound for the random integer
     * @param upper_inclusive The upper inclusive bound for the random integer
     * @throws Exception if lower_inclusive > upper_inclusive
     * @return A uniformly distributed integer in [lower_inclusive, upper_inclusive]
     */
    template <std::integral IntegerType>
        requires(!std::same_as<std::remove_cv_t<IntegerType>, bool>)
    [[nodiscard]] static IntegerType get_random_uniform_integer(const KeyType key, const IntegerType lower_inclusive, const IntegerType upper_inclusive) {
        Exception::check(lower_inclusive <= upper_inclusive,
            "utility::RandomHolder::get_random_uniform_integer: Random number from invalid interval [{}, {}]", lower_inclusive, upper_inclusive);

        using distribution_type = detail::uniform_int_engine_type<IntegerType>;
        auto distribution = std::uniform_int_distribution<distribution_type>(
            static_cast<distribution_type>(lower_inclusive), static_cast<distribution_type>(upper_inclusive));
        return static_cast<IntegerType>(distribution(get_generator(key)));
    }

    /**
     * @brief Returns the desired number of distinct indices from a total number of elements, drawn uniformly.
     * @param key The key whose generator shall be used
     * @param number_indices The number of indices, must be <= number_elements
     * @param number_elements The total number of elements
     * @throws Exception if number_indices > number_elements
     * @return The vector of distinct indices in [0, number_elements) in no particular order
     */
    [[nodiscard]] static std::vector<std::size_t> get_random_uniform_indices(const KeyType key, const std::size_t number_indices, const std::size_t number_elements) {
        Exception::check(number_indices <= number_elements,
            "utility::RandomHolder::get_random_uniform_indices: Cannot get more indices ({}) than elements ({})", number_indices, number_elements);

        auto indices = std::vector<std::size_t>(number_elements);
        std::iota(indices.begin(), indices.end(), std::size_t{ 0 });
        ranges::shuffle(indices, get_generator(key));
        indices.resize(number_indices);

        return indices;
    }

    /**
     * @brief Generates a random float, normally distributed with the specified mean and standard deviation.
     *      Uses the generator associated with the key.
     * @tparam FloatType The floating point type used for mean, stddev, and the result
     * @param key The key whose generator shall be used
     * @param mean The mean of the normal distribution
     * @param stddev The standard deviation of the normal distribution, zero yields exactly @p mean
     * @throws Exception if stddev < 0
     * @return A normally distributed float with the specified mean and standard deviation
     */
    template <std::floating_point FloatType>
    [[nodiscard]] static FloatType get_random_normal_double(const KeyType key, const FloatType mean, const FloatType stddev) {
        Exception::check(FloatType{ 0 } <= stddev,
            "utility::RandomHolder::get_random_normal_double: Random number with invalid standard deviation {}", stddev);

        // A zero standard deviation describes a degenerate distribution whose only value is the mean.
        // std::normal_distribution requires a positive one, so it is never constructed with zero.
        if (stddev == FloatType{ 0 }) {
            return mean;
        }

        auto distribution = std::normal_distribution<FloatType>(mean, stddev);
        return distribution(get_generator(key));
    }

    /**
     * @brief Generates a random double, uniformly distributed in [lower_inclusive, upper_exclusive).
     *      Uses the generator associated with the key.
     * @tparam FloatType The floating point type used for the bounds and the result
     * @param key The key whose generator shall be used
     * @param lower_inclusive The lower inclusive bound for the random float
     * @param upper_exclusive The upper exclusive bound for the random float, must be finite
     * @throws Exception if lower_inclusive >= upper_exclusive or if upper_exclusive is not finite
     * @return A uniformly distributed float in [lower_inclusive, upper_exclusive)
     */
    template <std::floating_point FloatType>
    [[nodiscard]] static FloatType get_random_uniform_double(const KeyType key, const FloatType lower_inclusive, const FloatType upper_exclusive) {
        Exception::check(lower_inclusive < upper_exclusive,
            "utility::RandomHolder::get_random_uniform_double: Random number from invalid interval [{}, {})", lower_inclusive, upper_exclusive);
        Exception::check(upper_exclusive <= std::numeric_limits<FloatType>::max(),
            "utility::RandomHolder::get_random_uniform_double: upper_exclusive was not finite");

        auto distribution = std::uniform_real_distribution<FloatType>(lower_inclusive, upper_exclusive);
        return distribution(get_generator(key));
    }

    /**
     * @brief Fills all values in [begin, end) with uniformly distributed floats from [lower_inclusive, upper_exclusive).
     *      Uses the generator associated with the key.
     * @tparam Iterator The random access iterator type; its value type must be floating point
     * @param key The key whose generator shall be used
     * @param begin The iterator that marks the inclusive begin
     * @param end The iterator that marks the exclusive end
     * @param lower_inclusive The lower inclusive bound for the random floats
     * @param upper_exclusive The upper exclusive bound for the random floats
     * @throws Exception if lower_inclusive >= upper_exclusive
     */
    template <std::random_access_iterator Iterator>
        requires std::floating_point<std::iter_value_t<Iterator>>
    static void fill(const KeyType key, const Iterator begin, const Iterator end, const std::iter_value_t<Iterator> lower_inclusive, const std::iter_value_t<Iterator> upper_exclusive) {
        using value_type = std::iter_value_t<Iterator>;
        Exception::check(lower_inclusive < upper_exclusive,
            "utility::RandomHolder::fill: Random number from invalid interval [{}, {})", lower_inclusive, upper_exclusive);

        auto distribution = std::uniform_real_distribution<value_type>(lower_inclusive, upper_exclusive);
        auto& generator = get_generator(key);
        for (auto iterator = begin; iterator != end; ++iterator) {
            *iterator = distribution(generator);
        }
    }

    /**
     * @brief Fills all values in the range with uniformly distributed floats from [lower_inclusive, upper_exclusive).
     *      Uses the generator associated with the key.
     * @tparam Range The random access range type; its value type must be floating point
     * @param key The key whose generator shall be used
     * @param range The range to fill
     * @param lower_inclusive The lower inclusive bound for the random floats
     * @param upper_exclusive The upper exclusive bound for the random floats
     * @throws Exception if lower_inclusive >= upper_exclusive
     */
    template <ranges::random_access_range Range>
        requires std::floating_point<ranges::range_value_t<Range>>
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    static void fill(const KeyType key, Range&& range, const ranges::range_value_t<Range> lower_inclusive, const ranges::range_value_t<Range> upper_exclusive) {
        fill(key, ranges::begin(range), ranges::end(range), lower_inclusive, upper_exclusive);
    }

    /**
     * @brief Shuffles all values in [begin, end) such that all permutations have equal probability.
     *      Uses the generator associated with the key.
     * @tparam Iterator The random access iterator type
     * @param key The key whose generator shall be used
     * @param begin The iterator that marks the inclusive begin
     * @param end The iterator that marks the exclusive end
     */
    template <std::random_access_iterator Iterator>
    static void shuffle(const KeyType key, const Iterator begin, const Iterator end) {
        ranges::shuffle(begin, end, get_generator(key));
    }

    /**
     * @brief Shuffles all values in the range such that all permutations have equal probability.
     *      Uses the generator associated with the key.
     * @tparam Range The random access range type
     * @param key The key whose generator shall be used
     * @param range The range to shuffle
     */
    template <ranges::random_access_range Range>
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    static void shuffle(const KeyType key, Range&& range) {
        ranges::shuffle(range, get_generator(key));
    }

    /**
     * @brief Returns a range-v3 action closure that shuffles a range using the key's generator.
     * @param key The key whose generator shall be used
     * @return A shuffle action closure
     */
    [[nodiscard]] static auto shuffle_action(const KeyType key) {
        return ranges::actions::shuffle(get_generator(key));
    }

    /**
     * @brief Seeds the generator associated with the key on the calling thread.
     * @param key The key whose generator shall be seeded
     * @param seed The seed that should be used
     */
    static void seed(const KeyType key, const std::size_t seed) {
        get_generator(key).seed(static_cast<typename Engine::result_type>(seed));
    }

    /**
     * @brief Seeds the generators associated with the given keys on the calling thread.
     * @param keys The keys whose generators shall be seeded
     * @param seed The seed that should be used
     */
    static void seed_all(const std::initializer_list<KeyType> keys, const std::size_t seed) {
        for (const auto key : keys) {
            RandomHolder::seed(key, seed);
        }
    }

private:
    RandomHolder() = default;

    [[nodiscard]] static Engine& get_generator(const KeyType key) {
        return generators_[key];
    }

    static inline thread_local std::unordered_map<KeyType, Engine> generators_{};
};

} // namespace utility
