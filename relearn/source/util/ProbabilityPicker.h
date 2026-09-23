#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "util/Random.h"
#include "util/RelearnException.h"

#include <cpp-utility/ranges/Functional.hpp>

#include <range/v3/algorithm/all_of.hpp>
#include <range/v3/numeric/accumulate.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>

/**
 * This class provides the possibility to pick an element from a vector based on their probabilities.
 * This is useful with, e.g., picking nodes to connect to or synapses to delete.
 * The methods are generic in the value type, so they work with every floating point type,
 * i.e., with RelearnTypes::attraction_type as well as with a plain double.
 */
class ProbabilityPicker {
public:
    /**
     * @brief Given some probabilities and a random number, returns the index in the range such that the sum of
     *      the probabilities before the element is smaller than the random number and the same sum plus the picked
     *      element is larger or equal to the random number.
     *      If the random number is larger than the sum of probabilities, returns the last index with probability > 0.0.
     * @tparam Range A contiguous range of a floating point type, e.g., std::vector<double> or std::span<const attraction_type>
     * @param values The probabilities. Not negative, not empty, sum must be > 0.0
     * @param random_number The random number, not negative
     * @exception Throws a RelearnException if values is empty, some are negative, all were 0.0, or random_number < 0.0
     * @return The picked index, the probability was > 0.0
     */
    template <std::ranges::contiguous_range Range>
        requires std::floating_point<std::ranges::range_value_t<Range>>
    [[nodiscard]] static std::size_t pick_target(const Range& values, const std::ranges::range_value_t<Range> random_number) {
        using value_type = std::ranges::range_value_t<Range>;
        const auto probabilities = std::span<const value_type>{ values };

        RelearnException::check(!probabilities.empty(), "ProbabilityPicker::pick_target: There were no probabilities to pick from");
        RelearnException::check(random_number >= value_type{ 0.0 }, "ProbabilityPicker::pick_target: random_number was smaller than 0.0");
        RelearnException::check(ranges::all_of(probabilities, utility::greater_equal(value_type{ 0.0 })), "ProbabilityPicker::pick_target: Some probability was negative");

        if (!(value_type{ 0.0 } < random_number)) {
            // random_number is (denormalized-)zero: the loop below never enters its body (since
            // sum_probabilities starts at 0.0, which is never < a non-positive random_number), so
            // it cannot be used here. Pick the first entry with strictly positive probability
            // instead of blindly returning 0 -- index 0 can itself have probability 0 (e.g. a
            // self-excluded autapse candidate), so returning 0 unconditionally could pick a
            // zero-probability target.
            for (std::size_t i = 0; i < probabilities.size(); i++) {
                if (probabilities[i] > value_type{ 0.0 }) {
                    return i;
                }
            }

            RelearnException::fail("ProbabilityPicker::pick_target: All probabilities were <= 0.0");
        }

        auto counter = std::size_t{ 0 };
        auto sum_probabilities = value_type{ 0.0 };

        for (; counter < probabilities.size() && sum_probabilities < random_number; counter++) {
            sum_probabilities += probabilities[counter];
        }

        RelearnException::check(sum_probabilities > value_type{ 0.0 }, "ProbabilityPicker::pick_target: The sum of probabilities was <= 0.0");

        while (probabilities[counter - 1U] <= value_type{ 0.0 }) {
            // Ignore all probabilities that are <= 0.0
            counter--;
        }

        return counter - 1U;
    }

    /**
     * @brief Given some probabilities, picks one element based on its probability. Uses the PRNG associated with the key
     * @tparam Range A contiguous range of a floating point type, e.g., std::vector<double> or std::span<const attraction_type>
     * @param values The probabilities. Not negative, not empty, sum must be > 0.0
     * @param key The identifier or the PRNG
     * @exception Throws a RelearnException if values is empty, some are negative, or the total sum of probabilities is negative or 0.0
     * @return The picked index, the probability was > 0.0
     */
    template <std::ranges::contiguous_range Range>
        requires std::floating_point<std::ranges::range_value_t<Range>>
    [[nodiscard]] static std::size_t pick_target(const Range& values, const RandomHolderKey key) {
        using value_type = std::ranges::range_value_t<Range>;
        const auto probabilities = std::span<const value_type>{ values };

        RelearnException::check(!probabilities.empty(), "ProbabilityPicker::pick_target: There were no probabilities to pick from");

        const auto total_probability = ranges::accumulate(probabilities, value_type{ 0.0 });
        RelearnException::check(total_probability > value_type{ 0.0 }, "ProbabilityPicker::pick_target: total_probability was smaller than or equal to 0.0");

        const auto next = std::nextafter(total_probability, total_probability + static_cast<value_type>(Constants::eps));
        const auto random_number = static_cast<value_type>(RandomHolder::get_random_uniform_double(key, value_type{ 0 }, next));

        return pick_target(probabilities, random_number);
    }
};
