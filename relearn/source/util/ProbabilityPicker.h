#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "util/Random.h"
#include "util/RelearnException.h"

#include "cpp-utility/ranges/Functional.hpp"

#include <range/v3/algorithm/all_of.hpp>
#include <range/v3/numeric/accumulate.hpp>

#include <cmath>
#include <span>

/**
 * This class provides the possibility to pick an element from a vector based on their probabilities.
 * This is useful with, e.g., picking nodes to connect to or synapses to delete.
 */
class ProbabilityPicker {
public:
    /**
     * @brief Given some probabilities and a random number, returns the index in the span such that the sum of
     *      the probabilities before the element is smaller than the random number and the same sum plus the picked
     *      element is larger or equal to the random number.
     *      If the random number is larger than the sum of probabilities, returns the last index with probability > 0.0.
     * @param probabilities The probabilities. Not negative, not empty, sum must be > 0.0
     * @param random_number The random number, not negative
     * @exception Throws a RelearnException if probabilities is empty, some are negative, all were 0.0, or random_number < 0.0
     * @return The picked index, the probability was > 0.0
     */
    [[nodiscard]] static std::size_t pick_target(const std::span<const double> probabilities, const double random_number) {
        RelearnException::check(!probabilities.empty(), "ProbabilityPicker::pick_target: There were no probabilities to pick from");
        RelearnException::check(random_number >= 0.0, "ProbabilityPicker::pick_target: random_number was smaller than 0.0");
        RelearnException::check(ranges::all_of(probabilities, utility::greater_equal(0.0)), "ProbabilityPicker::pick_target: Some probability was negative");

        if (!(0.0 < random_number)) {
            // This exists for denormalized numbers
            return 0;
        }

        auto counter = std::size_t{ 0 };
        auto sum_probabilities = 0.0;

        for (; counter < probabilities.size() && sum_probabilities < random_number; counter++) {
            sum_probabilities += probabilities[counter];
        }

        RelearnException::check(sum_probabilities > 0.0, "ProbabilityPicker::pick_target: The sum of probabilities was <= 0.0");

        while (probabilities[counter - 1U] <= 0.0) {
            // Ignore all probabilities that are <= 0.0
            counter--;
        }

        return counter - 1U;
    }

    /**
     * @brief Given some probabilities, picks one element based on its probability. Uses the PRNG associated with the key
     * @param probabilities The probabilities. Not negative, not empty, sum must be > 0.0
     * @param key The identifier or the PRNG
     * @exception Throws a RelearnException if probabilities is empty, some are negative, or the total sum of probabilities is negative or 0.0
     * @return The picked index, the probability was > 0.0
     */
    [[nodiscard]] static std::size_t pick_target(const std::span<const double> probabilities, const RandomHolderKey key) {
        RelearnException::check(!probabilities.empty(), "ProbabilityPicker::pick_target: There were no probabilities to pick from");

        const auto total_probability = ranges::accumulate(probabilities, 0.0);
        RelearnException::check(total_probability > 0.0, "ProbabilityPicker::pick_target: total_probability was smaller than or equal to 0.0");

        const auto next = std::nextafter(total_probability, total_probability + Constants::eps);
        const auto random_number = RandomHolder::get_random_uniform_double(key, 0.0, next);

        return pick_target(probabilities, random_number);
    }
};
