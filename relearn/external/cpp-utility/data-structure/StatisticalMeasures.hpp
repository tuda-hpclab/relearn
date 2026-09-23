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

#include "cpp-utility/Exception.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <ostream>
#include <ranges>
#include <type_traits>

namespace utility {

/**
 * @brief This struct is used to aggregate different statistical parameters.
 *      var and std describe the population variance and standard deviation,
 *      i.e., they are normalized by the number of values
 */
struct StatisticalMeasures {
    double min{ 0.0 };
    double max{ 0.0 };
    double avg{ 0.0 };
    double var{ 0.0 };
    double std{ 0.0 };

    /**
     * @brief Calculates the statistical measures of the values in a single pass.
     *      Uses Welford's algorithm, so the variance is numerically stable
     * @tparam RangeType The type of the range with the values. The values must be arithmetic
     * @param values The finite values, must not be empty
     * @exception Throws an Exception if values is empty or contains a value that is not finite as a double
     * @return The statistical measures of the values
     */
    template <std::ranges::input_range RangeType>
        requires std::is_arithmetic_v<std::ranges::range_value_t<RangeType>>
    [[nodiscard]] static StatisticalMeasures calculate(const RangeType& values) {
        auto count = std::size_t{ 0 };
        auto mean = 0.0;
        auto sum_squared_distance = 0.0;

        auto minimum = std::numeric_limits<double>::infinity();
        auto maximum = -std::numeric_limits<double>::infinity();

        for (const auto& value : values) {
            const auto value_as_double = static_cast<double>(value);
            Exception::check(std::isfinite(value_as_double),
                             "StatisticalMeasures::calculate: values must be finite, found {}", value_as_double);

            count++;
            const auto delta = value_as_double - mean;
            mean += delta / static_cast<double>(count);
            const auto delta_after_update = value_as_double - mean;
            sum_squared_distance += delta * delta_after_update;

            minimum = std::min(minimum, value_as_double);
            maximum = std::max(maximum, value_as_double);
        }

        Exception::check(count > 0, "StatisticalMeasures::calculate: values must not be empty");

        // Roundoff can make a mathematically zero variance minimally negative.
        const auto variance = std::max(0.0, sum_squared_distance / static_cast<double>(count));
        const auto standard_deviation = std::sqrt(variance);

        return StatisticalMeasures{ minimum, maximum, mean, variance, standard_deviation };
    }

    /**
     * @brief Checks if two statistical measures are equal (with actually equal parameters!)
     * @return True iff all parameters are equal
     */
    [[nodiscard]] friend constexpr bool operator==(const StatisticalMeasures&, const StatisticalMeasures&) noexcept = default;

    /**
     * @brief Checks if two statistical measures are equal up to an absolute tolerance,
     *      i.e., whether no parameter differs by more than epsilon
     * @param other The other statistical measures
     * @param epsilon The non-negative maximum allowed absolute difference per parameter
     * @return True iff the two statistical measures are equal up to the tolerance
     */
    [[nodiscard]] bool almost_equal(const StatisticalMeasures& other, const double epsilon) const noexcept {
        return epsilon >= 0.0
               && std::abs(min - other.min) <= epsilon
               && std::abs(max - other.max) <= epsilon
               && std::abs(avg - other.avg) <= epsilon
               && std::abs(var - other.var) <= epsilon
               && std::abs(std - other.std) <= epsilon;
    }

    /**
     * @brief Prints the statistical measures to the ostream in the format [min: ..., max: ..., avg: ..., var: ..., std: ...]
     * @param output_stream The stream to which the object should be printed
     * @param measures The statistical measures that should be printed
     * @return A reference to output_stream that allows chaining.
     *      Is not marked as [[nodiscard]] as that typically does happen when chaining <<
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const StatisticalMeasures& measures) {
        return output_stream << "[min: " << measures.min << ", max: " << measures.max << ", avg: " << measures.avg
                             << ", var: " << measures.var << ", std: " << measures.std << ']';
    }
};

} // namespace utility

/**
 * @brief Formats StatisticalMeasures via fmt in the format [min: ..., max: ..., avg: ..., var: ..., std: ...].
 *      The format specification is applied to each parameter, e.g., "{:.2f}" formats
 *      every parameter with two decimal places
 */
template <>
struct fmt::formatter<utility::StatisticalMeasures> : fmt::formatter<double> {
    /**
     * @brief Formats the statistical measures into the output of the format context
     * @param measures The statistical measures that should be formatted
     * @param ctx The format context that provides the output iterator
     * @return The output iterator past the formatted statistical measures
     */
    auto format(const utility::StatisticalMeasures& measures, fmt::format_context& ctx) const {
        auto out = fmt::format_to(ctx.out(), "[min: ");
        ctx.advance_to(out);
        out = fmt::formatter<double>::format(measures.min, ctx);

        out = fmt::format_to(out, ", max: ");
        ctx.advance_to(out);
        out = fmt::formatter<double>::format(measures.max, ctx);

        out = fmt::format_to(out, ", avg: ");
        ctx.advance_to(out);
        out = fmt::formatter<double>::format(measures.avg, ctx);

        out = fmt::format_to(out, ", var: ");
        ctx.advance_to(out);
        out = fmt::formatter<double>::format(measures.var, ctx);

        out = fmt::format_to(out, ", std: ");
        ctx.advance_to(out);
        out = fmt::formatter<double>::format(measures.std, ctx);

        *out++ = ']';
        return out;
    }
};
