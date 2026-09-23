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

#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace utility {

/**
 * @brief Calculates the exclusive prefix sum of all values
 * @tparam Integral Must be an integral type
 * @param values The values
 * @exception Throws an Exception if a prefix cannot be represented by Integral
 * @return The prefix sum, i.e., {0, values[0], values[0] + values[1], ..., values[0] + ... + values[values.size() - 2]}
 */
template <std::integral Integral>
[[nodiscard]] std::vector<Integral> calculate_prefix_sum(const std::span<const Integral> values) {
    if (values.empty()) {
        return {};
    }

    auto prefix_sum = std::vector<Integral>(values.size(), Integral{ 0 });
    for (auto index = std::size_t{ 1 }; index < values.size(); ++index) {
        const auto previous = prefix_sum[index - 1];
        const auto value = values[index - 1];

        if constexpr (std::unsigned_integral<Integral>) {
            Exception::check(previous <= std::numeric_limits<Integral>::max() - value,
                             "calculate_prefix_sum: overflow while adding value {} at index {}", value, index - 1);
        } else if (value > Integral{ 0 }) {
            Exception::check(previous <= std::numeric_limits<Integral>::max() - value,
                             "calculate_prefix_sum: overflow while adding value {} at index {}", value, index - 1);
        } else if (value < Integral{ 0 }) {
            Exception::check(previous >= std::numeric_limits<Integral>::min() - value,
                             "calculate_prefix_sum: underflow while adding value {} at index {}", value, index - 1);
        }

        prefix_sum[index] = static_cast<Integral>(previous + value);
    }

    return prefix_sum;
}

} // namespace utility
