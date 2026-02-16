#pragma once

/*
 * This file is part of the CPP-Utility software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <concepts>
#include <numeric>
#include <span>
#include <vector>

namespace utility {

/**
 * @brief Calculates the prefix sum of all values
 * @tparam Integral Must be an integral type
 * @param values The values
 * @return The prefix sum, i.e., {0, values[0], values[0] + values[1], ..., values[0] + ... + values[values.size() - 2]}
 */
template <std::integral Integral>
[[nodiscard]] std::vector<Integral> calculate_prefix_sum(const std::span<const Integral> values) {
    if (values.empty()) {
        return {};
    }

    auto prefix_sum = std::vector<Integral>(values.size(), Integral{ 0 });
    std::inclusive_scan(values.begin(), values.end() - 1, prefix_sum.begin() + 1, std::plus{}, Integral{ 0 });

    return prefix_sum;
}

} // namespace utility
