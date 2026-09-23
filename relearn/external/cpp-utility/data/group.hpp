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

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

namespace utility {

/**
 * @brief Sorts the values and splits them into groups of consecutive elements.
 *      Example: [12, 3, 9, 4, 5, 13, 7, 15] yields [[3, 4, 5], [7], [9], [12, 13], [15]].
 *      The predicate is evaluated for adjacent elements of the sorted values,
 *      so duplicates only stay in the same group if the predicate treats equal elements as consecutive
 * @tparam T The type of the elements. Must be totally ordered
 * @tparam Predicate The type of the predicate
 * @param values The values that should be grouped
 * @param is_successor_or_equal Predicate that takes two elements and returns
 *      whether the second one is the direct successor of the first one or equal to it
 * @return The sorted groups of consecutive elements
 */
template <typename T, typename Predicate>
    requires std::totally_ordered<T> && std::predicate<Predicate, const T&, const T&>
[[nodiscard]] std::vector<std::vector<T>> group_consecutive_elements(std::vector<T> values, Predicate is_successor_or_equal) {
    std::ranges::sort(values);

    auto result = std::vector<std::vector<T>>{};

    if (values.empty()) {
        return result;
    }

    auto current_group = std::vector<T>{};
    current_group.push_back(values[0]);

    for (auto i = std::size_t{ 1 }; i < values.size(); ++i) {
        if (!is_successor_or_equal(values[i - 1], values[i])) {
            result.push_back(std::move(current_group));
            current_group.clear();
        }
        current_group.push_back(values[i]);
    }

    result.push_back(std::move(current_group));
    return result;
}

/**
 * @brief Sorts the values and splits them into groups of consecutive integers,
 *      i.e., adjacent elements that are equal or differ by exactly 1 end up in the same group.
 *      Example: [12, 3, 9, 4, 5, 13, 7, 15] yields [[3, 4, 5], [7], [9], [12, 13], [15]]
 * @tparam T The type of the elements. Must be integral
 * @param values The values that should be grouped
 * @return The sorted groups of consecutive integers
 */
template <std::integral T>
[[nodiscard]] std::vector<std::vector<T>> group_consecutive_elements(std::vector<T> values) {
    // the equality is checked first, so lhs + 1 is never evaluated for the maximum value
    // (the values are sorted, hence rhs is never smaller than lhs)
    const auto is_successor_or_equal = [](const T& lhs, const T& rhs) {
        return rhs == lhs || rhs == static_cast<T>(lhs + T{ 1 });
    };

    return group_consecutive_elements(std::move(values), is_successor_or_equal);
}

} // namespace utility
