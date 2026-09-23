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

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace utility {

namespace detail {

/** A function that maps a value of the range to the integral key of the part the value belongs to. */
template <typename KeyFunction, typename Range>
concept partition_key_function
    = std::invocable<KeyFunction, std::ranges::range_reference_t<Range>>
      && std::integral<std::remove_cvref_t<std::invoke_result_t<KeyFunction, std::ranges::range_reference_t<Range>>>>;

} // namespace detail

/**
 * @brief Calculates where each part of a range that is sorted by the keys of its values begins, i.e., turns those
 *      keys into offsets into the range. The values of key k occupy [partition[k], partition[k + 1]), so the result
 *      holds one offset more than there are keys and its last entry is the number of values.
 *      A key without values gets an empty part instead of being skipped, which keeps keys and offsets aligned.
 *      Example: the keys [0, 0, 2, 2, 2] of five values and number_keys = 4 yield [0, 2, 2, 5, 5],
 *      i.e., key 0 owns the values [0, 2), key 1 none, key 2 the values [2, 5), and key 3 none again
 * @tparam Range The type of the range with the values. Must be sorted by the keys of its values
 * @tparam KeyFunction The type of the key function
 * @tparam NumberKeys The integral type of the number of keys
 * @param values The values, sorted by their keys
 * @param get_key Function that returns the integral key of the part a value belongs to
 * @param number_keys The number of keys, i.e., the keys are expected in [0, number_keys)
 * @exception Throws an Exception if @p number_keys is negative or exceeds the maximum size of the result,
 *      if the values are not sorted by their keys, or if a key is negative or not smaller than @p number_keys
 * @return The number_keys + 1 offsets of the parts
 */
template <std::ranges::input_range Range, typename KeyFunction, std::integral NumberKeys>
    requires detail::partition_key_function<KeyFunction, Range>
[[nodiscard]] std::vector<std::size_t> create_partition(Range&& values, KeyFunction get_key, const NumberKeys number_keys) {
    const auto number_parts = safe_cast<std::size_t>(number_keys);

    auto partition = std::vector<std::size_t>{};

    Exception::check(number_parts < partition.max_size(), "create_partition: the number of keys ({}) exceeds the maximum size of the partition", number_parts);
    partition.reserve(number_parts + 1);

    auto index = std::size_t{ 0 };
    auto previous_key = std::size_t{ 0 };

    for (auto&& value : std::forward<Range>(values)) {
        // a negative key is not representable as an offset and is rejected by the cast
        const auto key = safe_cast<std::size_t>(std::invoke(get_key, std::forward<decltype(value)>(value)));

        Exception::check(key >= previous_key, "create_partition: the values must be sorted by their keys, found key {} after key {}", key, previous_key);
        Exception::check(key < number_parts, "create_partition: the key {} is not smaller than the number of keys {}", key, number_parts);

        // partition.size() is the next key that still needs an offset: the current key starts here,
        // and so do the keys in between, which have no values of their own
        while (partition.size() <= key) {
            partition.push_back(index);
        }

        previous_key = key;
        ++index;
    }

    // the keys behind the last value are empty, their parts start and end behind the last value
    while (partition.size() <= number_parts) {
        partition.push_back(index);
    }

    return partition;
}

/**
 * @brief Calculates where each part of a sorted range of keys begins, i.e., uses the values themselves as their keys.
 *      See create_partition(Range&&, KeyFunction, NumberKeys) for the shape of the result
 * @tparam Range The type of the range with the keys. Must be sorted
 * @tparam NumberKeys The integral type of the number of keys
 * @param keys The keys, sorted
 * @param number_keys The number of keys, i.e., the keys are expected in [0, number_keys)
 * @exception Throws an Exception if @p number_keys is negative or exceeds the maximum size of the result,
 *      if the keys are not sorted, or if a key is negative or not smaller than @p number_keys
 * @return The number_keys + 1 offsets of the parts
 */
template <std::ranges::input_range Range, std::integral NumberKeys>
    requires std::integral<std::remove_cvref_t<std::ranges::range_reference_t<Range>>>
[[nodiscard]] std::vector<std::size_t> create_partition(Range&& keys, const NumberKeys number_keys) {
    return create_partition(std::forward<Range>(keys), std::identity{}, number_keys);
}

} // namespace utility
