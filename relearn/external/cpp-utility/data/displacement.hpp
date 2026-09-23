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
#include "cpp-utility/data/prefix_sum.hpp"

#include <concepts>
#include <span>
#include <vector>

namespace utility {

/**
 * @brief Calculates the necessary displacements for a bunch of sizes such that the elements are tightly packed.
 *      For sizes = [a, b, c, ..., y, z] returns [0, a, a+b, ..., a+b+...+y], i.e., one displacement per size.
 *      Use calculate_offsets() to also get the offset behind the last part
 * @tparam count_type The type for the displacements/sizes
 * @param sizes The non-negative sizes, not empty
 * @exception Throws an Exception if sizes is empty, a size is negative, or a displacement overflows count_type
 * @return The displacements for the sizes
 */
template <std::integral count_type>
[[nodiscard]] std::vector<count_type> calculate_displacements(const std::span<const count_type> sizes) {
    Exception::check(!sizes.empty(), "Util::calculate_displacements: sizes must not be empty");

    if constexpr (std::signed_integral<count_type>) {
        for (const auto size : sizes) {
            Exception::check(size >= count_type{ 0 }, "Util::calculate_displacements: sizes must not be negative, found {}", size);
        }
    }

    return calculate_prefix_sum<count_type>(sizes);
}

/**
 * @brief Calculates the offsets of tightly packed parts with the given sizes, including the offset behind the last part.
 *      For sizes = [a, b, c, ..., y, z] returns [0, a, a+b, ..., a+b+...+y, a+b+...+z], i.e., one offset more than there
 *      are sizes. Part i occupies [offsets[i], offsets[i + 1]) and the last offset is the total number of elements.
 *      This is the shape create_partition() returns, and it degenerates to [0] for no sizes at all
 * @tparam count_type The type for the offsets/sizes
 * @param sizes The non-negative sizes, may be empty
 * @exception Throws an Exception if a size is negative or an offset overflows count_type
 * @return The sizes.size() + 1 offsets for the sizes
 */
template <std::integral count_type>
[[nodiscard]] std::vector<count_type> calculate_offsets(const std::span<const count_type> sizes) {
    if constexpr (std::signed_integral<count_type>) {
        for (const auto size : sizes) {
            Exception::check(size >= count_type{ 0 }, "Util::calculate_offsets: sizes must not be negative, found {}", size);
        }
    }

    // the exclusive prefix sum drops the total; a trailing size of zero turns the total into its last entry
    auto padded_sizes = std::vector<count_type>{};
    padded_sizes.reserve(sizes.size() + 1);
    padded_sizes.insert(padded_sizes.end(), sizes.begin(), sizes.end());
    padded_sizes.push_back(count_type{ 0 });

    return calculate_prefix_sum<count_type>(padded_sizes);
}

} // namespace utility
