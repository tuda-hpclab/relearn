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
#include <array>
#include <cstddef>
#include <utility>

namespace utility {

namespace details {

/** Returns the largest value in a non-empty index sequence. */
template <std::size_t... Indices>
    requires(sizeof...(Indices) > 0)
[[nodiscard]] constexpr std::size_t max_index(std::index_sequence<Indices...>) noexcept {
    return (std::max)({ Indices... });
}

/** Returns the smallest value in a non-empty index sequence. */
template <std::size_t... Indices>
    requires(sizeof...(Indices) > 0)
[[nodiscard]] constexpr std::size_t min_index(std::index_sequence<Indices...>) noexcept {
    return (std::min)({ Indices... });
}

/** Returns the value at compile-time position I in an index sequence. */
template <std::size_t I, std::size_t... Indices>
    requires(I < sizeof...(Indices))
[[nodiscard]] constexpr std::size_t get_index(std::index_sequence<Indices...>) noexcept {
    constexpr auto values = std::array<std::size_t, sizeof...(Indices)>{ Indices... };
    return values[I];
}

/** Checks every adjacent pair; empty and single-element sequences return true. */
template <std::size_t... Indices>
[[nodiscard]] constexpr bool is_strictly_increasing(std::index_sequence<Indices...>) noexcept {
    constexpr auto values = std::array<std::size_t, sizeof...(Indices)>{ Indices... };
    for (auto i = std::size_t{ 1 }; i < values.size(); ++i) {
        if (values[i - 1] >= values[i]) {
            return false;
        }
    }
    return true;
}

} // namespace details

/**
 * @brief Matches index packs in which every value is greater than its predecessor.
 * Empty and single-element packs are strictly increasing.
 */
template <std::size_t... Indices>
concept StrictlyIncreasing = details::is_strictly_increasing(std::index_sequence<Indices...>{});

} // namespace utility
