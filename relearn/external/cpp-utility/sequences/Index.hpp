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

#include <algorithm>
#include <cstdint>
#include <utility>

namespace utility {

namespace details {

template <std::size_t... Indices>
constexpr std::size_t max_index(std::index_sequence<Indices...>) {
    return (std::max)({ Indices... });
}

template <std::size_t... Indices>
constexpr std::size_t min_index(std::index_sequence<Indices...>) {
    return (std::min)({ Indices... });
}

template <std::size_t I, std::size_t... Indices>
constexpr std::size_t get_index(std::index_sequence<Indices...>) {
    constexpr std::size_t arr[] = { Indices... };
    return arr[I];
}

template <std::size_t... Indices>
constexpr bool is_strictly_increasing(std::index_sequence<Indices...>) {
    return true;
}

template <std::size_t Index>
constexpr bool is_strictly_increasing(std::index_sequence<Index>) {
    return true;
}

template <std::size_t Index1, std::size_t Index2>
constexpr bool is_strictly_increasing(std::index_sequence<Index1, Index2>) {
    constexpr auto first_value = get_index<0, Index1, Index2>(std::index_sequence<Index1, Index2>{});
    constexpr auto second_value = get_index<1, Index1, Index2>(std::index_sequence<Index1, Index2>{});

    return first_value < second_value;
}

template <std::size_t Index1, std::size_t Index2, std::size_t... Indices>
constexpr bool is_strictly_increasing(std::index_sequence<Index1, Index2, Indices...>) {
    constexpr auto first_value = get_index<0, Index1, Index2, Indices...>(std::index_sequence<Index1, Index2, Indices...>{});
    constexpr auto second_value = get_index<1, Index1, Index2, Indices...>(std::index_sequence<Index1, Index2, Indices...>{});

    constexpr auto first_result = first_value < second_value;
    constexpr auto recursive_result = is_strictly_increasing(std::index_sequence<Index2, Indices...>());

    return first_result && recursive_result;
}

} // namespace details

template <std::size_t... Indices>
concept StrictlyIncreasing = details::is_strictly_increasing(std::index_sequence<Indices...>{});

} // namespace utility
