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

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace utility {

namespace detail {
using std::get;

template <typename T>
concept has_tuple_size = requires {
    { std::tuple_size<std::remove_cvref_t<T>>::value } -> std::convertible_to<std::size_t>;
};

template <typename T>
concept has_extent = requires {
    requires std::integral<decltype(std::remove_cvref_t<T>::extent)>;
};

template <typename T>
concept has_adl_get = requires(T val) {
    { get<0>(val) };
};

template <typename T>
concept has_member_get = requires(T val) {
    { val.template get<0>() };
};

template <typename T>
concept Enum = std::is_enum_v<std::remove_cvref_t<T>>;

} // namespace detail

} // namespace utility
