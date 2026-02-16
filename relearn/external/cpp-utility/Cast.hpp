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

#include "cpp-utility/Exception.hpp"

#include <concepts>
#include <limits>
#include <utility>

namespace utility {

/**
 * @brief Casts a value between types; throws an Exception if the value cannot be represented in the new type
 * @tparam R The type to cast to
 * @tparam T The current type
 * @param value The value to cast
 * @exception Throws an Exception if R(val) is lossy
 * @return The cast value
 */
template <std::integral R, std::integral T>
    requires std::convertible_to<T, R>
[[nodiscard]] constexpr R save_cast(const T value) {
    constexpr auto flag_min = std::cmp_greater_equal(std::numeric_limits<T>::min(), std::numeric_limits<R>::min());
    constexpr auto flag_max = std::cmp_less_equal(std::numeric_limits<T>::max(), std::numeric_limits<R>::max());

    if constexpr (flag_min && flag_max) {
        return static_cast<R>(value);
    } else {
        const auto is_in_range = std::in_range<R, T>(value);
        if (!is_in_range) [[unlikely]] {
            Exception::fail("Value {} of type {} is not in the range of values representable of type {}", value, typeid(T).name(), typeid(R).name());
        }

        return static_cast<R>(value);
    }
}

} // namespace utility
