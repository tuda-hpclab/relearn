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

namespace utility {

/**
 * @brief Counts the number of digits necessary to print the value
 * @tparam T Must be integral
 * @param val The value to print
 * @return The number of digits of val
 */
template <std::integral T>
static constexpr unsigned int num_digits(const T val) noexcept {
    constexpr auto number_system_base = 10;
    constexpr auto number_system_base_converted = T{ 10 };

    auto num_digits = 1U;

    auto current_val = val;

    while (current_val >= number_system_base_converted) {
        ++num_digits;
        // NOLINTNEXTLINE
        current_val /= number_system_base;
    }

    return num_digits;
}

/**
 * @brief Calculates the faculty $value!$
 * @param value The value
 * @tparam T Type of which a faculty should be calculated. Must fulfill std::is_unsigned_v<T>
 * @return Returns the faculty of the parameter value.
 */
template <std::unsigned_integral T>
static constexpr T factorial(T value) noexcept {
    if (value < 2) {
        return 1;
    }

    auto result = T{ 1 };
    while (value > 1) {
        result *= value;
        value--;
    }

    return result;
}

} // namespace utility
