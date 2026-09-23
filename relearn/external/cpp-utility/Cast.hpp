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
#include <limits>
#include <typeinfo>
#include <utility>

namespace utility {

/**
 * @brief Converts an integral value without truncation or a signedness-induced value change.
 * @tparam R The type to cast to
 * @tparam T The current type
 * @param value The value to cast
 * @throws Exception if @p value is outside the range representable by @p R.
 * @return The value represented as @p R.
 */
template <std::integral R, std::integral T>
    requires std::convertible_to<T, R>
[[nodiscard]] constexpr R safe_cast(const T value) {
    if constexpr (std::same_as<T, R>) {
        return value;
    } else {
        constexpr auto flag_min = std::cmp_greater_equal(std::numeric_limits<T>::min(), std::numeric_limits<R>::min());
        constexpr auto flag_max = std::cmp_less_equal(std::numeric_limits<T>::max(), std::numeric_limits<R>::max());

        if constexpr (flag_min && flag_max) {
            return static_cast<R>(value);
        } else {
            const auto is_in_range = std::in_range<R>(value);
            if (!is_in_range) [[unlikely]] {
                Exception::fail(
                    "Value {} of type {} is not in the range of values representable of type {}", value, typeid(T).name(), typeid(R).name());
            }

            return static_cast<R>(value);
        }
    }
}

namespace detail {

template <bool RequireExactRepresentation, std::floating_point R, std::floating_point T>
    requires std::convertible_to<T, R>
[[nodiscard]] constexpr R floating_point_cast(const T value) {
    if constexpr (std::same_as<T, R>) {
        return value;
    } else {
        if (value != value) {
            if constexpr (std::numeric_limits<R>::has_quiet_NaN) {
                return static_cast<R>(value);
            } else {
                Exception::fail("Value of type {} is NaN, which is not representable by type {}", typeid(T).name(), typeid(R).name());
            }
        }

        if constexpr (std::numeric_limits<T>::has_infinity) {
            constexpr auto infinity = std::numeric_limits<T>::infinity();
            if (value == infinity || value == -infinity) {
                if constexpr (std::numeric_limits<R>::has_infinity) {
                    return static_cast<R>(value);
                } else {
                    Exception::fail("Infinite value of type {} is not representable by type {}", typeid(T).name(), typeid(R).name());
                }
            }
        }

        constexpr auto source_has_larger_range = std::numeric_limits<T>::max_exponent > std::numeric_limits<R>::max_exponent
                                                 || (std::numeric_limits<T>::max_exponent == std::numeric_limits<R>::max_exponent
                                                     && std::numeric_limits<T>::digits > std::numeric_limits<R>::digits);
        if constexpr (source_has_larger_range) {
            const auto lowest = static_cast<T>(std::numeric_limits<R>::lowest());
            const auto max = static_cast<T>(std::numeric_limits<R>::max());
            if (value < lowest || value > max) [[unlikely]] {
                Exception::fail("Value {} of type {} is not in the range of values representable by type {}", value, typeid(T).name(), typeid(R).name());
            }
        }

        const auto converted = static_cast<R>(value);
        if constexpr (RequireExactRepresentation) {
            if (static_cast<T>(converted) != value) [[unlikely]] {
                Exception::fail("Value {} of type {} cannot be represented exactly by type {}", value, typeid(T).name(), typeid(R).name());
            }
        }

        return converted;
    }
}

} // namespace detail

/**
 * @brief Converts a floating-point value without overflow, underflow, or loss of precision.
 * @tparam R The type to cast to
 * @tparam T The current type
 * @param value The value to cast
 * @throws Exception if @p value cannot be represented exactly by @p R.
 * @return The value represented as @p R.
 */
template <std::floating_point R, std::floating_point T>
    requires std::convertible_to<T, R>
[[nodiscard]] constexpr R safe_cast(const T value) {
    return detail::floating_point_cast<true, R>(value);
}

namespace detail {

template <typename R, typename T>
concept safely_castable = requires(const T& value) { utility::safe_cast<R>(value); };

template <typename R, typename T>
    requires safely_castable<R, T>
[[nodiscard]] constexpr R invoke_safe_cast(const T& value) {
    return utility::safe_cast<R>(value);
}

} // namespace detail

/**
 * @brief Converts an integral value without truncation or a signedness-induced value change.
 * @tparam R The type to cast to
 * @tparam T The current type
 * @param value The value to cast
 * @throws Exception if @p value is outside the range representable by @p R.
 * @return The value represented as @p R.
 */
template <std::integral R, std::integral T>
    requires std::convertible_to<T, R>
[[nodiscard]] constexpr R cast(const T value) {
    return safe_cast<R>(value);
}

/**
 * @brief Converts a floating-point value without overflow, allowing loss of precision.
 * @tparam R The type to cast to
 * @tparam T The current type
 * @param value The value to cast
 * @throws Exception if @p value is finite and outside the range representable by @p R, or if a special value is not supported by @p R.
 * @return The value represented as @p R.
 */
template <std::floating_point R, std::floating_point T>
    requires std::convertible_to<T, R>
[[nodiscard]] constexpr R cast(const T value) {
    return detail::floating_point_cast<false, R>(value);
}

/**
 * @brief Performs a compile-time cast between integral types.
 * @tparam R The destination integral type.
 * @tparam T The source integral type.
 * @param value The value to cast.
 * @return @p value represented as @p R.
 *
 * @note This function can only be invoked with compile-time constant values.
 * @note The conversion is performed using <tt>static_cast&lt;R&gt;</tt>.
 */
template<std::integral R, std::integral T>
    requires std::convertible_to<T, R>
[[nodiscard]] consteval R as(const T value) {
    return static_cast<R>(value);
}

/**
 * @brief Performs a compile-time cast between floating-point types.
 * @tparam R The destination floating-point type.
 * @tparam T The source floating-point type.
 * @param value The value to cast.
 * @return @p value represented as @p R.
 *
 * @note This function can only be invoked with compile-time constant values.
 * @note The conversion is performed using <tt>static_cast&lt;R&gt;</tt>.
 */
template<std::floating_point R, std::floating_point T>
    requires std::convertible_to<T, R>
[[nodiscard]] consteval R as(const T value) {
    return static_cast<R>(value);
}

} // namespace utility
