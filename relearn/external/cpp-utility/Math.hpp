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

#include <cmath>
#include <concepts>
#include <numeric>
#include <type_traits>

namespace utility {

/**
 * @brief Counts the base-10 digits of an integral value's magnitude.
 * The minus sign of a negative value is not counted; zero has one digit.
 * @tparam T Must be integral
 * @param val The value to print
 * @return The number of digits of val
 */
template <std::integral T>
constexpr unsigned int num_digits(const T val) noexcept {
    if constexpr (std::same_as<std::remove_cv_t<T>, bool>) {
        return 1U;
    } else {
        using unsigned_type = std::make_unsigned_t<T>;
        constexpr auto number_system_base = unsigned_type{ 10 };

        auto magnitude = static_cast<unsigned_type>(val);
        if constexpr (std::signed_integral<T>) {
            if (val < T{ 0 }) {
                magnitude = unsigned_type{ 0 } - magnitude;
            }
        }

        auto digits = 1U;
        while (magnitude >= number_system_base) {
            ++digits;
            magnitude /= number_system_base;
        }

        return digits;
    }
}

/**
 * @brief Calculates the factorial @p value! iteratively.
 * @param value The value
 * @tparam T Type whose factorial should be calculated. Must be an unsigned integral type
 * @return The factorial, modulo the range of @p T if multiplication overflows.
 */
template <std::unsigned_integral T>
constexpr T factorial(T value) noexcept {
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

/**
 * @brief Computes the ceiling of a / b, i.e., the smallest integer >= a/b
 * @tparam T Must be an unsigned integral type
 * @param a The dividend
 * @param b The divisor, must not be zero
 * @exception Throws an Exception if b is zero
 * @return ceil(a / b)
 */
template <std::unsigned_integral T>
[[nodiscard]] constexpr T div_ceil(const T a, const T b) {
    Exception::check(b != T{ 0 }, "div_ceil: divisor must not be zero");
    return a / b + (a % b != T{ 0 } ? T{ 1 } : T{ 0 });
}

/**
 * @brief Checks whether n is a power of two
 * @tparam T Must be an unsigned integral type
 * @param n The value to check
 * @return True iff n is a power of two (i.e., n > 0 and exactly one bit is set)
 */
template <std::unsigned_integral T>
[[nodiscard]] constexpr bool is_power_of_two(const T n) noexcept {
    return n > T{ 0 } && (n & (n - T{ 1 })) == T{ 0 };
}

/**
 * @brief Computes base^Exp using recursive squaring. Exp is a compile-time constant
 * @tparam Exp The exponent, a compile-time constant
 * @tparam T The type of the base, must be arithmetic
 * @param base The base value
 * @return base raised to the power Exp; arithmetic overflow is unchecked.
 */
template <std::size_t Exp, typename T>
    requires std::is_arithmetic_v<T>
[[nodiscard]] constexpr T integer_pow(const T base) noexcept {
    if constexpr (Exp == 0) {
        return T{ 1 };
    } else if constexpr (Exp == 1) {
        return base;
    } else if constexpr (Exp % 2 == 0) {
        const auto half = integer_pow<Exp / 2>(base);
        return half * half;
    } else {
        return base * integer_pow<Exp - 1>(base);
    }
}

/**
 * @brief Returns the sign of value: -1 if negative, 0 if zero, +1 if positive
 * @tparam T Must be arithmetic
 * @param value The value to check
 * @return -1, 0, or 1. A floating-point NaN returns 0.
 */
template <typename T>
    requires std::is_arithmetic_v<T>
[[nodiscard]] constexpr int sign(const T value) noexcept {
    return (T{ 0 } < value) - (value < T{ 0 });
}

/**
 * @brief Computes the binomial coefficient n choose k without overflowing an intermediate product when the result fits in @p T.
 * @tparam T Must be an unsigned integral type
 * @param n The total number of elements
 * @param k The number of elements to choose, must be <= n
 * @exception Throws an Exception if k > n
 * @return The binomial coefficient n choose k; overflow is unchecked if the result itself does not fit in @p T.
 */
template <std::unsigned_integral T>
    requires(!std::same_as<std::remove_cv_t<T>, bool>)
[[nodiscard]] constexpr T binomial_coefficient(const T n, const T k) {
    Exception::check(k <= n, "binomial_coefficient: k ({}) must not exceed n ({})", k, n);

    if (k == T{ 0 } || k == n) {
        return T{ 1 };
    }

    const auto k_eff = (k < n - k) ? k : n - k;

    auto result = T{ 1 };
    for (auto i = T{ 0 }; i < k_eff; ++i) {
        auto numerator = n - i;
        auto denominator = i + T{ 1 };

        const auto numerator_factor = std::gcd(numerator, denominator);
        numerator /= numerator_factor;
        denominator /= numerator_factor;

        const auto result_factor = std::gcd(result, denominator);
        result /= result_factor;

        result *= numerator;
    }
    return result;
}

/**
 * @brief Checks whether two floating-point values are equal within an absolute tolerance
 * @tparam T Must be a floating-point type
 * @param a The first value
 * @param b The second value
 * @param epsilon The non-negative maximum allowed absolute difference.
 * @return True iff epsilon is non-negative and the values compare equal or |a - b| <= epsilon.
 * Equal infinities compare equal; unequal infinities and NaNs never compare equal.
 */
template <std::floating_point T>
[[nodiscard]] bool almost_equal(const T a, const T b, const T epsilon) noexcept {
    if (!(epsilon >= T{ 0 })) {
        return false;
    }
    if (a == b) {
        return true;
    }
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return false;
    }
    return std::abs(a - b) <= epsilon;
}

} // namespace utility
