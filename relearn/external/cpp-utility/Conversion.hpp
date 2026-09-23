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

#include <fmt/core.h>

#include <concepts>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace utility {

namespace detail {

template <typename T>
concept StringConversionTarget = (std::integral<T> && !std::same_as<std::remove_cv_t<T>, bool>) || std::floating_point<T>;

template <typename CharT>
concept StringConversionCharacter = std::same_as<CharT, char> || std::same_as<CharT, wchar_t>;

/**
 * @brief Invokes a standard string-to-number conversion and rejects unconsumed input.
 * @param value The owned string passed to the standard conversion function.
 * @param conversion The standard conversion function to invoke.
 * @return The converted numeric value.
 * @throws std::invalid_argument if @p value is not a complete numeric representation.
 * @throws std::out_of_range if the represented value does not fit the conversion function's result type.
 */
template <typename String, typename Conversion>
[[nodiscard]] auto parse_complete(const String& value, Conversion&& conversion) {
    std::size_t converted_characters = 0;
    const auto result = std::forward<Conversion>(conversion)(value, &converted_characters);
    if (converted_characters != value.size()) {
        throw std::invalid_argument("The string does not contain exactly one numeric value");
    }

    return result;
}

/**
 * @brief Converts a narrow or wide string view to a supported numeric destination type.
 * @tparam T The requested destination type.
 * @tparam CharT The input character type.
 * @param value The textual numeric representation.
 * @return The value represented by @p value.
 * @throws std::invalid_argument if @p value is not a complete numeric representation or is negative for an unsigned destination.
 * @throws std::out_of_range if @p value is outside the range of @p T.
 */
template <StringConversionTarget T, StringConversionCharacter CharT>
[[nodiscard]] T convert_from_string_view(const std::basic_string_view<CharT> value) {
    const auto input = std::basic_string<CharT>{ value };

    if constexpr (std::same_as<T, float>) {
        return parse_complete(input, [](const auto& string, auto* position) { return std::stof(string, position); });
    } else if constexpr (std::same_as<T, double>) {
        return parse_complete(input, [](const auto& string, auto* position) { return std::stod(string, position); });
    } else if constexpr (std::same_as<T, long double>) {
        return parse_complete(input, [](const auto& string, auto* position) { return std::stold(string, position); });
    } else if constexpr (std::same_as<T, int>) {
        return parse_complete(input, [](const auto& string, auto* position) { return std::stoi(string, position); });
    } else if constexpr (std::same_as<T, long>) {
        return parse_complete(input, [](const auto& string, auto* position) { return std::stol(string, position); });
    } else if constexpr (std::same_as<T, long long>) {
        return parse_complete(input, [](const auto& string, auto* position) { return std::stoll(string, position); });
    } else if constexpr (std::same_as<T, unsigned long>) {
        if (value.find(CharT{ '-' }) != std::basic_string_view<CharT>::npos) {
            throw std::invalid_argument("A negative value cannot be converted to an unsigned type");
        }

        return parse_complete(input, [](const auto& string, auto* position) { return std::stoul(string, position); });
    } else if constexpr (std::same_as<T, unsigned long long>) {
        if (value.find(CharT{ '-' }) != std::basic_string_view<CharT>::npos) {
            throw std::invalid_argument("A negative value cannot be converted to an unsigned type");
        }

        return parse_complete(input, [](const auto& string, auto* position) { return std::stoull(string, position); });
    } else if constexpr (std::signed_integral<T>) {
        const auto converted = parse_complete(input, [](const auto& string, auto* position) { return std::stoll(string, position); });
        if (!std::in_range<T>(converted)) {
            throw std::out_of_range("The numeric value is outside the range of the destination type");
        }

        return static_cast<T>(converted);
    } else {
        if (value.find(CharT{ '-' }) != std::basic_string_view<CharT>::npos) {
            throw std::invalid_argument("A negative value cannot be converted to an unsigned type");
        }

        const auto converted = parse_complete(input, [](const auto& string, auto* position) { return std::stoull(string, position); });
        if (!std::in_range<T>(converted)) {
            throw std::out_of_range("The numeric value is outside the range of the destination type");
        }

        return static_cast<T>(converted);
    }
}

} // namespace detail

/**
 * @brief Converts a narrow string, string view, or C string to a numeric type.
 * @tparam T The requested integral or floating-point destination type, excluding bool.
 * @param value The complete textual numeric representation.
 * @return The value represented by @p value.
 * @throws std::invalid_argument if @p value is invalid, contains trailing characters, or is negative for an unsigned destination.
 * @throws std::out_of_range if @p value cannot be represented by @p T.
 */
template <detail::StringConversionTarget T>
[[nodiscard]] T convert_from_string(const std::string_view value) {
    return detail::convert_from_string_view<T>(value);
}

/**
 * @brief Converts a wide string, string view, or C string to a numeric type.
 * @tparam T The requested integral or floating-point destination type, excluding bool.
 * @param value The complete textual numeric representation.
 * @return The value represented by @p value.
 * @throws std::invalid_argument if @p value is invalid, contains trailing characters, or is negative for an unsigned destination.
 * @throws std::out_of_range if @p value cannot be represented by @p T.
 */
template <detail::StringConversionTarget T>
[[nodiscard]] T convert_from_string(const std::wstring_view value) {
    return detail::convert_from_string_view<T>(value);
}

/**
 * @brief Converts an integer to a zero-padded string with a minimum width.
 * The sign of a negative number counts towards the width.
 * @param number The integer to convert
 * @param nr_of_digits The minimum output width
 * @throws Exception if @p nr_of_digits is zero.
 * @return The formatted integer, padded with leading zeros if necessary
 */
[[nodiscard]] inline std::string format_int_with_leading_zeros(const int number, const unsigned int nr_of_digits) {
    Exception::check(nr_of_digits >= 1, "utility::format_int_with_leading_zeros: The minimum width must be at least 1");
    return fmt::format("{:0{}}", number, nr_of_digits);
}

} // namespace utility
