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
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace utility {

/**
 * @brief Splits a string view at a delimiter character.
 * Empty substrings at the beginning and between two delimiters are retained; a trailing empty substring is discarded.
 * @param string_view The string view to split
 * @param delim The delimiter character
 * @return The substrings; an empty input produces an empty vector
 */
[[nodiscard]] inline std::vector<std::string> split_string(const std::string_view string_view, const char delim) {
    auto result = std::vector<std::string>{};
    result.reserve(string_view.size());

    auto current_position = std::string_view::size_type{ 0 };

    while (current_position < string_view.size()) {
        auto delim_position = string_view.find(delim, current_position);
        if (delim_position == std::string_view::npos) {
            delim_position = string_view.size();
        }

        const auto substring = string_view.substr(current_position, delim_position - current_position);
        result.emplace_back(substring);

        if (delim_position == string_view.size()) {
            break;
        }

        current_position = delim_position + 1;
    }

    return result;
}

/**
 * @brief Splits a string at a delimiter character.
 * Empty substrings at the beginning and between two delimiters are retained; a trailing empty substring is discarded.
 * @param string The string to split
 * @param delim The delimiter character
 * @return The substrings; an empty input produces an empty vector
 */
[[nodiscard]] inline std::vector<std::string> split_string(const std::string& string, const char delim) {
    return split_string(std::string_view{ string }, delim);
}

/**
 * @brief Checks whether a non-empty string contains only decimal digits.
 * @param string_view The string view to check
 * @return true if and only if the string is non-empty and contains only characters from '0' to '9'
 */
[[nodiscard]] inline bool is_number(const std::string_view string_view) {
    return !string_view.empty() && std::ranges::all_of(string_view, [](const char character) {
        return std::isdigit(static_cast<unsigned char>(character));
    });
}

/**
 * @brief Converts the characters of a string to lower case in place.
 * @param string The string to modify
 */
inline void to_lower(std::string& string) {
    std::ranges::transform(string, string.begin(), [](const char character) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    });
}

} // namespace utility
