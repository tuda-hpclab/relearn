/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "StringUtil.h"

#include "RelearnException.h"

#include "cpp-utility/Cast.hpp"

#include <fmt/core.h>
#include <range/v3/algorithm/all_of.hpp>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

std::vector<std::string> StringUtil::split_string(const std::string& string, const char delim) {
    auto result = std::vector<std::string>{};
    result.reserve(string.size());

    auto ss = std::stringstream(string);
    auto item = std::string{};

    while (getline(ss, item, delim)) {
        result.emplace_back(std::move(item));
    }

    return result;
}

std::vector<std::string> StringUtil::split_string(const std::string_view string_view, const char delim) {
    auto result = std::vector<std::string>{};
    result.reserve(string_view.size());

    auto current_position = std::string::size_type{ 0 };

    while (true) {
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

bool StringUtil::is_number(const std::string_view s) {
    return ranges::all_of(s, [](const char c) { return std::isdigit(c); });
}

void StringUtil::to_lower(std::string& str) {
    std::ranges::transform(str, str.begin(),
                           [](const char c) { return static_cast<char>(std::tolower(c)); });
}

std::string StringUtil::format_int_with_leading_zeros(const int number, const unsigned int nr_of_digits) {
    RelearnException::check(nr_of_digits >= 1, "StringUtil::format_with_leading_zeros. Number must has at least 1 digit");
    return fmt::format("{1:0>0{0}}", nr_of_digits, number);
}
