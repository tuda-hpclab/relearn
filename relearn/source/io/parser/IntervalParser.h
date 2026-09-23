#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "io/LogFiles.h"
#include "types/BasicTypes.h"

#include <cpp-utility/Interval.hpp>

#include <mpi-wrapper/core/MPIRank.h>

#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * This class parses string that describe an interval or a plethora of intervals and returns them either
 * wrapped inside an std::optional or within an std::vector
 */
struct IntervalParser {
    using step_type = RelearnTypes::step_type;

    /**
     * @brief Parses an interval from a description. The description must have the form
     *      <begin>-<end>:<frequency> with <begin> <= <end>
     * @param description The description of the interval
     * @return An optional which is empty if no interval could be parsed, and contains the interval if the parsing succeeded
     */
    [[nodiscard]] static std::optional<utility::Interval<step_type>> parse_interval(const std::string_view description) {
        const auto hyphen_position = description.find('-');
        const auto colon_position = description.find(':');

        if (hyphen_position == std::string::npos || colon_position == std::string::npos) {
            return {};
        }

        const auto& begin_string = description.substr(0, hyphen_position);
        const auto& end_string = description.substr(hyphen_position + 1, colon_position - hyphen_position - 1);
        const auto& frequency_string = description.substr(colon_position + 1, description.size() - colon_position);

        auto begin = step_type{};
        const auto& [begin_ptr, begin_err] = std::from_chars(begin_string.data(), begin_string.data() + begin_string.size(), begin);

        auto end = step_type{};
        const auto& [end_ptr, end_err] = std::from_chars(end_string.data(), end_string.data() + end_string.size(), end);

        auto frequency = step_type{};
        const auto& [frequency_ptr, frequency_err] = std::from_chars(frequency_string.data(), frequency_string.data() + frequency_string.size(), frequency);

        const auto begin_ok = (begin_err == std::errc{}) && (begin_ptr == begin_string.data() + begin_string.size());
        const auto end_ok = (end_err == std::errc{}) && (end_ptr == end_string.data() + end_string.size());
        const auto frequency_ok = (frequency_err == std::errc{}) && (frequency_ptr == frequency_string.data() + frequency_string.size());

        if (begin_ok && end_ok && frequency_ok) {
            if (end < begin) {
                LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Parsed interval description has end before beginning : {}", description);
                return {};
            }

            return utility::Interval<step_type>{ .begin = begin, .end = end, .frequency = frequency };
        }

        LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Failed to parse string to match the pattern <std::uint64>-<std::uint64>:<std::uint64> : {}", description);
        return {};
    }

    /**
     * @brief Parses multiple intervals from the description. Each interval must have the form
     *      <begin>-<end>:<frequency> with ; separating the intervals
     * @param description The description of the intervals
     * @return A vector with all successfully parsed intervals
     */
    [[nodiscard]] static std::vector<utility::Interval<step_type>> parse_description_as_intervals(const std::string_view description) {
        auto intervals = std::vector<utility::Interval<step_type>>{};
        auto current_position = std::string::size_type{ 0 };

        while (true) {
            auto semicolon_position = description.find(';', current_position);
            if (semicolon_position == std::string_view::npos) {
                semicolon_position = description.size();
            }

            const auto substring = description.substr(current_position, semicolon_position - current_position);
            const auto opt_interval = parse_interval(substring);

            if (opt_interval.has_value()) {
                intervals.emplace_back(opt_interval.value());
            }

            if (semicolon_position == description.size()) {
                break;
            }

            current_position = semicolon_position + 1;
        }

        return intervals;
    }
};
