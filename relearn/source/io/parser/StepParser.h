#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Types.h"

#include "io/LogFiles.h"
#include "io/parser/IntervalParser.h"

#include "cpp-utility/Interval.hpp"

#include "mpi-wrapper/MPIRank.h"

#include <algorithm>
#include <functional>
#include <string_view>
#include <vector>

/**
 * This class parses string that describe at which step a function shall be executed, and returns an std::function which encapsulates this logic.
 * It uses intervals of the form [begin, end] (in predetermined step sizes) to check
 */
class StepParser {
public:
    /**
     * @brief Parses a description of intervals to a std::function which returns true whenever the current simulation step
     *      falls into one of the intervals. The format must be: <begin>-<end>:<frequency> with ; separating the intervals
     * @param description The description of the intervals
     * @return The function indicating if the event shall occur
     */
    [[nodiscard]] static std::function<bool(RelearnTypes::step_type)> generate_step_check_function(const std::string_view description) {
        auto intervals = IntervalParser::parse_description_as_intervals(description);
        auto function = generate_step_check_function(std::move(intervals));
        return function;
    }

    /**
     * @brief Converts a vector of intervals to a function which maps the current simulation step to
     *      whether or not it is matched by the intervals. If the intervals intersect themselves, the empty std::function is returned
     * @param intervals The intervals that specify if an event shall occur
     * @return A std::function object that maps the current step to true or false, indicating if the event shall occur
     */
    [[nodiscard]] static std::function<bool(RelearnTypes::step_type)> generate_step_check_function(std::vector<utility::Interval<RelearnTypes::step_type>> intervals) noexcept {
        const auto intervals_intersect = utility::Interval<RelearnTypes::step_type>::check_intervals_for_intersection(intervals);
        if (intervals_intersect) {
            LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "The intervals for the step parser intersected, discarding all.");
            return {};
        }

        const auto comparison = [](const utility::Interval<RelearnTypes::step_type>& first, const utility::Interval<RelearnTypes::step_type>& second) noexcept -> bool {
            return first.begin < second.begin;
        };

        std::ranges::sort(intervals, comparison);

        const auto step_check_function = [intervals = std::move(intervals)](RelearnTypes::step_type step) noexcept -> bool {
            return std::ranges::any_of(intervals, [step](const utility::Interval<RelearnTypes::step_type>& interval) { return interval.hits_step(step); });
        };

        return step_check_function;
    }
};
