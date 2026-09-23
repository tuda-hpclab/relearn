#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"

#include "io/LogFiles.h"
#include "io/parser/IntervalParser.h"
#include "neurons/NeuronsExtraInfo.h"
#include "types/BasicTypes.h"
#include "types/StimulusTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/Interval.hpp>
#include <cpp-utility/StringUtil.hpp>
#include <cpp-utility/ranges/Functional.hpp>
#include <cpp-utility/ranges/views/Optional.hpp>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class StimulusParser {
public:
    using step_type = RelearnTypes::step_type;
    using activity_type = RelearnTypes::activity_type;

    struct Stimulus {
        utility::Interval<step_type> interval{};
        activity_type stimulus_intensity{};
        std::unordered_set<NeuronID> matching_ids{};
        std::unordered_set<RelearnTypes::group_name> matching_group_names;
    };

    /**
     * @brief Parses one line into a stimulus. Ignores stimuli for other ranks than the current.
     * The line must have the format:
     *      <interval_description> <stimulus intensity> <neuron_id>*
     *      A neuron id must have the format: <rank>:<local_neuron_id> or must be an group name
     * @param line The line to parse
     * @param my_rank The mpi rank of the current process
     * @return Returns an optional Stimulus which is empty if parsing failed
     */
    [[nodiscard]] static std::optional<StimulusParser::Stimulus> parse_line(const std::string& line, const int my_rank) {
        auto ss = std::stringstream{ line };

        auto interval_description = std::string{};
        ss >> interval_description;

        if (!ss) {
            return {};
        }

        const auto& parsed_interval = IntervalParser::parse_interval(interval_description);
        if (!parsed_interval.has_value()) {
            return {};
        }

        auto intensity = activity_type{};
        ss >> intensity;

        if (!ss) {
            return {};
        }

        auto ids = std::unordered_set<NeuronID>{};
        ids.reserve(line.size());

        auto group_names = std::unordered_set<RelearnTypes::group_name>{};
        group_names.reserve(line.size());

        for (auto current_value = std::string{}; ss >> current_value;) {
            const auto& rank_neuron_id_vector = utility::split_string(current_value, ':');
            if (rank_neuron_id_vector.size() == 2) {
                // Neuron has format <rank>:<neuron_id>
                const auto rank = std::stoi(rank_neuron_id_vector[0]);
                if (rank != my_rank) {
                    continue;
                }
                const auto neuron_id = std::stoul(rank_neuron_id_vector[1]);
                ids.insert({ NeuronID{ neuron_id - 1 } });
            } else {
                RelearnException::check(!utility::is_number(current_value), "StimulusParser::parseLine:: Illegal neuron id {} in stimulus files. Must have the format <rank>:<neuron_id> or be an group name", current_value);
                // Neuron descriptor is an group name
                group_names.insert(current_value);
            }
        }

        return { StimulusParser::Stimulus{ .interval = parsed_interval.value(), .stimulus_intensity = intensity, .matching_ids = ids, .matching_group_names = group_names } };
    }

    /**
     * @brief Parses all lines as stimuli and discards those that fail or are for another mpi rank
     * @param lines The lines to parse
     * @param my_rank The mpi rank of the current process
     * @return All successfully parsed stimuli
     */
    [[nodiscard]] static std::vector<Stimulus> parse_lines(const std::vector<std::string>& lines, const int my_rank) {
        return lines
               | ranges::views::transform([my_rank](const auto& line) { return parse_line(line, my_rank); })
               | utility::views::optional_values
               | ranges::to_vector;
    }

    /**
     * @brief Converts the given stimuli to a function that allows easy checking of the current step and neuron id.
     *      If a the combination of step and neuron id hits a stimulus, it returns the intensity. Otherwise, returns 0.0.
     * @param stimuli The given stimuli, should not intersect.
     * @return The check function. Empty if the stimuli intersect
     */
    [[nodiscard]] static RelearnTypes::stimuli_function_type generate_stimulus_function(std::vector<StimulusParser::Stimulus> stimuli) {
        auto step_checker_function = [stimuli = std::move(stimuli)](step_type current_step) noexcept -> RelearnTypes::stimuli_list_type {
            const auto hits_current_step = [current_step](const utility::Interval<step_type>& interval) {
                return interval.hits_step(current_step);
            };

            return stimuli
                   | ranges::views::filter(hits_current_step, &Stimulus::interval)
                   | ranges::views::transform([](const Stimulus& stimulus) {
                         return std::pair<std::unordered_set<NeuronID>, RelearnTypes::activity_type>{ stimulus.matching_ids, stimulus.stimulus_intensity };
                     })
                   | ranges::to<RelearnTypes::stimuli_list_type>;
        };

        return step_checker_function;
    }
};
