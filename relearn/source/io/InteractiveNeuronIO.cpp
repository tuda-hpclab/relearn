/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "InteractiveNeuronIO.h"

#include "Types.h"
#include "Types2.h"

#include "io/parser/StimulusParser.h"
#include "neurons/LocalGroupTranslator.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/StringUtil.h"

#include "cpp-utility/ranges/views/IO.hpp"
#include "cpp-utility/ranges/views/Optional.hpp"

#include "mpi-wrapper/MPIRank.h"

#include <fmt/ranges.h>
#include <fmt/std.h>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/functional/not_fn.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/range/primitives.hpp>
#include <range/v3/view/cache1.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/transform.hpp>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

std::vector<std::pair<InteractiveNeuronIO::step_type, std::vector<NeuronID>>> InteractiveNeuronIO::load_enable_interrupts(const std::filesystem::path& path_to_file, const mpiPP::MPIRank my_rank) {
    auto file = std::ifstream{ path_to_file };

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "InteractiveNeuronIO::load_enable_interrupts: Opening the file '{}' was not successful", path_to_file);

    auto return_value = std::vector<std::pair<step_type, std::vector<NeuronID>>>{};

    for (const auto& line : ranges::getlines(file) | utility::views::filter_not_comment_not_empty_line) {
        auto sstream = std::stringstream(line);

        auto step = step_type{};
        auto delim = char{};

        const auto success = (sstream >> step) && (sstream >> delim);

        if (!success) {
            spdlog::info("Skipping line: \"{}\"", line);
            continue;
        }

        if (delim != 'e') {
            if (delim != 'd' && delim != 'c') {
                spdlog::error("Wrong deliminator: \"{}\"", line);
            }
            continue;
        }

        auto indices = std::vector<NeuronID>{};

        for (auto rank_neuron_string = std::string{}; sstream >> rank_neuron_string;) {
            const auto rank_neuron_vector = StringUtil::split_string(rank_neuron_string, ':');
            const auto rank = mpiPP::MPIRank{ std::stoi(rank_neuron_vector[0]) };
            const auto neuron_id = NeuronID(std::stoi(rank_neuron_vector[1]) - 1);

            if (rank == my_rank) {
                indices.emplace_back(neuron_id);
            }
        }

        return_value.emplace_back(step, std::move(indices));
    }

    return return_value;
}

std::vector<std::pair<InteractiveNeuronIO::step_type, std::vector<NeuronID>>> InteractiveNeuronIO::load_disable_interrupts(const std::filesystem::path& path_to_file, const mpiPP::MPIRank my_rank) {
    auto file = std::ifstream{ path_to_file };

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "InteractiveNeuronIO::load_disable_interrupts: Opening the file '{}' was not successful", path_to_file);

    auto return_value = std::vector<std::pair<step_type, std::vector<NeuronID>>>{};

    for (const auto& line : ranges::getlines(file) | utility::views::filter_not_comment_not_empty_line) {
        auto sstream = std::stringstream(line);

        auto step = step_type{};
        auto delim = char{};

        const auto success = (sstream >> step) && (sstream >> delim);

        if (!success) {
            spdlog::info("Skipping line: \"{}\"", line);
            continue;
        }

        if (delim != 'd') {
            if (delim != 'e' && delim != 'c') {
                spdlog::info("Wrong deliminator: \"{}\"", line);
            }
            continue;
        }

        auto indices = std::vector<NeuronID>{};

        for (auto rank_neuron_string = std::string{}; sstream >> rank_neuron_string;) {
            const auto rank_neuron_vector = StringUtil::split_string(rank_neuron_string, ':');
            const auto rank = mpiPP::MPIRank{ std::stoi(rank_neuron_vector[0]) };
            const auto neuron_id = NeuronID(std::stoi(rank_neuron_vector[1]) - 1);

            if (rank == my_rank) {
                indices.push_back(neuron_id);
            }
        }

        return_value.emplace_back(step, std::move(indices));
    }

    return return_value;
}

std::vector<std::pair<InteractiveNeuronIO::step_type, InteractiveNeuronIO::number_neurons_type>>
InteractiveNeuronIO::load_creation_interrupts(const std::filesystem::path& path_to_file) {
    auto file = std::ifstream{ path_to_file };

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good,
                            "InteractiveNeuronIO::load_creation_interrupts: Opening the file '{}' was not successful", path_to_file);

    const auto parse_line = [](const auto& line) {
        std::stringstream sstream(line);

        auto step = step_type{};
        auto delim = char{};
        auto count = number_neurons_type{};

        const auto success = (sstream >> step) && (sstream >> delim) && (sstream >> count);

        return std::tuple{ success, step, delim, count };
    };

    const auto is_creation_delimiter = [](const auto& line) {
        const auto delim = std::get<3>(line);
        if (delim == 'c') {
            return true;
        }
        if (delim != 'e' && delim != 'd') {
            spdlog::warn("Wrong deliminator: \"{}\"", line);
        }
        return false;
    };

    const auto is_parsing_successfull = [](const auto& line) {
        auto success = std::get<0>(line);
        if (!success) {
            spdlog::warn("Skipping line: \"{}\"", line);
        }
        return success;
    };

    return ranges::getlines(file)
           | utility::views::filter_not_comment_not_empty_line
           | ranges::views::transform(parse_line)
           | ranges::views::cache1
           | ranges::views::filter(is_parsing_successfull)
           | ranges::views::filter(is_creation_delimiter)
           | ranges::views::transform([](const auto& values) { return std::pair{ std::get<1>(values), std::get<3>(values) }; })
           | ranges::to_vector;
}

RelearnTypes::stimuli_function_type InteractiveNeuronIO::load_stimulus_interrupts(
    const std::filesystem::path& path_to_file, const mpiPP::MPIRank my_rank,
    const std::shared_ptr<LocalGroupTranslator>& local_group_translator) {
    RelearnException::check(my_rank.is_initialized(),
                            "InteractiveNeuronIO::load_stimulus_interrupts: my_rank was virtual");

    auto file = std::ifstream{ path_to_file };

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good,
                            "InteractiveNeuronIO::load_stimulus_interrupts: Opening the file '{}' was not successful", path_to_file);

    using Stimulus = StimulusParser::Stimulus;

    const auto num_neurons = local_group_translator->get_number_neurons_in_total();

    const auto parse_line = [&my_rank](const auto& line) { return StimulusParser::parse_line(line, my_rank.get_rank()); };

    const auto check_neuron_ids = ranges::views::transform([num_neurons](Stimulus stimulus) {
        ranges::for_each(stimulus.matching_ids, [num_neurons](const auto& neuron_id) {
            RelearnException::check(neuron_id.get_neuron_id() < num_neurons, "InteractiveNeuronIO::load_stimulus_interrupts: Invalid neuron id {}", neuron_id);
        });
        return stimulus;
    });

    const auto consolidate_stimulus = [&local_group_translator](Stimulus stimulus) {
        const auto get_ids_in_group = [&local_group_translator](const auto& group) {
            const auto& group_ids = local_group_translator->get_group_ids_for_matching_group_names(group);
            return local_group_translator->get_neuron_ids_in_groups(group_ids);
        };

        auto ids = ranges::views::concat(
                       stimulus.matching_ids,
                       stimulus.matching_group_names
                           | ranges::views::for_each(get_ids_in_group))
                   | ranges::to<std::unordered_set>;

        return Stimulus{ stimulus.interval, stimulus.stimulus_intensity, std::move(ids), {} };
    };

    auto stimuli = ranges::getlines(file)
                   | utility::views::filter_not_comment_not_empty_line
                   | ranges::views::transform(parse_line)
                   | utility::views::optional_values
                   | check_neuron_ids
                   | ranges::views::transform(consolidate_stimulus)
                   | ranges::views::cache1
                   | ranges::views::filter(ranges::not_fn(ranges::empty), &Stimulus::matching_ids)
                   | ranges::to_vector;

    return StimulusParser::generate_stimulus_function(std::move(stimuli));
}
