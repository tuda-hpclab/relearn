/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "CalciumIO.h"

#include "io/parser/MonitorParser.h"
#include "neurons/LocalGroupTranslator.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/data/intersection.hpp>

#include <fmt/std.h>

#include <mpi-wrapper/core/MPIRank.h>

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

std::pair<CalciumIO::initial_value_calculator, CalciumIO::target_value_calculator> CalciumIO::load_initial_and_target_function(const std::filesystem::path& path_to_file, const std::shared_ptr<const LocalGroupTranslator>& local_group_translator, const mpiPP::MPIRank my_rank) {
    auto file = std::ifstream{ path_to_file };

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "InteractiveNeuronIO::load_enable_interrupts: Opening the file '{}' was not successful", path_to_file);

    auto default_initial_calcium = std::optional<calcium_type>{};
    auto default_target_calcium = std::optional<calcium_type>{};

    auto id_to_initial = std::unordered_map<NeuronID::value_type, calcium_type>{};
    auto id_to_target = std::unordered_map<NeuronID::value_type, calcium_type>{};

    auto already_seen_neuron_ids = std::unordered_set<NeuronID>{};

    for (auto line = std::string{}; std::getline(file, line);) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        auto sstream = std::stringstream(line);

        auto description = std::string{};
        auto initial_calcium = calcium_type{};
        auto target_calcium = calcium_type{};

        const auto success = (sstream >> description) && (sstream >> initial_calcium) && (sstream >> target_calcium);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        if (description == "default") {
            RelearnException::check(!default_initial_calcium.has_value(),
                                    "CalciumIO: {} had more than one default neuron (with id 0)", path_to_file.string());

            default_initial_calcium = initial_calcium;
            default_target_calcium = target_calcium;

            continue;
        }

        const auto& parsed_ids = MonitorParser::parse_my_ids(description, my_rank, local_group_translator);

        const auto id_already_seen = utility::containers_intersect(already_seen_neuron_ids, parsed_ids);

        RelearnException::check(!id_already_seen, "CalciumIO::load_initial_and_target_function: A neuron is not supposed to be assigned to multiple calcium value pairs. Each neuron should only get assigned to one!");

        already_seen_neuron_ids.insert(parsed_ids.begin(), parsed_ids.end());

        for (const auto& neuron_id : parsed_ids) {
            const auto local_neuron_id = neuron_id.get_neuron_id();
            const auto& found_initial = id_to_initial.find(local_neuron_id) != id_to_initial.end();
            const auto& found_target = id_to_target.find(local_neuron_id) != id_to_target.end();

            RelearnException::check(!found_initial && !found_target, "CalciumIO: Found the neuron id {} twice", (local_neuron_id + 1));

            id_to_initial[local_neuron_id] = initial_calcium;
            id_to_target[local_neuron_id] = target_calcium;
        }
    }

    auto initial_calculator = [lookup = std::move(id_to_initial), default_initial = default_initial_calcium]([[maybe_unused]] mpiPP::MPIRank mpi_rank, NeuronID::value_type neuron_id) {
        const auto& contains = lookup.find(neuron_id) != lookup.end();
        if (contains) {
            const calcium_type initial = lookup.at(neuron_id);
            return initial;
        }

        RelearnException::check(default_initial.has_value(), "Initial Calcium Calculator: Got id {} but I don't have a default value", neuron_id);
        return default_initial.value();
    };

    auto target_calculator = [lookup = std::move(id_to_target), default_target = default_target_calcium]([[maybe_unused]] mpiPP::MPIRank mpi_rank, NeuronID::value_type neuron_id) {
        const auto& contains = lookup.find(neuron_id) != lookup.end();
        if (contains) {
            const calcium_type target = lookup.at(neuron_id);
            return target;
        }

        RelearnException::check(default_target.has_value(), "Target Calcium Calculator: Got id {} but I don't have a default value", neuron_id);
        return default_target.value();
    };

    return std::make_pair(std::move(initial_calculator), std::move(target_calculator));
}
