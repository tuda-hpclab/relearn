/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticElementsIO.h"

#include "io/parser/MonitorParser.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

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

namespace {
/**
 * The values of one line of the file, in the order in which the file spells them.
 * Only the minimum calcium of each element type is a calcium concentration, the others are
 * continuous numbers of synaptic elements.
 */
struct ParsedValues {
    RelearnTypes::calcium_type min_calcium_axons{};
    RelearnTypes::grown_type nu_axons{};
    RelearnTypes::grown_type vacant_retract_ratio_axons{};
    RelearnTypes::grown_type min_elements_axons{};
    RelearnTypes::grown_type max_elements_axons{};

    RelearnTypes::calcium_type min_calcium_den_exc{};
    RelearnTypes::grown_type nu_den_exc{};
    RelearnTypes::grown_type vacant_retract_ratio_den_exc{};
    RelearnTypes::grown_type min_elements_den_exc{};
    RelearnTypes::grown_type max_elements_den_exc{};

    RelearnTypes::calcium_type min_calcium_den_inh{};
    RelearnTypes::grown_type nu_den_inh{};
    RelearnTypes::grown_type vacant_retract_ratio_den_inh{};
    RelearnTypes::grown_type min_elements_den_inh{};
    RelearnTypes::grown_type max_elements_den_inh{};
};
} // namespace

SynapticElementsIO::Calculators
SynapticElementsIO::load_function_from_file(const std::filesystem::path& path_to_file, mpiPP::MPIRank my_rank,
                                            const std::shared_ptr<const LocalGroupTranslator>& local_group_translator) {
    auto file = std::ifstream{ path_to_file };

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good,
                            "SynapticElementsIO::load_function_from_file: Opening the file '{}' was not successful", std::filesystem::absolute(path_to_file));

    auto default_value = std::optional<ParsedValues>{};

    auto id_to_values = std::unordered_map<RelearnTypes::number_neurons_type, ParsedValues>{};

    for (auto line = std::string{}; std::getline(file, line);) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        auto sstream = std::stringstream(line);

        auto description = std::string{};
        auto values = ParsedValues{};

        const auto success = (sstream >> description)
                             && (sstream >> values.min_calcium_axons) && (sstream >> values.nu_axons) && (sstream >> values.vacant_retract_ratio_axons) && (sstream >> values.min_elements_axons) && (sstream >> values.max_elements_axons)
                             && (sstream >> values.min_calcium_den_exc) && (sstream >> values.nu_den_exc) && (sstream >> values.vacant_retract_ratio_den_exc) && (sstream >> values.min_elements_den_exc) && (sstream >> values.max_elements_den_exc)
                             && (sstream >> values.min_calcium_den_inh) && (sstream >> values.nu_den_inh) && (sstream >> values.vacant_retract_ratio_den_inh) && (sstream >> values.min_elements_den_inh) && (sstream >> values.max_elements_den_inh);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        if (description == "default") {
            RelearnException::check(!default_value.has_value(),
                                    "CalciumIO: {} had more than one default neuron (with group default)",
                                    path_to_file.string());

            default_value = values;
            continue;
        }

        const auto& parsed_ids = MonitorParser::parse_my_ids(description, my_rank, local_group_translator);

        for (const auto& neuron_id : parsed_ids) {
            const auto local_neuron_id = neuron_id.get_neuron_id();
            const auto& found = id_to_values.find(local_neuron_id) != id_to_values.end();

            RelearnException::check(!found, "CalciumIO: Found the neuron id {} twice", (local_neuron_id + 1));

            id_to_values[local_neuron_id] = values;
        }
    }

    auto construct_calculator = [id_to_values, default_value](auto ParsedValues::* member) {
        return [id_to_values, default_value, member](const RelearnTypes::number_neurons_type neuron_id) {
            const auto it = id_to_values.find(neuron_id);
            if (it != id_to_values.end()) {
                return it->second.*member;
            }

            RelearnException::check(default_value.has_value(),
                                    "Synaptic Elements IO Calculator: Got id {} but I don't have a default value",
                                    neuron_id);
            return default_value.value().*member;
        };
    };

    return Calculators{
        .axons = {
            .min_calcium = construct_calculator(&ParsedValues::min_calcium_axons),
            .nu = construct_calculator(&ParsedValues::nu_axons),
            .vacant_retract_ratio = construct_calculator(&ParsedValues::vacant_retract_ratio_axons),
            .min_elements = construct_calculator(&ParsedValues::min_elements_axons),
            .max_elements = construct_calculator(&ParsedValues::max_elements_axons),
        },
        .dendrites_excitatory = {
            .min_calcium = construct_calculator(&ParsedValues::min_calcium_den_exc),
            .nu = construct_calculator(&ParsedValues::nu_den_exc),
            .vacant_retract_ratio = construct_calculator(&ParsedValues::vacant_retract_ratio_den_exc),
            .min_elements = construct_calculator(&ParsedValues::min_elements_den_exc),
            .max_elements = construct_calculator(&ParsedValues::max_elements_den_exc),
        },
        .dendrites_inhibitory = {
            .min_calcium = construct_calculator(&ParsedValues::min_calcium_den_inh),
            .nu = construct_calculator(&ParsedValues::nu_den_inh),
            .vacant_retract_ratio = construct_calculator(&ParsedValues::vacant_retract_ratio_den_inh),
            .min_elements = construct_calculator(&ParsedValues::min_elements_den_inh),
            .max_elements = construct_calculator(&ParsedValues::max_elements_den_inh),
        },
    };
}
