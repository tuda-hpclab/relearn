/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticElementsIO.h"

#include "Types2.h"

#include "io/parser/MonitorParser.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "mpi-wrapper/MPIRank.h"

#include <spdlog/spdlog.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

std::array<SynapticElementsIO::neuron_id_to_calcium_calculator, 15>
SynapticElementsIO::load_function_from_file(const std::filesystem::path& path_to_file, mpiPP::MPIRank my_rank,
                                            const std::shared_ptr<const LocalGroupTranslator>& local_group_translator) {
    auto file = std::ifstream{ path_to_file };

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good,
                            "SynapticElementsIO::load_function_from_file: Opening the file '{}' was not successful", std::filesystem::absolute(path_to_file));

    auto default_value = std::optional<std::array<RelearnTypes::calcium_type, 15>>{};

    auto id_to_values = std::unordered_map<RelearnTypes::number_neurons_type, std::array<RelearnTypes::calcium_type, 15>>{};

    for (auto line = std::string{}; std::getline(file, line);) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        auto sstream = std::stringstream(line);

        auto description = std::string{};
        auto min_calcium_axons = double{};
        auto nu_axons = double{};
        auto vacant_retract_ratio_axons = double{};
        auto min_calcium_den_exc = double{};
        auto nu_den_exc = double{};
        auto vacant_retract_ratio_den_exc = double{};
        auto min_calcium_den_inh = double{};
        auto nu_den_inh = double{};
        auto vacant_retract_ratio_den_inh = double{};
        auto min_elements_axons = double{};
        auto max_elements_axons = double{};
        auto min_elements_den_exc = double{};
        auto max_elements_den_exc = double{};
        auto min_elements_den_inh = double{};
        auto max_elements_den_inh = double{};

        const auto success = (sstream >> description)
                             && (sstream >> min_calcium_axons) && (sstream >> nu_axons) && (sstream >> vacant_retract_ratio_axons) && (sstream >> min_elements_axons) && (sstream >> max_elements_axons)
                             && (sstream >> min_calcium_den_exc) && (sstream >> nu_den_exc) && (sstream >> vacant_retract_ratio_den_exc) && (sstream >> min_elements_den_exc) && (sstream >> max_elements_den_exc)
                             && (sstream >> min_calcium_den_inh) && (sstream >> nu_den_inh) && (sstream >> vacant_retract_ratio_den_inh) && (sstream >> min_elements_den_inh) && (sstream >> max_elements_den_inh);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        if (description == "default") {
            RelearnException::check(!default_value.has_value(),
                                    "CalciumIO: {} had more than one default neuron (with group default)",
                                    path_to_file.string());

            default_value = { min_calcium_axons, nu_axons, vacant_retract_ratio_axons, min_elements_axons, max_elements_axons,
                              min_calcium_den_exc, nu_den_exc, vacant_retract_ratio_den_exc, min_elements_den_exc, max_elements_den_exc,
                              min_calcium_den_inh, nu_den_inh, vacant_retract_ratio_den_inh, min_elements_den_inh, max_elements_den_inh };
            continue;
        }

        const auto& parsed_ids = MonitorParser::parse_my_ids(description, my_rank, local_group_translator);

        for (const auto& neuron_id : parsed_ids) {
            const auto local_neuron_id = neuron_id.get_neuron_id();
            const auto& found = id_to_values.find(local_neuron_id) != id_to_values.end();

            RelearnException::check(!found, "CalciumIO: Found the neuron id {} twice", (local_neuron_id + 1));

            id_to_values[local_neuron_id] = { min_calcium_axons, nu_axons, vacant_retract_ratio_axons, min_elements_axons, max_elements_axons,
                                              min_calcium_den_exc, nu_den_exc, vacant_retract_ratio_den_exc, min_elements_den_exc, max_elements_den_exc,
                                              min_calcium_den_inh, nu_den_inh, vacant_retract_ratio_den_inh, min_elements_den_inh, max_elements_den_inh };
        }
    }

    auto construct_calculator = [id_to_values, default_value](auto index) {
        return [id_to_values, default_value, index](const RelearnTypes::number_neurons_type neuron_id) {
            const auto it = id_to_values.find(neuron_id);
            if (it != id_to_values.end()) {
                const double value = it->second[index];
                return value;
            }

            RelearnException::check(default_value.has_value(),
                                    "Synaptic Elements IO Calculator: Got id {} but I don't have a default value",
                                    neuron_id);
            return default_value.value()[index];
        };
    };

    return {
        construct_calculator(0U), construct_calculator(1U), construct_calculator(2U), construct_calculator(3U), construct_calculator(4U),
        construct_calculator(5U), construct_calculator(6U), construct_calculator(7U), construct_calculator(8U), construct_calculator(9U),
        construct_calculator(10U), construct_calculator(11U), construct_calculator(12U), construct_calculator(13U), construct_calculator(14U)
    };
}
