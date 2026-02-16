/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronToSubdomainAssignment.h"

#include "Types.h"

#include "io/NeuronIO.h"
#include "neurons/LocalGroupTranslator.h"
#include "structure/Partition.h"
#include "util/NeuronFilePaths.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

void NeuronToSubdomainAssignment::initialize() {
    partition->set_boundary_correction_function(get_subdomain_boundary_fix());
    partition->calculate_and_set_subdomain_boundaries();

    fill_all_subdomains();

    initialized = true;

    const auto _number_local_neurons = get_number_neurons_in_subdomains();
    partition->set_number_local_neurons(_number_local_neurons);
}

void NeuronToSubdomainAssignment::initialize_groups_and_local_group_translator(std::optional<std::filesystem::path> opt_file_path) {
    if (opt_file_path.has_value()) {
        local_group_translator = std::make_shared<LocalGroupTranslator>(opt_file_path.value(), get_number_local_neurons());
    } else {
        local_group_translator = std::make_shared<LocalGroupTranslator>(get_number_local_neurons());
    }
}

void NeuronToSubdomainAssignment::write_neuron_positions_and_signals_to_file(const std::filesystem::path& file_path) const {
    NeuronIO::write_neuron_positions_and_signals(loaded_neurons, file_path, partition);
}

void NeuronToSubdomainAssignment::write_neuron_groups_to_file(const std::filesystem::path& file_path) const {
    NeuronIO::write_neuron_groups(file_path, local_group_translator);
}

void NeuronToSubdomainAssignment::write_neurons_to_files(const NeuronFilePaths& paths) const {
    NeuronIO::write_neurons(loaded_neurons, paths, local_group_translator, partition);
}
