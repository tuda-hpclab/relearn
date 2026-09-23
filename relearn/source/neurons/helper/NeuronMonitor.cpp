/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronMonitor.h"

#include "types/BasicTypes.h"
#include "util/RelearnException.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <variant>

void NeuronMonitor::set_output_path(std::filesystem::path path, const bool clear_contents) {
    output_path = std::move(path);

    if (clear_contents && mpiPP::MPIRank::root_rank() == mpiPP::MPIInfo::get_my_rank()) {
        auto error_code = std::error_code{};
        const auto number_files = std::filesystem::remove_all(output_path / "neuron_monitors", error_code);

        if (number_files == std::numeric_limits<decltype(number_files)>::max()) {
            std::cout << "Failed to remove the neuron monitors!\nGot error code: " << error_code.value() << "\n";
        }
    }
}

void NeuronMonitor::register_neuron(RelearnTypes::number_neurons_type neuron) {
    RelearnException::check(!output_path.empty(), "NeuronMonitor::register_neuron: Call set_output_path first");
    neurons_to_monitor.push_back(neuron);
}

void NeuronMonitor::record_data(const RelearnTypes::step_type current_step) {
    RelearnException::check(!output_path.empty(), "NeuronMonitor::register_neuron: Call set_output_path first");

    for (const auto& pre_processing : pre_processings) {
        pre_processing();
    }

    for (const auto neuron : neurons_to_monitor) {
        auto& information_for_neuron = information[neuron].emplace_back();
        information_for_neuron.reserve(callbacks.size());

        for (const auto& callback : callbacks) {
            information_for_neuron.push_back(callback(neuron));
        }
    }

    for (const auto& post_processing : post_processings) {
        post_processing();
    }

    steps.push_back(current_step);
}

void NeuronMonitor::flush_current_contents() {
    RelearnException::check(!output_path.empty(), "NeuronMonitor::register_neuron: Call set_output_path first");

    const auto& path = output_path / "neuron_monitors";
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    }

    for (const auto neuron : neurons_to_monitor) {
        const auto& file_path = path / (std::string(mpiPP::MPIInfo::get_my_rank_str()) + '_' + std::to_string(neuron + 1) + ".csv");
        if (!std::filesystem::exists(file_path)) {
            auto outfile = std::ofstream(file_path, std::ios_base::out);
            outfile << "# Step";
            for (const auto& paramter_name : parameter_names) {
                outfile << ';' << paramter_name;
            }
            outfile << '\n';
        }

        auto outfile = std::ofstream(file_path, std::ios_base::out | std::ios_base::app);

        auto& neuron_information = information[neuron];

        RelearnException::check(steps.size() == neuron_information.size(),
                                "NeuronMonitor::flush_current_contents: Sizes of the steps ({}) and the monitor for neuron {} do not match ({})",
                                steps.size(), neuron, neuron_information.size());

        for (auto i = std::size_t(0); i < steps.size(); ++i) {
            outfile << steps[i];
            for (const auto& parameter : neuron_information[i]) {
                std::visit([&outfile](const auto& param) { outfile << ';' << param; }, parameter);
            }
            outfile << '\n';
        }
    }

    steps.clear();
    information.clear();
}
