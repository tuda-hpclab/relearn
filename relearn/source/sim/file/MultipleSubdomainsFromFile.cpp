/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MultipleSubdomainsFromFile.h"

#include "io/NeuronIO.h"
#include "sim/Essentials.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "sim/file/MultipleFilesSynapseLoader.h"
#include "structure/Partition.h"
#include "types/SpaceTypes.h"
#include "util/File.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <algorithm>
#include <filesystem>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

MultipleSubdomainsFromFile::MultipleSubdomainsFromFile(const std::filesystem::path& path_to_neurons,
                                                       std::optional<std::filesystem::path> path_to_synapses, const std::shared_ptr<Partition>& _partition)
    : NeuronToSubdomainAssignment(_partition) {
    const auto path_to_file = Util::find_file_for_rank(path_to_neurons, _partition->get_my_mpi_rank(), "rank_", "_positions.txt");

    read_neurons_from_file(path_to_file);
    synapse_loader = std::make_shared<MultipleFilesSynapseLoader>(_partition, std::move(path_to_synapses));
}

void MultipleSubdomainsFromFile::print_essentials(const std::unique_ptr<Essentials>& essentials) {
    essentials->insert("Neurons-Placed", get_total_number_placed_neurons());
}

void MultipleSubdomainsFromFile::read_neurons_from_file(const std::filesystem::path& path_to_neurons) {

    auto [nodes, additional_infos, additional_position_infos] = NeuronIO::read_neuron_positions_and_signals(path_to_neurons);

    const auto check = [](space_type value) -> bool {
        const auto min = mpiPP::MPIReductions::reduce_min(value);
        const auto max = mpiPP::MPIReductions::reduce_max(value);
        return min == max;
    };

    auto min_x = additional_position_infos.sim_size.get_minimum().get_x();
    auto min_y = additional_position_infos.sim_size.get_minimum().get_y();
    auto min_z = additional_position_infos.sim_size.get_minimum().get_z();
    auto max_x = additional_position_infos.sim_size.get_maximum().get_x();
    auto max_y = additional_position_infos.sim_size.get_maximum().get_y();
    auto max_z = additional_position_infos.sim_size.get_maximum().get_z();

    const auto all_same_min_x = check(utility::cast<double>(min_x));
    const auto all_same_min_y = check(utility::cast<double>(min_y));
    const auto all_same_min_z = check(utility::cast<double>(min_z));
    const auto all_same_max_x = check(utility::cast<double>(max_x));
    const auto all_same_max_y = check(utility::cast<double>(max_y));
    const auto all_same_max_z = check(utility::cast<double>(max_z));

    RelearnException::check(all_same_min_x, "MultipleSubdomainsFromFile::read_neurons_from_file: min_x is different across the ranks! Mine: {}", min_x);
    RelearnException::check(all_same_min_y, "MultipleSubdomainsFromFile::read_neurons_from_file: min_y is different across the ranks! Mine: {}", min_y);
    RelearnException::check(all_same_min_z, "MultipleSubdomainsFromFile::read_neurons_from_file: min_z is different across the ranks! Mine: {}", min_z);
    RelearnException::check(all_same_max_x, "MultipleSubdomainsFromFile::read_neurons_from_file: max_x is different across the ranks! Mine: {}", max_x);
    RelearnException::check(all_same_max_y, "MultipleSubdomainsFromFile::read_neurons_from_file: max_y is different across the ranks! Mine: {}", max_y);
    RelearnException::check(all_same_max_z, "MultipleSubdomainsFromFile::read_neurons_from_file: max_z is different across the ranks! Mine: {}", max_z);

    const auto minimum = RelearnTypes::position_type{ min_x, min_y, min_z };
    const auto maximum = RelearnTypes::position_type{ max_x, max_y, max_z };

    const auto& [_1, _2, loaded_ex_neurons, loaded_in_neurons] = additional_infos;
    const auto total_num_neurons = loaded_ex_neurons + loaded_in_neurons;

    RelearnException::check(additional_position_infos.local_neurons == total_num_neurons,
                            "MultipleSubdomainsFromFile::read_neurons_from_file: Number of loaded neurons does not equals commented number {} vs {}",
                            additional_position_infos.local_neurons, total_num_neurons);

    partition->set_simulation_box_size({ minimum, maximum });

    set_number_local_neurons(total_num_neurons);
    set_total_number_placed_neurons(total_num_neurons);
    set_requested_number_neurons(total_num_neurons);
    set_number_placed_neurons(total_num_neurons);

    const auto ratio_excitatory_neurons = static_cast<percentage_type>(loaded_ex_neurons) / static_cast<percentage_type>(total_num_neurons);

    set_requested_ratio_excitatory_neurons(ratio_excitatory_neurons);
    set_ratio_placed_excitatory_neurons(ratio_excitatory_neurons);

    partition->set_total_number_neurons(total_num_neurons);

    set_loaded_nodes(std::move(nodes));

    additional_position_information = additional_position_infos;
}

void MultipleSubdomainsFromFile::fill_all_subdomains() {
    RelearnException::check(additional_position_information.subdomain_sizes.size() == partition->get_number_local_subdomains(),
                            "MultipleSubdomainsFromFile::read_neurons_from_file:Number of subdomains {} in positions file is not equal to the actual number {}",
                            additional_position_information.subdomain_sizes.size(), partition->get_number_local_subdomains());

    auto num_subdomains = partition->get_number_local_subdomains();
    const auto sim_size = additional_position_information.sim_size;
    for (auto i = 0U; i < num_subdomains; i++) {
        auto subdomain_bb = partition->get_subdomain_boundaries(i);
        const auto is_within_eps = subdomain_bb.almost_equal(additional_position_information.subdomain_sizes[i], static_cast<space_type>(Constants::eps));
        RelearnException::check(is_within_eps, "MultipleSubdomainsFromFile::read_neurons_from_file: Wrong subdomain boundaries for subdomain {} on rank {}. Expected: {}, found: {}",
                                i, mpiPP::MPIInfo::get_my_rank(), subdomain_bb, additional_position_information.subdomain_sizes[i]);

        RelearnException::check(subdomain_bb.get_minimum().check_in_box(sim_size.get_minimum(), sim_size.get_maximum())
                                    && subdomain_bb.get_maximum().check_in_box(sim_size.get_minimum(), sim_size.get_maximum()),
                                "MultipleSubdomainsFromFile::read_neurons_from_file: Subdomain outside of simulation box");
    }

    for (const auto& node : loaded_neurons) {
        auto contains = false;
        for (auto i = 0U; i < num_subdomains; i++) {
            const auto& subdomain_bb = additional_position_information.subdomain_sizes[i];
            if (node.pos.check_in_box(subdomain_bb.get_minimum(), subdomain_bb.get_maximum())) {
                contains = true;
                break;
            }
        }
        RelearnException::check(contains, "MultipleSubdomainsFromFile::read_neurons_from_file: Neuron {} outside of subdomains", node.id);
    }

    auto positions = std::vector<position_type>{};
    positions.reserve(loaded_neurons.size());
    std::ranges::transform(loaded_neurons, std::back_inserter(positions), [](const LoadedNeuron& node) { return node.pos; });

    const auto positions_set = std::set<position_type>(positions.begin(), positions.end());
    RelearnException::check(positions.size() == positions_set.size(), "MultipleSubdomainsFromFile::read_neurons_from_file: Same position occurs multiple times");
}
