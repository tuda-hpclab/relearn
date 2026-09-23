/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SubdomainFromNeuronPerRank.h"

#include "sim/Essentials.h"
#include "sim/LoadedNeuron.h"
#include "sim/random/BoxBasedRandomSubdomainAssignment.h"
#include "sim/random/RandomSynapseLoader.h"
#include "structure/Partition.h"
#include "util/Random.h"
#include "util/RandomHolderKey.h"
#include "util/RelearnException.h"

#include <range/v3/action/insert.hpp>

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

SubdomainFromNeuronPerRank::SubdomainFromNeuronPerRank(const SubdomainFromNeuronPerRank::number_neurons_type _number_neurons_per_rank,
                                                       const percentage_type fraction_excitatory_neurons, const space_type um_per_neuron, std::shared_ptr<Partition> _partition)
    : BoxBasedRandomSubdomainAssignment(_partition, fraction_excitatory_neurons, um_per_neuron)
    , number_neurons_per_rank(_number_neurons_per_rank) {

    RelearnException::check(_number_neurons_per_rank >= 1, "SubdomainFromNeuronPerRank::SubdomainFromNeuronPerRank: There must be at least one neuron per mpi rank!");

    const auto my_rank = static_cast<unsigned int>(_partition->get_my_mpi_rank().get_rank());
    const auto number_ranks = _partition->get_number_mpi_ranks();
    const auto number_local_subdomains = _partition->get_number_local_subdomains();

    RandomHolder::seed(RandomHolderKey::Subdomain, my_rank);

    const auto number_neurons = static_cast<number_neurons_type>(number_ranks) * _number_neurons_per_rank;
    const auto preliminary_number_neurons_per_subdomain = _number_neurons_per_rank / number_local_subdomains;
    const auto additional_neuron = (_number_neurons_per_rank % number_local_subdomains == 0) ? 0U : 1U;

    const auto number_neurons_per_subdomain = preliminary_number_neurons_per_subdomain + additional_neuron;

    // Calculate size of simulation box based on neuron density
    // number_neurons_per_subdomain^(1/3) == #neurons per dimension for one subdomain
    const auto number_boxes_per_subdomain_one_dimension = static_cast<number_neurons_type>(ceil(pow(static_cast<double>(number_neurons_per_subdomain), 1. / 3)));
    const auto number_boxes_one_dimension = _partition->get_number_subdomains_per_dimension() * number_boxes_per_subdomain_one_dimension;

    const auto simulation_box_length_ = static_cast<space_type>(number_boxes_one_dimension) * um_per_neuron;

    _partition->set_simulation_box_size({ { 0, 0, 0 }, position_type(simulation_box_length_) });

    set_number_local_neurons(_number_neurons_per_rank);
    set_requested_number_neurons(number_neurons);
    set_total_number_placed_neurons(number_neurons);

    set_number_placed_neurons(0);
    set_ratio_placed_excitatory_neurons(0.0);

    synapse_loader = std::make_shared<RandomSynapseLoader>(std::move(_partition));
}

void SubdomainFromNeuronPerRank::print_essentials(const std::unique_ptr<Essentials>& essentials) {
    essentials->insert("Neurons-Placed", get_total_number_placed_neurons());
    essentials->insert("Neurons-Placed-Per-Rank", number_neurons_per_rank);
}

void SubdomainFromNeuronPerRank::fill_all_subdomains() {
    RelearnException::check(!initialized, "SubdomainFromNeuronPerRank::fill_all_subdomains: The object is already initialized.");

    const auto number_local_subdomains = partition->get_number_local_subdomains();
    const auto preliminary_number_neurons_per_subdomain = number_neurons_per_rank / number_local_subdomains;

    number_neurons_type currently_placed_neurons = 0;
    number_neurons_type currently_placed_excitatory_neurons = 0;

    auto new_loaded_neurons = std::vector<LoadedNeuron>{};
    new_loaded_neurons.reserve(number_neurons_per_rank);

    for (auto i = 0U; i < number_local_subdomains; i++) {
        const auto additional_neuron = (i < number_neurons_per_rank % number_local_subdomains) ? 1U : 0U;
        const auto number_neurons_per_subdomain = preliminary_number_neurons_per_subdomain + additional_neuron;

        const auto& [min, max] = partition->get_subdomain_boundaries(i);

        auto [nodes, placed_excitatory_neurons] = place_neurons_in_box(min, max, number_neurons_per_subdomain, currently_placed_neurons);

        currently_placed_neurons += number_neurons_per_subdomain;
        currently_placed_excitatory_neurons += placed_excitatory_neurons;

        ranges::insert(new_loaded_neurons, new_loaded_neurons.end(), nodes);
    }

    set_loaded_nodes(std::move(new_loaded_neurons));

    set_number_placed_neurons(currently_placed_neurons);

    const auto fraction_excitatory_neurons = static_cast<percentage_type>(currently_placed_excitatory_neurons) / static_cast<percentage_type>(currently_placed_neurons);
    set_ratio_placed_excitatory_neurons(fraction_excitatory_neurons);
}
