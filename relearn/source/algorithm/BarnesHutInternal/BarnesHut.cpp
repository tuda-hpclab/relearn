/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BarnesHut.h"

#include "algorithm/Algorithm.h"
#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/CombinedAlgorithmsInternal/RequestEnums.h"
#include "algorithm/Connector.h"
#include "io/Event.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/helper/SynapseCreationResponse.h"
#include "neurons/synaptic_elements/Axons.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/CommunicationTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronID.h"
#include "util/Timers.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <tuple>

RelearnTypes::comm_map_creation<SynapseCreationRequest>
BarnesHut::find_target_neurons(const number_neurons_type number_neurons) {
    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto size_hint = std::min(number_neurons, static_cast<number_neurons_type>(number_ranks));
    auto synapse_creation_requests_outgoing = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks,
                                                                                                      size_hint);

    const auto& global_tree = get_octree();
    const auto& node_cache = global_tree->get_cache();
    auto* const root = get_octree_root();

    const auto& axons = synaptic_elements->get_axons();
    const auto signal_types = axons->get_signal_types();
    const auto& vacant_axons = axons->get_vacant_elements();

    const auto& probability_kernel = *this->kernel;

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(probability_kernel, node_cache, root, my_rank, number_neurons, disable_flags, signal_types, axons, vacant_axons, synapse_creation_requests_outgoing)
    for (auto neuron_id = number_neurons_type{ 0 }; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] != UpdateStatus::Enabled) {
            continue;
        }

        const auto id = NeuronID{ neuron_id };

        const auto dendrite_type_needed = signal_types[neuron_id];
        const auto number_vacant_axons = vacant_axons[neuron_id];
        if (number_vacant_axons == 0) {
            continue;
        }

        const auto& axon_position = axons->get_bouton_position(neuron_id);

        const auto& requests = BarnesHutBase<BarnesHutCell>::find_target_neurons(probability_kernel, node_cache,
                                                                                 { my_rank, id }, axon_position,
                                                                                 number_vacant_axons, root,
                                                                                 ElementType::Dendrite,
                                                                                 dendrite_type_needed,
                                                                                 acceptance_criterion);
        for (const auto& [target_rank, creation_request] : requests) {
#pragma omp critical(BHrequests)
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }

        Event::create_and_print_counter_event("BH", {}, { { "ID:", std::to_string(neuron_id) }, { "Cache:", std::to_string(node_cache.get_cache_size()) }, { "Memory:", std::to_string(node_cache.get_memory_size()) } },
                                              true);
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    global_tree->clear_cache();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}

std::tuple<Algorithm::ResultType, RequestTypeEnum, DirectionEnum>
BarnesHut::find_target_neurons_for_combined_algorithms(const std::vector<NeuronID>& neuron_ids) {
    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto size_hint = std::min(neuron_ids.size(), static_cast<std::size_t>(number_ranks));
    auto synapse_creation_requests_outgoing = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks,
                                                                                                      size_hint);

    const auto& global_tree = get_octree();
    const auto& node_cache = global_tree->get_cache();
    auto* const root = get_octree_root();

    const auto& axons = synaptic_elements->get_axons();
    const auto signal_types = axons->get_signal_types();
    const auto& vacant_axons = axons->get_vacant_elements();

    const auto& probability_kernel = *this->kernel;

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(probability_kernel, node_cache, root, my_rank, neuron_ids, disable_flags, signal_types, axons, vacant_axons, synapse_creation_requests_outgoing)
    for (auto index = 0UL; index < neuron_ids.size(); ++index) {
        const auto id = neuron_ids[index];
        const auto neuron_id = id.get_neuron_id();
        if (disable_flags[neuron_id] != UpdateStatus::Enabled) {
            continue;
        }

        const auto dendrite_type_needed = signal_types[neuron_id];
        const auto number_vacant_axons = vacant_axons[neuron_id];
        if (number_vacant_axons == 0) {
            continue;
        }

        const auto& axon_position = axons->get_bouton_position(neuron_id);

        const auto& requests = BarnesHutBase<BarnesHutCell>::find_target_neurons(probability_kernel, node_cache,
                                                                                 { my_rank, id }, axon_position,
                                                                                 number_vacant_axons, root,
                                                                                 ElementType::Dendrite,
                                                                                 dendrite_type_needed,
                                                                                 acceptance_criterion);
        for (const auto& [target_rank, creation_request] : requests) {
#pragma omp critical(BHrequests)
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }

        Event::create_and_print_counter_event("BH", {}, { { "ID:", std::to_string(neuron_id) }, { "Cache:", std::to_string(node_cache.get_cache_size()) }, { "Memory:", std::to_string(node_cache.get_memory_size()) } },
                                              true);
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    global_tree->clear_cache();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return { synapse_creation_requests_outgoing, RequestTypeEnum::SynapseCreationRequest, DirectionEnum::Forward };
}

ForwardProcessRequestsResult<SynapseCreationResponse>
BarnesHut::process_requests(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests) {
    return ForwardConnector::process_requests(creation_requests, synaptic_elements);
}

PlasticDistantOutSynapses
BarnesHut::process_responses(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                             const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses) {
    return ForwardConnector::process_responses(creation_requests, creation_responses, synaptic_elements);
}