/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Naive.h"

#include "NaiveBase.h"

#include "algorithm/Algorithm.h"
#include "algorithm/CombinedAlgorithmsInternal/RequestEnums.h"
#include "algorithm/Connector.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/helper/SynapseCreationResponse.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/CommunicationTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/Timers.h"

#include <cpp-utility/ranges/Functional.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <range/v3/view/filter.hpp>

#include <algorithm>
#include <cstddef>
#include <tuple>

RelearnTypes::comm_map_creation<SynapseCreationRequest>
Naive::find_target_neurons(const number_neurons_type number_neurons) {
    const auto& global_tree = get_octree();
    const auto& node_cache = global_tree->get_cache();
    const auto& root = get_octree_root();

    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    const auto size_hint = std::min(static_cast<number_neurons_type>(number_ranks), number_neurons);
    auto synapse_creation_requests_outgoing = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks,
                                                                                                      size_hint);

    const auto signal_types = synaptic_elements->get_signal_types();
    const auto& vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    const auto& probability_kernel = *kernel;

    // For my neurons
    for (const auto id : NeuronIDRange::range(number_neurons) | ranges::views::filter(utility::equal_to(UpdateStatus::Enabled), utility::lookup(disable_flags, &NeuronID::get_neuron_id))) {
        const auto neuron_id = id.get_neuron_id();
        const auto dendrite_type_needed = signal_types[neuron_id];
        const auto number_vacant_axons = vacant_axons[neuron_id];
        if (number_vacant_axons == 0) {
            continue;
        }

        const auto& axon_position = extra_infos->get_position(id);

        const auto& requests = NaiveBase<NaiveCell>::find_target_neurons(probability_kernel, node_cache, id,
                                                                         axon_position, number_vacant_axons, root,
                                                                         dendrite_type_needed);
        for (const auto& [target_rank, creation_request] : requests) {
            synapse_creation_requests_outgoing.emplace_back(target_rank, creation_request);
        }
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    global_tree->clear_cache();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}

std::tuple<Algorithm::ResultType, RequestTypeEnum, DirectionEnum>
Naive::find_target_neurons_for_combined_algorithms(const std::vector<NeuronID>& neuron_ids) {
    const auto& global_tree = get_octree();
    const auto& node_cache = global_tree->get_cache();
    const auto& root = get_octree_root();

    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    const auto size_hint = std::min(static_cast<std::size_t>(number_ranks), neuron_ids.size());
    auto synapse_creation_requests_outgoing = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks,
                                                                                                      size_hint);

    const auto signal_types = synaptic_elements->get_signal_types();
    const auto& vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    const auto& probability_kernel = *kernel;

    // For my neurons
    for (const auto id : neuron_ids | ranges::views::filter(utility::equal_to(UpdateStatus::Enabled), utility::lookup(disable_flags, &NeuronID::get_neuron_id))) {
        const auto neuron_id = id.get_neuron_id();
        const auto dendrite_type_needed = signal_types[neuron_id];
        const auto number_vacant_axons = vacant_axons[neuron_id];
        if (number_vacant_axons == 0) {
            continue;
        }

        const auto& axon_position = extra_infos->get_position(id);

        const auto& requests = NaiveBase<NaiveCell>::find_target_neurons(probability_kernel, node_cache, id,
                                                                         axon_position, number_vacant_axons, root,
                                                                         dendrite_type_needed);
        for (const auto& [target_rank, creation_request] : requests) {
            synapse_creation_requests_outgoing.emplace_back(target_rank, creation_request);
        }
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    global_tree->clear_cache();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return { synapse_creation_requests_outgoing, RequestTypeEnum::SynapseCreationRequest, DirectionEnum::Forward };
}

ForwardProcessRequestsResult<SynapseCreationResponse>
Naive::process_requests(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests) {
    return ForwardConnector::process_requests(creation_requests, synaptic_elements);
}

PlasticDistantOutSynapses
Naive::process_responses(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                         const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses) {
    return ForwardConnector::process_responses(creation_requests, creation_responses, synaptic_elements);
}
