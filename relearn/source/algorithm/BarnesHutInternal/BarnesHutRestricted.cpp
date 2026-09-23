/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BarnesHutRestricted.h"

#include "algorithm/Algorithm.h"
#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/CombinedAlgorithmsInternal/RequestEnums.h"
#include "algorithm/Connector.h"
#include "io/Event.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/helper/SynapseCreationResponse.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/CommunicationTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/Random.h"
#include "util/RandomHolderKey.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/core/MPIRankRange.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <tuple>

std::tuple<Algorithm::ResultType, RequestTypeEnum, DirectionEnum>
BarnesHutRestricted::find_target_neurons_for_combined_algorithms(const std::vector<NeuronID>& /* neuron_ids */) {
    RelearnException::fail("BarnesHutRestricted::find_target_neurons_for_combined_algorithms: This method should not be called for this algorithm");
}

RelearnTypes::comm_map_creation<SynapseCreationRequest> BarnesHutRestricted::find_target_neurons(const number_neurons_type number_neurons) {
    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto size_hint = std::min(number_neurons, static_cast<number_neurons_type>(number_ranks));
    auto synapse_creation_requests_outgoing = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks, size_hint);

    auto* const root = get_octree_root();

    const auto signal_types = synaptic_elements->get_signal_types();
    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    const auto& probability_kernel = (*(this->get_kernel()));
    const auto& node_cache = get_octree()->get_cache();

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(root, my_rank, number_neurons, disable_flags, synapse_creation_requests_outgoing, signal_types, vacant_axons, node_cache, probability_kernel)
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

        RelearnException::check(possible_positions.size() > neuron_id, "BarnesHutRestricted::find_target_neurons: No position found for neuron {}", neuron_id);

        const auto& axon_positions = possible_positions[neuron_id];
        const auto number_axon_positions = axon_positions.size();

        RelearnException::check(number_axon_positions > 0, "BarnesHutRestricted::find_target_neurons: No axon positions found for neuron {}", neuron_id);

        const auto random_idx = RandomHolder::get_random_uniform_integer<std::size_t>(RandomHolderKey::Algorithm, 0, number_axon_positions - 1);

        const auto& axon_position = axon_positions[random_idx];

        const auto& requests = BarnesHutBase<BarnesHutCell>::find_target_neurons(probability_kernel, node_cache, { my_rank, id }, axon_position, number_vacant_axons, root, ElementType::Dendrite, dendrite_type_needed, acceptance_criterion);
        for (const auto& [target_rank, creation_request] : requests) {
#pragma omp critical(BHrequests)
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }

        Event::create_and_print_counter_event("BH", {}, { { "ID:", std::to_string(neuron_id) }, { "Cache:", std::to_string(node_cache.get_cache_size()) }, { "Memory:", std::to_string(node_cache.get_memory_size()) } }, true);
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    get_octree()->clear_cache();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}

ForwardProcessRequestsResult<SynapseCreationResponse> BarnesHutRestricted::process_requests(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests) {
    return ForwardConnector::process_requests(creation_requests, synaptic_elements);
}

PlasticDistantOutSynapses BarnesHutRestricted::process_responses(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests, const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses) {
    return ForwardConnector::process_responses(creation_requests, creation_responses, synaptic_elements);
}

void BarnesHutRestricted::update_possible_positions(const number_neurons_type number_additional_neurons) {
    RelearnException::check(network_graph != nullptr, "BarnesHutRestricted::update_possible_positions: network_graph was not set");

    const auto number_neurons_before = possible_positions.size();
    const auto number_neurons_now = number_neurons_before + number_additional_neurons;
    possible_positions.resize(number_neurons_now);

    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();
    const auto number_ranks_cast = mpiPP::MPIInfo::get_number_ranks_cast();

    auto position_translation = std::unordered_map<RankNeuronId, position_type>{};
    // This reservation is just a guess
    position_translation.reserve(number_neurons_now);

    // All distant neurons whose position we need to know
    auto position_requests = RelearnTypes::comm_map_creation<NeuronID>{ number_ranks, number_ranks_cast };
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_before, number_neurons_now)) {
        const auto& [distant_synapses, _] = network_graph->get_distant_out_edges(neuron_id.get_neuron_id());

        for (const auto& [rni, _w] : distant_synapses) {
            if (position_translation.find(rni) != position_translation.end()) {
                // It's enough to ask once
                continue;
            }

            position_translation[rni] = position_type{};
            position_requests.append(rni.get_rank(), rni.get_neuron_id());
        }
    }

    const auto received_requests = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(position_requests);

    auto position_responses = RelearnTypes::comm_map_creation<position_type>{ number_ranks, number_ranks_cast };
    position_responses.resize(received_requests.get_request_sizes());

    for (const auto mpi_rank : mpiPP::MPIRankRange::range(number_ranks)) {
        const auto size = received_requests.size(mpi_rank);
        if (size == 0) {
            continue;
        }

        const auto& requests = received_requests.get_requests(mpi_rank);
        for (auto i = std::size_t{ 0 }; i < requests.size(); i++) {
            const auto neuron_id = requests[i];
            const auto position = extra_infos->get_position(neuron_id);
            position_responses.set_request(mpi_rank, neuron_id.get_neuron_id(), position);
        }
    }

    const auto received_responses = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(position_responses);

    for (const auto mpi_rank : mpiPP::MPIRankRange::range(number_ranks)) {
        const auto size = position_requests.size(mpi_rank);
        if (size == 0) {
            continue;
        }

        const auto& requests = position_requests.get_requests(mpi_rank);
        const auto& responses = received_responses.get_requests(mpi_rank);
        for (auto i = std::size_t{ 0 }; i < received_responses.size(); i++) {
            const auto neuron_id = requests[i];
            const auto position = responses[i];
            position_translation[RankNeuronId{ mpi_rank, neuron_id }] = position;
        }
    }

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_before, number_neurons_now)) {
        const auto& [local_synapses, _1] = network_graph->get_local_out_edges(neuron_id.get_neuron_id());
        const auto& [distant_synapses, _2] = network_graph->get_distant_out_edges(neuron_id.get_neuron_id());

        auto& pos = possible_positions[neuron_id.get_neuron_id()];
        pos.reserve(local_synapses.size() + distant_synapses.size() + 1);

        const auto my_position = extra_infos->get_position(neuron_id);
        pos.emplace_back(my_position);

        for (const auto& [other_neuron, _w] : local_synapses) {
            if (other_neuron == neuron_id) {
                continue;
            }

            const auto position = extra_infos->get_position(other_neuron);
            pos.emplace_back(position);
        }

        for (const auto& [rni, _w] : distant_synapses) {
            const auto position = position_translation.at(rni);
            pos.emplace_back(position);
        }
    }
}
