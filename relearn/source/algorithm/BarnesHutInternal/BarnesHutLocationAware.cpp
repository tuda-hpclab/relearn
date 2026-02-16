/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BarnesHutLocationAware.h"

#include "Types.h"

#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/Connector.h"
#include "algorithm/Internal/octree/NodeCache.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/DistantNeuronRequests.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "util/MemoryHolder.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include <algorithm>
#include <utility>

RelearnTypes::comm_map_creation<DistantNeuronRequest> BarnesHutLocationAware::find_target_neurons(const number_neurons_type number_neurons) {
    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto size_hint = std::min(number_neurons, static_cast<number_neurons_type>(number_ranks));
    auto neuron_requests_outgoing = RelearnTypes::comm_map_creation<DistantNeuronRequest>(number_ranks, size_hint);

    const auto& global_tree = get_octree();
    const auto& node_cache = global_tree->get_cache();

    auto* const root = get_octree_root();
    const auto level_of_branch_nodes = get_level_of_branch_nodes();

    const auto signal_types = synaptic_elements->get_signal_types();
    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    const auto& probability_kernel = *this->kernel;

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(probability_kernel, node_cache, root, my_rank, number_neurons, level_of_branch_nodes, disable_flags, signal_types, vacant_axons, neuron_requests_outgoing)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] != UpdateStatus::Enabled) {
            continue;
        }

        const auto id = NeuronID{ neuron_id };

        const auto dendrite_type_needed = signal_types[neuron_id];
        const auto number_vacant_axons = vacant_axons[neuron_id];
        if (number_vacant_axons == 0) {
            continue;
        }

        const auto& axon_position = extra_infos->get_position(id);

        const auto& requests = BarnesHutBase<BarnesHutCell>::find_target_neurons_location_aware(probability_kernel, node_cache, { my_rank, id }, axon_position, number_vacant_axons,
                                                                                                root, ElementType::Dendrite, dendrite_type_needed, level_of_branch_nodes, acceptance_criterion);

        for (const auto& [target_rank, creation_request] : requests) {
#pragma omp critical(BHrequests)
            neuron_requests_outgoing.append(target_rank, creation_request);
        }
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    global_tree->clear_cache();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return neuron_requests_outgoing;
}

std::tuple<Algorithm::ResultType, RequestTypeEnum, DirectionEnum> BarnesHutLocationAware::find_target_neurons_for_combined_algorithms(const std::vector<NeuronID>& neuron_ids) {
    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto size_hint = std::min(neuron_ids.size(), static_cast<std::size_t>(number_ranks));
    auto neuron_requests_outgoing = RelearnTypes::comm_map_creation<DistantNeuronRequest>(number_ranks, size_hint);

    const auto& global_tree = get_octree();
    const auto& node_cache = global_tree->get_cache();

    auto* const root = get_octree_root();
    const auto level_of_branch_nodes = get_level_of_branch_nodes();

    const auto signal_types = synaptic_elements->get_signal_types();
    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    const auto& probability_kernel = *this->kernel;

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(probability_kernel, node_cache, root, my_rank, neuron_ids, level_of_branch_nodes, disable_flags, signal_types, vacant_axons, neuron_requests_outgoing)
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

        const auto& axon_position = extra_infos->get_position(id);

        const auto& requests = BarnesHutBase<BarnesHutCell>::find_target_neurons_location_aware(probability_kernel, node_cache, { my_rank, id }, axon_position, number_vacant_axons,
                                                                                                root, ElementType::Dendrite, dendrite_type_needed, level_of_branch_nodes, acceptance_criterion);

        for (const auto& [target_rank, creation_request] : requests) {
#pragma omp critical(BHrequests)
            neuron_requests_outgoing.append(target_rank, creation_request);
        }
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    global_tree->clear_cache();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return { neuron_requests_outgoing, RequestTypeEnum::DistantNeuronRequest, DirectionEnum::Forward };
}

std::pair<RelearnTypes::comm_map_creation<DistantNeuronResponse>, std::pair<PlasticLocalSynapses, PlasticDistantInSynapses>>
BarnesHutLocationAware::process_requests(const RelearnTypes::comm_map_creation<DistantNeuronRequest>& neuron_requests) {
    const auto number_ranks = neuron_requests.get_number_ranks();

    const auto size_hint = neuron_requests.size();
    auto neuron_responses = RelearnTypes::comm_map_creation<DistantNeuronResponse>(number_ranks, size_hint);

    if (neuron_requests.empty()) {
        return { neuron_responses, {} };
    }

    const auto& node_cache = get_octree()->get_cache();
    const auto& memory_holder = get_octree()->get_memory_holder();

    auto creation_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks, size_hint);
    creation_requests.resize(neuron_requests.get_request_sizes());

    const auto& probability_kernel = *this->kernel;

    for (const auto& [source_rank, requests] : neuron_requests) {
        const auto num_requests = requests.size();

        // All requests from a rank
        for (auto request_index = 0U; request_index < num_requests; request_index++) {
            const auto& current_request = requests[request_index];

            const auto source_neuron_id = current_request.get_source_id();
            const auto signal_type = current_request.get_signal_type();
            const auto target_neuron_type = current_request.get_target_neuron_type();

            if (target_neuron_type == DistantNeuronRequest::TargetNeuronType::Leaf) {
                const auto target_id = current_request.get_leaf_node_id();
                const auto target_neuron_id = NeuronID{ target_id };
                creation_requests.set_request(source_rank, request_index, SynapseCreationRequest{ target_neuron_id, source_neuron_id, signal_type });
                continue;
            }

            const auto rma_offset = current_request.get_rma_offset();
            const auto idx_offset = rma_offset;

            auto* const chosen_target = memory_holder.get_parent_from_offset(idx_offset);

            // Otherwise get target through local barnes hut
            const auto source_position = current_request.get_source_position();

            // If the local search is successful, create a SynapseCreationRequest
            if (const auto& local_search = BarnesHutBase<BarnesHutCell>::find_target_neuron(probability_kernel, node_cache, { mpiPP::MPIRank(static_cast<int>(request_index)), source_neuron_id }, source_position, chosen_target, ElementType::Dendrite, signal_type, acceptance_criterion); local_search.has_value()) {
                const auto& [target_rank, target_neuron_id] = local_search.value();

                creation_requests.set_request(source_rank, request_index, SynapseCreationRequest{ target_neuron_id, source_neuron_id, signal_type });
            } else {
                creation_requests.set_request(source_rank, request_index, SynapseCreationRequest{ source_neuron_id, source_neuron_id, signal_type });
            }
        }
    }

    // Pass the translated requests to the forward connector
    auto [creation_responses, synapses] = ForwardConnector::process_requests(creation_requests, synaptic_elements);

    // Translate the responses back by adding the found neuron id
    neuron_responses.resize(creation_responses.get_request_sizes());

    for (const auto& [source_rank, responses] : creation_responses) {
        const auto num_responses = responses.size();

        // All responses for a rank
        for (auto response_index = 0U; response_index < num_responses; response_index++) {
            const auto [target_neuron_id, source_neuron_id, dendrite_type_needed] = creation_requests.get_request(source_rank, response_index);
            const auto response = responses[response_index];

            neuron_responses.set_request(source_rank, response_index, DistantNeuronResponse{ target_neuron_id, response });
        }
    }

    return std::make_pair(neuron_responses, synapses);
}

PlasticDistantOutSynapses BarnesHutLocationAware::process_responses(const RelearnTypes::comm_map_creation<DistantNeuronRequest>& neuron_requests,
                                                                    const RelearnTypes::comm_map_creation<DistantNeuronResponse>& neuron_responses) {

    RelearnException::check(neuron_requests.size() == neuron_responses.size(), "BarnesHutLocationAware::process_responses: Requests and Responses had different sizes");

    const auto number_ranks = neuron_requests.get_number_ranks();
    const auto size_hint = neuron_requests.size();

    auto creation_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks, size_hint);
    auto creation_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>(number_ranks, size_hint);

    const auto& sizes = neuron_requests.get_request_sizes();

    creation_requests.resize(sizes);
    creation_responses.resize(sizes);

    for (const auto& [rank, requests] : neuron_requests) {
        if (!sizes.contains(rank)) {
            continue;
        }

        const auto& responses = neuron_responses.get_requests(rank);

        for (auto index = 0U; index < requests.size(); index++) {
            const auto source_neuron_id = requests[index].get_source_id();
            const auto signal_type = requests[index].get_signal_type();
            const auto target_neuron_id = responses[index].get_source_id();
            const auto creation_response = responses[index].get_creation_response();

            if (creation_response == SynapseCreationResponse::Succeeded) {
                // If the creation succeeded set the corresponding target neuron
                creation_requests.set_request(rank, index, SynapseCreationRequest{ target_neuron_id, source_neuron_id, signal_type });
            } else {
                // Otherwise set the source as the target
                creation_requests.set_request(rank, index, SynapseCreationRequest{ source_neuron_id, source_neuron_id, signal_type });
            }

            creation_responses.set_request(rank, index, creation_response);
        }
    }

    return ForwardConnector::process_responses(creation_requests, creation_responses, synaptic_elements);
}
