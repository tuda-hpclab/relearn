/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BarnesHutLocationAwareModified.h"

#include "algorithm/Algorithm.h"
#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/CombinedAlgorithmsInternal/RequestEnums.h"
#include "algorithm/Connector.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/DistantNeuronRequests.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/CommunicationTypes.h"
#include "types/SynapseTypes.h"
#include "util/MemoryHolder.h"
#include "util/NeuronID.h"
#include "util/ProbabilityPicker.h"
#include "util/RandomHolderKey.h"
#include "util/RelearnException.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/patterns/CommunicationMap.h>

#include <range/v3/algorithm/transform.hpp>
#include <range/v3/range/conversion.hpp>

#include <algorithm>
#include <span>
#include <tuple>

std::tuple<Algorithm::ResultType, RequestTypeEnum, DirectionEnum>
BarnesHutLocationAwareModified::find_target_neurons_for_combined_algorithms(const std::vector<NeuronID>& /* neuron_ids */) {
    RelearnException::fail("BarnesHutLocationAwareModified::find_target_neurons_for_combined_algorithms: This method should not be called for this algorithm");
}

RelearnTypes::comm_map_creation<DistantNeuronRequest> BarnesHutLocationAwareModified::find_target_neurons(const number_neurons_type number_neurons) {
    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto size_hint = std::min(number_neurons, static_cast<number_neurons_type>(number_ranks));
    auto neuron_requests_outgoing = RelearnTypes::comm_map_creation<DistantNeuronRequest>(number_ranks, size_hint);

    // prepare probability picking by making a map mapping local neuron_ids to rma_offsets of other ranks and their respective probability
    auto neuron_to_prob = std::unordered_map<NeuronID::value_type, std::tuple<std::vector<NeuronID::value_type>, std::vector<RelearnTypes::attraction_type>, std::vector<mpiPP::MPIRank>>>();

    const auto signal_types = synaptic_elements->get_signal_types();
    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    const auto& probability_kernel = (*(this->get_kernel()));
    const auto& memory_holder = get_octree()->get_memory_holder();

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(number_neurons, disable_flags, neuron_requests_outgoing, my_rank, neuron_to_prob, vacant_axons, signal_types, probability_kernel)
    for (auto neuron_id = number_neurons_type{ 0 }; neuron_id < number_neurons; ++neuron_id) { // step 1 // NOLINT(openmp-exception-escape) - only throwable calls are RelearnException::check on already-validated algorithm invariants
        if (disable_flags[neuron_id] != UpdateStatus::Enabled) {
            continue;
        }

        const auto id = NeuronID(neuron_id);

        const auto dendrite_type_needed = signal_types[neuron_id];
        const auto number_vacant_axons = vacant_axons[neuron_id];
        if (number_vacant_axons == 0) { // step 1
            continue;
        }

        const auto& axon_position = extra_infos->get_position(id);

        for (auto& branch_node : get_octree()->get_local_branch_nodes()) { // step 2
            if (branch_node->get_cell().get_number_dendrites_for(dendrite_type_needed) == 0) {
                continue;
            }

            if (BarnesHutBase<BarnesHutCell>::test_acceptance_criterion(axon_position, branch_node, ElementType::Dendrite, dendrite_type_needed, acceptance_criterion) == BarnesHutBase<BarnesHutCell>::AcceptanceStatus::Accept) { // step 3a
                const auto prob_to_connect = Kernel<AdditionalCellAttributes>::calculate_attractiveness_to_connect(probability_kernel, { my_rank, id }, axon_position, branch_node, ElementType::Dendrite, dendrite_type_needed);
                if (prob_to_connect == 0) {
                    continue;
                }
#pragma omp critical(remember_prob)
                {
                    auto& [rma_offsets, probs, ranks] = neuron_to_prob[neuron_id];
                    rma_offsets.push_back(branch_node->get_cell_neuron_id().get_rma_offset());
                    probs.push_back(prob_to_connect);
                    ranks.push_back(branch_node->get_mpi_rank());
                }
            } else { // step 3b
                const auto branch_node_index = branch_node->get_cell_neuron_id().get_rma_offset();
                const auto branch_node_rank = branch_node->get_mpi_rank();
                const auto request = DistantNeuronRequest(id, axon_position, branch_node_index, DistantNeuronRequest::TargetNeuronType::VirtualNode, dendrite_type_needed);
#pragma omp critical(BHrequests)
                neuron_requests_outgoing.append(branch_node_rank, request);
            }
        }
    }
    // step 3
    const auto& incoming_requests = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(neuron_requests_outgoing);

    // step 4
    // collect probabilities and remember connected neurons
    // first NeuronID::value_type is source_neuron_id.get_neuron_id() (i.e. a real neuron id) and second is rma_offset of local branch node
    auto outgoing_probabilities = RelearnTypes::comm_map_creation<std::tuple<NeuronID::value_type, NeuronID::value_type, RelearnTypes::attraction_type>>(number_ranks);
    // iterate over every rank
    for (auto& [source_rank, current_requests] : incoming_requests) {
        // iterate over every request
        for (const auto& current_request : current_requests) {
            const auto source_neuron_id = current_request.get_source_id();
            const auto signal_type = current_request.get_signal_type();

            const auto rma_offset = current_request.get_rma_offset();

            auto* const chosen_target = memory_holder->get_parent_from_offset(rma_offset); // parent or node itself?

            // get target through local barnes hut
            const auto source_position = current_request.get_source_position();

            const auto total_connection_probability = Kernel<AdditionalCellAttributes>::calculate_attractiveness_to_connect(probability_kernel, { source_rank, source_neuron_id }, source_position, chosen_target, ElementType::Dendrite, signal_type);
            if (total_connection_probability == RelearnTypes::attraction_type{ 0 }) {
                continue;
            }
            // remember probability corresponding to source_neuron_id and rma_offset
            outgoing_probabilities.append(source_rank, { source_neuron_id.get_neuron_id(), rma_offset, total_connection_probability });
        }
    }
    // step 5
    // get probabilities from other ranks
    const auto& incoming_probabilities = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(outgoing_probabilities);

    for (const auto& [rank, tuples] : incoming_probabilities) {
        for (const auto& [source_neuron_id, rma_offset, probability] : tuples) {
            auto& [rma_offsets, probs, ranks] = neuron_to_prob[source_neuron_id];
            rma_offsets.push_back(rma_offset);
            probs.push_back(probability);
            ranks.push_back(rank);
        }
    }

    // step 6
    auto neuron_requests_outgoing_final = RelearnTypes::comm_map_creation<DistantNeuronRequest>(number_ranks, size_hint);
#pragma omp parallel for default(none) shared(number_neurons, disable_flags, neuron_to_prob, neuron_requests_outgoing_final, vacant_axons, signal_types)
    for (auto neuron_id = number_neurons_type{ 0 }; neuron_id < number_neurons; ++neuron_id) { // NOLINT(openmp-exception-escape) - probability map is fully built and read-only here; ProbabilityPicker only throws on an empty span, which contains() guards against
        if (disable_flags[neuron_id] != UpdateStatus::Enabled) {
            continue;
        }

        if (!neuron_to_prob.contains(neuron_id)) {
            continue;
        }

        const auto id = NeuronID(neuron_id);

        const auto number_vacant_axons = vacant_axons[neuron_id];
        const auto& axon_position = extra_infos->get_position(id);
        const auto dendrite_type_needed = signal_types[neuron_id];
        // step 7
        // for every free axon
        for (auto i = 0U; i < number_vacant_axons; i++) {
            // step 8
            const auto& [rma_offsets, probabilities, ranks] = neuron_to_prob.at(neuron_id);
            const auto idx = ProbabilityPicker::pick_target(std::span{ probabilities.begin(), probabilities.end() }, RandomHolderKey::Algorithm);
            const auto& chosen_rma_offset = rma_offsets[idx];
            const auto& request = DistantNeuronRequest(id, axon_position, chosen_rma_offset, DistantNeuronRequest::TargetNeuronType::VirtualNode, dendrite_type_needed);
#pragma omp critical(BHrequests_final)
            neuron_requests_outgoing_final.append(ranks[idx], request);
        }
    }
    return neuron_requests_outgoing_final;
}

ForwardProcessRequestsResult<DistantNeuronResponse>
BarnesHutLocationAwareModified::process_requests(const RelearnTypes::comm_map_creation<DistantNeuronRequest>& neuron_requests) {
    const auto number_ranks = neuron_requests.get_number_ranks();

    const auto size_hint = neuron_requests.size();
    auto neuron_responses = RelearnTypes::comm_map_creation<DistantNeuronResponse>(number_ranks, size_hint);

    if (neuron_requests.empty()) {
        return { neuron_responses, 0, {} };
    }

    auto creation_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks, size_hint);
    creation_requests.resize(neuron_requests.get_request_sizes());

    const auto& memory_holder = get_octree()->get_memory_holder();
    const auto& node_cache = get_octree()->get_cache();
    const auto& probability_kernel = (*(this->get_kernel()));

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
            auto* const chosen_target = memory_holder->get_parent_from_offset(rma_offset);

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
    auto res = ForwardConnector::process_requests(creation_requests, synaptic_elements);

    // Translate the responses back by adding the found neuron id
    neuron_responses.resize(res.responses.get_request_sizes());

    for (const auto& [source_rank, responses] : res.responses) {
        const auto num_responses = responses.size();

        // All responses for a rank
        for (auto response_index = 0U; response_index < num_responses; response_index++) {
            const auto [target_neuron_id, source_neuron_id, dendrite_type_needed] = creation_requests.get_request(source_rank, response_index);
            const auto response = responses[response_index];

            neuron_responses.set_request(source_rank, response_index, DistantNeuronResponse{ target_neuron_id, response });
        }
    }

    return { neuron_responses, res.number_created_synapses, res.synapses };
}

PlasticDistantOutSynapses BarnesHutLocationAwareModified::process_responses(const RelearnTypes::comm_map_creation<DistantNeuronRequest>& neuron_requests,
                                                                            const RelearnTypes::comm_map_creation<DistantNeuronResponse>& neuron_responses) {

    RelearnException::check(neuron_requests.size() == neuron_responses.size(), "BarnesHutLocationAware::process_responses: Requests and Responses had different sizes");

    const auto number_ranks = neuron_requests.get_number_ranks();

    const auto size_hint = neuron_requests.size();
    auto creation_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks, size_hint);
    creation_requests.resize(neuron_requests.get_request_sizes());

    auto creation_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>(number_ranks, size_hint);
    creation_responses.resize(neuron_responses.get_request_sizes());

    for (const auto& [rank, requests] : neuron_requests) {
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
