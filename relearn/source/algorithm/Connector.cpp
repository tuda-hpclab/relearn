/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Connector.h"

#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/helper/SynapseCreationResponse.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/CommunicationTypes.h"
#include "types/SynapseTypes.h"
#include "util/Random.h"
#include "util/RandomHolderKey.h"
#include "util/RelearnException.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/core/MPIRankRange.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

ForwardProcessRequestsResult<SynapseCreationResponse>
ForwardConnector::process_requests(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                                   const std::shared_ptr<SynapticElements>& synaptic_elements) {

    const auto synaptic_elements_empty = synaptic_elements != nullptr;
    RelearnException::check(synaptic_elements_empty,
                            "ForwardConnector::process_requests: The synaptic elements are empty");

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();
    const auto number_ranks = creation_requests.get_number_ranks();
    const auto number_neurons = synaptic_elements->get_size();

    const auto size_hint = creation_requests.size();
    auto responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>(number_ranks, size_hint);

    if (creation_requests.empty()) {
        return { responses, 0, {} };
    }

    responses.resize(creation_requests.get_request_sizes());

    const auto total_number_requests = creation_requests.get_total_number_requests();
    auto created_synapses = std::uint64_t{};

    auto local_synapses = PlasticLocalSynapses{};
    local_synapses.reserve(total_number_requests);

    auto distant_synapses = PlasticDistantInSynapses{};
    distant_synapses.reserve(total_number_requests);

    auto indices = std::vector<std::tuple<mpiPP::MPIRank, std::size_t, NeuronID>>{};
    indices.reserve(creation_requests.get_total_number_requests());
    for (const auto& [source_rank, requests] : creation_requests) {
        for (auto request_index = 0U; request_index < requests.size(); request_index++) {
            const auto& request = requests[request_index];
            indices.emplace_back(source_rank, request_index, request.get_target());
        }
    }

    RandomHolder::shuffle(RandomHolderKey::Connector, indices);
    // We need to shuffle the request indices so we do not prefer those from smaller MPI ranks and lower neuron ids
    // const auto indices = creation_requests
    //                     | ranges::views::for_each([](const auto& request) {
    //                           const auto& [source_rank, requests] = request;
    //                           return ranges::views::zip(
    //                               ranges::views::repeat(source_rank),
    //                               ranges::views::indices(ranges::size(requests)));
    //                       })
    //                     | ranges::to<std::vector<std::pair<mpiPP::MPIRank, std::size_t>>>
    //                     | RandomHolder::shuffleAction(RandomHolderKey::Connector);

    for (const auto& [source_rank, request_index, target_neuron] : indices) {
        const auto& [target_neuron_id, source_neuron_id, dendrite_type_needed] = creation_requests.get_request(
            source_rank, request_index);

        if (source_rank == my_rank && target_neuron_id == source_neuron_id) {
            responses.set_request(source_rank, request_index, SynapseCreationResponse::Failed);
            continue;
        }

        RelearnException::check(target_neuron_id.get_neuron_id() < number_neurons,
                                "ForwardConnector::process_requests: target_neuron_id exceeds my neurons");

        const auto synaptic_elements_type = get_synaptic_element_type(ElementType::Dendrite, dendrite_type_needed);
        const auto number_free_elements = synaptic_elements->get_vacant_elements(
            synaptic_elements_type)[target_neuron_id.get_neuron_id()];
        if (number_free_elements == 0) {
            // Other axons were faster and came first
            responses.set_request(source_rank, request_index, SynapseCreationResponse::Failed);
            continue;
        }

        // Increment number of connected dendrites
        synaptic_elements->connect_elements(1, target_neuron_id.get_neuron_id(), synaptic_elements_type);

        // Set response to "connected" (success)
        responses.set_request(source_rank, request_index, SynapseCreationResponse::Succeeded);

        const auto weight = (SignalType::Inhibitory == dendrite_type_needed) ? -1 : 1;
        if (source_rank == my_rank) {
            local_synapses.emplace_back(target_neuron_id, source_neuron_id, weight);
            continue;
        }

        distant_synapses.emplace_back(target_neuron_id, RankNeuronId{ source_rank, source_neuron_id }, weight);
    }
    created_synapses = local_synapses.size() + distant_synapses.size();
    return { responses, created_synapses, { local_synapses, distant_synapses } };
}

PlasticDistantOutSynapses
ForwardConnector::process_responses(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                                    const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses,
                                    const std::shared_ptr<SynapticElements>& synaptic_elements) {

    const auto synaptic_elements_empty = synaptic_elements != nullptr;
    RelearnException::check(synaptic_elements_empty,
                            "ForwardConnector::process_responses: The synaptic elements are empty");

    RelearnException::check(creation_requests.get_number_ranks() == creation_responses.get_number_ranks(),
                            "ForwardConnector::process_responses: Requests and Responses had a different number of ranks");
    RelearnException::check(creation_requests.size() == creation_responses.size(),
                            "ForwardConnector::process_responses: Requests and Responses had different sizes");

    if (creation_requests.empty()) {
        return {};
    }

    RelearnException::check(creation_requests.size() < std::numeric_limits<int>::max(),
                            "ForwardConnector::process_responses: Too many requests: {}", creation_requests.size());

    for (const auto rank : mpiPP::MPIRankRange::range(creation_requests.get_number_ranks())) {
        RelearnException::check(creation_requests.size(rank) == creation_responses.size(rank),
                                "ForwardConnector::process_responses: Requests and Responses for rank {} had different sizes",
                                rank);
    }

    const auto number_neurons = synaptic_elements->get_size();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto total_number_responses = creation_requests.get_total_number_requests();

    auto synapses = PlasticDistantOutSynapses{};
    synapses.reserve(total_number_responses);

    // Process the responses of all mpi ranks
    for (const auto& [target_rank, requests] : creation_responses) {
        const auto num_requests = requests.size();

        // All responses from a rank
        for (auto request_index = 0U; request_index < num_requests; request_index++) {
            const auto connected = requests[request_index];
            if (connected == SynapseCreationResponse::Failed) {
                continue;
            }

            const auto& [target_neuron_id, source_neuron_id, dendrite_type_needed] = creation_requests.get_request(
                target_rank, request_index);

            RelearnException::check(source_neuron_id.get_neuron_id() < number_neurons,
                                    "ForwardConnector::process_responses: The source neuron id was too large: {} vs {}",
                                    source_neuron_id.get_neuron_id(), number_neurons);

            const auto number_free_elements = synaptic_elements->get_vacant_elements(
                SynapticElementType::Axon)[source_neuron_id.get_neuron_id()];
            RelearnException::check(number_free_elements > 0,
                                    "ForwardConnector::process_responses: The source neuron did not have a vacant element: {}",
                                    source_neuron_id);

            // Increment number of connected axons
            synaptic_elements->connect_elements(1, source_neuron_id.get_neuron_id(), SynapticElementType::Axon);

            if (target_rank == my_rank) {
                // I have already created the synapse in the network if the response comes from myself
                continue;
            }

            // Mark this synapse for later use (must be added to the network graph)
            const auto weight = (SignalType::Inhibitory == dendrite_type_needed) ? -1 : +1;
            synapses.emplace_back(RankNeuronId{ target_rank, target_neuron_id }, source_neuron_id, weight);
        }
    }

    return synapses;
}

BackwardProcessRequestsResult<SynapseCreationResponse>
BackwardConnector::process_requests(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                                    const std::shared_ptr<SynapticElements>& synaptic_elements) {

    const auto synaptic_elements_empty = synaptic_elements != nullptr;
    RelearnException::check(synaptic_elements_empty,
                            "BackwardConnector::process_requests: The synaptic elements are empty");

    const auto number_neurons = synaptic_elements->get_size();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();
    const auto number_ranks = creation_requests.get_number_ranks();

    const auto size_hint = creation_requests.size();
    auto responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>(number_ranks, size_hint);
    if (creation_requests.empty()) {
        return { responses, 0, {} };
    }

    responses.resize(creation_requests.get_request_sizes());

    const auto total_number_requests = creation_requests.get_total_number_requests();

    auto local_synapses = PlasticLocalSynapses{};
    local_synapses.reserve(total_number_requests);

    auto distant_synapses = PlasticDistantOutSynapses{};
    distant_synapses.reserve(total_number_requests);

    auto indices = std::vector<std::pair<mpiPP::MPIRank, std::size_t>>{};
    indices.reserve(creation_requests.get_total_number_requests());

    for (const auto& [source_rank, requests] : creation_requests) {
        for (auto request_index = 0U; request_index < requests.size(); request_index++) {
            indices.emplace_back(source_rank, request_index);
        }
    }

    // We need to shuffle the request indices so we do not prefer those from smaller MPI ranks and lower neuron ids
    RandomHolder::shuffle(RandomHolderKey::Connector, indices);

    // const auto indices = creation_requests
    //                      | ranges::views::for_each([](const auto& request) {
    //                            const auto& [source_rank, requests] = request;
    //                            return ranges::views::zip(
    //                                ranges::views::repeat(source_rank),
    //                                ranges::views::iota(0U, ranges::size(requests)));
    //                        })
    //                      | ranges::to<std::vector<std::pair<mpiPP::MPIRank, unsigned int>>>
    //                      | RandomHolder::shuffleAction(RandomHolderKey::Connector);

    const auto& signal_types = synaptic_elements->get_signal_types();

    for (const auto& [source_rank, request_index] : indices) {
        const auto& [target_neuron_id, source_neuron_id, axon_type_needed] = creation_requests.get_request(source_rank,
                                                                                                           request_index);

        RelearnException::check(target_neuron_id.get_neuron_id() < number_neurons,
                                "BackwardConnector::process_requests: target_neuron_id exceeds my neurons");
        RelearnException::check(signal_types[target_neuron_id.get_neuron_id()] == axon_type_needed,
                                "BackwardConnector::process_requests: Request had the wrong signal type");

        const auto weight = (SignalType::Inhibitory == axon_type_needed) ? -1 : 1;
        const auto number_free_elements = synaptic_elements->get_vacant_elements(
            SynapticElementType::Axon)[target_neuron_id.get_neuron_id()];

        if (number_free_elements == 0) {
            // Other axons were faster and came first
            responses.set_request(source_rank, request_index, SynapseCreationResponse::Failed);
            continue;
        }

        // Increment number of connected dendrites
        synaptic_elements->connect_elements(1, target_neuron_id.get_neuron_id(), SynapticElementType::Axon);

        // Set response to "connected" (success)
        responses.set_request(source_rank, request_index, SynapseCreationResponse::Succeeded);

        if (source_rank == my_rank) {
            local_synapses.emplace_back(source_neuron_id, target_neuron_id, weight);
            continue;
        }

        distant_synapses.emplace_back(RankNeuronId{ source_rank, source_neuron_id }, target_neuron_id, weight);
    }

    const auto created_synapses = local_synapses.size() + distant_synapses.size();
    return { responses, created_synapses, { local_synapses, distant_synapses } };
}

PlasticDistantInSynapses
BackwardConnector::process_responses(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                                     const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses,
                                     const std::shared_ptr<SynapticElements>& synaptic_elements) {

    const auto synaptic_elements_empty = synaptic_elements != nullptr;
    RelearnException::check(synaptic_elements_empty,
                            "BackwardConnector::process_responses: The synaptic elements are empty");

    RelearnException::check(creation_requests.get_number_ranks() == creation_responses.get_number_ranks(),
                            "BackwardConnector::process_responses: Requests and Responses had a different number of ranks");
    RelearnException::check(creation_requests.size() == creation_responses.size(),
                            "BackwardConnector::process_responses: Requests and Responses had different sizes");

    if (creation_requests.empty()) {
        return {};
    }

    RelearnException::check(creation_requests.size() < std::numeric_limits<int>::max(),
                            "BackwardConnector::process_responses: Too many requests: {}", creation_requests.size());

    for (const auto rank : mpiPP::MPIRankRange::range(creation_requests.get_number_ranks())) {
        RelearnException::check(creation_requests.size(rank) == creation_responses.size(rank),
                                "BackwardConnector::process_responses: Requests and Responses for rank {} had different sizes",
                                rank);
    }

    const auto number_neurons = synaptic_elements->get_size();

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto total_number_responses = creation_requests.get_total_number_requests();

    auto synapses = PlasticDistantInSynapses{};
    synapses.reserve(total_number_responses);

    // Process the responses of all mpi ranks
    for (const auto& [target_rank, requests] : creation_responses) {
        const auto num_requests = requests.size();

        // All responses from a rank
        for (auto request_index = 0U; request_index < num_requests; request_index++) {
            const auto connected = requests[request_index];
            if (connected == SynapseCreationResponse::Failed) {
                continue;
            }

            const auto& [target_neuron_id, source_neuron_id, axon_type_needed] = creation_requests.get_request(
                target_rank, request_index);

            RelearnException::check(source_neuron_id.get_neuron_id() < number_neurons,
                                    "BackwardConnector::process_responses: The source neuron id was too large: {} vs {}",
                                    source_neuron_id.get_neuron_id(), number_neurons);

            const auto synaptic_element_type = get_synaptic_element_type(ElementType::Dendrite, axon_type_needed);
            // Increment number of connected dendrites
            synaptic_elements->connect_elements(1, source_neuron_id.get_neuron_id(), synaptic_element_type);

            if (target_rank == my_rank) {
                // I have already created the synapse in the network if the response comes from myself
                continue;
            }

            // Mark this synapse for later use (must be added to the network graph)
            const auto weight = (SignalType::Inhibitory == axon_type_needed) ? -1 : +1;
            synapses.emplace_back(source_neuron_id, RankNeuronId{ target_rank, target_neuron_id }, weight);
        }
    }

    return synapses;
}
