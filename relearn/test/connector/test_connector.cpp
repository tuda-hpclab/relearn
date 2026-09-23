/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_connector.h"

#include "algorithm/Connector.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/CommunicationTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"

#include "adapter/synaptic_elements/SynapticElementsAdapter.h"

#include "factory/connector/connector_factory.h"
#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/synaptic_elements/synaptic_elements_factory.h"

#include <cpp-utility/data/vectorify.hpp>

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/core/MPIRankRange.h>

#include <range/v3/algorithm/sort.hpp>
#include <range/v3/view/indices.hpp>

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

TEST_F(ConnectorTest, testForwardConnectorProcessRequestsExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_1 = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_neurons_2 = NeuronIdFactory::get_random_number_neurons(mt);

    const auto final_number_neurons = number_neurons_1 == number_neurons_2 ? number_neurons_2 + 1 : number_neurons_2;

    const auto number_ranks_1 = MPIRankFactory::get_random_number_ranks(mt) + 1;
    const auto number_ranks_2 = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto final_number_ranks = number_ranks_1 == static_cast<int>(number_neurons_2) ? static_cast<int>(number_neurons_2) + 1 : number_ranks_2;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(final_number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    const auto empty = std::shared_ptr<SynapticElements>{ nullptr };

    const auto incoming_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks_1 };

    ASSERT_THROW_NO_PRINT(std::ignore = ForwardConnector::process_requests(incoming_requests, empty), RelearnException);

    const auto incoming_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>{ number_ranks_1 };
    ASSERT_THROW_NO_PRINT(std::ignore = ForwardConnector::process_responses(incoming_requests, incoming_responses, empty), RelearnException);

    const auto wrong_incoming_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>{ final_number_ranks };
    ASSERT_THROW_NO_PRINT(std::ignore = ForwardConnector::process_responses(incoming_requests, wrong_incoming_responses, empty), RelearnException);
}

TEST_F(ConnectorTest, testForwardConnectorProcessRequestsEmptyMap) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    // The following copies are intentional
    const auto previous_connected_excitatory = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::DendriteExcitatory));
    const auto previous_grown_excitatory = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::DendriteExcitatory));
    const auto previous_deltas_excitatory = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::DendriteExcitatory));

    const auto previous_connected_inhibitory = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::DendriteInhibitory));
    const auto previous_grown_inhibitory = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::DendriteInhibitory));
    const auto previous_deltas_inhibitory = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::DendriteInhibitory));

    const auto incoming_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };

    const auto [responses, created_synapses, synapses] = ForwardConnector::process_requests(incoming_requests, synaptic_elements);
    const auto [local_synapses, distant_in_synapses] = synapses;

    ASSERT_EQ(responses.size(), incoming_requests.size());
    ASSERT_EQ(responses.get_number_ranks(), incoming_requests.get_number_ranks());
    ASSERT_EQ(responses.get_total_number_requests(), 0);

    ASSERT_TRUE(local_synapses.empty());
    ASSERT_TRUE(distant_in_synapses.empty());

    const auto& now_connected_excitatory = synaptic_elements->get_connected_elements(SynapticElementType::DendriteExcitatory);
    const auto& now_grown_excitatory = synaptic_elements->get_grown_elements(SynapticElementType::DendriteExcitatory);
    const auto& now_deltas_excitatory = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::DendriteExcitatory));

    const auto& now_connected_inhibitory = synaptic_elements->get_connected_elements(SynapticElementType::DendriteInhibitory);
    const auto& now_grown_inhibitory = synaptic_elements->get_grown_elements(SynapticElementType::DendriteInhibitory);
    const auto& now_deltas_inhibitory = synaptic_elements->get_deltas(SynapticElementType::DendriteInhibitory);

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        ASSERT_EQ(previous_connected_excitatory[neuron_id], now_connected_excitatory[neuron_id]);
        ASSERT_EQ(previous_grown_excitatory[neuron_id], now_grown_excitatory[neuron_id]);
        ASSERT_EQ(previous_deltas_excitatory[neuron_id], now_deltas_excitatory[neuron_id]);

        ASSERT_EQ(previous_connected_inhibitory[neuron_id], now_connected_inhibitory[neuron_id]);
        ASSERT_EQ(previous_grown_inhibitory[neuron_id], now_grown_inhibitory[neuron_id]);
        ASSERT_EQ(previous_deltas_inhibitory[neuron_id], now_deltas_inhibitory[neuron_id]);
    }
}

TEST_F(ConnectorTest, testForwardConnectorProcessRequestsMatchingRequests) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    auto incoming_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };

    auto number_excitatory_requests = 0U;
    auto number_inhibitory_requests = 0U;

    auto excitatory_requests = std::map<NeuronID, std::vector<SynapseCreationRequest>>{};
    auto inhibitory_requests = std::map<NeuronID, std::vector<SynapseCreationRequest>>{};

    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    for (const auto& id : NeuronIDRange::range(number_neurons)) {
        const auto number_vacant_excitatory = vacant_excitatory_dendrites[id.get_neuron_id()];
        number_excitatory_requests += number_vacant_excitatory;

        for (const auto neuron_id : NeuronIDRange::range(number_vacant_excitatory)) {
            const auto scr = SynapseCreationRequest(id, neuron_id, SignalType::Excitatory);
            incoming_requests.append(mpiPP::MPIRank(1), scr);

            excitatory_requests[id].emplace_back(scr);
        }

        const auto number_vacant_inhibitory = vacant_inhibitory_dendrites[id.get_neuron_id()];
        number_inhibitory_requests += number_vacant_inhibitory;

        for (const auto neuron_id : NeuronIDRange::range(number_vacant_inhibitory)) {
            const auto scr = SynapseCreationRequest(id, neuron_id, SignalType::Inhibitory);
            incoming_requests.append(mpiPP::MPIRank(1), scr);

            inhibitory_requests[id].emplace_back(scr);
        }
    }

    const auto [responses, created_synapses, synapses] = ForwardConnector::process_requests(incoming_requests, synaptic_elements);
    const auto [local_synapses, distant_in_synapses] = synapses;

    ASSERT_EQ(incoming_requests.size(), responses.size());

    const auto& request_sizes = incoming_requests.get_request_sizes();
    const auto& response_sizes = responses.get_request_sizes();

    // For each saved rank: The number of responses matches the number of requests
    ASSERT_EQ(request_sizes.size(), response_sizes.size());
    for (const auto& [rank, size] : request_sizes) {
        const auto found_in_responses = response_sizes.contains(rank);
        ASSERT_TRUE(found_in_responses);

        ASSERT_EQ(size, response_sizes.at(rank));
    }

    for (const auto& [rank, resps] : responses) {
        for (const auto resp : resps) {
            ASSERT_EQ(resp, SynapseCreationResponse::Succeeded);
        }
    }

    ASSERT_EQ(local_synapses.size(), 0);
    ASSERT_EQ(distant_in_synapses.size(), number_excitatory_requests + number_inhibitory_requests);

    for (const auto& id : NeuronIDRange::range(number_neurons)) {
        const auto number_vacant_excitatory = vacant_excitatory_dendrites[id.get_neuron_id()];
        ASSERT_EQ(number_vacant_excitatory, 0);

        const auto number_vacant_inhibitory = vacant_inhibitory_dendrites[id.get_neuron_id()];
        ASSERT_EQ(number_vacant_inhibitory, 0);
    }

    for (const auto& [target_id, source_id, weight] : distant_in_synapses) {
        const auto& [source_rank, source_neuron_id] = source_id;

        ASSERT_EQ(source_rank, mpiPP::MPIRank(1));
        ASSERT_EQ(std::abs(weight), 1);
    }
}

TEST_F(ConnectorTest, testForwardConnectorProcessRequestsSelfRequests) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    // The following copies are intentional
    const auto previous_connected_excitatory = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::DendriteExcitatory));
    const auto previous_grown_excitatory = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::DendriteExcitatory));
    const auto previous_deltas_excitatory = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::DendriteExcitatory));

    const auto previous_connected_inhibitory = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::DendriteInhibitory));
    const auto previous_grown_inhibitory = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::DendriteInhibitory));
    const auto previous_deltas_inhibitory = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::DendriteInhibitory));

    auto incoming_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };

    auto excitatory_requests = std::map<NeuronID, std::vector<SynapseCreationRequest>>{};
    auto inhibitory_requests = std::map<NeuronID, std::vector<SynapseCreationRequest>>{};

    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    for (const auto& id : NeuronIDRange::range(number_neurons)) {
        const auto number_vacant_excitatory = vacant_excitatory_dendrites[id.get_neuron_id()];

        for (auto i = 0U; i < number_vacant_excitatory; i++) {
            const auto scr = SynapseCreationRequest(id, id, SignalType::Excitatory);
            incoming_requests.append(mpiPP::MPIRank(0), scr);

            excitatory_requests[id].emplace_back(scr);
        }

        const auto number_vacant_inhibitory = vacant_inhibitory_dendrites[id.get_neuron_id()];

        for (auto i = 0U; i < number_vacant_inhibitory; i++) {
            const auto scr = SynapseCreationRequest(id, id, SignalType::Inhibitory);
            incoming_requests.append(mpiPP::MPIRank(0), scr);

            inhibitory_requests[id].emplace_back(scr);
        }
    }

    auto [responses, created_synapses, synapses] = ForwardConnector::process_requests(incoming_requests, synaptic_elements);
    const auto [local_synapses, distant_in_synapses] = synapses;

    ASSERT_EQ(incoming_requests.size(), responses.size());

    const auto& request_sizes = incoming_requests.get_request_sizes();
    const auto& response_sizes = responses.get_request_sizes();

    // For each saved rank: The number of responses matches the number of requests
    ASSERT_EQ(request_sizes.size(), response_sizes.size());
    for (const auto& [rank, size] : request_sizes) {
        const auto found_in_responses = response_sizes.contains(rank);
        ASSERT_TRUE(found_in_responses);

        ASSERT_EQ(size, response_sizes.at(rank));
    }

    for (const auto& [rank, resps] : responses) {
        for (const auto resp : resps) {
            ASSERT_EQ(resp, SynapseCreationResponse::Failed);
        }
    }

    ASSERT_TRUE(local_synapses.empty());
    ASSERT_TRUE(distant_in_synapses.empty());

    const auto& now_connected_excitatory = synaptic_elements->get_connected_elements(SynapticElementType::DendriteExcitatory);
    const auto& now_grown_excitatory = synaptic_elements->get_grown_elements(SynapticElementType::DendriteExcitatory);
    const auto& now_deltas_excitatory = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::DendriteExcitatory));

    const auto& now_connected_inhibitory = synaptic_elements->get_connected_elements(SynapticElementType::DendriteInhibitory);
    const auto& now_grown_inhibitory = synaptic_elements->get_grown_elements(SynapticElementType::DendriteInhibitory);
    const auto& now_deltas_inhibitory = synaptic_elements->get_deltas(SynapticElementType::DendriteInhibitory);

    for (auto i = 0U; i < number_neurons; i++) {
        ASSERT_EQ(previous_connected_excitatory[i], now_connected_excitatory[i]);
        ASSERT_EQ(previous_grown_excitatory[i], now_grown_excitatory[i]);
        ASSERT_EQ(previous_deltas_excitatory[i], now_deltas_excitatory[i]);

        ASSERT_EQ(previous_connected_inhibitory[i], now_connected_inhibitory[i]);
        ASSERT_EQ(previous_grown_inhibitory[i], now_grown_inhibitory[i]);
        ASSERT_EQ(previous_deltas_inhibitory[i], now_deltas_inhibitory[i]);
    }
}

TEST_F(ConnectorTest, testForwardConnectorProcessRequestsIncoming) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 5;
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    // The following copies are intentional
    const auto previous_connected_excitatory = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::DendriteExcitatory));
    const auto previous_grown_excitatory = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::DendriteExcitatory));

    const auto previous_connected_inhibitory = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::DendriteInhibitory));
    const auto previous_grown_inhibitory = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::DendriteInhibitory));

    const auto& [incoming_requests, number_excitatory_requests, number_inhibitory_requests]
        = ConnectorFactory::create_incoming_requests(number_ranks, 0, number_neurons, 0, 9, mt);

    const auto [responses, created_synapses, synapses] = ForwardConnector::process_requests(incoming_requests, synaptic_elements);
    auto [local_synapses, distant_in_synapses] = synapses;

    // There are as many requests as responses
    ASSERT_EQ(incoming_requests.size(), responses.size());

    const auto& request_sizes = incoming_requests.get_request_sizes();
    const auto& response_sizes = responses.get_request_sizes();

    // For each saved rank: The number of responses matches the number of requests
    ASSERT_EQ(request_sizes.size(), response_sizes.size());
    for (const auto& [rank, size] : request_sizes) {
        const auto found_in_responses = response_sizes.contains(rank);
        ASSERT_TRUE(found_in_responses);

        ASSERT_EQ(size, response_sizes.at(rank));
    }

    const auto& now_connected_excitatory = synaptic_elements->get_connected_elements(SynapticElementType::DendriteExcitatory);
    const auto& now_grown_excitatory = synaptic_elements->get_grown_elements(SynapticElementType::DendriteExcitatory);

    const auto& now_connected_inhibitory = synaptic_elements->get_connected_elements(SynapticElementType::DendriteInhibitory);
    const auto& now_grown_inhibitory = synaptic_elements->get_grown_elements(SynapticElementType::DendriteInhibitory);

    auto newly_connected_excitatory_dendrites = std::vector<unsigned int>(number_neurons, 0);
    auto newly_connected_inhibitory_dendrites = std::vector<unsigned int>(number_neurons, 0);

    // The grown elements did not change. There are now not less connected then before, and not more than grown
    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        ASSERT_EQ(previous_grown_excitatory[neuron_id], now_grown_excitatory[neuron_id]) << neuron_id;
        ASSERT_EQ(previous_grown_inhibitory[neuron_id], now_grown_inhibitory[neuron_id]) << neuron_id;

        ASSERT_GE(now_connected_excitatory[neuron_id], previous_connected_excitatory[neuron_id]) << neuron_id;
        ASSERT_GE(now_connected_inhibitory[neuron_id], previous_connected_inhibitory[neuron_id]) << neuron_id;

        ASSERT_LE(now_connected_excitatory[neuron_id], static_cast<unsigned int>(now_grown_excitatory[neuron_id])) << neuron_id;
        ASSERT_LE(now_connected_inhibitory[neuron_id], static_cast<unsigned int>(now_grown_inhibitory[neuron_id])) << neuron_id;

        newly_connected_excitatory_dendrites[neuron_id] = now_connected_excitatory[neuron_id] - previous_connected_excitatory[neuron_id];
        newly_connected_inhibitory_dendrites[neuron_id] = now_connected_inhibitory[neuron_id] - previous_connected_inhibitory[neuron_id];
    }

    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    // If there are still vacant elements, then all requests are connected
    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto vacant_excitatory_elements = vacant_excitatory_dendrites[neuron_id.get_neuron_id()];
        if (vacant_excitatory_elements > 0) {
            ASSERT_EQ(newly_connected_excitatory_dendrites[neuron_id.get_neuron_id()], number_excitatory_requests[neuron_id.get_neuron_id()]) << neuron_id;
        }

        if (const auto vacant_inhibitory_elements = vacant_inhibitory_dendrites[neuron_id.get_neuron_id()];
            vacant_inhibitory_elements > 0) {
            ASSERT_EQ(newly_connected_inhibitory_dendrites[neuron_id.get_neuron_id()], number_inhibitory_requests[neuron_id.get_neuron_id()]) << neuron_id;
        }
    }

    auto accepted_excitatory_requests = std::vector<unsigned int>(number_neurons, 0);
    auto accepted_inhibitory_requests = std::vector<unsigned int>(number_neurons, 0);

    auto expected_local_synapses = PlasticLocalSynapses{};
    auto expected_distant_in_synapses = PlasticDistantInSynapses{};

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    // Extract things from the return value
    for (const auto rank : mpiPP::MPIRankRange::range(number_ranks)) {
        const auto found_in_requests = request_sizes.contains(rank);
        if (!found_in_requests) {
            continue;
        }

        for (auto index = 0U; index < request_sizes.at(rank); index++) {
            const auto& [target_index, source_index, signal_type] = incoming_requests.get_request(rank, index);
            const auto& response = responses.get_request(rank, index);

            if (response == SynapseCreationResponse::Failed) {
                continue;
            }

            const auto& target_id = target_index.get_neuron_id();

            if (signal_type == SignalType::Excitatory) {
                accepted_excitatory_requests[target_id]++;
            } else {
                accepted_inhibitory_requests[target_id]++;
            }

            const auto weight = signal_type == SignalType::Excitatory ? 1 : -1;

            if (rank.get_rank() == my_rank.get_rank()) {
                expected_local_synapses.emplace_back(target_index, source_index, weight);
            } else {
                expected_distant_in_synapses.emplace_back(target_index, RankNeuronId{ rank, source_index }, weight);
            }
        }
    }

    // The sizes of the return values match the number of accepted responses
    ASSERT_EQ(local_synapses.size(), expected_local_synapses.size());
    ASSERT_EQ(distant_in_synapses.size(), expected_distant_in_synapses.size());

    ranges::sort(local_synapses);
    ranges::sort(distant_in_synapses);
    ranges::sort(expected_local_synapses);
    ranges::sort(expected_distant_in_synapses);

    // All and only the accepted local synapses are returned
    for (const auto neuron_id : ranges::views::indices(local_synapses.size())) {
        const auto& [target_1, source_1, weight_1] = local_synapses[neuron_id];
        const auto& [target_2, source_2, weight_2] = expected_local_synapses[neuron_id];

        ASSERT_EQ(target_1, target_2) << neuron_id;
        ASSERT_EQ(source_1, source_2) << neuron_id;
        ASSERT_EQ(weight_1, weight_2) << neuron_id;
    }

    // All and only the accepted distant in synapses are returned
    for (const auto neuron_id : ranges::views::indices(distant_in_synapses.size())) {
        const auto& [target_1, source_1, weight_1] = distant_in_synapses[neuron_id];
        const auto& [target_2, source_2, weight_2] = expected_distant_in_synapses[neuron_id];

        ASSERT_EQ(target_1, target_2) << neuron_id;
        ASSERT_EQ(source_1, source_2) << neuron_id;
        ASSERT_EQ(weight_1, weight_2) << neuron_id;
    }

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        ASSERT_EQ(accepted_excitatory_requests[neuron_id], newly_connected_excitatory_dendrites[neuron_id]) << neuron_id;
        ASSERT_EQ(accepted_inhibitory_requests[neuron_id], newly_connected_inhibitory_dendrites[neuron_id]) << neuron_id;
    }
}

TEST_F(ConnectorTest, testForwardConnectorProcessResponsesExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    const auto number_ranks_requests = MPIRankFactory::get_random_number_ranks(mt) + 1;
    auto number_ranks_responses = MPIRankFactory::get_random_number_ranks(mt) + 1;
    if (number_ranks_responses == number_ranks_requests) {
        number_ranks_responses++;
    }

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    const auto empty = std::shared_ptr<SynapticElements>{ nullptr };

    const auto outgoing_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks_requests };
    const auto wrong_outgoing_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks_responses };
    const auto incoming_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>{ number_ranks_requests };
    const auto wrong_incoming_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>{ number_ranks_responses };

    ASSERT_THROW_NO_PRINT(std::ignore = ForwardConnector::process_responses(outgoing_requests, incoming_responses, empty), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = ForwardConnector::process_responses(wrong_outgoing_requests, wrong_incoming_responses, empty), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = ForwardConnector::process_responses(outgoing_requests, wrong_incoming_responses, empty), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = ForwardConnector::process_responses(wrong_outgoing_requests, incoming_responses, empty), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = ForwardConnector::process_responses(outgoing_requests, wrong_incoming_responses, synaptic_elements), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = ForwardConnector::process_responses(wrong_outgoing_requests, incoming_responses, synaptic_elements), RelearnException);
}

TEST_F(ConnectorTest, testForwardConnectorProcessResponsesDifferentSizes) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    auto outgoing_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };
    auto incoming_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>{ number_ranks };

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    for (const auto id : NeuronIDRange::range(number_neurons)) {
        const auto signal_type = signal_types[id.get_neuron_id()];
        const auto free_elements = vacant_axons[id.get_neuron_id()];

        for (auto i = 0U; i < free_elements; i++) {
            const auto scr = SynapseCreationRequest(id, NeuronID{ i }, signal_type);
            outgoing_requests.append(mpiPP::MPIRank(1), scr);

            incoming_responses.append(mpiPP::MPIRank(1), SynapseCreationResponse::Failed);
        }
    }

    if (outgoing_requests.empty()) {
        return;
    }

    ASSERT_THROW_NO_PRINT(std::ignore = ForwardConnector::process_responses(outgoing_requests,
                                                                            RelearnTypes::comm_map_creation<SynapseCreationResponse>{ number_ranks }, synaptic_elements);
                          , RelearnException);

    for (auto i = 0U; i < 10U; i++) {
        incoming_responses.append(mpiPP::MPIRank(1), SynapseCreationResponse::Failed);
        ASSERT_THROW_NO_PRINT(std::ignore = ForwardConnector::process_responses(outgoing_requests, incoming_responses, synaptic_elements);, RelearnException);
    }
}

TEST_F(ConnectorTest, testForwardConnectorProcessResponsesAllException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    // The following copies are intentional
    const auto previous_connected_axon = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::Axon));
    const auto previous_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));
    const auto previous_deltas_axon = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::Axon));

    auto outgoing_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };
    auto incoming_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>{ number_ranks };

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    for (const auto id : NeuronIDRange::range(number_neurons)) {
        const auto signal_type = signal_types[id.get_neuron_id()];
        const auto free_elements = vacant_axons[id.get_neuron_id()];

        for (auto i = 0U; i < free_elements; i++) {
            const auto scr = SynapseCreationRequest(id, NeuronID{ i }, signal_type);
            outgoing_requests.append(mpiPP::MPIRank(1), scr);

            incoming_responses.append(mpiPP::MPIRank(1), SynapseCreationResponse::Failed);
        }
    }

    if (outgoing_requests.empty()) {
        return;
    }

    const auto synapses = ForwardConnector::process_responses(outgoing_requests, incoming_responses, synaptic_elements);

    ASSERT_TRUE(synapses.empty());

    const auto now_connected_axon = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::Axon));
    const auto now_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));
    const auto now_deltas_axon = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::Axon));

    ASSERT_EQ(previous_connected_axon, now_connected_axon);
    ASSERT_EQ(previous_grown_axon, now_grown_axon);
    ASSERT_EQ(previous_deltas_axon, now_deltas_axon);
}

TEST_F(ConnectorTest, testForwardConnectorProcessResponsesAllSuccessful) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    // The following copies are intentional
    const auto previous_connected_axon = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::Axon));
    const auto previous_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));
    const auto previous_deltas_axon = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::Axon));

    auto outgoing_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };
    auto incoming_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>{ number_ranks };

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    for (const auto id : NeuronIDRange::range(number_neurons)) {
        const auto signal_type = signal_types[id.get_neuron_id()];
        const auto free_elements = vacant_axons[id.get_neuron_id()];

        for (auto i = 0U; i < free_elements; i++) {
            const auto scr = SynapseCreationRequest(NeuronID{ i }, id, signal_type);
            outgoing_requests.append(mpiPP::MPIRank(1), scr);

            incoming_responses.append(mpiPP::MPIRank(1), SynapseCreationResponse::Succeeded);
        }
    }

    if (outgoing_requests.empty()) {
        return;
    }

    const auto synapses = ForwardConnector::process_responses(outgoing_requests, incoming_responses, synaptic_elements);

    ASSERT_EQ(synapses.size(), outgoing_requests.get_total_number_requests());

    const auto now_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));
    ASSERT_EQ(previous_grown_axon, now_grown_axon);

    const auto deltas = synaptic_elements->get_deltas(SynapticElementType::Axon);
    for (const auto id : NeuronIDRange::range(number_neurons)) {
        ASSERT_LT(deltas[id.get_neuron_id()], 1.0);
    }

    auto map = std::map<NeuronID, unsigned int>{};
    for (const auto& [target, source_id, weight] : synapses) {
        const auto& [target_rank, target_id] = target;

        ASSERT_EQ(target_rank, mpiPP::MPIRank(1));
        map[source_id] += static_cast<unsigned int>(std::abs(weight));

        if (signal_types[source_id.get_neuron_id()] == SignalType::Excitatory) {
            ASSERT_GT(weight, 0);
        } else {
            ASSERT_LT(weight, 0);
        }
    }

    for (const auto id : NeuronIDRange::range(number_neurons)) {
        ASSERT_EQ(synaptic_elements->get_connected_elements(SynapticElementType::Axon)[id.get_neuron_id()], map[id] + previous_connected_axon[id.get_neuron_id()]);
    }
}

TEST_F(ConnectorTest, testForwardConnectorProcessResponsesAllLocal) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    // The following copies are intentional
    const auto previous_connected_axon = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::Axon));
    const auto previous_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));
    const auto previous_deltas_axon = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::Axon));

    auto outgoing_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };
    auto incoming_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>{ number_ranks };

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    for (const auto id : NeuronIDRange::range(number_neurons)) {
        const auto signal_type = signal_types[id.get_neuron_id()];
        const auto free_elements = vacant_axons[id.get_neuron_id()];

        for (auto i = 0U; i < free_elements; i++) {
            const auto scr = SynapseCreationRequest(NeuronID{ i }, id, signal_type);
            outgoing_requests.append(mpiPP::MPIRank(0), scr);

            incoming_responses.append(mpiPP::MPIRank(0), SynapseCreationResponse::Succeeded);
        }
    }

    if (outgoing_requests.empty()) {
        return;
    }

    const auto synapses = ForwardConnector::process_responses(outgoing_requests, incoming_responses, synaptic_elements);
    ASSERT_TRUE(synapses.empty());

    const auto now_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));
    ASSERT_EQ(previous_grown_axon, now_grown_axon);

    const auto deltas = synaptic_elements->get_deltas(SynapticElementType::Axon);
    for (const auto id : NeuronIDRange::range(number_neurons)) {
        ASSERT_LT(deltas[id.get_neuron_id()], 1.0);
    }
}

TEST_F(ConnectorTest, testForwardConnectorProcessResponsesEmpty) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    // The following copies are intentional
    const auto previous_connected_axon = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::Axon));
    const auto previous_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));
    const auto previous_deltas_axon = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::Axon));

    auto outgoing_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };
    auto incoming_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>{ number_ranks };

    const auto synapses = ForwardConnector::process_responses(outgoing_requests, incoming_responses, synaptic_elements);

    ASSERT_TRUE(synapses.empty());

    const auto now_connected_axon = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::Axon));
    const auto now_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));
    const auto now_deltas_axon = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::Axon));

    ASSERT_EQ(previous_connected_axon, now_connected_axon);
    ASSERT_EQ(previous_grown_axon, now_grown_axon);
    ASSERT_EQ(previous_deltas_axon, now_deltas_axon);
}

TEST_F(ConnectorTest, testForwardConnectorProcessResponsesTooLargeIds) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    // The following copies are intentional
    const auto previous_connected_axon = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::Axon));
    const auto previous_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));
    const auto previous_deltas_axon = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::Axon));

    auto outgoing_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };
    auto incoming_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>{ number_ranks };

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    for (const auto id : NeuronIDRange::range(number_neurons)) {
        const auto signal_type = signal_types[id.get_neuron_id()];
        const auto free_elements = vacant_axons[id.get_neuron_id()];

        for (auto i = 0U; i < free_elements; i++) {
            const auto scr = SynapseCreationRequest(NeuronID{ i }, NeuronID(number_neurons + id.get_neuron_id()), signal_type);
            outgoing_requests.append(mpiPP::MPIRank(1), scr);

            incoming_responses.append(mpiPP::MPIRank(1), SynapseCreationResponse::Succeeded);
        }
    }

    if (outgoing_requests.empty()) {
        return;
    }

    ASSERT_THROW_NO_PRINT(std::ignore = ForwardConnector::process_responses(outgoing_requests, incoming_responses, synaptic_elements);, RelearnException);
}

TEST_F(ConnectorTest, testForwardConnectorProcessResponsesDoubleException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    // The following copies are intentional
    const auto previous_connected_axon = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::Axon));
    const auto previous_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));
    const auto previous_deltas_axon = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::Axon));

    auto outgoing_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };
    auto incoming_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>{ number_ranks };

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    for (const auto id : NeuronIDRange::range(number_neurons)) {
        const auto signal_type = signal_types[id.get_neuron_id()];
        const auto free_elements = vacant_axons[id.get_neuron_id()];

        for (auto i = 0U; i < free_elements; i++) {
            const auto scr = SynapseCreationRequest(NeuronID{ i }, id, signal_type);
            outgoing_requests.append(mpiPP::MPIRank(1), scr);
            outgoing_requests.append(mpiPP::MPIRank(1), scr);

            incoming_responses.append(mpiPP::MPIRank(1), SynapseCreationResponse::Succeeded);
            incoming_responses.append(mpiPP::MPIRank(1), SynapseCreationResponse::Succeeded);
        }
    }

    if (outgoing_requests.empty()) {
        return;
    }

    ASSERT_THROW_NO_PRINT(std::ignore = ForwardConnector::process_responses(outgoing_requests, incoming_responses, synaptic_elements);, RelearnException);
}

TEST_F(ConnectorTest, testForwardConnectorProcessResponsesIgnoreException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    // The following copies are intentional
    const auto previous_connected_axon = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::Axon));
    const auto previous_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));
    const auto previous_deltas_axon = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::Axon));

    auto outgoing_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };
    auto incoming_responses = RelearnTypes::comm_map_creation<SynapseCreationResponse>{ number_ranks };

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    for (const auto id : NeuronIDRange::range(number_neurons)) {
        const auto signal_type = signal_types[id.get_neuron_id()];
        const auto free_elements = vacant_axons[id.get_neuron_id()];

        for (auto i = 0U; i < free_elements; i++) {
            const auto scr = SynapseCreationRequest(NeuronID{ i }, id, signal_type);
            outgoing_requests.append(mpiPP::MPIRank(1), scr);
            outgoing_requests.append(mpiPP::MPIRank(1), scr);

            incoming_responses.append(mpiPP::MPIRank(1), SynapseCreationResponse::Succeeded);
            incoming_responses.append(mpiPP::MPIRank(1), SynapseCreationResponse::Failed);
        }
    }

    if (outgoing_requests.empty()) {
        return;
    }

    const auto synapses = ForwardConnector::process_responses(outgoing_requests, incoming_responses, synaptic_elements);

    ASSERT_EQ(synapses.size(), outgoing_requests.get_total_number_requests() / 2);

    const auto now_connected_axon = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::Axon));
    const auto now_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));

    ASSERT_EQ(previous_grown_axon, now_grown_axon);

    const auto deltas = synaptic_elements->get_deltas(SynapticElementType::Axon);
    for (const auto id : NeuronIDRange::range(number_neurons)) {
        ASSERT_LT(deltas[id.get_neuron_id()], 1.0);
    }

    auto map = std::map<NeuronID, unsigned int>{};

    for (const auto& [target, source_id, weight] : synapses) {
        const auto& [target_rank, target_id] = target;

        ASSERT_EQ(target_rank, mpiPP::MPIRank(1));
        map[source_id] += static_cast<unsigned int>(std::abs(weight));

        if (signal_types[source_id.get_neuron_id()] == SignalType::Excitatory) {
            ASSERT_GT(weight, 0);
        } else {
            ASSERT_LT(weight, 0);
        }
    }

    for (const auto id : NeuronIDRange::range(number_neurons)) {
        ASSERT_EQ(now_connected_axon[id.get_neuron_id()], map[id] + previous_connected_axon[id.get_neuron_id()]);
    }
}

TEST_F(ConnectorTest, testBackwardConnectorProcessRequestsExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;
    const auto incoming_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };

    const auto empty = std::shared_ptr<SynapticElements>(nullptr);

    ASSERT_THROW_NO_PRINT(std::ignore = BackwardConnector::process_requests(incoming_requests, empty), RelearnException);
}

TEST_F(ConnectorTest, testBackwardConnectorProcessRequestsEmptyMap) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    // The following copies are intentional
    const auto previous_connected_axon = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::Axon));
    const auto previous_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));
    const auto previous_deltas_axon = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::Axon));

    const auto incoming_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };

    auto [responses, created_synapses, synapses] = BackwardConnector::process_requests(incoming_requests, synaptic_elements);
    const auto [local_synapses, distant_in_synapses] = synapses;

    ASSERT_EQ(responses.size(), incoming_requests.size());
    ASSERT_EQ(responses.get_number_ranks(), incoming_requests.get_number_ranks());
    ASSERT_EQ(responses.get_total_number_requests(), 0);

    ASSERT_TRUE(local_synapses.empty());
    ASSERT_TRUE(distant_in_synapses.empty());

    const auto now_connected_axon = utility::vectorify_span(synaptic_elements->get_connected_elements(SynapticElementType::Axon));
    const auto now_grown_axon = utility::vectorify_span(synaptic_elements->get_grown_elements(SynapticElementType::Axon));
    const auto now_deltas_axon = utility::vectorify_span(synaptic_elements->get_deltas(SynapticElementType::Axon));

    for (auto i = 0U; i < number_neurons; i++) {
        ASSERT_EQ(previous_connected_axon[i], now_connected_axon[i]);
        ASSERT_EQ(previous_grown_axon[i], now_grown_axon[i]);
        ASSERT_EQ(previous_deltas_axon[i], now_deltas_axon[i]);
    }
}

TEST_F(ConnectorTest, testBackwardConnectorProcessRequestsMatchingRequests) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    const auto signal_types = SynapticElementsFactory::get_excitatory_signal_types(number_neurons);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    auto incoming_requests = RelearnTypes::comm_map_creation<SynapseCreationRequest>{ number_ranks };

    auto number_excitatory_requests = 0U;
    auto number_inhibitory_requests = 0U;

    auto excitatory_requests = std::map<NeuronID, std::vector<SynapseCreationRequest>>{};
    auto inhibitory_requests = std::map<NeuronID, std::vector<SynapseCreationRequest>>{};

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);

    for (const auto& id : NeuronIDRange::range(number_neurons)) {
        const auto number_vacant_elements = vacant_axons[id.get_neuron_id()];

        const auto number_vacant_excitatory = signal_types[id.get_neuron_id()] == SignalType::Excitatory ? number_vacant_elements : 0;
        number_excitatory_requests += number_vacant_excitatory;

        for (auto i = 0U; i < number_vacant_excitatory; i++) {
            const auto scr = SynapseCreationRequest(id, NeuronID{ i }, SignalType::Excitatory);
            incoming_requests.append(mpiPP::MPIRank(1), scr);

            excitatory_requests[id].emplace_back(scr);
        }

        const auto number_vacant_inhibitory = signal_types[id.get_neuron_id()] == SignalType::Inhibitory ? number_vacant_elements : 0;
        number_inhibitory_requests += number_vacant_inhibitory;

        for (auto i = 0U; i < number_vacant_inhibitory; i++) {
            const auto scr = SynapseCreationRequest(id, NeuronID{ i }, SignalType::Inhibitory);
            incoming_requests.append(mpiPP::MPIRank(1), scr);

            inhibitory_requests[id].emplace_back(scr);
        }
    }

    auto [responses, created_synapses, synapses] = BackwardConnector::process_requests(incoming_requests, synaptic_elements);
    const auto [local_synapses, distant_in_synapses] = synapses;

    ASSERT_EQ(incoming_requests.size(), responses.size());

    const auto& request_sizes = incoming_requests.get_request_sizes();
    const auto& response_sizes = responses.get_request_sizes();

    // For each saved rank: The number of responses matches the number of requests
    ASSERT_EQ(request_sizes.size(), response_sizes.size());
    for (const auto& [rank, size] : request_sizes) {
        const auto found_in_responses = response_sizes.contains(rank);
        ASSERT_TRUE(found_in_responses);

        ASSERT_EQ(size, response_sizes.at(rank));
    }

    for (const auto& [rank, resps] : responses) {
        for (const auto resp : resps) {
            ASSERT_EQ(resp, SynapseCreationResponse::Succeeded);
        }
    }

    ASSERT_EQ(local_synapses.size(), 0);
    ASSERT_EQ(distant_in_synapses.size(), number_excitatory_requests + number_inhibitory_requests);

    for (const auto& id : NeuronIDRange::range(number_neurons)) {
        const auto number_vacant_axons = vacant_axons[id.get_neuron_id()];
        ASSERT_EQ(number_vacant_axons, 0);
    }

    for (const auto& [target_id, source_id, weight] : distant_in_synapses) {
        const auto& [target_rank, target_neuron_id] = target_id;

        ASSERT_EQ(target_rank, mpiPP::MPIRank(1));
        ASSERT_EQ(std::abs(weight), 1);
    }
}
