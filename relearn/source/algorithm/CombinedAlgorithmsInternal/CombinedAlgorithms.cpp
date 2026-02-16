/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "algorithm/CombinedAlgorithmsInternal/CombinedAlgorithms.h"


[[nodiscard]] std::tuple<PlasticLocalSynapses, PlasticDistantInSynapses, PlasticDistantOutSynapses> CombinedAlgorithms::update_connectivity([[maybe_unused]] const number_neurons_type number_neurons) {
    auto local_synapses_total = PlasticLocalSynapses{};
    auto distant_in_synapses_total = PlasticDistantInSynapses{};
    auto out_synapses_total = PlasticDistantOutSynapses{};
    for (const auto& tuple : _indices_and_neurons) {
        const auto& [index, neuron_ids] = tuple;
        const auto& alg_ptr = algorithm_ptrs[index];

        const auto& [requests_in_variant, request_type, direction] = alg_ptr->find_target_neurons_for_combined_algorithms(neuron_ids);

        switch (request_type) {
        case RequestTypeEnum::SynapseCreationRequest: {
            const auto& synapse_creation_requests_outgoing = std::get<RelearnTypes::comm_map_creation<SynapseCreationRequest>>(requests_in_variant);
            const auto& synapse_creation_requests_incoming = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(synapse_creation_requests_outgoing);
            switch (direction) {
            case DirectionEnum::Forward: {
                auto [responses_outgoing, synapses] = process_requests_forward(synapse_creation_requests_incoming);
                auto& [local_synapses, distant_in_synapses] = synapses;
                const auto responses_incoming = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(responses_outgoing);
                auto out_synapses = process_responses_forward(synapse_creation_requests_outgoing, responses_incoming);

                local_synapses_total.insert(std::end(local_synapses_total), std::begin(local_synapses), std::end(local_synapses));
                distant_in_synapses_total.insert(std::end(distant_in_synapses_total), std::begin(distant_in_synapses), std::end(distant_in_synapses));
                out_synapses_total.insert(std::end(out_synapses_total), std::begin(out_synapses), std::end(out_synapses));
                break;
            }
            case DirectionEnum::Backward: {
                auto [responses_outgoing, synapses] = process_requests_backward(synapse_creation_requests_incoming);
                auto& [local_synapses, distant_out_synapses] = synapses;
                const auto responses_incoming = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(responses_outgoing);
                auto distant_in_synapses = process_responses_backward(synapse_creation_requests_outgoing, responses_incoming);

                local_synapses_total.insert(std::end(local_synapses_total), std::begin(local_synapses), std::end(local_synapses));
                distant_in_synapses_total.insert(std::end(distant_in_synapses_total), std::begin(distant_in_synapses), std::end(distant_in_synapses));
                out_synapses_total.insert(std::end(out_synapses_total), std::begin(distant_out_synapses), std::end(distant_out_synapses));
                break;
            }
            default:
                RelearnException::fail("CombinedAlgorithms::update_connectivity: The direction is unknown!");
            }
        }
        break;
        case RequestTypeEnum::DistantNeuronRequest: {
            const auto& synapse_creation_requests_outgoing = std::get<RelearnTypes::comm_map_creation<DistantNeuronRequest>>(requests_in_variant);
            const auto& synapse_creation_requests_incoming = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(synapse_creation_requests_outgoing);
            const auto& alg_as_location_aware_bh = std::static_pointer_cast<BarnesHutLocationAware>(alg_ptr);
            auto [responses_outgoing, synapses] = alg_as_location_aware_bh->process_requests(synapse_creation_requests_incoming);
            auto& [local_synapses, distant_in_synapses] = synapses;
            const auto responses_incoming = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(responses_outgoing);
            auto out_synapses = alg_as_location_aware_bh->process_responses(synapse_creation_requests_outgoing, responses_incoming);

            local_synapses_total.insert(std::end(local_synapses_total), std::begin(local_synapses), std::end(local_synapses));
            distant_in_synapses_total.insert(std::end(distant_in_synapses_total), std::begin(distant_in_synapses), std::end(distant_in_synapses));
            out_synapses_total.insert(std::end(out_synapses_total), std::begin(out_synapses), std::end(out_synapses));
            break;
        }
        default:
            RelearnException::fail("CombinedAlgorithms::update_connectivity: The request type is unknown!");
        }
    }

    return {
        std::move(local_synapses_total), std::move(distant_in_synapses_total), std::move(out_synapses_total)
    };
}


[[nodiscard]] std::pair<RelearnTypes::comm_map_creation<SynapseCreationResponse>, std::pair<PlasticLocalSynapses, PlasticDistantInSynapses>> CombinedAlgorithms::process_requests_forward(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests) {
    return ForwardConnector::process_requests(creation_requests, synaptic_elements);
}


[[nodiscard]] PlasticDistantOutSynapses CombinedAlgorithms::process_responses_forward(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests, const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses) {
    return ForwardConnector::process_responses(creation_requests, creation_responses, synaptic_elements);
}

[[nodiscard]] std::pair<RelearnTypes::comm_map_creation<SynapseCreationResponse>, std::pair<PlasticLocalSynapses, PlasticDistantOutSynapses>> CombinedAlgorithms::process_requests_backward(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests) {
    return BackwardConnector::process_requests(creation_requests, synaptic_elements);
}

[[nodiscard]] PlasticDistantInSynapses CombinedAlgorithms::process_responses_backward(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests, const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses) {
    return BackwardConnector::process_responses(creation_requests, creation_responses, synaptic_elements);
}
