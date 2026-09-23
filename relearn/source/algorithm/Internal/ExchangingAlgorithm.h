#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "algorithm/Algorithm.h"
#include "algorithm/Connector.h"
#include "neurons/NetworkGraph.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/BasicTypes.h"
#include "types/CommunicationTypes.h"
#include "types/SynapseTypes.h"
#include "util/Timers.h"

#include <mpi-wrapper/patterns/MPIAdvancedCommunicationPatterns.h>

#include <memory>
#include <utility>

/** Result of ForwardAlgorithm::process_requests_aware(): the responses to each request, the target neuron ids they resolved to, and how many synapses were created. */
struct ProcessRequestsAwareResult {
    DeviceArray<SynapseCreationResponse> responses;
    DeviceArray<CudaConfig::number_neurons_type> target_ids;
    std::uint64_t number_created_synapses{};
};

/**
 * This class manages the exchange of requests and responses, and their distribution on all MPI ranks
 *      It connects from axons to dendrites
 * @tparam RequestType The type of creation requests
 * @tparam ResponseType The type of creation responses
 */
template <typename RequestType, typename ResponseType>
class ForwardCPUAlgorithm : public Algorithm {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Constructs a new object
     */
    ForwardCPUAlgorithm()
        : Algorithm() { }

    /**
     * @brief Updates the connectivity with the algorithm. Already updates the synaptic elements, i.e., the axons and dendrites (both excitatory and inhibitory).
     *      Does not update the network graph. Performs communication with MPI
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return A tuple with the created synapses that must be committed to the network graph
     */
    [[nodiscard]] ConnectivityUpdateResult update_connectivity(const number_neurons_type number_neurons) final {
        Timers::start(TimerRegion::CREATE_SYNAPSES);

        Timers::start(TimerRegion::FIND_TARGET_NEURONS);
        const auto& synapse_creation_requests_outgoing = find_target_neurons(number_neurons);
        Timers::stop_and_add(TimerRegion::FIND_TARGET_NEURONS);

        Timers::start(TimerRegion::EXCHANGE_CREATION_REQUESTS);
        const auto& synapse_creation_requests_incoming = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(synapse_creation_requests_outgoing);
        Timers::stop_and_add(TimerRegion::EXCHANGE_CREATION_REQUESTS);

        Timers::start(TimerRegion::PROCESS_CREATION_REQUESTS);
        auto [responses_outgoing, number_created_synapses, synapses] = process_requests(synapse_creation_requests_incoming);
        auto& [local_synapses, distant_in_synapses] = synapses;
        Timers::stop_and_add(TimerRegion::PROCESS_CREATION_REQUESTS);

        Timers::start(TimerRegion::CREATE_CREATION_RESPONSES);
        const auto& responses_incoming = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(responses_outgoing);
        Timers::stop_and_add(TimerRegion::CREATE_CREATION_RESPONSES);

        Timers::start(TimerRegion::PROCESS_CREATION_RESPONSES);
        auto out_synapses = process_responses(synapse_creation_requests_outgoing, responses_incoming);
        Timers::stop_and_add(TimerRegion::PROCESS_CREATION_RESPONSES);

        Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES);

        return {
            number_created_synapses, std::move(local_synapses), std::move(distant_in_synapses), std::move(out_synapses)
        };
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        Algorithm::record_memory_footprint(footprint);

        const auto my_footprint = sizeof(*this) - sizeof(Algorithm);
        footprint->emplace("ForwardAlgorithm", my_footprint);
    }

protected:
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */

    [[nodiscard]] virtual RelearnTypes::comm_map_creation<RequestType> find_target_neurons(number_neurons_type number_neurons) = 0;

    /**
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] virtual ForwardProcessRequestsResult<ResponseType>
    process_requests(const RelearnTypes::comm_map_creation<RequestType>& creation_requests) = 0;

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] virtual PlasticDistantOutSynapses process_responses(const RelearnTypes::comm_map_creation<RequestType>& creation_requests,
                                                                      const RelearnTypes::comm_map_creation<ResponseType>& creation_responses)
        = 0;
};

/**
 * This class manages the exchange of requests and responses, and their distribution on all MPI ranks
 *      It connects from dendrites to axons
 * @tparam RequestType The type of creation requests
 * @tparam ResponseType The type of creation responses
 */
template <typename RequestType, typename ResponseType>
class BackwardAlgorithm : public Algorithm {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Constructs a new object
     * @exception Throws a RelearnException if octree is nullptr
     */
    BackwardAlgorithm()
        : Algorithm() { }

    /**
     * @brief Updates the connectivity with the algorithm. Already updates the synaptic elements, i.e., the axons and dendrites (both excitatory and inhibitory).
     *      Does not update the network graph. Performs communication with MPI
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return A tuple with the created synapses that must be committed to the network graph
     */
    [[nodiscard]] ConnectivityUpdateResult update_connectivity(const number_neurons_type number_neurons) override {
#ifdef RELEARN_CUDA_ENABLED
        CPU_NOT_SUPPORTED
#endif
        Timers::start(TimerRegion::CREATE_SYNAPSES);

        Timers::start(TimerRegion::FIND_TARGET_NEURONS);
        const auto& synapse_creation_requests_outgoing = find_target_neurons(number_neurons);
        Timers::stop_and_add(TimerRegion::FIND_TARGET_NEURONS);

        Timers::start(TimerRegion::EXCHANGE_CREATION_REQUESTS);
        const auto& synapse_creation_requests_incoming = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(synapse_creation_requests_outgoing);
        Timers::stop_and_add(TimerRegion::EXCHANGE_CREATION_REQUESTS);

        Timers::start(TimerRegion::PROCESS_CREATION_REQUESTS);
        auto [responses_outgoing, created_synapse, synapses] = process_requests(synapse_creation_requests_incoming);
        auto& [local_synapses, distant_out_synapses] = synapses;
        Timers::stop_and_add(TimerRegion::PROCESS_CREATION_REQUESTS);

        Timers::start(TimerRegion::CREATE_CREATION_RESPONSES);
        const auto& responses_incoming = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(responses_outgoing);
        Timers::stop_and_add(TimerRegion::CREATE_CREATION_RESPONSES);

        Timers::start(TimerRegion::PROCESS_CREATION_RESPONSES);
        auto distant_in_synapses = process_responses(synapse_creation_requests_outgoing, responses_incoming);
        Timers::stop_and_add(TimerRegion::PROCESS_CREATION_RESPONSES);

        Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES);

        return {
            created_synapse, std::move(local_synapses), std::move(distant_in_synapses), std::move(distant_out_synapses)
        };
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        Algorithm::record_memory_footprint(footprint);

        const auto my_footprint = sizeof(*this) - sizeof(Algorithm);
        footprint->emplace("BackwardAlgorithm", my_footprint);
    }

protected:
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */
    [[nodiscard]] virtual RelearnTypes::comm_map_creation<RequestType> find_target_neurons(number_neurons_type number_neurons) = 0;

    /**
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all synapses from other ranks
     */
    [[nodiscard]] virtual BackwardProcessRequestsResult<ResponseType>
    process_requests(const RelearnTypes::comm_map_creation<RequestType>& creation_requests) = 0;

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses to this MPI rank from other MPI ranks
     */
    [[nodiscard]] virtual PlasticDistantInSynapses process_responses(const RelearnTypes::comm_map_creation<RequestType>& creation_requests,
                                                                     const RelearnTypes::comm_map_creation<ResponseType>& creation_responses)
        = 0;
};
