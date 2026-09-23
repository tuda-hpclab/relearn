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

#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/CommunicationTypes.h"
#include "types/SynapseTypes.h"

#include <memory>
#include <utility>

class NetworkGraph;

/** All local synapses plus all distant-in synapses created by processing one batch of creation requests. */
struct LocalAndDistantInSynapses {
    PlasticLocalSynapses local_synapses;
    PlasticDistantInSynapses distant_in_synapses;
};

/** All local synapses plus all distant-out synapses created by processing one batch of creation requests. */
struct LocalAndDistantOutSynapses {
    PlasticLocalSynapses local_synapses;
    PlasticDistantOutSynapses distant_out_synapses;
};

/** Result of ForwardConnector::process_requests() / ForwardAlgorithm::process_requests(): the responses to each request, how many synapses were created, and the created synapses themselves. */
template <typename ResponseType>
struct ForwardProcessRequestsResult {
    RelearnTypes::comm_map_creation<ResponseType> responses{};
    std::uint64_t number_created_synapses{};
    LocalAndDistantInSynapses synapses;
};

/** Result of BackwardConnector::process_requests() / BackwardAlgorithm::process_requests(): the responses to each request, how many synapses were created, and the created synapses themselves. */
template <typename ResponseType>
struct BackwardProcessRequestsResult {
    RelearnTypes::comm_map_creation<ResponseType> responses{};
    std::uint64_t number_created_synapses{};
    LocalAndDistantOutSynapses synapses;
};

/**
 * This class commits SynapseCreationRequests and SynapseCreationResponses to the synaptic elements.
 * It assumes that a request is made by an axon and targets a dendrite. It does not perform communication with MPI.
 */
class ForwardConnector {
public:
    /**
     * @brief Connects as many of the creation requests as possible locally, and commits the changes to the synaptic elements.
     *      Picks the order of the requests randomly. A request is from an axon to a dendrite
     * @param creation_requests The creation requests from all MPI ranks to the current rank
     * @param synaptic_elements The synaptic elements
     * @exception Throws a RelearnException if (a) One of the pointers is empty, (b) They have different sizes, (c) One target has an id larger than the number of elements in the pointers
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] static ForwardProcessRequestsResult<SynapseCreationResponse>
    process_requests(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                     const std::shared_ptr<SynapticElements>& synaptic_elements);

    /**
     * @brief Processes all incoming responses from the MPI ranks locally, and commits the changes to the synaptic elements.
     *      A response is from a request from an axon to a dendrite
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @param synaptic_elements The synaptic elements
     * @exception Throws a RelearnException if (a) The axons are empty, (b) The requests and responses don't have the same size,
     *      (c) One of the source ids that are accepted are too large, (d) An accepted request targets an axon with not enough vacant elements
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] static PlasticDistantOutSynapses process_responses(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                                                                     const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses,
                                                                     const std::shared_ptr<SynapticElements>& synaptic_elements);
};

/**
 * This class commits SynapseCreationRequests and SynapseCreationResponses to the synaptic elements.
 * It assumes that a request is made by a dendrite and targets an axon. It does not perform communication with MPI.
 */
class BackwardConnector {
public:
    /**
     * @brief Connects as many of the creation requests as possible locally, and commits the changes to the synaptic elements.
     *      Picks the order of the requests randomly. A request is from a dendrite to an axon
     * @param creation_requests The creation requests from all MPI ranks to the current rank
     * @param synaptic_elements The synaptic elements
     * @exception Throws a RelearnException if (a) The pointer is empty, (b) One target has an id larger than the number of elements in the pointers, (c) The signal type of a request does not match that of the axon
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses from the local rank
     */
    [[nodiscard]] static BackwardProcessRequestsResult<SynapseCreationResponse>
    process_requests(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests, const std::shared_ptr<SynapticElements>& synaptic_elements);

    /**
     * @brief Processes all incoming responses from the MPI ranks locally, and commits the changes to the synaptic elements.
     *      A response is from a request from a dendrite to an axon
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @param synaptic_elements The synaptic elements
     * @exception Throws a RelearnException if (a) The axons are empty, (b) One of the source ids that are accepted are too large
     * @return All synapses to this MPI rank from other MPI ranks
     */
    [[nodiscard]] static PlasticDistantInSynapses process_responses(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                                                                    const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses,
                                                                    const std::shared_ptr<SynapticElements>& synaptic_elements);
};
