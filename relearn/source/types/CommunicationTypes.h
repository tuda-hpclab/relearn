#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <mpi-wrapper/patterns/CommunicationVector.h>

// The containers the ranks exchange their requests in, one alias per kind of request. They all name the same
// container today; naming them apart keeps the choice per kind of request open.

namespace RelearnTypes {

/** @brief Exchanges the requests for the groups a neuron belongs to */
template <typename RequestType>
using comm_map_group = mpiPP::CommunicationVector<RequestType>;

/** @brief Exchanges the requests that create synapses */
template <typename RequestType>
using comm_map_creation = mpiPP::CommunicationVector<RequestType>;

/** @brief Exchanges the requests that delete synapses */
template <typename RequestType>
using comm_map_deletion = mpiPP::CommunicationVector<RequestType>;

/** @brief Exchanges the fired status of the neurons */
template <typename RequestType>
using comm_map_firing = mpiPP::CommunicationVector<RequestType>;

/** @brief Exchanges the positions of the neurons */
template <typename RequestType>
using comm_map_position = mpiPP::CommunicationVector<RequestType>;

} // namespace RelearnTypes
