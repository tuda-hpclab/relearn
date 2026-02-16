#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <mpi-wrapper/CommunicationMap.h>
#include <mpi-wrapper/CommunicationVector.h>

namespace RelearnTypes {
// These types are using MPI

template <typename RequestType>
using comm_map_group = mpiPP::CommunicationVector<RequestType>;

template <typename RequestType>
using comm_map_creation = mpiPP::CommunicationVector<RequestType>;

template <typename RequestType>
using comm_map_deletion = mpiPP::CommunicationVector<RequestType>;

template <typename RequestType>
using comm_map_firing = mpiPP::CommunicationVector<RequestType>;

template <typename RequestType>
using comm_map_position = mpiPP::CommunicationVector<RequestType>;

} // namespace RelearnTypes