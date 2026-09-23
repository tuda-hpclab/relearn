#pragma once

/*
 * This file is part of the MPI-Wrapper software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "mpi-wrapper/communicator/MPICommunicator.h"
#include "mpi-wrapper/core/MPICounters.h"
#include "mpi-wrapper/core/MPITypes.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

#include <mpi.h>

#include <span>

namespace mpiPP {

namespace MPIBroadcasts {

/**
 * @brief Broadcasts the provided value from the sender to all other ranks
 * @tparam T The type of data to broadcast
 * @param value The value to broadcast on sender; ignored on every other rank
 * @param sender The rank which sends the value
 * @param communicator The communicator to broadcast within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The broadcasted value from sender
 */
template <MPICompatible T>
T broadcast(const T value, const MPIRank sender, const MPICommunicator& communicator = MPICommunicator::World) {
    auto buffer = T{};
    if (communicator.get_my_rank() == sender) {
        buffer = value;
    }

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Bcast(&buffer, 1, type, sender.get_rank(), communicator.get());
    utility::Exception::check(error_code == 0, "Broadcasting one value returned the error: {}", error_code);

    MPICounters::add_to_received(sizeof(T));
    if (communicator.get_my_rank() == sender) {
        MPICounters::add_to_sent(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(T));
    }

    return buffer;
}

/**
 * @brief Broadcasts the provided values from the sender to all other ranks
 * @tparam T The type of data to broadcast
 * @tparam extent The extent of the input span, is usually deduced automatically
 * @param values The local values, can be of any size on the non-sender ranks
 * @param sender The rank which sends the values
 * @param communicator The communicator to broadcast within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The broadcasted values from sender
 */
template <MPICompatible T, std::size_t extent>
std::vector<T> broadcast(const std::span<const T, extent> values, const MPIRank sender, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto span_size = values.size();
    const auto vector_size = broadcast(span_size, sender, communicator);

    auto buffer = std::vector<T>(vector_size);
    if (communicator.get_my_rank() == sender) {
        std::copy(values.begin(), values.end(), buffer.begin());
    }

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Bcast(buffer.data(), utility::safe_cast<int>(vector_size), type, sender.get_rank(), communicator.get());
    utility::Exception::check(error_code == 0, "Broadcasting multiple values returned the error: {}", error_code);

    const auto data_bytes = vector_size * sizeof(T);
    MPICounters::add_to_received(data_bytes);
    if (communicator.get_my_rank() == sender) {
        MPICounters::add_to_sent(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * data_bytes);
    }

    return buffer;
}

} // namespace MPIBroadcasts

} // namespace mpiPP
