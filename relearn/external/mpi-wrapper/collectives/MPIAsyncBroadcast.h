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
#include "mpi-wrapper/core/MPIRank.h"
#include "mpi-wrapper/core/MPITypes.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

#include <mpi.h>

#include <span>

namespace mpiPP {

namespace MPIBroadcasts {
/**
 * @brief Starts broadcasting the buffer from the sender to all other ranks without blocking. On the
 *      sender the buffer holds the data to send; on the other ranks it receives it. Unlike the
 *      blocking span overload the size is NOT negotiated: every rank must pass a buffer of the same
 *      size. The buffer must stay valid until the returned request completes (via wait/test).
 * @tparam T The type of data to broadcast
 * @param buffer The data to send on the sender, the receive buffer on all other ranks
 * @param sender The rank which sends the values
 * @param communicator The communicator to broadcast within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return A token that can be waited on
 */
template <MPICompatible T>
[[nodiscard]] MPI_Request async_broadcast(const std::span<T> buffer, const MPIRank sender, const MPICommunicator& communicator = MPICommunicator::World) {
    auto request = MPI_Request{};

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto size_cast = utility::safe_cast<int>(buffer.size());
    const auto error_code = MPI_Ibcast(buffer.data(), size_cast, type, sender.get_rank(), communicator.get(), &request);
    utility::Exception::check(error_code == 0, "Asynchronously broadcasting values returned the error: {}", error_code);

    MPICounters::add_to_received(buffer.size_bytes());
    if (communicator.get_my_rank() == sender) {
        MPICounters::add_to_sent(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * buffer.size_bytes());
    }

    return request;
}
} // namespace MPIBroadcasts

} // namespace mpiPP
