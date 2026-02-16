#pragma once

/*
 * This file is part of the MPI-Wrapper software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "mpi-wrapper/MPICounters.h"
#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"
#include "mpi-wrapper/MPITypes.h"

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"

#include <mpi.h>

#include <span>

namespace mpiPP {

namespace MPICommunication {
/**
 * @brief Sends the buffer to the specified rank. Must have a matching call on the other rank
 * @tparam T The type of data to send
 * @param buffer The data to send
 * @param target Where to send the data
 * @param tag A tag to distinguish data transfers, default is 0
 * @exception Throws an Exception if target is not in MPI_COMM_WORLD or mpi reports an error
 */
template <MPICompatible T>
void send(const std::span<const T> buffer, const MPIRank target, const int tag = 0) {
    const auto number_ranks = MPIInfo::get_number_ranks();
    const auto target_rank = target.get_rank();
    utility::Exception::check(target_rank < number_ranks, "MPICommunication::send: There are {} ranks, but should send to {}", number_ranks, target_rank);

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto size_cast = utility::save_cast<int>(buffer.size());
    const auto error_code = MPI_Send(buffer.data(), size_cast, type, target_rank, tag, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "MPICommunication::send: Error code is: ", error_code);

    MPICounters::add_to_sent(buffer.size_bytes());
}

/**
 * @brief Received data into the buffer from the specified rank. Must have a matching call on the other rank
 * @tparam T The type of data to send
 * @param buffer The buffer for the data
 * @param source From where to receive the data
 * @param tag A tag to distinguish data transfers, default is 0
 * @exception Throws an Exception if source is not in MPI_COMM_WORLD or mpi reports an error
 */
template <MPICompatible T>
void receive(const std::span<T> buffer, const MPIRank source, const int tag = 0) {
    const auto number_ranks = MPIInfo::get_number_ranks();
    const auto source_rank = source.get_rank();
    utility::Exception::check(source_rank < number_ranks, "MPICommunication::receive: There are {} ranks, but should receive from {}", number_ranks, source_rank);

    auto status = MPI_Status{};

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto size_cast = utility::save_cast<int>(buffer.size());
    const auto error_code = MPI_Recv(buffer.data(), size_cast, type, source_rank, tag, MPI_COMM_WORLD, &status);
    utility::Exception::check(error_code == 0, "MPICommunication::receive: Error code is: ", error_code);

    MPICounters::add_to_received(buffer.size_bytes());
}

/**
 * @brief Sends the data element to the specified rank. Must have a matching call on the other rank
 * @tparam T The type of data to send
 * @param value The data element to send
 * @param target Where to send the data
 * @param tag A tag to distinguish data transfers, default is 0
 * @exception Throws an Exception if target is not in MPI_COMM_WORLD or mpi reports an error
 */
template <MPICompatible T>
void send(const T value, const MPIRank target, const int tag = 0) {
    send(std::span{ &value, 1 }, target, tag);
}

/**
 * @brief Receives one data element from the specified rank. Must have a matching call on the other rank
 * @tparam T The type of data to send
 * @param source From where to receive the data
 * @param tag A tag to distinguish data transfers, default is 0
 * @exception Throws an Exception if source is not in MPI_COMM_WORLD or mpi reports an error
 */
template <MPICompatible T>
[[nodiscard]] T receive(const MPIRank source, const int tag = 0) {
    auto value = T{};
    receive<T>(std::span<T>{ &value, 1 }, source, tag);
    return value;
}

/**
 * @brief Sends the buffer to the specified rank. Must have a matching call on the other rank.
 *      Returns the control flow without necessarily completing the request
 * @tparam T The type of data to send
 * @param buffer The data to send
 * @param target Where to send the data
 * @param tag A tag to distinguish data transfers, default is 0
 * @exception Throws an Exception if target is not in MPI_COMM_WORLD or mpi reports an error
 * @return Returns a token that can be waited on
 */
template <MPICompatible T>
[[nodiscard]] MPI_Request async_send(const std::span<const T> buffer, const MPIRank target, const int tag = 0) {
    const auto number_ranks = MPIInfo::get_number_ranks();
    const auto target_rank = target.get_rank();
    utility::Exception::check(target_rank < number_ranks, "MPICommunication::send: There are {} ranks, but should send to {}", number_ranks, target_rank);

    auto request = MPI_Request{};

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto size_cast = utility::save_cast<int>(buffer.size());
    const auto error_code = MPI_Isend(buffer.data(), size_cast, type, target_rank, tag, MPI_COMM_WORLD, &request);
    utility::Exception::check(error_code == 0, "MPICommunication::send: Error code is: ", error_code);

    MPICounters::add_to_sent(buffer.size_bytes());

    return request;
}

/**
 * @brief Received data into the buffer from the specified rank. Must have a matching call on the other rank.
 *      Returns the control flow without necessarily completing the request
 * @tparam T The type of data to send
 * @param buffer The buffer for the data
 * @param source From where to receive the data
 * @param tag A tag to distinguish data transfers, default is 0
 * @exception Throws an Exception if source is not in MPI_COMM_WORLD or mpi reports an error
 * @return Returns a token that can be waited on
 */
template <MPICompatible T>
[[nodiscard]] MPI_Request async_receive(const std::span<T> buffer, const MPIRank source, const int tag = 0) {
    const auto number_ranks = MPIInfo::get_number_ranks();
    const auto source_rank = source.get_rank();
    utility::Exception::check(source_rank < number_ranks, "MPICommunication::receive: There are {} ranks, but should receive from {}", number_ranks, source_rank);

    auto request = MPI_Request{};

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto size_cast = utility::save_cast<int>(buffer.size());
    const auto error_code = MPI_Irecv(buffer.data(), size_cast, type, source_rank, tag, MPI_COMM_WORLD, &request);
    utility::Exception::check(error_code == 0, "MPICommunication::receive: Error code is: ", error_code);

    MPICounters::add_to_received(buffer.size_bytes());

    return request;
}
} // namespace MPICommunication

} // namespace mpiPP
