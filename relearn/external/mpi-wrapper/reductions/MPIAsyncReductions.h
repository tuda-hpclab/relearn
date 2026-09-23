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

#include <concepts>
#include <span>

namespace mpiPP {

namespace detail {
/**
 * @brief Starts a non-blocking all-reduce of the send buffer into the receive buffer with the given
 *      operation. Shared implementation of the async_all_reduce_* functions; both buffers and their
 *      matching sizes must stay valid until the returned request completes.
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_all_reduce(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPI_Op operation, const MPICommunicator& communicator) {
    utility::Exception::check(send_buffer.size() == receive_buffer.size(), "MPIReductions::async_all_reduce: send and receive buffers differ in size: {} vs {}", send_buffer.size(), receive_buffer.size());

    auto request = MPI_Request{};
    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto count = utility::safe_cast<int>(send_buffer.size());
    const auto error_code = MPI_Iallreduce(send_buffer.data(), receive_buffer.data(), count, type, operation, communicator.get(), &request);
    utility::Exception::check(error_code == 0, "Asynchronously all-reducing returned the error: {}", error_code);

    MPICounters::add_to_sent(send_buffer.size_bytes());
    MPICounters::add_to_received(receive_buffer.size_bytes());

    return request;
}

/**
 * @brief Starts a non-blocking reduce of the send buffer into the receive buffer on the root rank
 *      with the given operation. Shared implementation of the async_reduce_* functions; both buffers
 *      and their matching sizes must stay valid until the returned request completes.
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_reduce(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPI_Op operation, const MPICommunicator& communicator) {
    utility::Exception::check(send_buffer.size() == receive_buffer.size(), "MPIReductions::async_reduce: send and receive buffers differ in size: {} vs {}", send_buffer.size(), receive_buffer.size());

    auto request = MPI_Request{};
    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto count = utility::safe_cast<int>(send_buffer.size());
    const auto error_code = MPI_Ireduce(send_buffer.data(), receive_buffer.data(), count, type, operation, 0, communicator.get(), &request);
    utility::Exception::check(error_code == 0, "Asynchronously reducing returned the error: {}", error_code);

    MPICounters::add_to_sent(send_buffer.size_bytes());
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * send_buffer.size_bytes());
    }

    return request;
}
} // namespace detail

namespace MPIReductions {
// These are span-based, so a single function serves both the scalar case (a span of length 1) and
// the componentwise case (a longer span). Every rank must pass buffers of the same size; the send
// and receive buffers must stay valid until the returned request completes (via wait/test).

/**
 * @brief Starts a non-blocking all-reduce that yields the componentwise sum on all ranks.
 * @tparam T The type of data to reduce
 * @param send_buffer The local values
 * @param receive_buffer The buffer for the reduced values, must have the same size as send_buffer
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the buffer sizes differ or mpi reports an error
 * @return A token that can be waited on
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_all_reduce_sum(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_all_reduce(send_buffer, receive_buffer, MPI_SUM, communicator);
}

/**
 * @brief Starts a non-blocking all-reduce that yields the componentwise minimum on all ranks.
 * @see async_all_reduce_sum
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_all_reduce_min(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_all_reduce(send_buffer, receive_buffer, MPI_MIN, communicator);
}

/**
 * @brief Starts a non-blocking all-reduce that yields the componentwise maximum on all ranks.
 * @see async_all_reduce_sum
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_all_reduce_max(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_all_reduce(send_buffer, receive_buffer, MPI_MAX, communicator);
}

/**
 * @brief Starts a non-blocking all-reduce that yields the componentwise product on all ranks.
 * @see async_all_reduce_sum
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_all_reduce_prod(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_all_reduce(send_buffer, receive_buffer, MPI_PROD, communicator);
}

/**
 * @brief Starts a non-blocking all-reduce that yields the componentwise bitwise and on all ranks.
 * @see async_all_reduce_sum
 */
template <MPIReductionArithmetic T>
    requires std::integral<T>
[[nodiscard]] MPI_Request async_all_reduce_bitwise_and(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_all_reduce(send_buffer, receive_buffer, MPI_BAND, communicator);
}

/**
 * @brief Starts a non-blocking all-reduce that yields the componentwise bitwise or on all ranks.
 * @see async_all_reduce_sum
 */
template <MPIReductionArithmetic T>
    requires std::integral<T>
[[nodiscard]] MPI_Request async_all_reduce_bitwise_or(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_all_reduce(send_buffer, receive_buffer, MPI_BOR, communicator);
}

/**
 * @brief Starts a non-blocking all-reduce that yields the componentwise bitwise exclusive or on all ranks.
 * @see async_all_reduce_sum
 */
template <MPIReductionArithmetic T>
    requires std::integral<T>
[[nodiscard]] MPI_Request async_all_reduce_bitwise_xor(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_all_reduce(send_buffer, receive_buffer, MPI_BXOR, communicator);
}

/**
 * @brief Starts a non-blocking reduce that yields the componentwise sum on the root rank.
 * @tparam T The type of data to reduce
 * @param send_buffer The local values
 * @param receive_buffer The buffer for the reduced values on the root rank (untouched on others),
 *      must have the same size as send_buffer
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the buffer sizes differ or mpi reports an error
 * @return A token that can be waited on
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_reduce_sum(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_reduce(send_buffer, receive_buffer, MPI_SUM, communicator);
}

/**
 * @brief Starts a non-blocking reduce that yields the componentwise minimum on the root rank.
 * @see async_reduce_sum
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_reduce_min(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_reduce(send_buffer, receive_buffer, MPI_MIN, communicator);
}

/**
 * @brief Starts a non-blocking reduce that yields the componentwise maximum on the root rank.
 * @see async_reduce_sum
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_reduce_max(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_reduce(send_buffer, receive_buffer, MPI_MAX, communicator);
}

/**
 * @brief Starts a non-blocking reduce that yields the componentwise product on the root rank.
 * @see async_reduce_sum
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_reduce_prod(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_reduce(send_buffer, receive_buffer, MPI_PROD, communicator);
}

/**
 * @brief Starts a non-blocking reduce that yields the componentwise bitwise and on the root rank.
 * @see async_reduce_sum
 */
template <MPIReductionArithmetic T>
    requires std::integral<T>
[[nodiscard]] MPI_Request async_reduce_bitwise_and(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_reduce(send_buffer, receive_buffer, MPI_BAND, communicator);
}

/**
 * @brief Starts a non-blocking reduce that yields the componentwise bitwise or on the root rank.
 * @see async_reduce_sum
 */
template <MPIReductionArithmetic T>
    requires std::integral<T>
[[nodiscard]] MPI_Request async_reduce_bitwise_or(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_reduce(send_buffer, receive_buffer, MPI_BOR, communicator);
}

/**
 * @brief Starts a non-blocking reduce that yields the componentwise bitwise exclusive or on the root rank.
 * @see async_reduce_sum
 */
template <MPIReductionArithmetic T>
    requires std::integral<T>
[[nodiscard]] MPI_Request async_reduce_bitwise_xor(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_reduce(send_buffer, receive_buffer, MPI_BXOR, communicator);
}
} // namespace MPIReductions

} // namespace mpiPP
