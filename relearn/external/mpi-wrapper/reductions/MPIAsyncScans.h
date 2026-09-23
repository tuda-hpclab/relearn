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

namespace detail {
/**
 * @brief Shared implementation of the non-blocking inclusive scans. The buffers and their matching
 *      sizes must stay valid until the returned request completes.
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_scan(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPI_Op operation, const MPICommunicator& communicator) {
    utility::Exception::check(send_buffer.size() == receive_buffer.size(), "MPIScans::async_inclusive_scan: send and receive buffers differ in size: {} vs {}", send_buffer.size(), receive_buffer.size());

    auto request = MPI_Request{};
    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto count = utility::safe_cast<int>(send_buffer.size());
    const auto error_code = MPI_Iscan(send_buffer.data(), receive_buffer.data(), count, type, operation, communicator.get(), &request);
    utility::Exception::check(error_code == 0, "Asynchronous inclusive scan returned the error: {}", error_code);

    MPICounters::add_to_sent(send_buffer.size_bytes());
    MPICounters::add_to_received(receive_buffer.size_bytes());

    return request;
}

/**
 * @brief Shared implementation of the non-blocking exclusive scans. Unlike the blocking exclusive
 *      scans, the result on the root rank is left as MPI defines it (undefined) rather than
 *      normalized to the neutral element: the wrapper cannot post-process the buffer before the
 *      operation completes. The buffers and their matching sizes must stay valid until the returned
 *      request completes.
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_exscan(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPI_Op operation, const MPICommunicator& communicator) {
    utility::Exception::check(send_buffer.size() == receive_buffer.size(), "MPIScans::async_exclusive_scan: send and receive buffers differ in size: {} vs {}", send_buffer.size(), receive_buffer.size());

    auto request = MPI_Request{};
    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto count = utility::safe_cast<int>(send_buffer.size());
    const auto error_code = MPI_Iexscan(send_buffer.data(), receive_buffer.data(), count, type, operation, communicator.get(), &request);
    utility::Exception::check(error_code == 0, "Asynchronous exclusive scan returned the error: {}", error_code);

    MPICounters::add_to_sent(send_buffer.size_bytes());
    MPICounters::add_to_received(receive_buffer.size_bytes());

    return request;
}
} // namespace detail

namespace MPIScans {
// Span-based, so a single function serves both the scalar case (a span of length 1) and the
// componentwise case (a longer span). Every rank must pass buffers of the same size; the buffers
// must stay valid until the returned request completes (via wait/test).

/**
 * @brief Starts a non-blocking inclusive prefix sum: after completion rank i holds the componentwise
 *      sum of the local values of ranks 0..i.
 * @tparam T The type of data to scan
 * @param send_buffer The local values
 * @param receive_buffer The buffer for the result, must have the same size as send_buffer
 * @param communicator The communicator to scan within, defaults to the world communicator
 * @exception Throws an Exception if the buffer sizes differ or mpi reports an error
 * @return A token that can be waited on
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_inclusive_scan_sum(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_scan(send_buffer, receive_buffer, MPI_SUM, communicator);
}

/**
 * @brief Starts a non-blocking inclusive prefix minimum (ranks 0..i).
 * @see async_inclusive_scan_sum
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_inclusive_scan_min(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_scan(send_buffer, receive_buffer, MPI_MIN, communicator);
}

/**
 * @brief Starts a non-blocking inclusive prefix maximum (ranks 0..i).
 * @see async_inclusive_scan_sum
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_inclusive_scan_max(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_scan(send_buffer, receive_buffer, MPI_MAX, communicator);
}

/**
 * @brief Starts a non-blocking exclusive prefix sum: after completion rank i holds the componentwise
 *      sum of the local values of ranks 0..i-1. Note: unlike the blocking exclusive_scan_sum, the
 *      result on the root rank is left as MPI defines it (undefined), not normalized to zero, since
 *      the wrapper cannot post-process the buffer before the operation completes.
 * @tparam T The type of data to scan
 * @param send_buffer The local values
 * @param receive_buffer The buffer for the result, must have the same size as send_buffer
 * @param communicator The communicator to scan within, defaults to the world communicator
 * @exception Throws an Exception if the buffer sizes differ or mpi reports an error
 * @return A token that can be waited on
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_exclusive_scan_sum(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_exscan(send_buffer, receive_buffer, MPI_SUM, communicator);
}

/**
 * @brief Starts a non-blocking exclusive prefix minimum (ranks 0..i-1). The root-rank result is
 *      left undefined, see async_exclusive_scan_sum.
 * @see async_exclusive_scan_sum
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_exclusive_scan_min(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_exscan(send_buffer, receive_buffer, MPI_MIN, communicator);
}

/**
 * @brief Starts a non-blocking exclusive prefix maximum (ranks 0..i-1). The root-rank result is
 *      left undefined, see async_exclusive_scan_sum.
 * @see async_exclusive_scan_sum
 */
template <MPIReductionArithmetic T>
[[nodiscard]] MPI_Request async_exclusive_scan_max(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    return detail::async_exscan(send_buffer, receive_buffer, MPI_MAX, communicator);
}
} // namespace MPIScans

} // namespace mpiPP
