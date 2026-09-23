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

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>

namespace mpiPP {

namespace MPICollectives {
/**
 * @brief Starts gathering all the data on each rank without blocking. Uses the provided buffers.
 *      Each rank contributes send_buffer and, after completion, receive_buffer holds the
 *      contributions of all ranks in rank order. Every rank must pass the same send_buffer size,
 *      and both buffers must stay valid until the returned request completes (via wait/test).
 * @tparam T The type of data to gather
 * @param send_buffer The local contribution
 * @param receive_buffer The buffer for the gathered data, must hold (number of ranks) * send_buffer.size() elements
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception if the receive buffer has the wrong size or mpi returns an error code
 * @return A token that can be waited on
 */
template <MPICompatible T>
[[nodiscard]] MPI_Request async_all_gather(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto expected = utility::safe_cast<std::size_t>(number_ranks) * send_buffer.size();
    utility::Exception::check(receive_buffer.size() == expected, "MPICollectives::async_all_gather: receive buffer must hold {} elements but holds {}", expected, receive_buffer.size());

    auto request = MPI_Request{};
    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto count = utility::safe_cast<int>(send_buffer.size());
    const auto error_code = MPI_Iallgather(send_buffer.data(), count, type, receive_buffer.data(), count, type, communicator.get(), &request);
    utility::Exception::check(error_code == 0, "MPICollectives::async_all_gather: Error code received: {}", error_code);

    MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(number_ranks) * send_buffer.size_bytes());
    MPICounters::add_to_received(receive_buffer.size_bytes());

    return request;
}

/**
 * @brief Starts exchanging one value per rank between all ranks without blocking. Uses the provided
 *      buffers: src[i] is sent to rank i, and after completion tgt[k] holds the value received from
 *      rank k. Both buffers must have exactly (number of ranks) elements and must stay valid until
 *      the returned request completes (via wait/test).
 * @tparam T The type of data to exchange
 * @param src The elements to exchange, one per rank
 * @param tgt The buffer for the received values, one per rank
 * @param communicator The communicator to exchange within, defaults to the world communicator
 * @exception Throws an Exception if a buffer size is not equal to the number of ranks or mpi returns an error code
 * @return A token that can be waited on
 */
template <MPICompatible T>
[[nodiscard]] MPI_Request async_all_to_all(const std::span<const T> src, const std::span<T> tgt, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto number_ranks_cast = utility::safe_cast<std::size_t>(number_ranks);
    utility::Exception::check(number_ranks_cast == src.size(), "MPICollectives::async_all_to_all: Sizes do not match: {} vs {}", number_ranks_cast, src.size());
    utility::Exception::check(number_ranks_cast == tgt.size(), "MPICollectives::async_all_to_all: Sizes do not match: {} vs {}", number_ranks_cast, tgt.size());

    auto request = MPI_Request{};
    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const int error_code = MPI_Ialltoall(src.data(), 1, type, tgt.data(), 1, type, communicator.get(), &request);
    utility::Exception::check(error_code == 0, "MPICollectives::async_all_to_all: Error code received: {}", error_code);

    MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(number_ranks_cast) * sizeof(T));
    MPICounters::add_to_received(utility::safe_cast<std::uint64_t>(number_ranks_cast) * sizeof(T));

    return request;
}

/**
 * @brief Starts gathering all the data on the root rank without blocking. Uses the provided buffers.
 *      Each rank contributes send_buffer; after completion receive_buffer on the root holds the
 *      contributions of all ranks in rank order. Every rank must pass the same send_buffer size;
 *      the buffers must stay valid until the returned request completes (via wait/test).
 * @tparam T The type of data to gather
 * @param send_buffer The local contribution
 * @param receive_buffer The buffer for the gathered data on the root rank, must hold
 *      (number of ranks) * send_buffer.size() elements. Ignored (may be empty) on all other ranks
 * @param root The rank where to gather to, is default MPIRank::root_rank()
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception if the receive buffer has the wrong size on the root or mpi returns an error code
 * @return A token that can be waited on
 */
template <MPICompatible T>
[[nodiscard]] MPI_Request async_gather(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPIRank root = MPIRank::root_rank(), const MPICommunicator& communicator = MPICommunicator::World) {
    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto count = utility::safe_cast<int>(send_buffer.size());
    const auto is_root = communicator.get_my_rank() == root;
    auto request = MPI_Request{};

    if (is_root) {
        const auto number_ranks = communicator.get_number_ranks();
        const auto expected = utility::safe_cast<std::size_t>(number_ranks) * send_buffer.size();
        utility::Exception::check(receive_buffer.size() == expected, "MPICollectives::async_gather: receive buffer must hold {} elements but holds {}", expected, receive_buffer.size());

        const auto error_code = MPI_Igather(send_buffer.data(), count, type, receive_buffer.data(), count, type, root.get_rank(), communicator.get(), &request);
        utility::Exception::check(error_code == 0, "MPICollectives::async_gather: Error code received: {}", error_code);
    } else {
        const auto error_code = MPI_Igather(send_buffer.data(), count, type, nullptr, 0, type, root.get_rank(), communicator.get(), &request);
        utility::Exception::check(error_code == 0, "MPICollectives::async_gather: Error code received: {}", error_code);
    }

    MPICounters::add_to_sent(send_buffer.size_bytes());
    if (is_root) {
        const auto number_ranks = communicator.get_number_ranks();
        MPICounters::add_to_received(utility::safe_cast<std::uint64_t>(number_ranks) * send_buffer.size_bytes());
    }

    return request;
}

// --- Variable-count (V) variants ----------------------------------------------------------------
// In addition to the data buffers, the counts and displacements spans are read by MPI throughout
// the operation, so they too must stay valid until the returned request completes (via wait/test).

/**
 * @brief Starts gathering a variable amount of data on each rank without blocking. Uses the provided
 *      buffers: each rank contributes send_buffer, and after completion receive_buffer holds the
 *      contribution of rank k at offset displacements[k] with receive_counts[k] elements.
 * @tparam T The type of data to gather
 * @param send_buffer The local contribution
 * @param receive_buffer The buffer for the gathered data, large enough for the counts and displacements
 * @param receive_counts The number of elements received from each rank, one per rank
 * @param displacements Where the elements received from each rank start in receive_buffer, one per rank
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception if the counts or displacements do not have one entry per rank or mpi returns an error code
 * @return A token that can be waited on
 */
template <MPICompatible T>
[[nodiscard]] MPI_Request async_all_gather_v(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const std::span<const int> receive_counts, const std::span<const int> displacements, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto number_ranks_cast = utility::safe_cast<std::size_t>(number_ranks);
    utility::Exception::check(receive_counts.size() == number_ranks_cast, "MPICollectives::async_all_gather_v: expected {} receive counts but got {}", number_ranks_cast, receive_counts.size());
    utility::Exception::check(displacements.size() == number_ranks_cast, "MPICollectives::async_all_gather_v: expected {} displacements but got {}", number_ranks_cast, displacements.size());

    auto number_received = std::uint64_t{ 0 };
    for (auto rank = std::size_t{ 0 }; rank < number_ranks_cast; ++rank) {
        utility::Exception::check(receive_counts[rank] >= 0, "MPICollectives::async_all_gather_v: negative receive count {} for rank {}", receive_counts[rank], rank);
        utility::Exception::check(displacements[rank] >= 0, "MPICollectives::async_all_gather_v: negative displacement {} for rank {}", displacements[rank], rank);
        const auto end = utility::safe_cast<std::uint64_t>(displacements[rank]) + utility::safe_cast<std::uint64_t>(receive_counts[rank]);
        utility::Exception::check(end <= receive_buffer.size(), "MPICollectives::async_all_gather_v: block for rank {} ends at {} but receive buffer holds {} elements", rank, end, receive_buffer.size());
        number_received += utility::safe_cast<std::uint64_t>(receive_counts[rank]);
    }

    auto request = MPI_Request{};
    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto send_count = utility::safe_cast<int>(send_buffer.size());
    const auto error_code = MPI_Iallgatherv(send_buffer.data(), send_count, type, receive_buffer.data(), receive_counts.data(), displacements.data(), type, communicator.get(), &request);
    utility::Exception::check(error_code == 0, "MPICollectives::async_all_gather_v: Error code received: {}", error_code);

    MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(number_ranks) * send_buffer.size_bytes());
    MPICounters::add_to_received(number_received * sizeof(T));

    return request;
}

/**
 * @brief Starts exchanging a variable amount of data between all ranks without blocking. Uses the
 *      provided buffers: the block for rank k starts in send_buffer at send_displacements[k] with
 *      send_counts[k] elements; after completion the block from rank i is in receive_buffer at
 *      receive_displacements[i] with receive_counts[i] elements.
 * @tparam T The type of data to exchange
 * @param send_buffer The local data to send
 * @param send_counts The number of elements sent to each rank, one per rank
 * @param send_displacements Where the elements sent to each rank start in send_buffer, one per rank
 * @param receive_buffer The buffer for the received data
 * @param receive_counts The number of elements received from each rank, one per rank
 * @param receive_displacements Where the elements received from each rank start in receive_buffer, one per rank
 * @param communicator The communicator to exchange within, defaults to the world communicator
 * @exception Throws an Exception if any counts or displacements do not have one entry per rank or mpi returns an error code
 * @return A token that can be waited on
 */
template <MPICompatible T>
[[nodiscard]] MPI_Request async_all_to_all_v(const std::span<const T> send_buffer, const std::span<const int> send_counts, const std::span<const int> send_displacements, const std::span<T> receive_buffer, const std::span<const int> receive_counts, const std::span<const int> receive_displacements, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto number_ranks_cast = utility::safe_cast<std::size_t>(number_ranks);
    utility::Exception::check(send_counts.size() == number_ranks_cast, "MPICollectives::async_all_to_all_v: expected {} send counts but got {}", number_ranks_cast, send_counts.size());
    utility::Exception::check(send_displacements.size() == number_ranks_cast, "MPICollectives::async_all_to_all_v: expected {} send displacements but got {}", number_ranks_cast, send_displacements.size());
    utility::Exception::check(receive_counts.size() == number_ranks_cast, "MPICollectives::async_all_to_all_v: expected {} receive counts but got {}", number_ranks_cast, receive_counts.size());
    utility::Exception::check(receive_displacements.size() == number_ranks_cast, "MPICollectives::async_all_to_all_v: expected {} receive displacements but got {}", number_ranks_cast, receive_displacements.size());

    auto number_sent = std::uint64_t{ 0 };
    auto number_received = std::uint64_t{ 0 };
    for (auto rank = std::size_t{ 0 }; rank < number_ranks_cast; ++rank) {
        utility::Exception::check(send_counts[rank] >= 0, "MPICollectives::async_all_to_all_v: negative send count {} for rank {}", send_counts[rank], rank);
        utility::Exception::check(send_displacements[rank] >= 0, "MPICollectives::async_all_to_all_v: negative send displacement {} for rank {}", send_displacements[rank], rank);
        utility::Exception::check(receive_counts[rank] >= 0, "MPICollectives::async_all_to_all_v: negative receive count {} for rank {}", receive_counts[rank], rank);
        utility::Exception::check(receive_displacements[rank] >= 0, "MPICollectives::async_all_to_all_v: negative receive displacement {} for rank {}", receive_displacements[rank], rank);

        const auto send_end = utility::safe_cast<std::uint64_t>(send_displacements[rank]) + utility::safe_cast<std::uint64_t>(send_counts[rank]);
        const auto receive_end = utility::safe_cast<std::uint64_t>(receive_displacements[rank]) + utility::safe_cast<std::uint64_t>(receive_counts[rank]);
        utility::Exception::check(send_end <= send_buffer.size(), "MPICollectives::async_all_to_all_v: block for rank {} ends at {} but send buffer holds {} elements", rank, send_end, send_buffer.size());
        utility::Exception::check(receive_end <= receive_buffer.size(), "MPICollectives::async_all_to_all_v: block for rank {} ends at {} but receive buffer holds {} elements", rank, receive_end, receive_buffer.size());
        number_sent += utility::safe_cast<std::uint64_t>(send_counts[rank]);
        number_received += utility::safe_cast<std::uint64_t>(receive_counts[rank]);
    }

    auto request = MPI_Request{};
    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Ialltoallv(send_buffer.data(), send_counts.data(), send_displacements.data(), type, receive_buffer.data(), receive_counts.data(), receive_displacements.data(), type, communicator.get(), &request);
    utility::Exception::check(error_code == 0, "MPICollectives::async_all_to_all_v: Error code received: {}", error_code);

    MPICounters::add_to_sent(number_sent * sizeof(T));
    MPICounters::add_to_received(number_received * sizeof(T));

    return request;
}

/**
 * @brief Starts gathering a variable amount of data on the root rank without blocking. Uses the
 *      provided buffers: each rank contributes send_buffer, and after completion receive_buffer on
 *      the root holds the contribution of rank k at offset displacements[k] with receive_counts[k]
 *      elements. On non-root ranks the receive buffer, counts and displacements are ignored.
 * @tparam T The type of data to gather
 * @param send_buffer The local contribution
 * @param receive_buffer The buffer for the gathered data on the root rank, ignored on other ranks
 * @param receive_counts The number of elements received from each rank (root only), one per rank
 * @param displacements Where the elements received from each rank start in receive_buffer (root only), one per rank
 * @param root The rank where to gather to, is default MPIRank::root_rank()
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception if the counts or displacements do not have one entry per rank on the root or mpi returns an error code
 * @return A token that can be waited on
 */
template <MPICompatible T>
[[nodiscard]] MPI_Request async_gather_v(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const std::span<const int> receive_counts, const std::span<const int> displacements, const MPIRank root = MPIRank::root_rank(), const MPICommunicator& communicator = MPICommunicator::World) {
    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto send_count = utility::safe_cast<int>(send_buffer.size());
    const auto is_root = communicator.get_my_rank() == root;
    auto request = MPI_Request{};

    if (is_root) {
        const auto number_ranks_cast = utility::safe_cast<std::size_t>(communicator.get_number_ranks());
        utility::Exception::check(receive_counts.size() == number_ranks_cast, "MPICollectives::async_gather_v: expected {} receive counts but got {}", number_ranks_cast, receive_counts.size());
        utility::Exception::check(displacements.size() == number_ranks_cast, "MPICollectives::async_gather_v: expected {} displacements but got {}", number_ranks_cast, displacements.size());

        for (auto rank = std::size_t{ 0 }; rank < number_ranks_cast; ++rank) {
            utility::Exception::check(receive_counts[rank] >= 0, "MPICollectives::async_gather_v: negative receive count {} for rank {}", receive_counts[rank], rank);
            utility::Exception::check(displacements[rank] >= 0, "MPICollectives::async_gather_v: negative displacement {} for rank {}", displacements[rank], rank);
            const auto end = utility::safe_cast<std::uint64_t>(displacements[rank]) + utility::safe_cast<std::uint64_t>(receive_counts[rank]);
            utility::Exception::check(end <= receive_buffer.size(), "MPICollectives::async_gather_v: block for rank {} ends at {} but receive buffer holds {} elements", rank, end, receive_buffer.size());
        }

        const auto error_code = MPI_Igatherv(send_buffer.data(), send_count, type, receive_buffer.data(), receive_counts.data(), displacements.data(), type, root.get_rank(), communicator.get(), &request);
        utility::Exception::check(error_code == 0, "MPICollectives::async_gather_v: Error code received: {}", error_code);
    } else {
        const auto error_code = MPI_Igatherv(send_buffer.data(), send_count, type, nullptr, nullptr, nullptr, type, root.get_rank(), communicator.get(), &request);
        utility::Exception::check(error_code == 0, "MPICollectives::async_gather_v: Error code received: {}", error_code);
    }

    MPICounters::add_to_sent(send_buffer.size_bytes());
    if (is_root) {
        const auto number_received = std::accumulate(receive_counts.begin(), receive_counts.end(), std::uint64_t{ 0 });
        MPICounters::add_to_received(number_received * sizeof(T));
    }

    return request;
}
} // namespace MPICollectives

} // namespace mpiPP
