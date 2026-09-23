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

namespace MPIScatters {
/**
 * @brief Starts scattering a fixed number of values per rank from the sender without blocking. Uses
 *      the provided buffers: on the sender send_buffer holds the data for all ranks in rank order;
 *      after completion receive_buffer holds this rank's chunk. Every rank must pass the same
 *      receive_buffer size, and both buffers must stay valid until the returned request completes
 *      (via wait/test).
 * @tparam T The type of data to scatter
 * @param send_buffer The values to scatter on the sender, must hold (number of ranks) * receive_buffer.size()
 *      elements. Ignored (may be empty) on all other ranks
 * @param receive_buffer The buffer for the values destined for this rank
 * @param sender The rank which scatters the values, is default MPIRank::root_rank()
 * @param communicator The communicator to scatter within, defaults to the world communicator
 * @exception Throws an Exception if sender is too large, if the send buffer has the wrong size on the sender,
 *      or mpi returns an error code
 * @return A token that can be waited on
 */
template <MPICompatible T>
[[nodiscard]] MPI_Request async_scatter(const std::span<const T> send_buffer, const std::span<T> receive_buffer, const MPIRank sender = MPIRank::root_rank(), const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto sender_rank = sender.get_rank();
    utility::Exception::check(sender_rank < number_ranks, "MPIScatters::async_scatter: There are {} ranks, but should scatter from {}", number_ranks, sender_rank);

    const auto is_sender = communicator.get_my_rank() == sender;
    if (is_sender) {
        const auto expected = utility::safe_cast<std::size_t>(number_ranks) * receive_buffer.size();
        utility::Exception::check(send_buffer.size() == expected, "MPIScatters::async_scatter: send buffer must hold {} elements but holds {}", expected, send_buffer.size());
    }

    auto request = MPI_Request{};
    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto count = utility::safe_cast<int>(receive_buffer.size());
    const auto error_code = MPI_Iscatter(send_buffer.data(), count, type, receive_buffer.data(), count, type, sender_rank, communicator.get(), &request);
    utility::Exception::check(error_code == 0, "MPIScatters::async_scatter: Error code received: {}", error_code);

    if (is_sender) {
        MPICounters::add_to_sent(send_buffer.size_bytes());
    }
    MPICounters::add_to_received(receive_buffer.size_bytes());

    return request;
}

/**
 * @brief Starts scattering a variable amount of data per rank from the sender without blocking. Uses
 *      the provided buffers: on the sender the block for rank k starts in send_buffer at
 *      displacements[k] with send_counts[k] elements; after completion receive_buffer holds this
 *      rank's chunk. The counts and displacements are ignored on non-sender ranks. In addition to
 *      the data buffers, the counts and displacements spans must stay valid until the returned
 *      request completes (via wait/test).
 * @tparam T The type of data to scatter
 * @param send_buffer The values to scatter on the sender, ignored on other ranks
 * @param send_counts The number of elements sent to each rank (sender only), one per rank
 * @param displacements Where the elements for each rank start in send_buffer (sender only), one per rank
 * @param receive_buffer The buffer for the values destined for this rank
 * @param sender The rank which scatters the values, is default MPIRank::root_rank()
 * @param communicator The communicator to scatter within, defaults to the world communicator
 * @exception Throws an Exception if sender is too large, if the counts or displacements do not have one entry
 *      per rank on the sender, or mpi returns an error code
 * @return A token that can be waited on
 */
template <MPICompatible T>
[[nodiscard]] MPI_Request async_scatter_v(const std::span<const T> send_buffer, const std::span<const int> send_counts, const std::span<const int> displacements, const std::span<T> receive_buffer, const MPIRank sender = MPIRank::root_rank(), const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto sender_rank = sender.get_rank();
    utility::Exception::check(sender_rank < number_ranks, "MPIScatters::async_scatter_v: There are {} ranks, but should scatter from {}", number_ranks, sender_rank);

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto receive_count = utility::safe_cast<int>(receive_buffer.size());
    const auto is_sender = communicator.get_my_rank() == sender;
    auto request = MPI_Request{};

    if (is_sender) {
        const auto number_ranks_cast = utility::safe_cast<std::size_t>(number_ranks);
        utility::Exception::check(send_counts.size() == number_ranks_cast, "MPIScatters::async_scatter_v: expected {} send counts but got {}", number_ranks_cast, send_counts.size());
        utility::Exception::check(displacements.size() == number_ranks_cast, "MPIScatters::async_scatter_v: expected {} displacements but got {}", number_ranks_cast, displacements.size());

        for (auto rank = std::size_t{ 0 }; rank < number_ranks_cast; ++rank) {
            utility::Exception::check(send_counts[rank] >= 0, "MPIScatters::async_scatter_v: negative send count {} for rank {}", send_counts[rank], rank);
            utility::Exception::check(displacements[rank] >= 0, "MPIScatters::async_scatter_v: negative displacement {} for rank {}", displacements[rank], rank);
            const auto end = utility::safe_cast<std::uint64_t>(displacements[rank]) + utility::safe_cast<std::uint64_t>(send_counts[rank]);
            utility::Exception::check(end <= send_buffer.size(), "MPIScatters::async_scatter_v: block for rank {} ends at {} but send buffer holds {} elements", rank, end, send_buffer.size());
        }

        const auto error_code = MPI_Iscatterv(send_buffer.data(), send_counts.data(), displacements.data(), type, receive_buffer.data(), receive_count, type, sender_rank, communicator.get(), &request);
        utility::Exception::check(error_code == 0, "MPIScatters::async_scatter_v: Error code received: {}", error_code);
    } else {
        const auto error_code = MPI_Iscatterv(nullptr, nullptr, nullptr, type, receive_buffer.data(), receive_count, type, sender_rank, communicator.get(), &request);
        utility::Exception::check(error_code == 0, "MPIScatters::async_scatter_v: Error code received: {}", error_code);
    }

    if (is_sender) {
        const auto number_sent = std::accumulate(send_counts.begin(), send_counts.end(), std::uint64_t{ 0 });
        MPICounters::add_to_sent(number_sent * sizeof(T));
    }
    MPICounters::add_to_received(receive_buffer.size_bytes());

    return request;
}
} // namespace MPIScatters

} // namespace mpiPP
