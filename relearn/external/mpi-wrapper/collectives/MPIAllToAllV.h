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

#include "mpi-wrapper/collectives/MPIAllToAll.h"
#include "mpi-wrapper/communicator/MPICommunicator.h"
#include "mpi-wrapper/core/MPICounters.h"
#include "mpi-wrapper/core/MPITypes.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>
#include <cpp-utility/data/displacement.hpp>
#include <cpp-utility/data/reorganize.hpp>

#include <mpi.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <vector>

namespace mpiPP {

namespace MPICollectives {
namespace detail {
template <typename Counts, typename Displacements>
[[nodiscard]] std::size_t validate_all_to_all_v_layout(const Counts& counts, const Displacements& displacements, const std::size_t number_ranks, const char* direction) {
    utility::Exception::check(std::ranges::size(counts) == number_ranks, "MPICollectives::all_to_all_v: Expected {} {} counts, got {}", number_ranks, direction, std::ranges::size(counts));
    utility::Exception::check(std::ranges::size(displacements) == number_ranks, "MPICollectives::all_to_all_v: Expected {} {} displacements, got {}", number_ranks, direction, std::ranges::size(displacements));

    auto required_size = std::size_t{ 0 };
    for (auto rank = std::size_t{ 0 }; rank < number_ranks; ++rank) {
        utility::Exception::check(counts[rank] >= 0, "MPICollectives::all_to_all_v: Negative {} count {} for rank {}", direction, counts[rank], rank);
        utility::Exception::check(displacements[rank] >= 0, "MPICollectives::all_to_all_v: Negative {} displacement {} for rank {}", direction, displacements[rank], rank);
        required_size = std::max(required_size, utility::safe_cast<std::size_t>(counts[rank]) + utility::safe_cast<std::size_t>(displacements[rank]));
    }
    return required_size;
}
} // namespace detail

/**
 * @brief Exchanges a variable amount of data between all MPI ranks using caller-provided buffers.
 *      Before the call:
 *          rank i has the block destined for rank k in src, starting at sendDispls[k] with sendCounts[k] elements
 *      After the call:
 *          rank k has the block received from rank i in dest, starting at recvDispls[i] with recvCounts[i] elements
 * @tparam T The type of data to exchange
 * @param src The buffer of the local data elements to send, is accessed for each rank at the specified displacement
 * @param sendCounts The nonnegative number of elements sent to each rank; must point to one entry per rank
 * @param sendDispls The nonnegative source displacement for each rank; must contain one entry per rank
 * @param dest The destination buffer where the received data elements are stored, is accessed for each rank at the specified displacement
 * @param recvCounts The nonnegative number of elements received from each rank; must point to one entry per rank
 * @param recvDispls The nonnegative destination displacement for each rank; must contain one entry per rank
 * @param communicator The communicator to exchange within, defaults to the world communicator
 * @exception Throws an Exception if any of sendCounts, sendDispls, recvCounts, recvDispls is nullptr,
 *      if src is nullptr although elements should be sent, if dest is nullptr although elements should be received,
 *      a count or displacement is negative, or MPI returns an error code
 */
template <MPICompatible T>
void all_to_all_v(const T* src, const int* const sendCounts, const int* const sendDispls, T* dest, const int* const recvCounts, const int* const recvDispls, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto my_rank = communicator.get_my_rank();
    const auto number_ranks = communicator.get_number_ranks();

    // The count and displacement arrays are always read in full and must be valid.
    utility::Exception::check(sendCounts != nullptr, "MPICollectives::all_to_all_v: sendCounts is nullptr on rank {}", my_rank);
    utility::Exception::check(sendDispls != nullptr, "MPICollectives::all_to_all_v: sendDispls is nullptr on rank {}", my_rank);
    utility::Exception::check(recvCounts != nullptr, "MPICollectives::all_to_all_v: recvCounts is nullptr on rank {}", my_rank);
    utility::Exception::check(recvDispls != nullptr, "MPICollectives::all_to_all_v: recvDispls is nullptr on rank {}", my_rank);

    // The data buffers are only dereferenced according to the counts, so they may be nullptr for empty transfers
    auto number_sent = std::uint64_t{ 0 };
    auto number_received = std::uint64_t{ 0 };
    for (auto rank = 0; rank < number_ranks; ++rank) {
        utility::Exception::check(sendCounts[rank] >= 0, "MPICollectives::all_to_all_v: Negative send count {} for rank {}", sendCounts[rank], rank);
        utility::Exception::check(sendDispls[rank] >= 0, "MPICollectives::all_to_all_v: Negative send displacement {} for rank {}", sendDispls[rank], rank);
        utility::Exception::check(recvCounts[rank] >= 0, "MPICollectives::all_to_all_v: Negative receive count {} for rank {}", recvCounts[rank], rank);
        utility::Exception::check(recvDispls[rank] >= 0, "MPICollectives::all_to_all_v: Negative receive displacement {} for rank {}", recvDispls[rank], rank);
        number_sent += utility::safe_cast<std::uint64_t>(sendCounts[rank]);
        number_received += utility::safe_cast<std::uint64_t>(recvCounts[rank]);
    }
    utility::Exception::check(src != nullptr || number_sent == 0,
                              "MPICollectives::all_to_all_v: src is nullptr on rank {}, but {} elements should be sent", my_rank, number_sent);
    utility::Exception::check(dest != nullptr || number_received == 0,
                              "MPICollectives::all_to_all_v: dest is nullptr on rank {}, but {} elements should be received", my_rank, number_received);

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Alltoallv(src, sendCounts, sendDispls, type, dest, recvCounts, recvDispls, type, communicator.get());
    utility::Exception::check(error_code == 0, "All-To-All-Ving all values returned the error: {}", error_code);

    MPICounters::add_to_sent(number_sent * sizeof(T));
    MPICounters::add_to_received(number_received * sizeof(T));
}

/**
 * @brief Exchanges a variable amount of data between all MPI ranks using caller-provided ranges.
 *      Before the call:
 *          rank i has the block destined for rank k in src, starting at sendDispls[k] with sendCounts[k] elements
 *      After the call:
 *          rank k has the block received from rank i in dest, starting at recvDispls[i] with recvCounts[i] elements
 *      Offers more convenient parameter types
 * @tparam SourceRange A contiguous input range
 * @tparam TargetRange A borrowed contiguous output range with the same element type as SourceRange
 * @param src The local data elements to send; must cover every block described by sendCounts and sendDispls
 * @param sendCounts The nonnegative number of elements sent to each rank; must contain one entry per rank
 * @param sendDispls The nonnegative source displacement for each rank; must contain one entry per rank
 * @param dest The destination buffer; must cover every block described by recvCounts and recvDispls
 * @param recvCounts The nonnegative number of elements received from each rank; must contain one entry per rank
 * @param recvDispls The nonnegative destination displacement for each rank; must contain one entry per rank
 * @param communicator The communicator to exchange within, defaults to the world communicator
 * @exception Throws an Exception if a count or displacement range does not contain exactly one entry per rank,
 *      a count or displacement is negative, a data buffer is too small for its layout, or MPI reports an error
 */
template <MPICompatibleRange SourceRange, MPICompatibleRangeOfType<std::ranges::range_value_t<SourceRange>> TargetRange>
    requires std::ranges::borrowed_range<TargetRange>
void all_to_all_v(const SourceRange& src, const MPICompatibleRangeOfType<int> auto& sendCounts, const MPICompatibleRangeOfType<int> auto& sendDispls,
                  TargetRange&& dest, const MPICompatibleRangeOfType<int> auto& recvCounts, const MPICompatibleRangeOfType<int> auto& recvDispls, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks_cast();
    const auto required_source_size = detail::validate_all_to_all_v_layout(sendCounts, sendDispls, number_ranks, "send");
    const auto required_destination_size = detail::validate_all_to_all_v_layout(recvCounts, recvDispls, number_ranks, "receive");
    utility::Exception::check(std::ranges::size(src) >= required_source_size, "MPICollectives::all_to_all_v: Source has {} elements, but the layout requires {}", std::ranges::size(src), required_source_size);
    utility::Exception::check(std::ranges::size(dest) >= required_destination_size, "MPICollectives::all_to_all_v: Destination has {} elements, but the layout requires {}", std::ranges::size(dest), required_destination_size);
    all_to_all_v<std::ranges::range_value_t<SourceRange>>(src.data(), sendCounts.data(), sendDispls.data(), dest.data(), recvCounts.data(), recvDispls.data(), communicator);
}

/**
 * @brief Exchanges a variable amount of data between all MPI ranks and returns a newly allocated receive buffer.
 *      Before the call:
 *          rank i has the block destined for rank k in src, starting at sendDispls[k] with sendCounts[k] elements
 *      After the call:
 *          rank k has the block received from rank i in <return>, starting at recvDispls[i] with recvCounts[i] elements
 * @tparam T The type of data to exchange
 * @param src The buffer of the local data elements to send, is accessed for each rank at the specified displacement
 * @param sendCounts The number of elements sent to each rank. Is accessed at 0, ..., number_ranks-1
 * @param sendDispls The displacement for the elements in src, i.e., where the elements sent to rank k start in src.
 *      Is accessed at 0, ..., number_ranks-1
 * @param recvCounts The number of elements received from each rank. Is accessed at 0, ..., number_ranks-1
 * @param recvDispls The displacement for the elements in <return>, i.e., where the elements received from rank i start in <return>.
 *      Is accessed at 0, ..., number_ranks-1
 * @param communicator The communicator to exchange within, defaults to the world communicator
 * @exception Throws an Exception if any of sendCounts, sendDispls, recvCounts, recvDispls is nullptr,
 *      if src is nullptr although elements should be sent, or MPI returns an error code
 * @return A vector large enough to include every receive block at its specified displacement
 */
template <MPICompatible T>
[[nodiscard]] std::vector<T> all_to_all_v(const T* src, const int* const sendCounts, const int* const sendDispls, const int* const recvCounts, const int* const recvDispls, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto my_rank = communicator.get_my_rank();
    const auto number_ranks = communicator.get_number_ranks();

    // recvCounts and recvDispls are dereferenced here to size the buffer; the remaining checks are
    // performed by the buffer overload below
    utility::Exception::check(recvCounts != nullptr, "MPICollectives::all_to_all_v: recvCounts is nullptr on rank {}", my_rank);
    utility::Exception::check(recvDispls != nullptr, "MPICollectives::all_to_all_v: recvDispls is nullptr on rank {}", my_rank);

    auto buffer_size = std::size_t{ 0 };
    for (auto rank = 0; rank < number_ranks; rank++) {
        const auto last_element = utility::safe_cast<std::size_t>(recvCounts[rank]) + utility::safe_cast<std::size_t>(recvDispls[rank]);
        buffer_size = std::max(buffer_size, last_element);
    }

    auto buffer = std::vector<T>(buffer_size);

    all_to_all_v(src, sendCounts, sendDispls, buffer.data(), recvCounts, recvDispls, communicator);

    return buffer;
}

/**
 * @brief Exchanges a variable amount of data between all MPI ranks. Calculates the sizes and the displacements
 *      automatically, such that the data is tightly packed in order of ranks.
 *      Call arguments:
 *          rank i calls with [[data for rank 0], [data for rank 1], ...]
 *      Return values:
 *          <return> = [[data from rank 0], [data from rank 1], ...] on every rank
 * @tparam T The type of data to exchange
 * @param src The local values to send, src[k] holds the data destined for rank k
 * @param communicator The communicator to exchange within, defaults to the world communicator
 * @exception Throws an Exception if src.size() is not equal to the number of ranks or MPI returns an error code
 * @return All received values, split per source rank
 */
template <MPICompatible T>
[[nodiscard]] std::vector<std::vector<T>> all_to_all_v(const std::vector<std::vector<T>>& src, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto number_ranks_cast = utility::safe_cast<std::size_t>(number_ranks);
    utility::Exception::check(number_ranks_cast == src.size(), "MPICollectives::all_to_all_v: Sizes do not match: {} vs {}", number_ranks_cast, src.size());

    auto send_counts = std::vector<int>{};
    send_counts.reserve(src.size());
    for (const auto& part : src) {
        send_counts.emplace_back(utility::safe_cast<int>(part.size()));
    }

    // exchange the per-rank counts so each rank learns how much it will receive from every other rank
    const auto recv_counts = MPICollectives::all_to_all<int>(send_counts, communicator);

    const auto send_displacements = utility::calculate_displacements<int>(send_counts);
    const auto recv_displacements = utility::calculate_displacements<int>(recv_counts);

    auto total_send = std::size_t{ 0 };
    for (const auto& part : src) {
        total_send += part.size();
    }

    auto flattened_send = std::vector<T>{};
    flattened_send.reserve(total_send);
    for (const auto& part : src) {
        flattened_send.insert(flattened_send.end(), part.begin(), part.end());
    }

    auto total_recv = std::size_t{ 0 };
    for (const auto count : recv_counts) {
        total_recv += utility::safe_cast<std::size_t>(count);
    }

    auto flattened_recv = std::vector<T>(total_recv);

    all_to_all_v<T>(flattened_send.data(), send_counts.data(), send_displacements.data(), flattened_recv.data(), recv_counts.data(), recv_displacements.data(), communicator);

    return utility::reorganize_data<T, const int, const int>(flattened_recv, recv_counts, recv_displacements);
}
} // namespace MPICollectives

} // namespace mpiPP
