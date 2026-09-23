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

#include "mpi-wrapper/collectives/MPIGather.h"
#include "mpi-wrapper/communicator/MPICommunicator.h"
#include "mpi-wrapper/core/MPICounters.h"
#include "mpi-wrapper/core/MPIRank.h"
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
[[nodiscard]] std::size_t validate_gatherv_layout(const Counts& counts, const Displacements& displacements, const std::size_t number_ranks) {
    utility::Exception::check(std::ranges::size(counts) == number_ranks, "MPICollectives::gatherv: Expected {} receive counts, got {}", number_ranks, std::ranges::size(counts));
    utility::Exception::check(std::ranges::size(displacements) == number_ranks, "MPICollectives::gatherv: Expected {} displacements, got {}", number_ranks, std::ranges::size(displacements));

    auto required_size = std::size_t{ 0 };
    for (auto rank = std::size_t{ 0 }; rank < number_ranks; ++rank) {
        utility::Exception::check(counts[rank] >= 0, "MPICollectives::gatherv: Negative receive count {} for rank {}", counts[rank], rank);
        utility::Exception::check(displacements[rank] >= 0, "MPICollectives::gatherv: Negative displacement {} for rank {}", displacements[rank], rank);
        required_size = std::max(required_size, utility::safe_cast<std::size_t>(counts[rank]) + utility::safe_cast<std::size_t>(displacements[rank]));
    }
    return required_size;
}
} // namespace detail

/**
 * @brief Gathers a variable amount of data into a caller-provided buffer on the specified root rank.
 *      Before the call:
 *          rank k has n values
 *      After the call:
 *          rank root stores each rank's block in dest at the corresponding displacement
 * @tparam T The type of data to gather
 * @param src The buffer of the local data elements, is accessed at 0, ..., count-1
 * @param count The number of data elements the current rank sends
 * @param dest The destination buffer where the data elements are gathered. Only relevant at the specified root rank
 * @param destCounts The number of elements from each rank that are received. Is accessed at 0, ..., number_ranks-1.
 *      Is ignored at each rank but the specified root rank
 * @param displs The displacement for the elements in dest, i.e., where the received elements of rank k start in dest.
 *      Is ignored at each rank but the specified root rank
 * @param root The rank that receives the data
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception if root is too large, if src is nullptr although count > 0,
 *      on root if destCounts or displs is nullptr or dest is nullptr although elements should be received,
 *      a count or displacement is negative, or MPI returns an error code
 */
template <MPICompatible T>
void gatherv(const T* src, const std::integral auto count, T* dest, const int* const destCounts, const int* const displs, const MPIRank root, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto my_rank = communicator.get_my_rank();
    const auto number_ranks = communicator.get_number_ranks();
    const auto root_rank = root.get_rank();
    const auto count_cast = utility::safe_cast<int>(count);
    utility::Exception::check(root_rank < number_ranks, "MPICollectives::gatherv: There are {} ranks, but should gather to {}", number_ranks, root_rank);

    // The data buffers are only dereferenced according to the counts, so they may be nullptr for empty transfers.
    // The count and displacement arrays are always read in full and must be valid.
    utility::Exception::check(src != nullptr || count_cast == 0, "MPICollectives::gatherv: src is nullptr on rank {}, but {} elements should be sent", my_rank, count_cast);

    auto number_received = std::uint64_t{ 0 };
    if (my_rank == root) {
        utility::Exception::check(destCounts != nullptr, "MPICollectives::gatherv: destCounts is nullptr on rank {}", my_rank);
        utility::Exception::check(displs != nullptr, "MPICollectives::gatherv: displs is nullptr on rank {}", my_rank);

        for (auto rank = 0; rank < number_ranks; ++rank) {
            utility::Exception::check(destCounts[rank] >= 0, "MPICollectives::gatherv: Negative receive count {} for rank {}", destCounts[rank], rank);
            utility::Exception::check(displs[rank] >= 0, "MPICollectives::gatherv: Negative displacement {} for rank {}", displs[rank], rank);
            number_received += utility::safe_cast<std::uint64_t>(destCounts[rank]);
        }
        utility::Exception::check(dest != nullptr || number_received == 0,
                                  "MPICollectives::gatherv: dest is nullptr on rank {}, but {} elements should be received", my_rank, number_received);
    }

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Gatherv(src, count_cast, type, dest, destCounts, displs, type, root.get_rank(), communicator.get());
    utility::Exception::check(error_code == 0, "Gatherving all values returned the error: {}", error_code);

    MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(count) * sizeof(T));

    if (my_rank == root) {
        MPICounters::add_to_received(number_received * sizeof(T));
    }
}

/**
 * @brief Gathers a variable amount of data into a caller-provided range on the specified root rank.
 *      Before the call:
 *          rank k has n values
 *      After the call:
 *          rank root stores each rank's block in dest at the corresponding displacement
 *      Offers more convenient parameter types
 * @tparam SourceRange A contiguous input range
 * @tparam TargetRange A borrowed contiguous output range with the same element type as SourceRange
 * @param src The local data elements to gather at the specified root rank
 * @param dest The destination buffer where the data elements are gathered. Only relevant at the specified root rank
 * @param destCounts The nonnegative number of elements received from each rank; on root, must contain one entry per rank
 * @param displs The nonnegative destination displacement for each rank; on root, must contain one entry per rank
 * @param root The rank that receives the data
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception if root is too large, a layout range has the wrong size, a count or displacement
 *      is negative, the destination is too small for the layout, or MPI reports an error
 */
template <MPICompatibleRange SourceRange, MPICompatibleRangeOfType<std::ranges::range_value_t<SourceRange>> TargetRange>
    requires std::ranges::borrowed_range<TargetRange>
void gatherv(const SourceRange& src, TargetRange&& dest, const MPICompatibleRangeOfType<int> auto& destCounts,
             const MPICompatibleRangeOfType<int> auto& displs, const MPIRank root, const MPICommunicator& communicator = MPICommunicator::World) {
    if (communicator.get_my_rank() == root) {
        const auto required_size = detail::validate_gatherv_layout(destCounts, displs, communicator.get_number_ranks_cast());
        utility::Exception::check(std::ranges::size(dest) >= required_size, "MPICollectives::gatherv: Destination has {} elements, but the layout requires {}", std::ranges::size(dest), required_size);
    }
    gatherv<std::ranges::range_value_t<SourceRange>>(src.data(), src.size(), dest.data(), destCounts.data(), displs.data(), root, communicator);
}

/**
 * @brief Gathers a variable amount of data and returns the receive buffer on the specified root rank.
 *      Before the call:
 *          rank k has n values
 *      After the call:
 *          rank root has n values in <return>, starting at the specified displacement
 * @tparam T The type of data to gather
 * @param src The buffer of the local data elements, is accessed at 0, ..., count-1
 * @param count The number of data elements the current rank sends
 * @param destCounts The number of elements from each rank that are received. Is accessed at 0, ..., number_ranks-1.
 *      Is ignored at each rank but the specified root rank
 * @param displs The displacement for the elements in <return>, i.e., where the received elements of rank k start in <return>.
 *      Is ignored at each rank but the specified root rank
 * @param root The rank that receives the data
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception if root is too large, src is nullptr although count > 0,
 *      on root if destCounts or displs is nullptr, a count or displacement is negative,
 *      or MPI returns an error code
 * @return On root, a vector large enough to include every block at its specified displacement; empty elsewhere
 */
template <MPICompatible T>
[[nodiscard]] std::vector<T> gatherv(const T* src, const std::integral auto count, const int* const destCounts, const int* const displs, const MPIRank root, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto my_rank = communicator.get_my_rank();
    const auto number_ranks = communicator.get_number_ranks();

    auto buffer = std::vector<T>{};

    if (root == my_rank) {
        utility::Exception::check(destCounts != nullptr, "MPICollectives::gatherv: destCounts is nullptr on rank {}", my_rank);
        utility::Exception::check(displs != nullptr, "MPICollectives::gatherv: displs is nullptr on rank {}", my_rank);

        auto buffer_size = std::size_t{ 0 };
        for (auto rank = 0; rank < number_ranks; rank++) {
            const auto last_element = utility::safe_cast<std::size_t>(destCounts[rank]) + utility::safe_cast<std::size_t>(displs[rank]);
            buffer_size = std::max(buffer_size, last_element);
        }

        buffer.resize(buffer_size);

        gatherv(src, count, buffer.data(), destCounts, displs, root, communicator);
    } else {
        gatherv<T>(src, count, nullptr, nullptr, nullptr, root, communicator);
    }

    return buffer;
}

/**
 * @brief Gathers a variable amount of data and returns the receive buffer on the specified root rank.
 *      Before the call:
 *          rank k has n values
 *      After the call:
 *          rank root has n values in <return>, starting at the specified displacement
 *      Offers more convenient parameter types
 * @tparam SourceRange A contiguous input range
 * @param src The local data elements to gather at the specified root rank
 * @param destCounts The number of elements from each rank that are received. Is accessed at 0, ..., number_ranks-1.
 *      Is ignored at each rank but the specified root rank
 * @param displs The displacement for the elements in <return>, i.e., where the received elements of rank k start in <return>.
 *      Is ignored at each rank but the specified root rank
 * @param root The rank that receives the data
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception if root is too large, a layout range has the wrong size, a count or displacement
 *      is negative, or MPI reports an error
 * @return On root, a vector large enough to include every block at its specified displacement; empty elsewhere
 */
template <MPICompatibleRange SourceRange>
[[nodiscard]] std::vector<std::ranges::range_value_t<SourceRange>> gatherv(const SourceRange& src, const MPICompatibleRangeOfType<int> auto& destCounts,
                                                                           const MPICompatibleRangeOfType<int> auto& displs, const MPIRank root, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto my_rank = communicator.get_my_rank();
    const auto number_ranks = communicator.get_number_ranks();
    const auto number_ranks_cast = utility::safe_cast<std::size_t>(number_ranks);

    auto buffer = std::vector<std::ranges::range_value_t<SourceRange>>{};

    if (root == my_rank) {
        const auto buffer_size = detail::validate_gatherv_layout(destCounts, displs, number_ranks_cast);

        buffer.resize(buffer_size);

        gatherv<std::ranges::range_value_t<SourceRange>>(src.data(), src.size(), buffer.data(), destCounts.data(), displs.data(), root, communicator);
    } else {
        gatherv<std::ranges::range_value_t<SourceRange>>(src.data(), src.size(), nullptr, nullptr, nullptr, root, communicator);
    }

    return buffer;
}

/**
 * @brief Gathers a variable amount of data on the specified root rank. Calculates the sizes and the displacements
 *      automatically, such that the data is tightly packed in order of ranks.
 *      Call arguments:
 *          rank 0 calls with [a, b, c]
 *          rank 1 calls with [d, e]
 *          ...
 *      Return values:
 *          <return> = [[a, b, c], [d, e], ...] on the root rank; empty everywhere else
 * @tparam SourceRange A contiguous input range
 * @param src The local values to gather
 * @param root The rank that receives the data
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @return All gathered values grouped by source rank on root; an empty vector on every other rank
 */
template <MPICompatibleRange SourceRange>
[[nodiscard]] std::vector<std::vector<std::ranges::range_value_t<SourceRange>>> gatherv(SourceRange&& src, const MPIRank root, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto local_number_elements = std::ranges::size(src);
    const auto data_size_per_rank = MPICollectives::gather(utility::safe_cast<int>(local_number_elements), root, communicator);

    const auto my_rank = communicator.get_my_rank();
    if (my_rank == root) {
        const auto displacements = utility::calculate_displacements<int>(data_size_per_rank);
        const auto gathered_data_elements = gatherv(src, data_size_per_rank, displacements, root, communicator);
        return utility::reorganize_data<std::ranges::range_value_t<SourceRange>, const int, const int>(gathered_data_elements, data_size_per_rank, displacements);
    }

    gatherv<std::ranges::range_value_t<SourceRange>>(src.data(), src.size(), nullptr, nullptr, nullptr, root, communicator);
    return {};
}
} // namespace MPICollectives

} // namespace mpiPP
