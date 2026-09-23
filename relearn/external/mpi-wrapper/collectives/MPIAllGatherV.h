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

#include "mpi-wrapper/collectives/MPIAllGather.h"
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
[[nodiscard]] std::size_t validate_all_gatherv_layout(const Counts& counts, const Displacements& displacements, const std::size_t number_ranks) {
    utility::Exception::check(counts.size() == number_ranks, "MPICollectives::all_gatherv: Expected {} receive counts, got {}", number_ranks, counts.size());
    utility::Exception::check(displacements.size() == number_ranks, "MPICollectives::all_gatherv: Expected {} displacements, got {}", number_ranks, displacements.size());

    auto required_size = std::size_t{ 0 };
    for (auto rank = std::size_t{ 0 }; rank < number_ranks; ++rank) {
        utility::Exception::check(counts[rank] >= 0, "MPICollectives::all_gatherv: Negative receive count {} for rank {}", counts[rank], rank);
        utility::Exception::check(displacements[rank] >= 0, "MPICollectives::all_gatherv: Negative displacement {} for rank {}", displacements[rank], rank);
        required_size = std::max(required_size, utility::safe_cast<std::size_t>(counts[rank]) + utility::safe_cast<std::size_t>(displacements[rank]));
    }
    return required_size;
}
} // namespace detail

/**
 * @brief Gathers a variable amount of data on each rank. Uses the provided buffer.
 *      Before the call:
 *          rank k has n values
 *      After the call:
 *          all ranks have n values in dest, starting at the specified displacement
 * @tparam T The type of data to gather
 * @param src The buffer of the local data elements, is accessed at 0, ..., count-1
 * @param count The number of data elements the current rank sends
 * @param dest The destination buffer where the data elements are gathered
 * @param destCounts The number of elements from each rank that are received. Is accessed at 0, ..., number_ranks-1
 * @param displs The displacement for the elements in dest, i.e., where the received elements of rank k start in dest
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception for invalid counts or displacements, missing non-empty buffers,
 *     missing layout arrays, or an MPI error
 */
template <MPICompatible T>
void all_gatherv(const T* src, const std::integral auto count, T* dest, const int* const destCounts, const int* const displs, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto my_rank = communicator.get_my_rank();
    const auto count_cast = utility::safe_cast<int>(count);
    utility::Exception::check(src != nullptr || count_cast == 0, "MPICollectives::all_gatherv: src is nullptr on rank {}, but {} elements should be sent", my_rank, count_cast);
    utility::Exception::check(destCounts != nullptr, "MPICollectives::all_gatherv: destCounts is nullptr on rank {}", my_rank);
    utility::Exception::check(displs != nullptr, "MPICollectives::all_gatherv: displs is nullptr on rank {}", my_rank);

    const auto number_ranks = communicator.get_number_ranks();
    auto number_received = std::uint64_t{ 0 };
    for (auto rank = 0; rank < number_ranks; ++rank) {
        utility::Exception::check(destCounts[rank] >= 0, "MPICollectives::all_gatherv: Negative receive count {} for rank {}", destCounts[rank], rank);
        utility::Exception::check(displs[rank] >= 0, "MPICollectives::all_gatherv: Negative displacement {} for rank {}", displs[rank], rank);
        number_received += utility::safe_cast<std::uint64_t>(destCounts[rank]);
    }
    utility::Exception::check(dest != nullptr || number_received == 0,
                              "MPICollectives::all_gatherv: dest is nullptr on rank {}, but {} elements should be received", my_rank, number_received);

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allgatherv(src, count_cast, type, dest, destCounts, displs, type, communicator.get());
    utility::Exception::check(error_code == 0, "All-Gatherving all values returned the error: {}", error_code);

    MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(number_ranks) * utility::safe_cast<std::uint64_t>(count) * sizeof(T));
    MPICounters::add_to_received(number_received * sizeof(T));
}

/**
 * @brief Gathers a variable amount of data on each rank. Uses the provided buffer.
 *      Before the call:
 *          rank k has n values
 *      After the call:
 *          all ranks have n values in dest, starting at the specified displacement
 * @tparam SourceRange The type of input-data range, must provide .data() and .size()
 * @tparam TargetRange The type of output-data range, must provide .data()
 * @param src The local data elements to gather
 * @param dest The destination buffer where the data elements are gathered
 * @param destCounts One non-negative receive count per communicator rank
 * @param displs One non-negative destination displacement per communicator rank
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception if a layout has the wrong length or invalid entries, dest is too
 *     small for the layout, or MPI reports an error
 */
template <MPICompatibleRange SourceRange, MPICompatibleRangeOfType<std::ranges::range_value_t<SourceRange>> TargetRange>
    requires std::ranges::borrowed_range<TargetRange>
void all_gatherv(const SourceRange& src, TargetRange&& dest, const MPICompatibleRangeOfType<int> auto& destCounts,
                 const MPICompatibleRangeOfType<int> auto& displs, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto required_size = detail::validate_all_gatherv_layout(destCounts, displs, communicator.get_number_ranks_cast());
    utility::Exception::check(dest.size() >= required_size, "MPICollectives::all_gatherv: Destination has {} elements, but the layout requires {}", dest.size(), required_size);
    all_gatherv<std::ranges::range_value_t<SourceRange>>(src.data(), src.size(), dest.data(), destCounts.data(), displs.data(), communicator);
}

/**
 * @brief Gathers a variable amount of data on each rank.
 *      Before the call:
 *          rank k has n values
 *      After the call:
 *          all ranks have n values in <return>, starting at the specified displacement
 * @tparam T The type of data to gather
 * @param src The buffer of the local data elements, is accessed at 0, ..., count-1
 * @param count The number of data elements the current rank sends
 * @param destCounts The number of elements from each rank that are received. Is accessed at 0, ..., number_ranks-1
 * @param displs The displacement for the elements in <return>, i.e., where the received elements of rank k start in <return>
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception for invalid counts or displacements, a missing non-empty source,
 *     missing layout arrays, or an MPI error
 * @return A vector sized through the largest displacement-plus-count entry
 */
template <MPICompatible T>
[[nodiscard]] std::vector<T> all_gatherv(const T* src, const std::integral auto count, const int* const destCounts, const int* const displs, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto my_rank = communicator.get_my_rank();
    const auto number_ranks = communicator.get_number_ranks();
    const auto count_cast = utility::safe_cast<int>(count);
    utility::Exception::check(src != nullptr || count_cast == 0, "MPICollectives::all_gatherv: src is nullptr on rank {}, but {} elements should be sent", my_rank, count_cast);
    utility::Exception::check(destCounts != nullptr, "MPICollectives::all_gatherv: destCounts is nullptr on rank {}", my_rank);
    utility::Exception::check(displs != nullptr, "MPICollectives::all_gatherv: displs is nullptr on rank {}", my_rank);

    auto buffer = std::vector<T>{};

    auto buffer_size = std::size_t{ 0 };
    for (auto rank = 0; rank < number_ranks; rank++) {
        const auto last_element = utility::safe_cast<std::size_t>(destCounts[rank]) + utility::safe_cast<std::size_t>(displs[rank]);
        buffer_size = std::max(buffer_size, last_element);
    }

    buffer.resize(buffer_size);

    all_gatherv(src, count, buffer.data(), destCounts, displs, communicator);

    return buffer;
}

/**
 * @brief Gathers a variable amount of data on each rank.
 *      Before the call:
 *          rank k has n values
 *      After the call:
 *          all ranks have n values in <return>, starting at the specified displacement
 * @tparam SourceRange The type of input-data range, must provide .data() and .size()
 * @param src The local data elements to gather
 * @param destCounts The number of elements from each rank that are received. Is accessed at 0, ..., number_ranks-1
 * @param displs The displacement for the elements in <return>, i.e., where the received elements of rank k start in <return>
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception if a layout has the wrong length or invalid entries, or MPI reports an error
 * @return A vector sized through the largest displacement-plus-count entry
 */
template <MPICompatibleRange SourceRange>
[[nodiscard]] std::vector<std::ranges::range_value_t<SourceRange>> all_gatherv(const SourceRange& src, const MPICompatibleRangeOfType<int> auto& destCounts,
                                                                               const MPICompatibleRangeOfType<int> auto& displs, const MPICommunicator& communicator = MPICommunicator::World) {
    auto buffer = std::vector<std::ranges::range_value_t<SourceRange>>{};

    const auto number_ranks = communicator.get_number_ranks();
    const auto number_ranks_cast = utility::safe_cast<std::size_t>(number_ranks);
    const auto buffer_size = detail::validate_all_gatherv_layout(destCounts, displs, number_ranks_cast);

    buffer.resize(buffer_size);

    all_gatherv<std::ranges::range_value_t<SourceRange>>(src.data(), src.size(), buffer.data(), destCounts.data(), displs.data(), communicator);

    return buffer;
}

/**
 * @brief Gathers a variable amount of data on each rank. Calculates the sizes and the displacements
 *      automatically, such that the data is tightly packed in order of ranks.
 *      Call arguments:
 *          rank 0 calls with [a, b, c]
 *          rank 1 calls with [d, e]
 *          ...
 *      Return values:
 *          <return> = [[a, b, c], [d, e], ...] on every rank
 * @tparam SourceRange The type of input-data range, must provide .data() and .size()
 * @param src The local values to gather
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @return All gathered values
 */
template <MPICompatibleRange SourceRange>
[[nodiscard]] std::vector<std::vector<std::ranges::range_value_t<SourceRange>>> all_gatherv(const SourceRange& src, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto local_number_elements = std::ranges::size(src);
    const auto data_size_per_rank = MPICollectives::all_gather(utility::safe_cast<int>(local_number_elements), communicator);
    const auto displacements = utility::calculate_displacements<int>(data_size_per_rank);

    const auto gathered_data_elements = all_gatherv(src, data_size_per_rank, displacements, communicator);

    return utility::reorganize_data<std::ranges::range_value_t<SourceRange>, const int, const int>(gathered_data_elements, data_size_per_rank, displacements);
}
} // namespace MPICollectives

} // namespace mpiPP
