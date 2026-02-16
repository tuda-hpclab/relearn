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
#include "mpi-wrapper/MPITypes.h"
#include "mpi-wrapper/collectives/MPIAllGather.h"

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"
#include "cpp-utility/data/displacement.hpp"
#include "cpp-utility/data/reorganize.hpp"

#include <mpi.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <vector>

namespace mpiPP {

namespace MPICollectives {
/**
 * @brief Gatheres a variable amount of data on each rank. Uses the provided buffer.
 *      Before the call:
 *          rank k has n values
 *      After the call:
 *          all ranks have n values in dest, starting at the specified displacement
 * @tparam T The type of data to gather
 * @param src The buffer of the local data elements, is accessed at 0, ..., count-1
 * @param count The number of data elements the current rank sends
 * @param dest The destination buffer where the data elements are gathered
 * @param destCounts The number of elements from each rank that are received. Is accessed at 0, ..., number_ranks-1
 * @param displs he displacement for the elements in dest, i.e., where the received elements of rank k start in dest
 * @exception Throws an Exception if root_rank is too large, src is nullptr,
 *     if any of dest, destCounts, displs is nullptr, or mpi returns an error code
 */
template <MPICompatible T>
void all_gatherv(const T* src, const std::integral auto count, T* dest, const int* const destCounts, const int* const displs) {
    const auto my_rank = MPIInfo::get_my_rank();
    utility::Exception::check(src != nullptr, "MPICollectives::all_gatherv: src is nullptr on rank {}", my_rank);
    utility::Exception::check(dest != nullptr, "MPICollectives::all_gatherv: destCounts is nullptr on rank {}", my_rank);
    utility::Exception::check(destCounts != nullptr, "MPICollectives::all_gatherv: destCounts is nullptr on rank {}", my_rank);
    utility::Exception::check(displs != nullptr, "MPICollectives::all_gatherv: displs is nullptr on rank {}", my_rank);

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allgatherv(src, utility::save_cast<int>(count), type, dest, destCounts, displs, type, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "All-Gatherving all values returned the error: {}", error_code);

    const auto number_ranks = MPIInfo::get_number_ranks();
    MPICounters::add_to_sent(utility::save_cast<std::uint64_t>(number_ranks) * utility::save_cast<std::uint64_t>(count) * sizeof(T));

    const auto number_received = std::accumulate(destCounts, destCounts + number_ranks, 0);
    MPICounters::add_to_received(utility::save_cast<std::uint64_t>(number_received) * sizeof(T));
}

/**
 * @brief Gatheres a variable amount of data on each rank. Uses the provided buffer.
 *      Before the call:
 *          rank k has n values
 *      After the call:
 *          all ranks have n values in dest, starting at the specified displacement
 * @tparam SourceRange The type of input-data range, must provide .data() and .size()
 * @tparam TargetRange The type of output-data range, must provide .data()
 * @param src The local data elements to gather
 * @param dest The destination buffer where the data elements are gathered
 * @param destCounts The number of elements from each rank that are received. Is accessed at 0, ..., number_ranks-1
 * @param displs The displacement for the elements in dest, i.e., where the received elements of rank k start in dest
 */
template <MPICompatibleRange SourceRange, MPICompatibleRangeOfType<std::ranges::range_value_t<SourceRange>> TargetRange>
    requires std::ranges::borrowed_range<TargetRange>
void all_gatherv(const SourceRange& src, TargetRange&& dest, const MPICompatibleRangeOfType<int> auto& destCounts,
                 const MPICompatibleRangeOfType<int> auto& displs) {
    all_gatherv<std::ranges::range_value_t<SourceRange>>(src.data(), src.size(), dest.data(), destCounts.data(), displs.data());
}

/**
 * @brief Gatheres a variable amount of data on each rank.
 *      Before the call:
 *          rank k has n values
 *      After the call:
 *          all ranks have n values in <return>, starting at the specified displacement
 * @tparam T The type of data to gather
 * @param src The buffer of the local data elements, is accessed at 0, ..., count-1
 * @param count The number of data elements the current rank sends
 * @param destCounts The number of elements from each rank that are received. Is accessed at 0, ..., number_ranks-1
 * @param displs he displacement for the elements in <return>, i.e., where the received elements of rank k start in <return>
 * @exception Throws an Exception if root_rank is too large, src is nullptr,
 *     if any of destCounts, displs is nullptr, or mpi returns an error code
 * @return A vector with the gathered elements
 */
template <MPICompatible T>
[[nodiscard]] std::vector<T> all_gatherv(const T* src, const std::integral auto count, const int* const destCounts, const int* const displs) {
    const auto my_rank = MPIInfo::get_my_rank();
    const auto number_ranks = MPIInfo::get_number_ranks();
    utility::Exception::check(src != nullptr, "MPICollectives::all_gatherv: src is nullptr on rank {}", my_rank);
    utility::Exception::check(destCounts != nullptr, "MPICollectives::all_gatherv: destCounts is nullptr on rank {}", my_rank);
    utility::Exception::check(displs != nullptr, "MPICollectives::all_gatherv: displs is nullptr on rank {}", my_rank);

    auto buffer = std::vector<T>{};

    auto buffer_size = std::size_t{ 0 };
    for (auto rank = 0; rank < number_ranks; rank++) {
        const auto last_element = utility::save_cast<std::size_t>(destCounts[rank]) + utility::save_cast<std::size_t>(displs[rank]);
        buffer_size = std::max(buffer_size, last_element);
    }

    buffer.resize(buffer_size);

    all_gatherv(src, count, buffer.data(), destCounts, displs);

    return buffer;
}

/**
 * @brief Gatheres a variable amount of data on each rank.
 *      Before the call:
 *          rank k has n values
 *      After the call:
 *          all ranks have n values in <return>, starting at the specified displacement
 * @tparam SourceRange The type of input-data range, must provide .data() and .size()
 * @param src The local data elements to gather
 * @param destCounts The number of elements from each rank that are received. Is accessed at 0, ..., number_ranks-1
 * @param displs The displacement for the elements in <return>, i.e., where the received elements of rank k start in <return>
 * @return A vector with the gathered elements
 */
template <MPICompatibleRange SourceRange>
[[nodiscard]] std::vector<std::ranges::range_value_t<SourceRange>> all_gatherv(const SourceRange& src, const MPICompatibleRangeOfType<int> auto& destCounts,
                                                                               const MPICompatibleRangeOfType<int> auto& displs) {
    auto buffer = std::vector<std::ranges::range_value_t<SourceRange>>{};

    const auto number_ranks = MPIInfo::get_number_ranks();
    const auto number_ranks_cast = utility::save_cast<std::size_t>(number_ranks);

    auto buffer_size = std::size_t{ 0 };
    for (auto rank = std::size_t{ 0 }; rank < number_ranks_cast; rank++) {
        const auto last_element = utility::save_cast<std::size_t>(destCounts[rank]) + utility::save_cast<std::size_t>(displs[rank]);
        buffer_size = std::max(buffer_size, last_element);
    }

    buffer.resize(buffer_size);

    all_gatherv<std::ranges::range_value_t<SourceRange>>(src.data(), src.size(), buffer.data(), destCounts.data(), displs.data());

    return buffer;
}

/**
 * @brief Gatheres a variable amount of data on each rank. Calculates the sizes and the displacements
 *      automatically, such that the data is tightly packed in order of ranks.
 *      Call arguments:
 *          rank 0 calls with [a, b, c]
 *          rank 1 calls with [d, e]
 *          ...
 *      Return values:
 *          <return> = [[a, b, c], [d, e], ...] on every rank
 * @tparam SourceRange The type of input-data range, must provide .data() and .size()
 * @param src The local values to gather
 * @return All gathered values
 */
template <MPICompatibleRange SourceRange>
[[nodiscard]] std::vector<std::vector<std::ranges::range_value_t<SourceRange>>> all_gatherv(const SourceRange& src) {
    const auto local_number_elements = std::ranges::size(src);
    const auto data_size_per_rank = MPICollectives::all_gather(utility::save_cast<int>(local_number_elements));
    const auto displacements = utility::calculate_displacements<int>(data_size_per_rank);

    const auto gathered_data_elements = all_gatherv(src, data_size_per_rank, displacements);

    return utility::reorganize_data<std::ranges::range_value_t<SourceRange>, const int, const int>(gathered_data_elements, data_size_per_rank, displacements);
}
} // namespace MPICollectives

} // namespace mpiPP
