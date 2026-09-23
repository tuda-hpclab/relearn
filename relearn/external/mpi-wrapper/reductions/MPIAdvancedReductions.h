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
#include <cpp-utility/data-structure/Histogram.hpp>
#include <cpp-utility/data/prefix_sum.hpp>

#include <mpi.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mpiPP {

namespace MPIAdvancedReductions {

/**
 * @brief The globally reduced representation of a utility::Histogram.
 *
 * The cpp-utility histogram deliberately exposes its bins as read-only views and has no constructor from
 * serialized bins. This value keeps the exact reduced representation so it can be inspected without changing
 * the original histogram API.
 * @tparam DataType The arithmetic type used for the bin borders
 */
template <typename DataType>
    requires std::is_arithmetic_v<DataType>
class HistogramReduction {
public:
    /**
     * @brief Returns the lower border of every reduced bin
     * @return The bin borders in ascending bin order
     */
    [[nodiscard]] std::span<const DataType> get_borders() const noexcept {
        return borders_;
    }

    /**
     * @brief Returns the summed count for every reduced bin
     * @return The counts corresponding to get_borders()
     */
    [[nodiscard]] std::span<const std::size_t> get_counts() const noexcept {
        return counts_;
    }

    /**
     * @brief Returns the number of bins in the reduced histogram
     * @return The common size of get_borders() and get_counts()
     */
    [[nodiscard]] std::size_t num_bins() const noexcept {
        return counts_.size();
    }

private:
    template <typename T>
        requires std::is_arithmetic_v<T>
    friend HistogramReduction<T> reduce_histogram(const utility::Histogram<T>&, const MPICommunicator&);

    std::vector<DataType> borders_{};
    std::vector<std::size_t> counts_{};
};

/**
 * @brief Collectively sums equal-bin histograms and returns their complete reduced representation on every rank.
 *
 * Empty upper bins may be omitted locally: a shorter histogram contributes zero to the missing bins. Every
 * overlapping bin border must be identical on all ranks; otherwise every rank throws before the count reduction.
 * @tparam DataType The arithmetic type used for the bin borders
 * @param local_histogram The local histogram contribution
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the bin count exceeds MPI's int limit, borders differ, a count cannot be
 *      represented by the reduction wire format, the reduced count does not fit std::size_t, or MPI reports an error
 * @return The shared borders and component-wise summed counts on every rank
 */
template <typename DataType>
    requires std::is_arithmetic_v<DataType>
[[nodiscard]] HistogramReduction<DataType> reduce_histogram(const utility::Histogram<DataType>& local_histogram, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto local_borders = local_histogram.get_borders();
    const auto local_counts = local_histogram.get_counts();
    const auto local_bin_count = local_counts.size();

    const auto local_bin_count_fits = local_bin_count <= utility::safe_cast<std::size_t>(std::numeric_limits<int>::max());
    auto all_bin_counts_fit = int{ 0 };
    const auto bin_count_validity_error = MPI_Allreduce(&local_bin_count_fits, &all_bin_counts_fit, 1, MPI_C_BOOL, MPI_LAND, communicator.get());
    utility::Exception::check(bin_count_validity_error == MPI_SUCCESS, "MPIAdvancedReductions::reduce_histogram: Reducing bin-count validity returned error code {}", bin_count_validity_error);
    utility::Exception::check(all_bin_counts_fit != 0, "MPIAdvancedReductions::reduce_histogram: A histogram has more bins than MPI can address with an int count");

    const auto local_bin_count_wire = utility::safe_cast<std::uint64_t>(local_bin_count);
    auto global_bin_count_wire = std::uint64_t{ 0 };
    const auto maximum_bin_count_error = MPI_Allreduce(&local_bin_count_wire, &global_bin_count_wire, 1, MPI_UINT64_T, MPI_MAX, communicator.get());
    utility::Exception::check(maximum_bin_count_error == MPI_SUCCESS, "MPIAdvancedReductions::reduce_histogram: Reducing the maximum bin count returned error code {}", maximum_bin_count_error);
    const auto global_bin_count = utility::safe_cast<std::size_t>(global_bin_count_wire);

    const auto my_rank = communicator.get_my_rank().get_rank();
    const auto local_owner = local_bin_count == global_bin_count ? my_rank : std::numeric_limits<int>::max();
    auto border_owner = int{ 0 };
    const auto owner_error = MPI_Allreduce(&local_owner, &border_owner, 1, MPI_INT, MPI_MIN, communicator.get());
    utility::Exception::check(owner_error == MPI_SUCCESS, "MPIAdvancedReductions::reduce_histogram: Selecting the border owner returned error code {}", owner_error);

    auto borders = std::vector<DataType>(global_bin_count);
    if (my_rank == border_owner) {
        std::ranges::copy(local_borders, borders.begin());
    }
    const auto broadcast_borders_error = MPI_Bcast(borders.data(), utility::safe_cast<int>(global_bin_count), MPITypes::convert_type_to_mpi_type<DataType>(), border_owner, communicator.get());
    utility::Exception::check(broadcast_borders_error == MPI_SUCCESS, "MPIAdvancedReductions::reduce_histogram: Broadcasting bin borders returned error code {}", broadcast_borders_error);

    const auto local_borders_match = std::ranges::equal(local_borders, borders | std::views::take(local_bin_count));
    auto all_borders_match = int{ 0 };
    const auto border_validity_error = MPI_Allreduce(&local_borders_match, &all_borders_match, 1, MPI_C_BOOL, MPI_LAND, communicator.get());
    utility::Exception::check(border_validity_error == MPI_SUCCESS, "MPIAdvancedReductions::reduce_histogram: Reducing border validity returned error code {}", border_validity_error);
    utility::Exception::check(all_borders_match != 0, "MPIAdvancedReductions::reduce_histogram: Histograms have different borders and cannot be summed");

    const auto local_counts_fit = std::ranges::all_of(local_counts, [](const auto count) {
        if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
            return count <= std::numeric_limits<std::uint64_t>::max();
        }
        static_cast<void>(count);
        return true;
    });
    auto all_counts_fit = int{ 0 };
    const auto count_validity_error = MPI_Allreduce(&local_counts_fit, &all_counts_fit, 1, MPI_C_BOOL, MPI_LAND, communicator.get());
    utility::Exception::check(count_validity_error == MPI_SUCCESS, "MPIAdvancedReductions::reduce_histogram: Reducing count validity returned error code {}", count_validity_error);
    utility::Exception::check(all_counts_fit != 0, "MPIAdvancedReductions::reduce_histogram: A bin count cannot be represented as uint64_t");

    auto local_counts_wire = std::vector<std::uint64_t>(global_bin_count, 0);
    std::ranges::transform(local_counts, local_counts_wire.begin(), [](const auto count) {
        return utility::safe_cast<std::uint64_t>(count);
    });
    auto reduced_counts_wire = std::vector<std::uint64_t>(global_bin_count, 0);
    const auto reduce_counts_error = MPI_Allreduce(local_counts_wire.data(), reduced_counts_wire.data(), utility::safe_cast<int>(global_bin_count), MPI_UINT64_T, MPI_SUM, communicator.get());
    utility::Exception::check(reduce_counts_error == MPI_SUCCESS, "MPIAdvancedReductions::reduce_histogram: Reducing bin counts returned error code {}", reduce_counts_error);

    const auto reduced_counts_fit = std::ranges::all_of(reduced_counts_wire, [](const auto count) {
        if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
            return count <= std::numeric_limits<std::size_t>::max();
        }
        static_cast<void>(count);
        return true;
    });
    utility::Exception::check(reduced_counts_fit, "MPIAdvancedReductions::reduce_histogram: A reduced bin count does not fit std::size_t");

    auto result = HistogramReduction<DataType>{};
    result.borders_ = std::move(borders);
    result.counts_.resize(global_bin_count);
    std::ranges::transform(reduced_counts_wire, result.counts_.begin(), [](const auto count) {
        return utility::safe_cast<std::size_t>(count);
    });
    return result;
}

/**
 * @brief Collectively forms the union of all uint64 keys and sums the value for each key across the
 *      communicator. Missing local keys contribute zero; every rank receives the complete result.
 * @param local_map The local key-value contributions
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if a local or global key count exceeds MPI's int limit, or MPI reports an error
 * @return The global key union with values summed across all ranks
 */
inline std::unordered_map<std::uint64_t, std::uint64_t> reduce_map(const std::unordered_map<std::uint64_t, std::uint64_t>& local_map, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto my_rank = communicator.get_my_rank().get_rank();
    const auto number_ranks = communicator.get_number_ranks();

    // These are the keys the current MPI rank has locally
    auto keys = std::vector<std::uint64_t>{};
    keys.reserve(local_map.size());
    for (const auto& [key, value] : local_map) {
        keys.emplace_back(key);
    }

    // MPI rank 0 now knows how many keys there are altogether (counting duplicates)
    auto number_keys_on_mpi_ranks = std::vector<int>{};
    if (my_rank == 0) {
        number_keys_on_mpi_ranks.resize(utility::safe_cast<std::size_t>(number_ranks), 0);
    }

    const auto number_local_keys = utility::safe_cast<int>(local_map.size());
    const auto gather_counts_error = MPI_Gather(&number_local_keys, 1, MPI_INT, number_keys_on_mpi_ranks.data(), 1, MPI_INT, 0, communicator.get());
    utility::Exception::check(gather_counts_error == MPI_SUCCESS, "MPIAdvancedReductions::reduce_map: Gathering key counts returned error code {}", gather_counts_error);

    auto number_values = int{ 0 };
    auto counts_fit = int{ 1 };
    if (my_rank == 0) {
        const auto number_values_wide = std::reduce(number_keys_on_mpi_ranks.begin(), number_keys_on_mpi_ranks.end(), std::uint64_t{ 0 });
        counts_fit = number_values_wide <= utility::safe_cast<std::uint64_t>(std::numeric_limits<int>::max());
        if (counts_fit != 0) {
            number_values = utility::safe_cast<int>(number_values_wide);
        }
    }
    const auto counts_fit_error = MPI_Bcast(&counts_fit, 1, MPI_INT, 0, communicator.get());
    utility::Exception::check(counts_fit_error == MPI_SUCCESS, "MPIAdvancedReductions::reduce_map: Broadcasting key-count validity returned error code {}", counts_fit_error);
    utility::Exception::check(counts_fit != 0, "MPIAdvancedReductions::reduce_map: The global number of keys exceeds the MPI int count limit");

    // MPI rank 0 now has all keys (counting duplicates)
    auto keys_on_mpi_ranks = std::vector<std::uint64_t>{};
    if (my_rank == 0) {
        keys_on_mpi_ranks.resize(utility::safe_cast<std::size_t>(number_values), 0);
    }
    auto prefix_sum = utility::calculate_prefix_sum<int>(number_keys_on_mpi_ranks);

    const auto gather_keys_error = MPI_Gatherv(keys.data(), number_local_keys, MPI_UINT64_T, keys_on_mpi_ranks.data(), number_keys_on_mpi_ranks.data(), prefix_sum.data(),
                                               MPI_UINT64_T, 0, communicator.get());
    utility::Exception::check(gather_keys_error == MPI_SUCCESS, "MPIAdvancedReductions::reduce_map: Gathering keys returned error code {}", gather_keys_error);

    // Removing the duplicates on MPI rank 0
    auto distinct_keys_set = std::unordered_set<std::uint64_t>{ keys_on_mpi_ranks.begin(), keys_on_mpi_ranks.end() };
    auto distinct_keys_vec = std::vector<std::uint64_t>{ distinct_keys_set.begin(), distinct_keys_set.end() };

    // Now every MPI rank knows how many distinct keys there are
    auto number_distinct_keys_send = distinct_keys_vec.size();
    const auto broadcast_count_error = MPI_Bcast(&number_distinct_keys_send, 1, MPITypes::convert_type_to_mpi_type<std::size_t>(), 0, communicator.get());
    utility::Exception::check(broadcast_count_error == MPI_SUCCESS, "MPIAdvancedReductions::reduce_map: Broadcasting the distinct-key count returned error code {}", broadcast_count_error);

    // Now every MPI rank knows the distinct keys
    auto global_keys = std::vector<std::uint64_t>{};

    if (my_rank == 0) {
        global_keys = std::move(distinct_keys_vec);
    } else {
        global_keys.resize(number_distinct_keys_send, 0);
    }

    const auto broadcast_keys_error = MPI_Bcast(global_keys.data(), utility::safe_cast<int>(global_keys.size()), MPI_UINT64_T, 0, communicator.get());
    utility::Exception::check(broadcast_keys_error == MPI_SUCCESS, "MPIAdvancedReductions::reduce_map: Broadcasting distinct keys returned error code {}", broadcast_keys_error);

    // Now every MPI rank knows the local values for the keys
    auto global_values = std::vector<std::uint64_t>{};
    for (const auto& key : global_keys) {
        const auto it = local_map.find(key);
        if (it == local_map.end()) {
            global_values.emplace_back(0);
        } else {
            global_values.emplace_back(it->second);
        }
    }

    // Now every MPI rank has the summed values for the keys
    auto summed_global_values = std::vector<std::uint64_t>{};
    summed_global_values.resize(number_distinct_keys_send);
    const auto reduce_values_error = MPI_Allreduce(global_values.data(), summed_global_values.data(), utility::safe_cast<int>(number_distinct_keys_send), MPI_UINT64_T, MPI_SUM,
                                                   communicator.get());
    utility::Exception::check(reduce_values_error == MPI_SUCCESS, "MPIAdvancedReductions::reduce_map: Reducing values returned error code {}", reduce_values_error);

    // Finally, put everything back into a map
    auto global_map = std::unordered_map<std::uint64_t, std::uint64_t>{};
    global_map.reserve(number_distinct_keys_send);
    for (auto i = std::size_t(0); i < number_distinct_keys_send; i++) {
        global_map[global_keys[i]] = summed_global_values[i];
    }

    return global_map;
}

} // namespace MPIAdvancedReductions

} // namespace mpiPP
