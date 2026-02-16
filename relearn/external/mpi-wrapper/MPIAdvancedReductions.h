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

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"
#include "cpp-utility/data/prefix_sum.hpp"

#include <mpi.h>

#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mpiPP {

namespace MPIAdvancedReductions {

static std::unordered_map<std::uint64_t, std::uint64_t> reduce_map(const std::unordered_map<std::uint64_t, std::uint64_t>& local_map) {
    const auto my_rank = MPIInfo::get_my_rank().get_rank();
    const auto number_ranks = MPIInfo::get_number_ranks();

    // These are the keys the current MPI rank has locally
    auto keys = std::vector<std::uint64_t>{};
    keys.reserve(local_map.size());
    for (const auto& [key, value] : local_map) {
        keys.emplace_back(key);
    }

    // MPI rank 0 now knows how many keys there are alltogether (counting duplicates)
    auto number_keys_on_mpi_ranks = std::vector<int>{};
    if (my_rank == 0) {
        number_keys_on_mpi_ranks.resize(utility::save_cast<std::size_t>(number_ranks), 0);
    }

    const auto number_local_keys = utility::save_cast<int>(local_map.size());
    MPI_Gather(&number_local_keys, 1, MPI_INT, number_keys_on_mpi_ranks.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    const auto number_values = std::reduce(number_keys_on_mpi_ranks.begin(), number_keys_on_mpi_ranks.end(), 0);

    // MPI rank 0 now has all keys (counting duplicates)
    auto keys_on_mpi_ranks = std::vector<std::uint64_t>{};
    if (my_rank == 0) {
        keys_on_mpi_ranks.resize(utility::save_cast<std::size_t>(number_values), 0);
    }
    auto prefix_sum = utility::calculate_prefix_sum<int>(number_keys_on_mpi_ranks);

    MPI_Gatherv(keys.data(), number_local_keys, MPI_UINT64_T, keys_on_mpi_ranks.data(), number_keys_on_mpi_ranks.data(), prefix_sum.data(),
                MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // Removing the duplicates on MPI rank 0
    auto distinct_keys_set = std::unordered_set<std::uint64_t>{ keys_on_mpi_ranks.begin(), keys_on_mpi_ranks.end() };
    auto distinct_keys_vec = std::vector<std::uint64_t>{ distinct_keys_set.begin(), distinct_keys_set.end() };

    // Now every MPI rank knows how many distinct keys there are
    auto number_distinct_keys_send = distinct_keys_vec.size();
    MPI_Bcast(&number_distinct_keys_send, 1, MPITypes::convert_type_to_mpi_type<std::size_t>(), 0, MPI_COMM_WORLD);

    // Now every MPI rank knows the distinct keys
    auto global_keys = std::vector<std::uint64_t>{};

    if (my_rank == 0) {
        global_keys = std::move(distinct_keys_vec);
    } else {
        global_keys.resize(number_distinct_keys_send, 0);
    }

    MPI_Bcast(global_keys.data(), utility::save_cast<int>(global_keys.size()), MPI_UINT64_T, 0, MPI_COMM_WORLD);

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
    MPI_Allreduce(global_values.data(), summed_global_values.data(), utility::save_cast<int>(number_distinct_keys_send), MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);

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
