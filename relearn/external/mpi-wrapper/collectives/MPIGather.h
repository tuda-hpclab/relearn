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
#include "mpi-wrapper/MPIRank.h"
#include "mpi-wrapper/MPITypes.h"

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"

#include <mpi.h>

#include <cstddef>
#include <span>
#include <vector>

namespace mpiPP {

namespace MPICollectives {
/**
 * @brief Gatheres all the data on the root rank.
 *      Before the call:
 *          rank k has value v
 *      After the call:
 *          rank 0 has <ret>[k] = v
 * @tparam T The type of data to gather
 * @param own_data The element of the current rank
 * @param root The rank where to gather to, is default MPIRank::root_rank()
 * @exception Throws an Exception if mpi returns an error code
 * @return The vector of the gathered elements on the root rank; an empty vector on all other ranks
 */
template <MPICompatible T>
[[nodiscard]] std::vector<T> gather(const T own_data, const MPIRank root = MPIRank::root_rank()) {
    const auto number_ranks = MPIInfo::get_number_ranks();
    const auto my_rank = MPIInfo::get_my_rank();
    auto results = std::vector<T>(utility::save_cast<std::size_t>(number_ranks) * (my_rank == root));

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Gather(&own_data, 1, type, results.data(), 1, type, root.get_rank(), MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "Gathering values returned the error : {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    if (my_rank == root) {
        MPICounters::add_to_received(utility::save_cast<std::uint64_t>(number_ranks) * sizeof(T));
    }

    return results;
}

/**
 * @brief Gatheres all the data on the root rank. Uses the provided buffer.
 *      Before the call:
 *          rank k has value v
 *      After the call:
 *          rank 0 has tgt[k] = v
 * @tparam T The type of data to gather
 * @param own_data The element of the current rank
 * @param root The rank where to gather to, is default MPIRank::root_rank()
 * @param tgt The buffer where to gather the elements. Used only on the root rank
 * @exception Throws an Exception if tgt.size() is not equal to the number of mpi ranks (checked only for the root rank) or mpi returns an error code
 */
template <MPICompatible T>
void gather(const T own_data, const std::span<T> tgt, const MPIRank root = MPIRank::root_rank()) {
    const auto type = MPITypes::convert_type_to_mpi_type<T>();

    if (MPIInfo::is_root_rank()) {
        const auto number_ranks = MPIInfo::get_number_ranks();
        const auto number_ranks_cast = utility::save_cast<std::size_t>(number_ranks);
        const auto count_tgt = tgt.size();
        utility::Exception::check(number_ranks_cast == count_tgt, "MPICollectives::gather: Sizes do not match: {} vs {}", number_ranks_cast, count_tgt);

        const auto error_code = MPI_Gather(&own_data, 1, type, tgt.data(), 1, type, root.get_rank(), MPI_COMM_WORLD);
        utility::Exception::check(error_code == 0, "Gathering values returned the error : {}", error_code);
    } else {
        const auto error_code = MPI_Gather(&own_data, 1, type, nullptr, 0, type, root.get_rank(), MPI_COMM_WORLD);
        utility::Exception::check(error_code == 0, "Gathering values returned the error : {}", error_code);
    }

    MPICounters::add_to_sent(sizeof(T));
    if (MPIInfo::get_my_rank() == root) {
        const auto number_ranks = MPIInfo::get_number_ranks();
        MPICounters::add_to_received(utility::save_cast<std::uint64_t>(number_ranks) * sizeof(T));
    }
}
} // namespace MPICollectives

} // namespace mpiPP
