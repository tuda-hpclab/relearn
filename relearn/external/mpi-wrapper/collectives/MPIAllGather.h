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

#include <cstddef>
#include <span>
#include <vector>

namespace mpiPP {

namespace MPICollectives {
/**
 * @brief Gathers all the data on each rank.
 *      Before the call:
 *          rank k has value v
 *      After the call:
 *          All ranks have <ret>[k] = v
 * @tparam T The type of data to gather
 * @param own_data The element of the current rank
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception if mpi returns an error code
 * @return The vector of the gathered elements
 */
template <MPICompatible T>
[[nodiscard]] std::vector<T> all_gather(const T own_data, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    auto results = std::vector<T>(utility::safe_cast<std::size_t>(number_ranks));

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allgather(&own_data, 1, type, results.data(), 1, type, communicator.get());
    utility::Exception::check(error_code == 0, "Gathering all values returned the error : {}", error_code);

    MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(number_ranks) * sizeof(T));
    MPICounters::add_to_received(utility::safe_cast<std::uint64_t>(number_ranks) * sizeof(T));

    return results;
}

/**
 * @brief Gathers all the data on each rank. Uses the provided buffer.
 *      Before the call:
 *          rank k has value v
 *      After the call:
 *          All ranks have tgt[k] = v
 * @tparam T The type of data to gather
 * @param own_data The element of the current rank
 * @param tgt The buffer where to gather the elements
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception if tgt.size() is not equal to the number of mpi ranks or mpi returns an error code
 */
template <MPICompatible T>
void all_gather(const T own_data, const std::span<T> tgt, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto number_ranks_cast = utility::safe_cast<std::size_t>(number_ranks);
    const auto count_tgt = tgt.size();
    utility::Exception::check(number_ranks_cast == count_tgt, "MPICollectives::all_gather: Sizes do not match: {} vs {}", number_ranks_cast, count_tgt);

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allgather(&own_data, 1, type, tgt.data(), 1, type, communicator.get());
    utility::Exception::check(error_code == 0, "Gathering all values returned the error : {}", error_code);

    MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(number_ranks_cast) * sizeof(T));
    MPICounters::add_to_received(utility::safe_cast<std::uint64_t>(number_ranks_cast) * sizeof(T));
}

/**
 * @brief Gathers all the data on each rank. Uses the same send and receive buffer.
 *      Before the call:
 *          rank k owns number_elements values in buffer at indices
 *          [k * number_elements, (k + 1) * number_elements)
 *      After the call:
 *          every rank has all of those blocks in rank order
 * @tparam T The type of data to gather
 * @param buffer Where to gather the elements
 * @param number_elements The number of elements to gather per rank
 * @param communicator The communicator to gather within, defaults to the world communicator
 * @exception Throws an Exception if buffer.size() is not equal to (the number of mpi ranks) multiplied with number_elements
 *       or mpi returns an error code
 */
template <MPICompatible T>
void all_gather_inline(const std::span<T> buffer, const int number_elements = 1, const MPICommunicator& communicator = MPICommunicator::World) {
    utility::Exception::check(number_elements > 0, "MPICollectives::all_gather_inline: number_elements must be greater than 0 but was {}", number_elements);

    const auto number_ranks = communicator.get_number_ranks();
    const auto size_cast = utility::safe_cast<int>(buffer.size());
    utility::Exception::check(number_ranks * number_elements == size_cast,
                              "MPICollectives::all_gather_inline: Size of span does not match number of ranks ({} != {} * {})", size_cast, number_ranks, number_elements);

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allgather(MPI_IN_PLACE, 0, type, buffer.data(), number_elements, type, communicator.get());
    utility::Exception::check(error_code == 0, "Inline-gathering all values returned the error : {}", error_code);

    MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(buffer.size()) * sizeof(T));
    MPICounters::add_to_received(utility::safe_cast<std::uint64_t>(buffer.size()) * sizeof(T));
}
} // namespace MPICollectives

} // namespace mpiPP
