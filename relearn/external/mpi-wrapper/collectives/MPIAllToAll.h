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

#include <span>
#include <vector>

namespace mpiPP {

namespace MPICollectives {
/**
 * @brief Exchanges values between all mpi ranks.
 *      Before the call:
 *          rank k has src[i] = v
 *      After the call:
 *          rank i has <ret>[k] = v
 * @tparam T The type of data to exchange
 * @param src The elements to exchange
 * @param communicator The communicator to exchange within, defaults to the world communicator
 * @exception Throws an Exception if src.size() is not equal to the number of ranks or mpi returns an error code
 * @return The exchanged values
 */
template <MPICompatible T>
[[nodiscard]] std::vector<T> all_to_all(const std::span<const T> src, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto number_ranks_cast = utility::safe_cast<std::size_t>(number_ranks);
    const auto count_src = src.size();
    utility::Exception::check(number_ranks_cast == count_src, "MPICollectives::all_to_all: Sizes do not match: {} vs {}", number_ranks_cast, count_src);

    auto dst = std::vector<T>{};
    dst.resize(count_src);

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const int error_code = MPI_Alltoall(src.data(), 1, type, dst.data(), 1, type, communicator.get());
    utility::Exception::check(error_code == 0, "MPICollectives::all_to_all: Error code received: {}", error_code);

    MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(number_ranks_cast) * sizeof(T));
    MPICounters::add_to_received(utility::safe_cast<std::uint64_t>(number_ranks_cast) * sizeof(T));

    return dst;
}

/**
 * @brief Exchanges values between all mpi ranks. Uses the provided buffer.
 *      Before the call:
 *          rank k has src[i] = v
 *      After the call:
 *          rank i has tgt[k] = v
 * @tparam T The type of data to exchange
 * @param src The elements to exchange
 * @param tgt The range where to store the received values
 * @param communicator The communicator to exchange within, defaults to the world communicator
 * @exception Throws an Exception if src.size() or tgt.size() are not equal to the number of ranks or mpi returns an error code
 */
template <MPICompatible T>
void all_to_all(const std::span<const T> src, const std::span<T> tgt, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto number_ranks_cast = utility::safe_cast<std::size_t>(number_ranks);
    const auto count_src = src.size();
    const auto count_tgt = tgt.size();

    utility::Exception::check(number_ranks_cast == count_src, "MPICollectives::all_to_all: Sizes do not match: {} vs {}", number_ranks_cast, count_src);
    utility::Exception::check(number_ranks_cast == count_tgt, "MPICollectives::all_to_all: Sizes do not match: {} vs {}", number_ranks_cast, count_tgt);

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const int error_code = MPI_Alltoall(src.data(), 1, type, tgt.data(), 1, type, communicator.get());
    utility::Exception::check(error_code == 0, "MPICollectives::all_to_all: Error code received: {}", error_code);

    MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(number_ranks_cast) * sizeof(T));
    MPICounters::add_to_received(utility::safe_cast<std::uint64_t>(number_ranks_cast) * sizeof(T));
}
} // namespace MPICollectives

} // namespace mpiPP
