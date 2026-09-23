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

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

#include <mpi.h>

#include <span>
#include <vector>

namespace mpiPP {

namespace MPISynchronization {
/**
 * @brief Waits for the specified request to complete
 * @param request The request
 * @exception Throws an Exception if mpi reports an error
 */
inline void wait(MPI_Request& request) {
    const auto error_code = MPI_Wait(&request, MPI_STATUS_IGNORE);
    utility::Exception::check(error_code == MPI_SUCCESS, "MPISynchronization::wait: Returned error code {}", error_code);
}

/**
 * @brief Tests the specified request for completion
 * @param request The request
 * @exception Throws an Exception if mpi reports an error
 * @return True iff the request is finished
 */
[[nodiscard]] inline bool test(MPI_Request& request) {
    auto is_ready = 0;

    const auto error_code = MPI_Test(&request, &is_ready, MPI_STATUS_IGNORE);
    utility::Exception::check(error_code == MPI_SUCCESS, "MPISynchronization::test: Returned error code {}", error_code);

    return static_cast<bool>(is_ready);
}

/**
 * @brief Waits for all of the specified requests to complete
 * @param request The requests
 * @exception Throws an Exception if mpi reports an error
 */
inline void wait_all(const std::span<MPI_Request> requests) {
    if (requests.empty()) {
        return;
    }

    auto statuses = std::vector<MPI_Status>{ requests.size() };

    const auto size_cast = utility::safe_cast<int>(requests.size());
    const auto error_code = MPI_Waitall(size_cast, requests.data(), statuses.data());
    utility::Exception::check(error_code == MPI_SUCCESS, "MPISynchronization::wait_all: Returned error code {}", error_code);
}

/**
 * @brief Blocks until all ranks have called barrier()
 * @param communicator The communicator to synchronize within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 */
inline void barrier(const MPICommunicator& communicator = MPICommunicator::World) {
    const auto error_code = MPI_Barrier(communicator.get());
    utility::Exception::check(error_code == MPI_SUCCESS, "MPISynchronization::barrier: Returned error code {}", error_code);
}
} // namespace MPISynchronization

} // namespace mpiPP
