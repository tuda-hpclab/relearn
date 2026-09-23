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

#include <cpp-utility/Exception.hpp>

#include <mpi.h>

namespace mpiPP {

namespace MPISynchronization {
/**
 * @brief Starts a barrier without blocking. Every rank must call this and eventually complete the
 *      returned request (via wait/test); a rank may do unrelated work in between. This is the
 *      non-blocking counterpart of barrier().
 * @param communicator The communicator to synchronize within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return A token that can be waited on
 */
[[nodiscard]] inline MPI_Request async_barrier(const MPICommunicator& communicator = MPICommunicator::World) {
    auto request = MPI_Request{};

    const auto error_code = MPI_Ibarrier(communicator.get(), &request);
    utility::Exception::check(error_code == MPI_SUCCESS, "MPISynchronization::async_barrier: Returned error code {}", error_code);

    return request;
}
} // namespace MPISynchronization

} // namespace mpiPP
