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
#include "mpi-wrapper/reductions/MPIReductions.h"

#include <cpp-utility/Status.hpp>

#include <cstdint>

namespace mpiPP {

/**
 * @brief Adapts an MPICommunicator to utility::BasicStatus.
 *
 * The constructor of utility::BasicStatus performs one collective reduction through this reporter; all
 * participating ranks must therefore construct their MPIProgress in the same order. The communicator must outlive
 * this reporter and every MPIProgress that stores it.
 */
class MPIProgressReporter {
public:
    /**
     * @brief Uses the given communicator for the progress total and for selecting its root rank
     * @param communicator The communicator whose rank zero writes the progress output
     */
    explicit MPIProgressReporter(const MPICommunicator& communicator = MPICommunicator::World) noexcept
        : communicator_(&communicator) { }

    /**
     * @brief Reduces the local work amount to the communicator root
     * @param value The local amount of work
     * @return The global sum on the root rank and zero on all other ranks
     */
    [[nodiscard]] std::uint64_t reduce_sum(const std::uint64_t value) const {
        return MPIReductions::reduce_sum(value, *communicator_);
    }

    /**
     * @brief Returns whether this rank is the reporting rank of the configured communicator
     * @return True exactly on communicator rank zero
     */
    [[nodiscard]] bool is_root() const {
        return communicator_->is_root_rank();
    }

private:
    const MPICommunicator* communicator_;
};

/**
 * @brief A distributed progress reporter that writes only on rank zero of its MPIProgressReporter communicator
 */
using MPIProgress = utility::BasicStatus<MPIProgressReporter>;

} // namespace mpiPP
