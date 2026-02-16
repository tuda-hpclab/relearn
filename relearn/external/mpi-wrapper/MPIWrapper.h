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

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPITypes.h"

#include "cpp-utility/Exception.hpp"

#include <mpi.h>

namespace mpiPP {

/**
 * @brief This class provides the hook for initializing and finalizing
 */
class MPIWrapper {
public:
    /**
     * @brief Initializes the mpi implementation. Must be called before other calls to the wrapping classes
     * @param argc The number of arguments to the program
     * @param argv The arguments to the program
     */
    static void init(int argc, char** argv) {
        const auto error_code = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level_provided);
        utility::Exception::check(error_code == 0, "MPIWrapper::init: error_code is {}", error_code);

        int number_ranks = 1;
        int my_rank = 0;

        // NOLINTNEXTLINE
        MPI_Comm_size(MPI_COMM_WORLD, &number_ranks);
        // NOLINTNEXTLINE
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

        MPIInfo::init(number_ranks, my_rank);
        MPITypes::init();
    }

    /**
     * @brief Finalizes all mpi interaction. Must be called before the end of the program
     */
    static void finalize() {
        MPITypes::finalize();
        MPIInfo::finalize();

        const auto error_code = MPI_Finalize();
        utility::Exception::check(error_code == 0, "MPIWrapper::finalize: error_code is {}", error_code);
    }

private:
    static inline int thread_level_provided{ -1 }; // Thread level provided by MPI
};

} // namespace mpiPP
