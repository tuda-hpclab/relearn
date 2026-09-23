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

#include "mpi-wrapper/core/MPIInfo.h"
#include "mpi-wrapper/core/MPITypes.h"

#include <cpp-utility/Exception.hpp>

#include <mpi.h>

namespace mpiPP {

/**
 * @brief The level of threading support requested from / provided by the MPI implementation,
 *      in increasing order of guarantees. Mirrors the MPI_THREAD_* constants.
 */
enum class MPIThreadLevel {
    Single,     ///< Only a single thread will execute (MPI_THREAD_SINGLE)
    Funneled,   ///< Only the thread that called init will make MPI calls (MPI_THREAD_FUNNELED)
    Serialized, ///< Multiple threads may make MPI calls, but not concurrently (MPI_THREAD_SERIALIZED)
    Multiple,   ///< Multiple threads may make MPI calls concurrently (MPI_THREAD_MULTIPLE)
};

/**
 * @brief Converts a thread level to the matching MPI_THREAD_* constant.
 * @param level The thread level
 * @return The matching MPI_THREAD_* constant
 */
[[nodiscard]] inline int to_mpi_thread_level(const MPIThreadLevel level) {
    if (level == MPIThreadLevel::Single) {
        return MPI_THREAD_SINGLE;
    }
    if (level == MPIThreadLevel::Funneled) {
        return MPI_THREAD_FUNNELED;
    }
    if (level == MPIThreadLevel::Serialized) {
        return MPI_THREAD_SERIALIZED;
    }
    if (level == MPIThreadLevel::Multiple) {
        return MPI_THREAD_MULTIPLE;
    }
    utility::Exception::fail("to_mpi_thread_level: Unknown MPI thread level {}.", static_cast<int>(level));
}

/**
 * @brief Converts an MPI_THREAD_* constant to the matching thread level.
 * @param level The MPI_THREAD_* constant
 * @exception Throws an Exception if the constant is not a known thread level
 * @return The matching thread level
 */
[[nodiscard]] inline MPIThreadLevel from_mpi_thread_level(const int level) {
    if (level == MPI_THREAD_SINGLE) {
        return MPIThreadLevel::Single;
    }
    if (level == MPI_THREAD_FUNNELED) {
        return MPIThreadLevel::Funneled;
    }
    if (level == MPI_THREAD_SERIALIZED) {
        return MPIThreadLevel::Serialized;
    }
    if (level == MPI_THREAD_MULTIPLE) {
        return MPIThreadLevel::Multiple;
    }
    utility::Exception::fail("from_mpi_thread_level: Unknown MPI thread level {}.", level);
}

/**
 * @brief This class provides the hook for initializing and finalizing
 */
class MPIWrapper {
public:
    /**
     * @brief Initializes the mpi implementation. Must be called before other calls to the wrapping classes
     * @param argc The number of arguments to the program
     * @param argv The arguments to the program
     * @param requested_thread_level The level of threading support to request, defaults to Multiple.
     *      The level actually granted may be lower and can be queried with get_provided_thread_level.
     */
    static void init(int argc, char** argv, const MPIThreadLevel requested_thread_level = MPIThreadLevel::Multiple) {
        const auto error_code = MPI_Init_thread(&argc, &argv, to_mpi_thread_level(requested_thread_level), &thread_level_provided);
        utility::Exception::check(error_code == 0, "MPIWrapper::init: error_code is {}", error_code);

        const auto world_error_handler_code = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        utility::Exception::check(world_error_handler_code == MPI_SUCCESS,
                                  "MPIWrapper::init: Setting MPI_ERRORS_RETURN on MPI_COMM_WORLD returned error code {}", world_error_handler_code);
        const auto self_error_handler_code = MPI_Comm_set_errhandler(MPI_COMM_SELF, MPI_ERRORS_RETURN);
        utility::Exception::check(self_error_handler_code == MPI_SUCCESS,
                                  "MPIWrapper::init: Setting MPI_ERRORS_RETURN on MPI_COMM_SELF returned error code {}", self_error_handler_code);

        int number_ranks = 1;
        int my_rank = 0;

        const auto size_error_code = MPI_Comm_size(MPI_COMM_WORLD, &number_ranks);
        utility::Exception::check(size_error_code == MPI_SUCCESS, "MPIWrapper::init: MPI_Comm_size returned error code {}", size_error_code);
        const auto rank_error_code = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        utility::Exception::check(rank_error_code == MPI_SUCCESS, "MPIWrapper::init: MPI_Comm_rank returned error code {}", rank_error_code);

        MPIInfo::init(number_ranks, my_rank);
        MPITypes::init();
    }

    /**
     * @brief Returns the level of threading support the MPI implementation actually provided at
     *      initialization. This may be lower than the level requested in init.
     * @exception Throws an Exception if the wrapper has not been initialized yet
     * @return The provided thread level
     */
    [[nodiscard]] static MPIThreadLevel get_provided_thread_level() {
        utility::Exception::check(thread_level_provided != -1, "MPIWrapper::get_provided_thread_level: MPI has not been initialized.");
        return from_mpi_thread_level(thread_level_provided);
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
