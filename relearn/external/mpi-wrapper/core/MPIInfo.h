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

#include "mpi-wrapper/core/MPIRank.h"

#include <cpp-utility/Exception.hpp>
#include <cpp-utility/Math.hpp>

#include <string>
#include <string_view>

namespace mpiPP {

class MPIWrapper;

/**
 * This class provides fast and initialized access to a small portion of MPI.
 * Must not be used before calling MPIWrapper::init(...) and after calling
 * MPIWrapper::finalize(...).
 */
class MPIInfo {
public:
    // MPIWrapper should be able to call init(...), but no other class
    friend class MPIWrapper;

    /**
     * @brief Returns the current mpi rank in a typesafe manner
     * @exception Throws an Exception if init(...) has not been called before
     * @return The current rank
     */
    [[nodiscard]] static MPIRank get_my_rank() {
        utility::Exception::check(my_rank >= 0, "MPIInfo::get_my_rank: Not initialized before");
        return MPIRank{ my_rank };
    }

    /**
     * @brief Checks if the curent mpi rank is the root rank (rank 0)
     * @exception Throws an Exception if init(...) has not been called before
     * @return True iff the current rank is the root rank
     */
    [[nodiscard]] static bool is_root_rank() {
        utility::Exception::check(my_rank >= 0, "MPIInfo::is_root_rank: Not initialized before");
        return my_rank == 0;
    }

    /**
     * @brief Returns the current MPI rank's id as string
     * @exception Throws an Exception if init(...) has not been called before
     * @return The current MPI rank's id as string
     */
    [[nodiscard]] static std::string_view get_my_rank_str() {
        utility::Exception::check(my_rank >= 0, "MPIInfo::get_my_rank_str: Not initialized before");
        return my_rank_string;
    }

    /**
     * @brief Returns the number of mpi ranks
     * @exception Throws an Exception if init(...) has not been called before
     * @return The number of mpi ranks, >0
     */
    [[nodiscard]] static int get_number_ranks() {
        utility::Exception::check(number_ranks > 0, "MPIInfo::get_number_ranks: Not initialized before");
        return number_ranks;
    }

    /**
     * @brief Returns the number of mpi ranks cast to std::size_t
     * @exception Throws an Exception if init(...) has not been called before
     * @return The number of mpi ranks, >0
     */
    [[nodiscard]] static std::size_t get_number_ranks_cast() {
        utility::Exception::check(number_ranks > 0, "MPIInfo::get_number_ranks_cast: Not initialized before");
        return static_cast<std::size_t>(number_ranks);
    }

private:
    static void init(const int _number_ranks, const int _my_rank) {
        utility::Exception::check(MPIInfo::number_ranks < 1, "MPIInfo::init: Has been initialized before");

        my_rank = _my_rank;
        number_ranks = _number_ranks;

        const unsigned int num_digits = utility::num_digits(number_ranks - 1);
        my_rank_string = fmt::format("{1:0>{0}}", num_digits, my_rank);
    }

    static void finalize() { }

    static inline std::string my_rank_string{ "-1" };
    static inline int my_rank{ -1 };
    static inline int number_ranks{ -1 };
};

} // namespace mpiPP
