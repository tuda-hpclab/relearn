#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "mpi-wrapper/MPIRank.h"

#include <random>

class MPIRankFactory {
public:
    constexpr static int upper_bound_num_ranks = 32;

    static int get_random_number_ranks(std::mt19937& mt);

    static int get_random_number_ranks(std::mt19937& mt, int minimum);

    static int get_adjusted_random_number_ranks(std::mt19937& mt);

    static int get_adjusted_random_number_ranks(std::mt19937& mt, int minimum);

    static mpiPP::MPIRank get_random_mpi_rank(std::mt19937& mt);

    static mpiPP::MPIRank get_random_mpi_rank(size_t number_ranks, std::mt19937& mt);

    static mpiPP::MPIRank get_random_mpi_rank(size_t number_ranks, mpiPP::MPIRank except, std::mt19937& mt);

    static mpiPP::MPIRank get_random_mpi_rank(int number_ranks, std::mt19937& mt);

    static mpiPP::MPIRank get_random_mpi_rank(int number_ranks, mpiPP::MPIRank except, std::mt19937& mt);

private:
    static size_t round_to_next_exponent(size_t numToRound, size_t exponent);
};
