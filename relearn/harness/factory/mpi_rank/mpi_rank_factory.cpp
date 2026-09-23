/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "mpi_rank_factory.h"

#include "factory/random/random_factory.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <cmath>
#include <cstddef>
#include <random>

int MPIRankFactory::get_random_number_ranks(std::mt19937& mt) {
    return get_random_number_ranks(mt, 1);
}

int MPIRankFactory::get_random_number_ranks(std::mt19937& mt, int minimum) {
    return RandomFactory::get_random_integer<int>(minimum, upper_bound_num_ranks, mt);
}

int MPIRankFactory::get_adjusted_random_number_ranks(std::mt19937& mt) {
    const auto random_rank = get_random_number_ranks(mt);
    return static_cast<int>(round_to_next_exponent(static_cast<std::size_t>(random_rank), 2));
}

int MPIRankFactory::get_adjusted_random_number_ranks(std::mt19937& mt, int minimum) {
    const auto random_rank = get_random_number_ranks(mt, minimum);
    return static_cast<int>(round_to_next_exponent(static_cast<std::size_t>(random_rank), 2));
}

mpiPP::MPIRank MPIRankFactory::get_random_mpi_rank(std::mt19937& mt) {
    const auto rank = RandomFactory::get_random_integer<int>(0, upper_bound_num_ranks - 1, mt);
    return mpiPP::MPIRank(rank);
}

mpiPP::MPIRank MPIRankFactory::get_random_mpi_rank(int number_ranks, std::mt19937& mt) {
    const auto rank = RandomFactory::get_random_integer<int>(0, number_ranks - 1, mt);
    return mpiPP::MPIRank(rank);
}

mpiPP::MPIRank MPIRankFactory::get_random_mpi_rank(int number_ranks, mpiPP::MPIRank except, std::mt19937& mt) {
    RelearnException::check(number_ranks > 1, "mpiPP::MPIRank MPIRankFactory::get_random_mpi_rank: number_ranks must be larger than 1");
    auto mpi_rank = mpiPP::MPIRank{ RandomFactory::get_random_integer<int>(0, number_ranks - 1, mt) };

    while (mpi_rank == except) {
        const auto rank = RandomFactory::get_random_integer<int>(0, number_ranks - 1, mt);
        mpi_rank = mpiPP::MPIRank(rank);
    }

    return mpi_rank;
}

size_t MPIRankFactory::round_to_next_exponent(size_t numToRound, size_t exponent) {
    auto log = std::log(static_cast<double>(numToRound)) / std::log(static_cast<double>(exponent));
    auto rounded_exp = std::ceil(log);
    auto new_val = std::pow(static_cast<double>(exponent), rounded_exp);
    return static_cast<size_t>(new_val);
}
