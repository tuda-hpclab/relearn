/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_rank_neuron_id.h"

#include "neurons/helper/RankNeuronId.h"
#include "util/RelearnException.h"

#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <iostream>

TEST_F(RankNeuronIdTest, testNeuronRankIdValid) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    for (auto i = 0; i < 1000; i++) {
        const auto rank = MPIRankFactory::get_random_mpi_rank(mt);
        const auto id = NeuronIdFactory::get_random_neuron_id(mt);

        const auto rni = RankNeuronId{ rank, NeuronID{ id } };

        ASSERT_EQ(rni.get_neuron_id(), NeuronID{ id });
        ASSERT_TRUE(rni.get_rank() == rank);
    }
}

TEST_F(RankNeuronIdTest, testNeuronRankIdInvalidId) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    for (auto i = 0; i < 1000; i++) {
        const auto rank = MPIRankFactory::get_random_mpi_rank(mt);

        const auto rni = RankNeuronId(rank, NeuronID::uninitialized_id());

        ASSERT_NO_THROW(std::ignore = rni.get_rank());
        ASSERT_THROW_NO_PRINT(std::ignore = rni.get_neuron_id(), RelearnException);
    }
}

TEST_F(RankNeuronIdTest, testNeuronRankIdEquality) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    for (auto i = 0; i < 1000; i++) {
        const auto rank_1 = MPIRankFactory::get_random_mpi_rank(mt);
        const auto id_1 = NeuronIdFactory::get_random_neuron_id(mt);

        const auto rank_2 = MPIRankFactory::get_random_mpi_rank(mt);
        const auto id_2 = NeuronIdFactory::get_random_neuron_id(mt);

        const auto rni_1 = RankNeuronId(rank_1, NeuronID{ id_1 });
        const auto rni_2 = RankNeuronId(rank_2, NeuronID{ id_2 });

        if (rank_1 == rank_2 && id_1 == id_2) {
            ASSERT_EQ(rni_1, rni_2);
        } else {
            ASSERT_NE(rni_1, rni_2);
        }
    }
}
