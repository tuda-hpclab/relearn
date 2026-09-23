/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_fired_status_approximator.h"

#include "neurons/firing/FiredStatusApproximator.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"

#include "factory/extra_info/extra_info_factory.h"
#include "factory/network_graph/network_graph_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/core/MPIRankRange.h>

#include <iostream>
#include <memory>
#include <tuple>

TEST_F(FiredStatusApproximatorTest, testConstructorThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_THROW_NO_PRINT(FiredStatusApproximator fsa(mpiPP::MPIRank::root_rank(), 0), RelearnException);
    ASSERT_THROW_NO_PRINT(FiredStatusApproximator fsa(mpiPP::MPIRank::root_rank(), -1), RelearnException);
    ASSERT_THROW_NO_PRINT(FiredStatusApproximator fsa(mpiPP::MPIRank::root_rank(), -4), RelearnException);
}

TEST_F(FiredStatusApproximatorTest, testConstructorNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_NO_THROW(FiredStatusApproximator fsa(mpiPP::MPIRank::root_rank(), 1));
    ASSERT_NO_THROW(FiredStatusApproximator fsa(mpiPP::MPIRank::root_rank(), 2));
    ASSERT_NO_THROW(FiredStatusApproximator fsa(mpiPP::MPIRank::root_rank(), 4));
    ASSERT_NO_THROW(FiredStatusApproximator fsa(mpiPP::MPIRank::root_rank(), 8));
    ASSERT_NO_THROW(FiredStatusApproximator fsa(mpiPP::MPIRank::root_rank(), 16));
    ASSERT_NO_THROW(FiredStatusApproximator fsa(mpiPP::MPIRank::root_rank(), 32));
    ASSERT_NO_THROW(FiredStatusApproximator fsa(mpiPP::MPIRank::root_rank(), 64));
    ASSERT_NO_THROW(FiredStatusApproximator fsa(mpiPP::MPIRank::root_rank(), 128));
}

TEST_F(FiredStatusApproximatorTest, testGetNumberRanks) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto fsa_1 = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 1);
    ASSERT_EQ(fsa_1.get_number_ranks(), 1);

    const auto fsa_2 = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 2);
    ASSERT_EQ(fsa_2.get_number_ranks(), 2);

    const auto fsa_4 = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 4);
    ASSERT_EQ(fsa_4.get_number_ranks(), 4);

    const auto fsa_8 = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 8);
    ASSERT_EQ(fsa_8.get_number_ranks(), 8);

    const auto fsa_16 = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 16);
    ASSERT_EQ(fsa_16.get_number_ranks(), 16);

    const auto fsa_32 = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 32);
    ASSERT_EQ(fsa_32.get_number_ranks(), 32);

    const auto fsa_64 = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 64);
    ASSERT_EQ(fsa_64.get_number_ranks(), 64);

    const auto fsa_128 = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 128);
    ASSERT_EQ(fsa_128.get_number_ranks(), 128);
}

TEST_F(FiredStatusApproximatorTest, testInitAndCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fsa = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 1);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_1 = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_2 = NeuronIdFactory::get_random_number_neurons(this->mt);

    ASSERT_THROW_NO_PRINT(fsa.init(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fsa.create_neurons(number_neurons_create_1), RelearnException);
    ASSERT_THROW_NO_PRINT(fsa.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fsa.create_neurons(number_neurons_create_2), RelearnException);
    ASSERT_THROW_NO_PRINT(fsa.init(0), RelearnException);

    ASSERT_EQ(fsa.get_number_local_neurons(), 0);

    fsa.init(number_neurons_init);

    ASSERT_EQ(fsa.get_number_local_neurons(), number_neurons_init);

    ASSERT_THROW_NO_PRINT(fsa.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(fsa.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fsa.init(0), RelearnException);

    ASSERT_EQ(fsa.get_number_local_neurons(), number_neurons_init);

    fsa.create_neurons(number_neurons_create_1);

    ASSERT_EQ(fsa.get_number_local_neurons(), number_neurons_init + number_neurons_create_1);

    ASSERT_THROW_NO_PRINT(fsa.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(fsa.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fsa.init(0), RelearnException);

    ASSERT_EQ(fsa.get_number_local_neurons(), number_neurons_init + number_neurons_create_1);

    fsa.create_neurons(number_neurons_create_2);

    ASSERT_EQ(fsa.get_number_local_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);

    ASSERT_THROW_NO_PRINT(fsa.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(fsa.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fsa.init(0), RelearnException);

    ASSERT_EQ(fsa.get_number_local_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
}

TEST_F(FiredStatusApproximatorTest, testSetExtraInfoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fsa = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 1);

    ASSERT_THROW_NO_PRINT(fsa.set_extra_infos({}), RelearnException);
}

TEST_F(FiredStatusApproximatorTest, testSetNetworkGraphThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fsa = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 1);

    ASSERT_THROW_NO_PRINT(fsa.set_network_graph({}), RelearnException);
}

TEST_F(FiredStatusApproximatorTest, testContainsThrow1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons_init);

    auto fsa = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 1);

    fsa.init(number_neurons_init);
    fsa.set_extra_infos(neurons_extra_info);
    fsa.set_network_graph(network_graph);

    for (const auto rank : mpiPP::MPIRankRange::range(1, 8)) {
        for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
            ASSERT_THROW_NO_PRINT(std::ignore = fsa.contains(rank, neuron_id), RelearnException);
        }
    }
}

TEST_F(FiredStatusApproximatorTest, testContainsThrow4) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons_init);

    auto fsa = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 4);

    fsa.init(number_neurons_init);
    fsa.set_extra_infos(neurons_extra_info);
    fsa.set_network_graph(network_graph);

    for (const auto rank : mpiPP::MPIRankRange::range(4, 8)) {
        for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
            ASSERT_THROW_NO_PRINT(std::ignore = fsa.contains(rank, neuron_id), RelearnException);
        }
    }
}

TEST_F(FiredStatusApproximatorTest, testContainsFalse1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons_init);

    auto fsa = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 1);

    fsa.init(number_neurons_init);
    fsa.set_extra_infos(neurons_extra_info);
    fsa.set_network_graph(network_graph);

    for (const auto rank : mpiPP::MPIRankRange::range(1)) {
        for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
            ASSERT_EQ(fsa.contains(rank, neuron_id), false);
        }
    }
}

TEST_F(FiredStatusApproximatorTest, testContainsFalse4) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons_init);

    auto fsa = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 4);

    fsa.init(number_neurons_init);
    fsa.set_extra_infos(neurons_extra_info);
    fsa.set_network_graph(network_graph);

    for (const auto rank : mpiPP::MPIRankRange::range(4)) {
        for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
            ASSERT_EQ(fsa.contains(rank, neuron_id), false);
        }
    }
}

TEST_F(FiredStatusApproximatorTest, testFootprintNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons_init);

    auto fsa = FiredStatusApproximator(mpiPP::MPIRank::root_rank(), 4);

    fsa.init(number_neurons_init);
    fsa.set_extra_infos(neurons_extra_info);
    fsa.set_network_graph(network_graph);

    const auto footprint = std::make_unique<utility::MemoryFootprint>(100);

    ASSERT_NO_THROW(fsa.record_memory_footprint(footprint));
}
