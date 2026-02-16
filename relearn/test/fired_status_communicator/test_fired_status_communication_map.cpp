/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_fired_status_communication_map.h"

#include "neurons/firing/FiredStatusCommunicationMap.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/extra_info/extra_info_factory.h"
#include "factory/network_graph/network_graph_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <tuple>

TEST_F(FiredStatusCommunicationMapTest, testConstructorThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_THROW_NO_PRINT(FiredStatusCommunicationMap fscm(0), RelearnException);
    ASSERT_THROW_NO_PRINT(FiredStatusCommunicationMap fscm(-1), RelearnException);
    ASSERT_THROW_NO_PRINT(FiredStatusCommunicationMap fscm(-4), RelearnException);
}

TEST_F(FiredStatusCommunicationMapTest, testConstructorNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_NO_THROW(FiredStatusCommunicationMap fscm(1));
    ASSERT_NO_THROW(FiredStatusCommunicationMap fscm(2));
    ASSERT_NO_THROW(FiredStatusCommunicationMap fscm(4));
    ASSERT_NO_THROW(FiredStatusCommunicationMap fscm(8));
    ASSERT_NO_THROW(FiredStatusCommunicationMap fscm(16));
    ASSERT_NO_THROW(FiredStatusCommunicationMap fscm(32));
    ASSERT_NO_THROW(FiredStatusCommunicationMap fscm(64));
    ASSERT_NO_THROW(FiredStatusCommunicationMap fscm(128));
}

TEST_F(FiredStatusCommunicationMapTest, testGetNumberRanks) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto fscm_1 = FiredStatusCommunicationMap(1);
    ASSERT_EQ(fscm_1.get_number_ranks(), 1);

    const auto fscm_2 = FiredStatusCommunicationMap(2);
    ASSERT_EQ(fscm_2.get_number_ranks(), 2);

    const auto fscm_4 = FiredStatusCommunicationMap(4);
    ASSERT_EQ(fscm_4.get_number_ranks(), 4);

    const auto fscm_8 = FiredStatusCommunicationMap(8);
    ASSERT_EQ(fscm_8.get_number_ranks(), 8);

    const auto fscm_16 = FiredStatusCommunicationMap(16);
    ASSERT_EQ(fscm_16.get_number_ranks(), 16);

    const auto fscm_32 = FiredStatusCommunicationMap(32);
    ASSERT_EQ(fscm_32.get_number_ranks(), 32);

    const auto fscm_64 = FiredStatusCommunicationMap(64);
    ASSERT_EQ(fscm_64.get_number_ranks(), 64);

    const auto fscm_128 = FiredStatusCommunicationMap(128);
    ASSERT_EQ(fscm_128.get_number_ranks(), 128);
}

TEST_F(FiredStatusCommunicationMapTest, testInitAndCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fscm = FiredStatusCommunicationMap(1);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_1 = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_2 = NeuronIdFactory::get_random_number_neurons(this->mt);

    ASSERT_THROW_NO_PRINT(fscm.init(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fscm.create_neurons(number_neurons_create_1), RelearnException);
    ASSERT_THROW_NO_PRINT(fscm.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fscm.create_neurons(number_neurons_create_2), RelearnException);
    ASSERT_THROW_NO_PRINT(fscm.init(0), RelearnException);

    ASSERT_EQ(fscm.get_number_local_neurons(), 0);

    fscm.init(number_neurons_init);

    ASSERT_EQ(fscm.get_number_local_neurons(), number_neurons_init);

    ASSERT_THROW_NO_PRINT(fscm.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(fscm.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fscm.init(0), RelearnException);

    ASSERT_EQ(fscm.get_number_local_neurons(), number_neurons_init);

    fscm.create_neurons(number_neurons_create_1);

    ASSERT_EQ(fscm.get_number_local_neurons(), number_neurons_init + number_neurons_create_1);

    ASSERT_THROW_NO_PRINT(fscm.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(fscm.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fscm.init(0), RelearnException);

    ASSERT_EQ(fscm.get_number_local_neurons(), number_neurons_init + number_neurons_create_1);

    fscm.create_neurons(number_neurons_create_2);

    ASSERT_EQ(fscm.get_number_local_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);

    ASSERT_THROW_NO_PRINT(fscm.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(fscm.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fscm.init(0), RelearnException);

    ASSERT_EQ(fscm.get_number_local_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
}

TEST_F(FiredStatusCommunicationMapTest, testSetExtraInfoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fscm = FiredStatusCommunicationMap(1);

    ASSERT_THROW_NO_PRINT(fscm.set_extra_infos({}), RelearnException);
}

TEST_F(FiredStatusCommunicationMapTest, testSetNetworkGraphThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fscm = FiredStatusCommunicationMap(1);

    ASSERT_THROW_NO_PRINT(fscm.set_network_graph({}), RelearnException);
}

TEST_F(FiredStatusCommunicationMapTest, testContainsThrow1) {
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

    auto fscm = FiredStatusCommunicationMap(1);

    fscm.init(number_neurons_init);
    fscm.set_extra_infos(neurons_extra_info);
    fscm.set_network_graph(network_graph);

    for (const auto rank : mpiPP::MPIRank::range(1, 8)) {
        for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
            ASSERT_THROW_NO_PRINT(std::ignore = fscm.contains(rank, neuron_id), RelearnException);
        }
    }
}

TEST_F(FiredStatusCommunicationMapTest, testContainsThrow4) {
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

    auto fscm = FiredStatusCommunicationMap(4);

    fscm.init(number_neurons_init);
    fscm.set_extra_infos(neurons_extra_info);
    fscm.set_network_graph(network_graph);

    for (const auto rank : mpiPP::MPIRank::range(4, 8)) {
        for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
            ASSERT_THROW_NO_PRINT(std::ignore = fscm.contains(rank, neuron_id), RelearnException);
        }
    }
}

TEST_F(FiredStatusCommunicationMapTest, testContainsFalse1) {
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

    auto fscm = FiredStatusCommunicationMap(1);

    fscm.init(number_neurons_init);
    fscm.set_extra_infos(neurons_extra_info);
    fscm.set_network_graph(network_graph);

    for (const auto rank : mpiPP::MPIRank::range(1)) {
        for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
            ASSERT_EQ(fscm.contains(rank, neuron_id), false);
        }
    }
}

TEST_F(FiredStatusCommunicationMapTest, testContainsFalse4) {
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

    auto fscm = FiredStatusCommunicationMap(4);

    fscm.init(number_neurons_init);
    fscm.set_extra_infos(neurons_extra_info);
    fscm.set_network_graph(network_graph);

    for (const auto rank : mpiPP::MPIRank::range(4)) {
        for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
            ASSERT_EQ(fscm.contains(rank, neuron_id), false);
        }
    }
}

TEST_F(FiredStatusCommunicationMapTest, testContainsRandom) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_ranks = 4;

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons_init);

    auto fscm = FiredStatusCommunicationMap(number_ranks);

    fscm.init(number_neurons_init);
    fscm.set_extra_infos(neurons_extra_info);
    fscm.set_network_graph(network_graph);

    auto fired_neurons = std::set<RankNeuronId>{};
    for (const auto rank : mpiPP::MPIRank::range(1, number_ranks)) {
        for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
            if (RandomFactory::get_random_bool(mt)) {
                fired_neurons.insert({ rank, neuron_id });
            }
        }
    }
    fscm.set_incoming_ids(fired_neurons);

    for (const auto rank : mpiPP::MPIRank::range(4)) {
        for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
            const auto fired = fscm.contains(rank, neuron_id);
            ASSERT_EQ(fscm.contains(rank, neuron_id), fired);
        }
    }
}

TEST_F(FiredStatusCommunicationMapTest, testFootprintNoThrow) {
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

    auto fscm = FiredStatusCommunicationMap(4);

    fscm.init(number_neurons_init);
    fscm.set_extra_infos(neurons_extra_info);
    fscm.set_network_graph(network_graph);

    const auto footprint = std::make_unique<utility::MemoryFootprint>(100);

    ASSERT_NO_THROW(fscm.record_memory_footprint(footprint));
}
