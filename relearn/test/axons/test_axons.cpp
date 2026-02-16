/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_axons.h"

#include "RelearnTest.hpp"
#include "Types.h"

#include "neurons/enums/SynapticElementType.h"
#include "neurons/synaptic_elements/Axons.h"
#include "neurons/synaptic_elements/MultiPositionAxons.h"
#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/extra_info/extra_info_factory.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <set>
#include <utility>
#include <vector>

TEST_F(AxonsTest, testConstruction) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto axons = Axons{};

    ASSERT_EQ(axons.get_size(), 0);
    ASSERT_EQ(axons.get_total_additions(), 0.0);
    ASSERT_EQ(axons.get_total_deletions(), 0.0);
    ASSERT_EQ(axons.get_signal_types().size(), 0);
    ASSERT_EQ(axons.get_grown_elements().size(), 0);
    ASSERT_EQ(axons.get_deltas().size(), 0);
    ASSERT_EQ(axons.get_vacant_elements().size(), 0);
    ASSERT_EQ(axons.get_connected_elements().size(), 0);
    ASSERT_EQ(axons.get_vacant_retract_ratio().size(), 0);
    ASSERT_EQ(axons.get_minimum_calcium().size(), 0);
}

TEST_F(AxonsTest, testEmptyInitAndCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto axons = Axons{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 13 };
    const auto number_neurons_create = RelearnTypes::number_neurons_type{ 15 };
    const auto number_neurons = number_neurons_init + number_neurons_create;

    axons.init(number_neurons_init);

    const auto signal_types = axons.get_signal_types();
    const auto excitatory_axons_id = axons.get_excitatory_axon_ids();
    const auto inhibitory_axons_id = axons.get_inhibitory_axon_ids();

    const auto grown_elements = axons.get_grown_elements();
    const auto deltas = axons.get_deltas();
    const auto vacant_elements = axons.get_vacant_elements();
    const auto connected_elements = axons.get_connected_elements();
    const auto retract_ratio = axons.get_vacant_retract_ratio();
    const auto minimum_calcum = axons.get_minimum_calcium();

    ASSERT_EQ(axons.get_total_additions(), 0.0);
    ASSERT_EQ(axons.get_total_deletions(), 0.0);

    ASSERT_EQ(signal_types.size(), number_neurons_init);
    ASSERT_EQ(excitatory_axons_id.size(), 0);
    ASSERT_EQ(inhibitory_axons_id.size(), 0);

    ASSERT_EQ(grown_elements.size(), number_neurons_init);
    ASSERT_EQ(deltas.size(), number_neurons_init);
    ASSERT_EQ(vacant_elements.size(), number_neurons_init);
    ASSERT_EQ(connected_elements.size(), number_neurons_init);
    ASSERT_EQ(retract_ratio.size(), number_neurons_init);
    ASSERT_EQ(minimum_calcum.size(), number_neurons_init);

    for (const auto signal_type : signal_types) {
        ASSERT_EQ(signal_type, SignalType::Excitatory);
    }

    for (const auto val : grown_elements) {
        ASSERT_EQ(val, 0.0);
    }

    for (const auto delta : deltas) {
        ASSERT_EQ(delta, 0.0);
    }

    for (const auto vacant : vacant_elements) {
        ASSERT_EQ(vacant, 0U);
    }

    for (const auto connected : connected_elements) {
        ASSERT_EQ(connected, 0U);
    }

    for (const auto ratio : retract_ratio) {
        ASSERT_EQ(ratio, 0.0);
    }

    for (const auto calcium : minimum_calcum) {
        ASSERT_EQ(calcium, 0.0);
    }

    axons.create_neurons(number_neurons_create);

    const auto new_signal_types = axons.get_signal_types();
    const auto new_excitatory_axons_id = axons.get_excitatory_axon_ids();
    const auto new_inhibitory_axons_id = axons.get_inhibitory_axon_ids();

    const auto new_grown_elements = axons.get_grown_elements();
    const auto new_deltas = axons.get_deltas();
    const auto new_vacant_elements = axons.get_vacant_elements();
    const auto new_connected_elements = axons.get_connected_elements();
    const auto new_retract_ratio = axons.get_vacant_retract_ratio();
    const auto new_minimum_calcum = axons.get_minimum_calcium();

    ASSERT_EQ(axons.get_total_additions(), 0.0);
    ASSERT_EQ(axons.get_total_deletions(), 0.0);

    ASSERT_EQ(new_signal_types.size(), number_neurons);
    ASSERT_EQ(new_excitatory_axons_id.size(), 0);
    ASSERT_EQ(new_inhibitory_axons_id.size(), 0);

    ASSERT_EQ(new_grown_elements.size(), number_neurons);
    ASSERT_EQ(new_deltas.size(), number_neurons);
    ASSERT_EQ(new_vacant_elements.size(), number_neurons);
    ASSERT_EQ(new_connected_elements.size(), number_neurons);
    ASSERT_EQ(new_retract_ratio.size(), number_neurons);
    ASSERT_EQ(new_minimum_calcum.size(), number_neurons);

    for (const auto signal_type : new_signal_types) {
        ASSERT_EQ(signal_type, SignalType::Excitatory);
    }

    for (const auto val : new_grown_elements) {
        ASSERT_EQ(val, 0.0);
    }

    for (const auto delta : new_deltas) {
        ASSERT_EQ(delta, 0.0);
    }

    for (const auto vacant : new_vacant_elements) {
        ASSERT_EQ(vacant, 0U);
    }

    for (const auto connected : new_connected_elements) {
        ASSERT_EQ(connected, 0U);
    }

    for (const auto ratio : new_retract_ratio) {
        ASSERT_EQ(ratio, 0.0);
    }

    for (const auto calcium : new_minimum_calcum) {
        ASSERT_EQ(calcium, 0.0);
    }
}

TEST_F(AxonsTest, testSetSignalTypes) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto axons = Axons{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 9 };
    const auto number_neurons_create = RelearnTypes::number_neurons_type{ 4 };
    const auto number_neurons = number_neurons_create + number_neurons_init;

    axons.init(number_neurons_init);

    auto signal_types = std::vector<SignalType>{};
    signal_types.emplace_back(SignalType::Excitatory);
    signal_types.emplace_back(SignalType::Inhibitory);
    signal_types.emplace_back(SignalType::Excitatory);
    signal_types.emplace_back(SignalType::Inhibitory);
    signal_types.emplace_back(SignalType::Excitatory);
    signal_types.emplace_back(SignalType::Excitatory);
    signal_types.emplace_back(SignalType::Inhibitory);
    signal_types.emplace_back(SignalType::Excitatory);

    ASSERT_THROW_NO_PRINT(axons.set_signal_types(signal_types);, RelearnException);

    signal_types.emplace_back(SignalType::Excitatory);

    axons.set_signal_types(signal_types);

    const auto retrieved_signal_types = axons.get_signal_types();
    const auto excitatory_axon_ids = axons.get_excitatory_axon_ids();
    const auto inhibitory_axon_ids = axons.get_inhibitory_axon_ids();

    ASSERT_EQ(retrieved_signal_types.size(), number_neurons_init);
    ASSERT_EQ(excitatory_axon_ids.size(), 6);
    ASSERT_EQ(inhibitory_axon_ids.size(), 3);

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        ASSERT_EQ(retrieved_signal_types[i], signal_types[i]);
    }

    ASSERT_EQ(excitatory_axon_ids[0], 0);
    ASSERT_EQ(excitatory_axon_ids[1], 2);
    ASSERT_EQ(excitatory_axon_ids[2], 4);
    ASSERT_EQ(excitatory_axon_ids[3], 5);
    ASSERT_EQ(excitatory_axon_ids[4], 7);
    ASSERT_EQ(excitatory_axon_ids[5], 8);

    ASSERT_EQ(inhibitory_axon_ids[0], 1);
    ASSERT_EQ(inhibitory_axon_ids[1], 3);
    ASSERT_EQ(inhibitory_axon_ids[2], 6);

    signal_types.emplace_back(SignalType::Inhibitory);
    ASSERT_THROW_NO_PRINT(axons.set_signal_types(signal_types);, RelearnException);

    signal_types.emplace_back(SignalType::Inhibitory);
    ASSERT_THROW_NO_PRINT(axons.set_signal_types(signal_types);, RelearnException);

    signal_types.emplace_back(SignalType::Inhibitory);
    ASSERT_THROW_NO_PRINT(axons.set_signal_types(signal_types);, RelearnException);

    signal_types.emplace_back(SignalType::Inhibitory);
    ASSERT_THROW_NO_PRINT(axons.set_signal_types(signal_types);, RelearnException);

    const auto new_retrieved_signal_types = axons.get_signal_types();
    const auto new_excitatory_axon_ids = axons.get_excitatory_axon_ids();
    const auto new_inhibitory_axon_ids = axons.get_inhibitory_axon_ids();

    ASSERT_EQ(new_retrieved_signal_types.size(), number_neurons_init);
    ASSERT_EQ(new_excitatory_axon_ids.size(), 6);
    ASSERT_EQ(new_inhibitory_axon_ids.size(), 3);

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        ASSERT_EQ(new_retrieved_signal_types[i], signal_types[i]);
    }

    ASSERT_EQ(new_excitatory_axon_ids[0], 0);
    ASSERT_EQ(new_excitatory_axon_ids[1], 2);
    ASSERT_EQ(new_excitatory_axon_ids[2], 4);
    ASSERT_EQ(new_excitatory_axon_ids[3], 5);
    ASSERT_EQ(new_excitatory_axon_ids[4], 7);
    ASSERT_EQ(new_excitatory_axon_ids[5], 8);

    ASSERT_EQ(new_inhibitory_axon_ids[0], 1);
    ASSERT_EQ(new_inhibitory_axon_ids[1], 3);
    ASSERT_EQ(new_inhibitory_axon_ids[2], 6);

    axons.create_neurons(number_neurons_create);
    axons.set_signal_types(signal_types);

    const auto last_retrieved_signal_types = axons.get_signal_types();
    const auto last_excitatory_axon_ids = axons.get_excitatory_axon_ids();
    const auto last_inhibitory_axon_ids = axons.get_inhibitory_axon_ids();

    ASSERT_EQ(last_retrieved_signal_types.size(), number_neurons);
    ASSERT_EQ(last_excitatory_axon_ids.size(), 6);
    ASSERT_EQ(last_inhibitory_axon_ids.size(), 7);

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons; i++) {
        ASSERT_EQ(last_retrieved_signal_types[i], signal_types[i]);
    }

    ASSERT_EQ(last_excitatory_axon_ids[0], 0);
    ASSERT_EQ(last_excitatory_axon_ids[1], 2);
    ASSERT_EQ(last_excitatory_axon_ids[2], 4);
    ASSERT_EQ(last_excitatory_axon_ids[3], 5);
    ASSERT_EQ(last_excitatory_axon_ids[4], 7);
    ASSERT_EQ(last_excitatory_axon_ids[5], 8);

    ASSERT_EQ(last_inhibitory_axon_ids[0], 1);
    ASSERT_EQ(last_inhibitory_axon_ids[1], 3);
    ASSERT_EQ(last_inhibitory_axon_ids[2], 6);
    ASSERT_EQ(last_inhibitory_axon_ids[3], 9);
    ASSERT_EQ(last_inhibitory_axon_ids[4], 10);
    ASSERT_EQ(last_inhibitory_axon_ids[5], 11);
    ASSERT_EQ(last_inhibitory_axon_ids[6], 12);
}

TEST_F(AxonsTest, testGetBoutons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto axons = Axons{};

    const auto number_neurons = RelearnTypes::number_neurons_type{ 6 };
    axons.init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto positions = std::vector<RelearnTypes::position_type>{};
    positions.emplace_back(0.0, 3.0, 1.0);
    positions.emplace_back(1.0, 3.0, 0.0);
    positions.emplace_back(2.0, 3.0, 1.0);
    positions.emplace_back(3.0, 3.0, 0.0);
    positions.emplace_back(4.0, 3.0, 1.0);
    positions.emplace_back(5.0, 3.0, 0.0);

    extra_info->set_positions(positions);
    axons.set_extra_infos(extra_info);

    for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < number_neurons; neuron_id++) {
        const auto number_boutons = axons.get_number_boutons(neuron_id);
        ASSERT_EQ(number_boutons, 1);
    }

    for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < number_neurons; neuron_id++) {
        const auto position = axons.get_bouton_position(neuron_id);
        ASSERT_EQ(position, positions[neuron_id]);
    }

    for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < number_neurons; neuron_id++) {
        const auto position = axons.get_bouton_position(neuron_id, 0);
        ASSERT_EQ(position, positions[neuron_id]);
    }

    for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < number_neurons; neuron_id++) {
        for (auto bouton_id = std::size_t{ 1 }; bouton_id < std::size_t{ 65 }; bouton_id++) {
            ASSERT_THROW_NO_PRINT(std::ignore = axons.get_bouton_position(neuron_id, bouton_id);, RelearnException);
        }
    }
}

TEST_F(AxonsTest, testMemoryFootprint) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto axons = Axons{};

    auto footprint = std::make_unique<utility::MemoryFootprint>(10);
    axons.record_memory_footprint(footprint);

    const auto& map = footprint->get_descriptions();

    ASSERT_EQ(map.size(), 2);
    ASSERT_NE(map.find("Axons"), map.end());
    ASSERT_NE(map.find("Axon Base"), map.end());

    ASSERT_GT(map.at("Axons"), 0);
    ASSERT_GT(map.at("Axon Base"), 0);
}

TEST_F(MultiPositionAxonsTest, testConstruction) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto axons = MultiPositionAxons{};

    ASSERT_EQ(axons.get_size(), 0);
    ASSERT_EQ(axons.get_total_additions(), 0.0);
    ASSERT_EQ(axons.get_total_deletions(), 0.0);
    ASSERT_EQ(axons.get_signal_types().size(), 0);
    ASSERT_EQ(axons.get_grown_elements().size(), 0);
    ASSERT_EQ(axons.get_deltas().size(), 0);
    ASSERT_EQ(axons.get_vacant_elements().size(), 0);
    ASSERT_EQ(axons.get_connected_elements().size(), 0);
    ASSERT_EQ(axons.get_vacant_retract_ratio().size(), 0);
    ASSERT_EQ(axons.get_minimum_calcium().size(), 0);
}

TEST_F(MultiPositionAxonsTest, testSetPositions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto axons = MultiPositionAxons{};

    const auto number_neurons = RelearnTypes::number_neurons_type{ 5 };
    axons.init(number_neurons);

    auto positions = std::vector<RelearnTypes::position_type>{};
    positions.emplace_back(0.0, 3.0, 1.0);
    positions.emplace_back(1.0, 3.0, 0.0);
    positions.emplace_back(2.0, 3.0, 1.0);
    positions.emplace_back(3.0, 3.0, 0.0);
    positions.emplace_back(4.0, 3.0, 1.0);
    positions.emplace_back(5.0, 3.0, 0.0);
    positions.emplace_back(6.0, 3.0, 1.0);
    positions.emplace_back(7.0, 3.0, 0.0);
    positions.emplace_back(8.0, 3.0, 1.0);
    positions.emplace_back(9.0, 3.0, 0.0);

    auto bouton_positions = std::vector<std::vector<RelearnTypes::position_type>>{};
    bouton_positions.resize(number_neurons, std::vector<RelearnTypes::position_type>{});

    bouton_positions[0].emplace_back(positions[0]);
    bouton_positions[0].emplace_back(positions[1]);

    bouton_positions[1].emplace_back(positions[1]);
    bouton_positions[1].emplace_back(positions[5]);
    bouton_positions[1].emplace_back(positions[8]);

    bouton_positions[2].emplace_back(positions[2]);
    bouton_positions[2].emplace_back(positions[1]);
    bouton_positions[2].emplace_back(positions[6]);
    bouton_positions[2].emplace_back(positions[8]);
    bouton_positions[2].emplace_back(positions[9]);

    bouton_positions[3].emplace_back(positions[3]);

    bouton_positions[4].emplace_back(positions[4]);
    bouton_positions[4].emplace_back(positions[7]);

    axons.set_bouton_positions(bouton_positions);

    ASSERT_EQ(axons.get_number_boutons(0), 2);
    ASSERT_EQ(axons.get_number_boutons(1), 3);
    ASSERT_EQ(axons.get_number_boutons(2), 5);
    ASSERT_EQ(axons.get_number_boutons(3), 1);
    ASSERT_EQ(axons.get_number_boutons(4), 2);

    ASSERT_EQ(axons.get_bouton_position(0, 0), positions[0]);
    ASSERT_EQ(axons.get_bouton_position(0, 1), positions[1]);

    ASSERT_EQ(axons.get_bouton_position(1, 0), positions[1]);
    ASSERT_EQ(axons.get_bouton_position(1, 1), positions[5]);
    ASSERT_EQ(axons.get_bouton_position(1, 2), positions[8]);

    ASSERT_EQ(axons.get_bouton_position(2, 0), positions[2]);
    ASSERT_EQ(axons.get_bouton_position(2, 1), positions[1]);
    ASSERT_EQ(axons.get_bouton_position(2, 2), positions[6]);
    ASSERT_EQ(axons.get_bouton_position(2, 3), positions[8]);
    ASSERT_EQ(axons.get_bouton_position(2, 4), positions[9]);

    ASSERT_EQ(axons.get_bouton_position(3, 0), positions[3]);

    ASSERT_EQ(axons.get_bouton_position(4, 0), positions[4]);
    ASSERT_EQ(axons.get_bouton_position(4, 1), positions[7]);

    for (auto bouton_id = std::size_t{ 2 }; bouton_id < std::size_t{ 65 }; bouton_id++) {
        ASSERT_THROW_NO_PRINT(std::ignore = axons.get_bouton_position(0, bouton_id);, RelearnException);
    }

    for (auto bouton_id = std::size_t{ 3 }; bouton_id < std::size_t{ 65 }; bouton_id++) {
        ASSERT_THROW_NO_PRINT(std::ignore = axons.get_bouton_position(1, bouton_id);, RelearnException);
    }

    for (auto bouton_id = std::size_t{ 5 }; bouton_id < std::size_t{ 65 }; bouton_id++) {
        ASSERT_THROW_NO_PRINT(std::ignore = axons.get_bouton_position(2, bouton_id);, RelearnException);
    }

    for (auto bouton_id = std::size_t{ 1 }; bouton_id < std::size_t{ 65 }; bouton_id++) {
        ASSERT_THROW_NO_PRINT(std::ignore = axons.get_bouton_position(3, bouton_id);, RelearnException);
    }

    for (auto bouton_id = std::size_t{ 2 }; bouton_id < std::size_t{ 65 }; bouton_id++) {
        ASSERT_THROW_NO_PRINT(std::ignore = axons.get_bouton_position(4, bouton_id);, RelearnException);
    }
}

TEST_F(MultiPositionAxonsTest, testGetRandomPositions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto axons = MultiPositionAxons{};

    const auto number_neurons = RelearnTypes::number_neurons_type{ 5 };
    axons.init(number_neurons);

    auto positions = std::vector<RelearnTypes::position_type>{};
    positions.emplace_back(0.0, 3.0, 1.0);
    positions.emplace_back(1.0, 3.0, 0.0);
    positions.emplace_back(2.0, 3.0, 1.0);
    positions.emplace_back(3.0, 3.0, 0.0);
    positions.emplace_back(4.0, 3.0, 1.0);
    positions.emplace_back(5.0, 3.0, 0.0);
    positions.emplace_back(6.0, 3.0, 1.0);
    positions.emplace_back(7.0, 3.0, 0.0);
    positions.emplace_back(8.0, 3.0, 1.0);
    positions.emplace_back(9.0, 3.0, 0.0);

    auto bouton_positions = std::vector<std::vector<RelearnTypes::position_type>>{};
    bouton_positions.resize(number_neurons, std::vector<RelearnTypes::position_type>{});

    bouton_positions[0].emplace_back(positions[0]);
    bouton_positions[0].emplace_back(positions[1]);

    bouton_positions[1].emplace_back(positions[1]);
    bouton_positions[1].emplace_back(positions[5]);
    bouton_positions[1].emplace_back(positions[8]);

    bouton_positions[2].emplace_back(positions[2]);
    bouton_positions[2].emplace_back(positions[1]);
    bouton_positions[2].emplace_back(positions[6]);
    bouton_positions[2].emplace_back(positions[8]);
    bouton_positions[2].emplace_back(positions[9]);

    bouton_positions[3].emplace_back(positions[3]);

    bouton_positions[4].emplace_back(positions[4]);
    bouton_positions[4].emplace_back(positions[7]);

    axons.set_bouton_positions(bouton_positions);

    const auto check_first_positions = [&axons, &positions]() {
        auto working_positions = std::set<RelearnTypes::position_type>{};

        for (auto i = 0; i < 10000; i++) {
            const auto position = axons.get_bouton_position(0);
            working_positions.emplace(position);
        }

        ASSERT_EQ(working_positions.size(), 2);

        auto working_contains = working_positions.contains(positions[0]);
        if (!working_contains) {
            std::cerr << "Did not retrieve position " << positions[0] << " for the first neuron. Maybe restart the test?" << '\n';
            ASSERT_TRUE(false);
        }

        working_contains = working_positions.contains(positions[1]);
        if (!working_contains) {
            std::cerr << "Did not retrieve position " << positions[1] << " for the first neuron. Maybe restart the test?" << '\n';
            ASSERT_TRUE(false);
        }
    };

    const auto check_second_positions = [&axons, &positions]() {
        auto working_positions = std::set<RelearnTypes::position_type>{};

        for (auto i = 0; i < 10000; i++) {
            const auto position = axons.get_bouton_position(1);
            working_positions.emplace(position);
        }

        ASSERT_EQ(working_positions.size(), 3);

        auto working_contains = working_positions.contains(positions[1]);
        if (!working_contains) {
            std::cerr << "Did not retrieve position " << positions[1] << " for the second neuron. Maybe restart the test?" << '\n';
            ASSERT_TRUE(false);
        }

        working_contains = working_positions.contains(positions[5]);
        if (!working_contains) {
            std::cerr << "Did not retrieve position " << positions[5] << " for the second neuron. Maybe restart the test?" << '\n';
            ASSERT_TRUE(false);
        }

        working_contains = working_positions.contains(positions[8]);
        if (!working_contains) {
            std::cerr << "Did not retrieve position " << positions[8] << " for the second neuron. Maybe restart the test?" << '\n';
            ASSERT_TRUE(false);
        }
    };

    const auto check_third_positions = [&axons, &positions]() {
        auto working_positions = std::set<RelearnTypes::position_type>{};

        for (auto i = 0; i < 10000; i++) {
            const auto position = axons.get_bouton_position(2);
            working_positions.emplace(position);
        }

        ASSERT_EQ(working_positions.size(), 5);

        auto working_contains = working_positions.contains(positions[2]);
        if (!working_contains) {
            std::cerr << "Did not retrieve position " << positions[2] << " for the third neuron. Maybe restart the test?" << '\n';
            ASSERT_TRUE(false);
        }

        working_contains = working_positions.contains(positions[1]);
        if (!working_contains) {
            std::cerr << "Did not retrieve position " << positions[1] << " for the third neuron. Maybe restart the test?" << '\n';
            ASSERT_TRUE(false);
        }

        working_contains = working_positions.contains(positions[6]);
        if (!working_contains) {
            std::cerr << "Did not retrieve position " << positions[6] << " for the third neuron. Maybe restart the test?" << '\n';
            ASSERT_TRUE(false);
        }

        working_contains = working_positions.contains(positions[8]);
        if (!working_contains) {
            std::cerr << "Did not retrieve position " << positions[8] << " for the third neuron. Maybe restart the test?" << '\n';
            ASSERT_TRUE(false);
        }

        working_contains = working_positions.contains(positions[9]);
        if (!working_contains) {
            std::cerr << "Did not retrieve position " << positions[9] << " for the third neuron. Maybe restart the test?" << '\n';
            ASSERT_TRUE(false);
        }
    };

    const auto check_forth_position = [&axons, &positions]() {
        auto working_positions = std::set<RelearnTypes::position_type>{};

        for (auto i = 0; i < 10000; i++) {
            const auto position = axons.get_bouton_position(3);
            working_positions.emplace(position);
        }

        ASSERT_EQ(working_positions.size(), 1);

        auto working_contains = working_positions.contains(positions[3]);
        if (!working_contains) {
            std::cerr << "Did not retrieve position " << positions[3] << " for the forth neuron. Maybe restart the test?" << '\n';
            ASSERT_TRUE(false);
        }
    };

    const auto check_fifth_position = [&axons, &positions]() {
        auto working_positions = std::set<RelearnTypes::position_type>{};

        for (auto i = 0; i < 10000; i++) {
            const auto position = axons.get_bouton_position(4);
            working_positions.emplace(position);
        }

        ASSERT_EQ(working_positions.size(), 2);

        auto working_contains = working_positions.contains(positions[4]);
        if (!working_contains) {
            std::cerr << "Did not retrieve position " << positions[4] << " for the fifth neuron. Maybe restart the test?" << '\n';
            ASSERT_TRUE(false);
        }

        working_contains = working_positions.contains(positions[7]);
        if (!working_contains) {
            std::cerr << "Did not retrieve position " << positions[7] << " for the fifth neuron. Maybe restart the test?" << '\n';
            ASSERT_TRUE(false);
        }
    };

    check_first_positions();
    check_second_positions();
    check_third_positions();
    check_forth_position();
    check_fifth_position();
}
