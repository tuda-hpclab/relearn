/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuron_extra_info.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/UpdateStatus.h"
#include "util/NeuronID.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <gtest/gtest.h>

#include <fmt/core.h>
#include <range/v3/view/indices.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>

void NeuronsExtraInfoTest::assert_empty(const NeuronsExtraInfo& extra_info, size_t number_neurons) {
    const auto& positions = extra_info.get_positions();

    const auto& positions_size = positions.size();

    ASSERT_EQ(0, positions_size) << positions_size;

    for ([[maybe_unused]] const auto i : NeuronID::range_id(number_neurons_out_of_scope)) {
        const auto neuron_id = NeuronIdFactory::get_random_neuron_id(number_neurons, 1, mt);

        ASSERT_THROW_NO_PRINT_MSG(std::ignore = extra_info.get_position(neuron_id), RelearnException, fmt::format("assert empty position {}", neuron_id));
    }
}

void NeuronsExtraInfoTest::assert_contains(const NeuronsExtraInfo& extra_info, size_t number_neurons, size_t num_neurons_check,
                                           const std::vector<Vec3d>& expected_positions) {

    const auto& expected_positions_size = expected_positions.size();

    ASSERT_EQ(num_neurons_check, expected_positions_size) << num_neurons_check << ' ' << expected_positions_size;

    const auto& actual_positions = extra_info.get_positions();

    const auto& positions_size = actual_positions.size();

    ASSERT_EQ(positions_size, number_neurons) << positions_size << ' ' << number_neurons;

    for (const auto neuron_id : NeuronID::range(num_neurons_check)) {
        ASSERT_EQ(expected_positions[neuron_id.get_neuron_id()], actual_positions[neuron_id.get_neuron_id()]) << neuron_id;
        ASSERT_EQ(expected_positions[neuron_id.get_neuron_id()], extra_info.get_position(neuron_id)) << neuron_id;
    }

    for ([[maybe_unused]] const auto i : ranges::views::indices(number_neurons_out_of_scope)) {
        const auto neuron_id = NeuronIdFactory::get_random_neuron_id(
            number_neurons, number_neurons, mt);

        ASSERT_THROW_NO_PRINT_MSG(std::ignore = extra_info.get_position(neuron_id), RelearnException, neuron_id);
    }
}

TEST_F(NeuronsExtraInfoTest, testConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto extra_info = NeuronsExtraInfo{};

    assert_empty(extra_info, NeuronIdFactory::upper_bound_num_neurons);

    auto empty_positions = std::vector<NeuronsExtraInfo::position_type>{};
    ASSERT_THROW_NO_PRINT(extra_info.set_positions(empty_positions), RelearnException);

    assert_empty(extra_info, NeuronIdFactory::upper_bound_num_neurons);

    const auto new_size = NeuronIdFactory::get_random_number_neurons(mt);

    auto full_positions = std::vector<NeuronsExtraInfo::position_type>{ new_size };
    ASSERT_THROW_NO_PRINT(extra_info.set_positions(full_positions), RelearnException);

    assert_empty(extra_info, NeuronIdFactory::upper_bound_num_neurons);

    ASSERT_EQ(extra_info.get_size(), 0);
}

TEST_F(NeuronsExtraInfoTest, testInit) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto extra_info = NeuronsExtraInfo{};

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    extra_info.init(number_neurons);
    assert_empty(extra_info, number_neurons);

    ASSERT_EQ(extra_info.get_size(), number_neurons);

    auto num_neurons_wrong = NeuronIdFactory::get_random_number_neurons(mt);
    if (num_neurons_wrong == number_neurons) {
        num_neurons_wrong++;
    }

    const auto positions_wrong = std::vector<Vec3d>(num_neurons_wrong);

    ASSERT_THROW_NO_PRINT(extra_info.set_positions(positions_wrong), RelearnException);

    assert_empty(extra_info, number_neurons);

    auto positions_right = SimulationFactory::get_random_positions<std::allocator<Vec3d>>(mt, number_neurons);
    extra_info.set_positions(positions_right);
    assert_contains(extra_info, number_neurons, number_neurons, positions_right);

    auto positions_right_2 = SimulationFactory::get_random_positions<std::allocator<Vec3d>>(mt, number_neurons);
    extra_info.set_positions(positions_right_2);
    assert_contains(extra_info, number_neurons, number_neurons, positions_right_2);

    ASSERT_THROW_NO_PRINT(extra_info.init(number_neurons), RelearnException);

    ASSERT_EQ(extra_info.get_size(), number_neurons);

    extra_info.set_positions(positions_right_2);

    assert_contains(extra_info, number_neurons, number_neurons, positions_right_2);
}

TEST_F(NeuronsExtraInfoTest, testCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto extra_info = NeuronsExtraInfo{};

    const auto num_neurons_init = NeuronIdFactory::get_random_number_neurons(mt);
    const auto num_neurons_create_1 = NeuronIdFactory::get_random_number_neurons(mt);
    const auto num_neurons_create_2 = NeuronIdFactory::get_random_number_neurons(mt);

    const auto num_neurons_total_1 = num_neurons_init + num_neurons_create_1;
    const auto num_neurons_total_2 = num_neurons_total_1 + num_neurons_create_2;

    extra_info.init(num_neurons_init);

    ASSERT_THROW_NO_PRINT(extra_info.create_neurons(num_neurons_create_1), RelearnException);

    assert_empty(extra_info, num_neurons_init);

    auto positions_right = SimulationFactory::get_random_positions<std::allocator<Vec3d>>(mt, num_neurons_init);

    extra_info.set_positions(positions_right);

    extra_info.create_neurons(num_neurons_create_1);

    ASSERT_EQ(extra_info.get_size(), num_neurons_init + num_neurons_create_1);

    assert_contains(extra_info, num_neurons_total_1, num_neurons_init, positions_right);

    auto positions_right_2 = SimulationFactory::get_random_positions<std::allocator<Vec3d>>(mt, num_neurons_total_1);

    extra_info.set_positions(positions_right_2);

    assert_contains(extra_info, num_neurons_total_1, num_neurons_total_1, positions_right_2);

    extra_info.create_neurons(num_neurons_create_2);

    ASSERT_EQ(extra_info.get_size(), num_neurons_init + num_neurons_create_1 + num_neurons_create_2);

    assert_contains(extra_info, num_neurons_total_2, num_neurons_total_1, positions_right_2);

    auto positions_right_3 = SimulationFactory::get_random_positions<std::allocator<Vec3d>>(mt, num_neurons_total_2);

    extra_info.set_positions(positions_right_3);

    assert_contains(extra_info, num_neurons_total_2, num_neurons_total_2, positions_right_3);

    ASSERT_THROW_NO_PRINT(extra_info.create_neurons(0), RelearnException);

    assert_contains(extra_info, num_neurons_total_2, num_neurons_total_2, positions_right_3);
}

TEST_F(NeuronsExtraInfoTest, testSetStatus) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto extra_info = NeuronsExtraInfo{};
    extra_info.init(number_neurons);

    auto enabled_neurons = std::vector<NeuronID>{};
    auto disabled_neurons = std::vector<NeuronID>{};
    auto static_neurons = std::vector<NeuronID>{};

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto random_number = RandomFactory::get_random_integer(0, 5, mt);
        if (random_number == 0) {
            static_neurons.emplace_back(neuron_id);
        } else if (random_number == 1) {
            disabled_neurons.emplace_back(neuron_id);
        } else {
            enabled_neurons.emplace_back(neuron_id);
        }
    }

    if (!enabled_neurons.empty()) {
        ASSERT_THROW_NO_PRINT(extra_info.set_enabled_neurons(enabled_neurons), RelearnException);
    }
    ASSERT_NO_THROW(extra_info.set_disabled_neurons(disabled_neurons));
    ASSERT_NO_THROW(extra_info.set_static_neurons(static_neurons));

    const auto status_flags = extra_info.get_disable_flags();
    ASSERT_EQ(status_flags.size(), number_neurons);

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto index = neuron_id.get_neuron_id();

        if (std::ranges::binary_search(enabled_neurons, neuron_id)) {
            ASSERT_EQ(status_flags[index], UpdateStatus::Enabled);
            ASSERT_TRUE(extra_info.does_update_electrical_actvity(neuron_id));
            ASSERT_TRUE(extra_info.does_update_plasticity(neuron_id));
        } else if (std::ranges::binary_search(disabled_neurons, neuron_id)) {
            ASSERT_EQ(status_flags[index], UpdateStatus::Disabled);
            ASSERT_FALSE(extra_info.does_update_electrical_actvity(neuron_id));
            ASSERT_FALSE(extra_info.does_update_plasticity(neuron_id));
        } else {
            ASSERT_EQ(status_flags[index], UpdateStatus::Static);
            ASSERT_TRUE(extra_info.does_update_electrical_actvity(neuron_id));
            ASSERT_FALSE(extra_info.does_update_plasticity(neuron_id));
        }
    }
}

TEST_F(NeuronsExtraInfoTest, testSetStatusShuffle) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto extra_info = NeuronsExtraInfo{};
    extra_info.init(number_neurons);

    auto enabled_neurons = std::vector<NeuronID>{};
    auto disabled_neurons = std::vector<NeuronID>{};
    auto static_neurons = std::vector<NeuronID>{};

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto random_number = RandomFactory::get_random_integer(0, 5, mt);
        if (random_number == 0) {
            static_neurons.emplace_back(neuron_id);
        } else if (random_number == 1) {
            disabled_neurons.emplace_back(neuron_id);
        } else {
            enabled_neurons.emplace_back(neuron_id);
        }
    }

    RandomFactory::shuffle(enabled_neurons, mt);
    RandomFactory::shuffle(disabled_neurons, mt);
    RandomFactory::shuffle(static_neurons, mt);

    if (!enabled_neurons.empty()) {
        ASSERT_THROW_NO_PRINT(extra_info.set_enabled_neurons(enabled_neurons), RelearnException);
    }
    ASSERT_NO_THROW(extra_info.set_disabled_neurons(disabled_neurons));
    ASSERT_NO_THROW(extra_info.set_static_neurons(static_neurons));

    std::ranges::sort(enabled_neurons);
    std::ranges::sort(disabled_neurons);
    std::ranges::sort(static_neurons);

    const auto status_flags = extra_info.get_disable_flags();
    ASSERT_EQ(status_flags.size(), number_neurons);

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto index = neuron_id.get_neuron_id();

        if (std::ranges::binary_search(enabled_neurons, neuron_id)) {
            ASSERT_EQ(status_flags[index], UpdateStatus::Enabled);
            ASSERT_TRUE(extra_info.does_update_electrical_actvity(neuron_id));
            ASSERT_TRUE(extra_info.does_update_plasticity(neuron_id));
        } else if (std::ranges::binary_search(disabled_neurons, neuron_id)) {
            ASSERT_EQ(status_flags[index], UpdateStatus::Disabled);
            ASSERT_FALSE(extra_info.does_update_electrical_actvity(neuron_id));
            ASSERT_FALSE(extra_info.does_update_plasticity(neuron_id));
        } else {
            ASSERT_EQ(status_flags[index], UpdateStatus::Static);
            ASSERT_TRUE(extra_info.does_update_electrical_actvity(neuron_id));
            ASSERT_FALSE(extra_info.does_update_plasticity(neuron_id));
        }
    }
}

TEST_F(NeuronsExtraInfoTest, testSetStatusOutOfBounds) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto extra_info = NeuronsExtraInfo{};
    extra_info.init(number_neurons);

    auto enabled_neurons = std::vector<NeuronID>{};
    auto disabled_neurons = std::vector<NeuronID>{};
    auto static_neurons = std::vector<NeuronID>{};

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto random_number = RandomFactory::get_random_integer(0, 5, mt);
        if (random_number == 0) {
            static_neurons.emplace_back(neuron_id);
        } else if (random_number == 1) {
            disabled_neurons.emplace_back(neuron_id);
        } else {
            enabled_neurons.emplace_back(neuron_id);
        }
    }

    static_neurons.emplace_back(number_neurons);
    enabled_neurons.emplace_back(number_neurons + 1);
    disabled_neurons.emplace_back(number_neurons + 2);

    ASSERT_THROW_NO_PRINT(extra_info.set_enabled_neurons(enabled_neurons), RelearnException);
    ASSERT_THROW_NO_PRINT(extra_info.set_disabled_neurons(disabled_neurons), RelearnException);
    ASSERT_THROW_NO_PRINT(extra_info.set_static_neurons(static_neurons), RelearnException);
}

TEST_F(NeuronsExtraInfoTest, testSetStatusRepeated) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = 5;

    auto extra_info = NeuronsExtraInfo{};
    extra_info.init(number_neurons);

    const auto status_flags = extra_info.get_disable_flags();

    extra_info.set_disabled_neurons(std::vector{ NeuronID(2) });
    extra_info.set_enabled_neurons(std::vector{ NeuronID(2) });

    for (auto i = 0U; i < number_neurons; i++) {
        ASSERT_EQ(status_flags[i], UpdateStatus::Enabled);
    }

    extra_info.set_static_neurons(std::vector{ NeuronID(2), NeuronID(3) });

    ASSERT_THROW_NO_PRINT(extra_info.set_enabled_neurons(std::vector{ NeuronID(3) }), RelearnException);
    ASSERT_THROW_NO_PRINT(extra_info.set_disabled_neurons(std::vector{ NeuronID(2) }), RelearnException);

    ASSERT_EQ(status_flags[0], UpdateStatus::Enabled);
    ASSERT_EQ(status_flags[1], UpdateStatus::Enabled);
    ASSERT_EQ(status_flags[2], UpdateStatus::Static);
    ASSERT_EQ(status_flags[3], UpdateStatus::Static);
    ASSERT_EQ(status_flags[4], UpdateStatus::Enabled);
}

TEST_F(NeuronsExtraInfoTest, testGetPositionsFor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto extra_info = NeuronsExtraInfo{};
    extra_info.init(number_neurons);

    auto positions = SimulationFactory::get_random_positions<std::allocator<Vec3d>>(mt, number_neurons);

    extra_info.set_positions(positions);

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt);

    auto cm = RelearnTypes::comm_map_position<NeuronID>(number_ranks, NeuronIdFactory::upper_bound_num_neurons);

    for (const auto rank : mpiPP::MPIRank::range(number_ranks)) {
        const auto number_neurons_for_rank = NeuronIdFactory::get_random_number_neurons(mt);
        for (auto it = 0U; it < number_neurons_for_rank; it++) {
            cm.emplace_back(rank, NeuronIdFactory::get_random_neuron_id(number_neurons, mt));
        }
    }

    auto results = extra_info.get_positions_for(cm);

    ASSERT_EQ(cm.size(), results.size());

    for (auto outer_it = 0; static_cast<std::size_t>(outer_it) < cm.size(); outer_it++) {
        const auto rank = mpiPP::MPIRank(outer_it);
        const auto& requests = cm.get_requests(rank);
        const auto& responses = results.get_requests(rank);

        ASSERT_EQ(requests.size(), responses.size());

        for (auto inner_it = 0U; inner_it < requests.size(); inner_it++) {
            const auto& expected_position = extra_info.get_position(requests[inner_it]);
            ASSERT_EQ(expected_position, responses[inner_it]);
        }
    }
}

TEST_F(NeuronsExtraInfoTest, testGetPositionsForException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto extra_info = NeuronsExtraInfo{};
    extra_info.init(number_neurons);

    auto positions = SimulationFactory::get_random_positions<std::allocator<Vec3d>>(mt, number_neurons);

    extra_info.set_positions(positions);

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt);

    auto cm = RelearnTypes::comm_map_position<NeuronID>(number_ranks, NeuronIdFactory::upper_bound_num_neurons);

    for (const auto rank : mpiPP::MPIRank::range(number_ranks)) {
        const auto number_neurons_for_rank = NeuronIdFactory::get_random_number_neurons(mt);
        for (auto it = 0U; it < number_neurons_for_rank; it++) {
            cm.emplace_back(rank, NeuronIdFactory::get_random_neuron_id(number_neurons, mt));
        }
    }

    const auto faulty_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mt);
    cm.emplace_back(faulty_rank, number_neurons);

    ASSERT_THROW_NO_PRINT(std::ignore = extra_info.get_positions_for(cm), RelearnException);
}
