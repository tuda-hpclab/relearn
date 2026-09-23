/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"
#include "test_barnes_hut.h"

#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/Cells.h"
#include "algorithm/Internal/octree/Cell.h"
#include "algorithm/Internal/octree/NodeCache.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "algorithm/Internal/octree/OctreeNodeHelper.h"
#include "algorithm/Kernel/Gaussian.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/DistantNeuronRequests.h"
#include "types/BasicTypes.h"
#include "util/MemoryHolder.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include "adapter/node_cache/NodeCacheAdapter.h"
#include "adapter/octree/OctreeAdapter.h"

#include "factory/memory_holder/memory_holder_factory.h"
#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neuron_types/neuron_types_factory.h"
#include "factory/octree/octree_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <cpp-utility/Cast.hpp>

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/core/MPIRankRange.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stack>
#include <tuple>
#include <vector>

TEST_F(BarnesHutBaseTest, testACException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto minimum = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto maximum = RelearnTypes::position_type{ 10.0, 10.0, 10.0 };

    const auto rank = MPIRankFactory::get_random_mpi_rank(1024, mt);
    const auto level = static_cast<RelearnTypes::level_type>(RandomFactory::get_random_integer<std::uint16_t>(0, 24, mt));
    const auto& neuron_id = NeuronIdFactory::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);

    auto node = OctreeNode<additional_cell_attributes>{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);

    node.set_cell_size(minimum, maximum);
    node.set_cell_neuron_position(node_position);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto source_position = RelearnTypes::position_type{ 15.0, 15.0, 15.0 };

    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                                                                                                             nullptr, ElementType::Dendrite, searched_signal_type, Constants::bh_default_theta),
                          RelearnException);

    const auto too_small_acceptance_criterion = RandomFactory::get_random_double(RelearnTypes::acceptance_criterion_type{ -1000 }, RelearnTypes::acceptance_criterion_type{ 0 }, mt);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                                                                                                             &node, ElementType::Dendrite, searched_signal_type, 0.0),
                          RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                                                                                                             &node, ElementType::Dendrite, searched_signal_type, too_small_acceptance_criterion),
                          RelearnException);

    const auto too_large_acceptance_criterion = RandomFactory::get_random_double(Constants::bh_max_theta + utility::cast<RelearnTypes::acceptance_criterion_type>(eps), RelearnTypes::acceptance_criterion_type{ 1000 }, mt);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                                                                                                             &node, ElementType::Dendrite, searched_signal_type, too_large_acceptance_criterion),
                          RelearnException);
}

TEST_F(BarnesHutBaseTest, testACLeafDendrites) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto minimum = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto maximum = RelearnTypes::position_type{ 10.0, 10.0, 10.0 };

    const auto rank = MPIRankFactory::get_random_mpi_rank(1024, mt);
    const auto level = static_cast<RelearnTypes::level_type>(RandomFactory::get_random_integer<std::uint16_t>(0, 24, mt));
    const auto& neuron_id = NeuronIdFactory::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);

    auto node = OctreeNode<additional_cell_attributes>{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);

    node.set_cell_size(minimum, maximum);
    node.set_cell_neuron_position(node_position);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);

    for (auto it = 0; it < 1000; it++) {
        const auto& position = SimulationFactory::get_random_position(mt);
        const auto number_free_elements = RandomFactory::get_random_integer<RelearnTypes::counter_type>(1, 1000, mt);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_dendrites(number_free_elements, 0);
        } else {
            node.set_cell_number_dendrites(0, number_free_elements);
        }

        const auto accept = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(accept, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_dendrites(0, number_free_elements);
        } else {
            node.set_cell_number_dendrites(number_free_elements, 0);
        }

        const auto discard = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(discard, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Discard);
    }
}

TEST_F(BarnesHutBaseTest, testACLeafSamePositionDendrites) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto minimum = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto maximum = RelearnTypes::position_type{ 10.0, 10.0, 10.0 };

    const auto rank = MPIRankFactory::get_random_mpi_rank(1024, mt);
    const auto level = static_cast<RelearnTypes::level_type>(RandomFactory::get_random_integer<std::uint16_t>(0, 24, mt));
    const auto& neuron_id = NeuronIdFactory::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);

    auto node = OctreeNode<additional_cell_attributes>{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);

    node.set_cell_size(minimum, maximum);
    node.set_cell_neuron_position(node_position);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);

    for (auto it = 0; it < 1000; it++) {
        const auto number_free_elements_excitatory = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 1000, mt);
        const auto number_free_elements_inhibitory = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 1000, mt);

        node.set_cell_number_dendrites(number_free_elements_excitatory, number_free_elements_inhibitory);

        const auto discard = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(node_position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(discard, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Discard);
    }
}

TEST_F(BarnesHutBaseTest, testACParentDendrite) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto& scaled_minimum = minimum / 10.0;
    const auto& scaled_maximum = maximum / 10.0;

    const auto rank = MPIRankFactory::get_random_mpi_rank(1024, mt);
    const auto level = static_cast<RelearnTypes::level_type>(RandomFactory::get_random_integer<std::uint16_t>(0, 24, mt));
    const auto& neuron_id = NeuronIdFactory::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationFactory::get_random_position_in_box(scaled_minimum, scaled_maximum, mt);

    auto node = OctreeNode<additional_cell_attributes>{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);
    node.set_parent();

    node.set_cell_size(scaled_minimum, scaled_maximum);
    node.set_cell_neuron_position(node_position);

    const auto& cell_dimensions = scaled_maximum - scaled_minimum;
    const auto& maximum_cell_dimension = cell_dimensions.get_maximum();

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto& position = SimulationFactory::get_random_position(mt);

        const auto distance = (node_position - position).calculate_2_norm<RelearnTypes::space_type>();
        const auto quotient = maximum_cell_dimension / distance;

        const auto number_free_elements = RandomFactory::get_random_integer<RelearnTypes::counter_type>(1, 1000, mt);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_dendrites(number_free_elements, 0);
        } else {
            node.set_cell_number_dendrites(0, number_free_elements);
        }

        const auto status = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

        if (acceptance_criterion > quotient) {
            ASSERT_EQ(status, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept);
        } else {
            ASSERT_EQ(status, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand);
        }

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_dendrites(0, number_free_elements);
        } else {
            node.set_cell_number_dendrites(number_free_elements, 0);
        }

        const auto discard = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(discard, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Discard);
    }
}

TEST_F(BarnesHutBaseTest, testNodesToConsiderException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    const auto position = RelearnTypes::position_type{ 0.0 };
    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);

    const auto too_small_acceptance_criterion = RandomFactory::get_random_double(RelearnTypes::acceptance_criterion_type{ -1000 }, RelearnTypes::acceptance_criterion_type{ 0 }, mt);
    const auto too_large_acceptance_criterion = RandomFactory::get_random_double(Constants::bh_max_theta + utility::cast<RelearnTypes::acceptance_criterion_type>(eps), RelearnTypes::acceptance_criterion_type{ 10000 }, mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, 0.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, Constants::bh_max_theta + utility::cast<RelearnTypes::acceptance_criterion_type>(eps)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, too_small_acceptance_criterion), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, too_large_acceptance_criterion), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, 0.0, true), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, Constants::bh_max_theta + utility::cast<RelearnTypes::acceptance_criterion_type>(eps), true), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, too_small_acceptance_criterion, true), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, too_large_acceptance_criterion, true), RelearnException);

    const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, nullptr, ElementType::Dendrite, searched_signal_type, Constants::bh_default_theta), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, nullptr, ElementType::Dendrite, searched_signal_type, acceptance_criterion), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, nullptr, ElementType::Dendrite, searched_signal_type, Constants::bh_default_theta, true), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, nullptr, ElementType::Dendrite, searched_signal_type, acceptance_criterion, true), RelearnException);
}

TEST_F(BarnesHutBaseTest, testNodesToConsiderLeaf) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto minimum = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto maximum = RelearnTypes::position_type{ 10.0, 10.0, 10.0 };

    const auto rank = MPIRankFactory::get_random_mpi_rank(1024, mt);
    const auto level = static_cast<RelearnTypes::level_type>(RandomFactory::get_random_integer<std::uint16_t>(0, 24, mt));
    const auto& neuron_id = NeuronIdFactory::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);

    auto node = OctreeNode<additional_cell_attributes>{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);

    node.set_cell_size(minimum, maximum);
    node.set_cell_neuron_position(node_position);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    for (auto it = 0; it < 1000; it++) {
        const auto& position = SimulationFactory::get_random_position(mt);
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto number_free_elements = RandomFactory::get_random_integer<RelearnTypes::counter_type>(1, 1000, mt);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_dendrites(number_free_elements, 0);
        } else {
            node.set_cell_number_dendrites(0, number_free_elements);
        }

        const auto accept_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(1, accept_nodes.size());
        ASSERT_EQ(&node, accept_nodes[0]);

        const auto accept_nodes_early = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion, true);
        ASSERT_EQ(1, accept_nodes_early.size());
        ASSERT_EQ(&node, accept_nodes_early[0]);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_dendrites(0, number_free_elements);
        } else {
            node.set_cell_number_dendrites(number_free_elements, 0);
        }

        const auto discard_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(discard_nodes.empty());

        const auto discard_nodes_early = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(discard_nodes_early.empty());
    }

    node.set_cell_number_dendrites(0, 0);

    for (auto it = 0; it < 1000; it++) {
        const auto& position = SimulationFactory::get_random_position(mt);
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);

        const auto accept_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(accept_nodes.empty());

        const auto discard_nodes_early = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(discard_nodes_early.empty());
    }
}

TEST_F(BarnesHutBaseTest, testNodesToConsiderNoDendrites) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_tree_no_dendrites<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto& position = SimulationFactory::get_random_position(mt);

        auto found_nodes_dendrite = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, false);
        ASSERT_TRUE(found_nodes_dendrite.empty());
    }
}

TEST_F(BarnesHutBaseTest, testNodesToConsiderNoElements) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_tree_no_synaptic_elements<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto& position = SimulationFactory::get_random_position(mt);

        auto found_nodes_dendrite = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, false);
        ASSERT_TRUE(found_nodes_dendrite.empty());
    }
}

TEST_F(BarnesHutBaseTest, testNodesToConsider) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto& position = SimulationFactory::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, false);
        std::ranges::sort(found_nodes);

        auto golden_nodes = std::vector<OctreeNode<additional_cell_attributes>*>{};
        golden_nodes.reserve(number_neurons);

        auto stack = std::stack<OctreeNode<additional_cell_attributes>*>{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }
}

TEST_F(BarnesHutBaseTest, testNodesToConsiderNoAxons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_tree_no_axons<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto& position = SimulationFactory::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, false);
        std::ranges::sort(found_nodes);

        auto golden_nodes = std::vector<OctreeNode<additional_cell_attributes>*>{};
        golden_nodes.reserve(number_neurons);

        auto stack = std::stack<OctreeNode<additional_cell_attributes>*>{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }
}

TEST_F(BarnesHutBaseTest, testNodesToConsiderDistributedTree) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto branching_level = SimulationFactory::get_small_positive_refinement_level(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);
    OctreeAdapter::mark_node_as_distributed(&root, branching_level);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto& position = SimulationFactory::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, false);
        std::ranges::sort(found_nodes);

        auto golden_nodes = std::vector<OctreeNode<additional_cell_attributes>*>{};
        golden_nodes.reserve(number_neurons);

        auto stack = std::stack<OctreeNode<additional_cell_attributes>*>{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }
}

TEST_F(BarnesHutBaseTest, testNodesToConsiderEarlyReturn) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto& position = SimulationFactory::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, true);
        std::ranges::sort(found_nodes);

        auto golden_nodes = std::vector<OctreeNode<additional_cell_attributes>*>{};
        golden_nodes.reserve(number_neurons);

        auto stack = std::stack<OctreeNode<additional_cell_attributes>*>{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }
}

TEST_F(BarnesHutBaseTest, testNodesToConsiderEarlyReturnDistributedTree) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto branching_level = SimulationFactory::get_small_positive_refinement_level(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);
    OctreeAdapter::mark_node_as_distributed(&root, branching_level);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto& position = SimulationFactory::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(node_cache, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, true);
        std::ranges::sort(found_nodes);

        auto golden_nodes = std::vector<OctreeNode<additional_cell_attributes>*>{};
        golden_nodes.reserve(number_neurons);

        auto stack = std::stack<OctreeNode<additional_cell_attributes>*>{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand && current->get_mpi_rank() != mpiPP::MPIRank(0)) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto neuron_id = NeuronID(1000000);
    const auto position = RelearnTypes::position_type{ 0.0 };

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto too_small_acceptance_criterion = RandomFactory::get_random_double(RelearnTypes::acceptance_criterion_type{ -1000 }, RelearnTypes::acceptance_criterion_type{ 0 }, mt);
    const auto too_large_acceptance_criterion = RandomFactory::get_random_double(Constants::bh_max_theta + utility::cast<RelearnTypes::acceptance_criterion_type>(eps), RelearnTypes::acceptance_criterion_type{ 10000 }, mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, { mpiPP::MPIRank::root_rank(), neuron_id }, position, nullptr, ElementType::Dendrite, searched_signal_type, Constants::bh_default_theta);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, { mpiPP::MPIRank::root_rank(), neuron_id }, position, &root, ElementType::Dendrite, searched_signal_type, 0.0);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, { mpiPP::MPIRank::root_rank(), neuron_id }, position, &root, ElementType::Dendrite, searched_signal_type, Constants::bh_max_theta + utility::cast<RelearnTypes::acceptance_criterion_type>(eps));, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, { mpiPP::MPIRank::root_rank(), neuron_id }, position, &root, ElementType::Dendrite, searched_signal_type, too_small_acceptance_criterion);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, { mpiPP::MPIRank::root_rank(), neuron_id }, position, &root, ElementType::Dendrite, searched_signal_type, too_large_acceptance_criterion);, RelearnException);
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronNoDendrites) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_tree_no_dendrites<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);
    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto searching_id = RankNeuronId{ mpiPP::MPIRank::root_rank(), neuron_id };

        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto& position = SimulationFactory::get_random_position(mt);

        auto found_target = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, searching_id, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

        ASSERT_FALSE(found_target.has_value());
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronNoChoice) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto root_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);

    auto root = OctreeNode<additional_cell_attributes>{};
    root.set_level(0);
    root.set_rank(mpiPP::MPIRank::root_rank());
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(0));
    root.set_cell_number_excitatory_dendrites(2);
    root.set_cell_number_inhibitory_dendrites(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (auto it = 0; it < 1000; it++) {
        const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto& position = SimulationFactory::get_random_position(mt);

        auto first_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, { mpiPP::MPIRank::root_rank(), NeuronID(0) }, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_FALSE(first_target_opt.has_value());

        auto second_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, { mpiPP::MPIRank::root_rank(), NeuronID(1) }, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(second_target_opt.has_value());

        auto [second_rank, second_id] = second_target_opt.value();
        ASSERT_EQ(second_rank, mpiPP::MPIRank::root_rank());
        ASSERT_EQ(second_id, NeuronID(0));
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronOneChoice) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto root_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);

    auto root = OctreeNode<additional_cell_attributes>{};
    root.set_level(0);
    root.set_rank(mpiPP::MPIRank::root_rank());
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(2));

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    const auto first_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);
    std::ignore = root.insert(first_position, NeuronID(0), memory_holder);

    const auto second_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);
    std::ignore = root.insert(second_position, NeuronID(1), memory_holder);

    auto* first_node = OctreeAdapter::find_node<additional_cell_attributes>({ mpiPP::MPIRank::root_rank(), NeuronID(0) }, &root);
    auto* second_node = OctreeAdapter::find_node<additional_cell_attributes>({ mpiPP::MPIRank::root_rank(), NeuronID(1) }, &root);

    first_node->set_cell_number_excitatory_dendrites(1);
    first_node->set_cell_number_inhibitory_dendrites(1);
    second_node->set_cell_number_excitatory_dendrites(2);
    second_node->set_cell_number_inhibitory_dendrites(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    auto first_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, { mpiPP::MPIRank::root_rank(), NeuronID(0) }, first_position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
    ASSERT_TRUE(first_target_opt.has_value());

    auto [first_rank, first_id] = first_target_opt.value();
    ASSERT_EQ(first_rank, mpiPP::MPIRank(0));
    ASSERT_EQ(first_id, NeuronID(1));

    auto second_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, { mpiPP::MPIRank::root_rank(), NeuronID(1) }, second_position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
    ASSERT_TRUE(second_target_opt.has_value());

    auto [second_rank, second_id] = second_target_opt.value();
    ASSERT_EQ(second_rank, mpiPP::MPIRank(0));
    ASSERT_EQ(second_id, NeuronID(0));
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronFullChoice) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);
    const auto& nodes = OctreeAdapter::find_nodes(&root);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto searching_id = RankNeuronId{ mpiPP::MPIRank::root_rank(), neuron_id };

        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        auto position = SimulationFactory::get_random_position(mt);

        if (nodes.contains(searching_id)) {
            position = nodes.at(searching_id)->get_cell().get_neuron_position().value();
        }

        const auto found_target = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, searching_id, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

        ASSERT_TRUE(found_target.has_value());

        const auto found_id = found_target.value();
        ASSERT_NE(searching_id, found_id);

        ASSERT_TRUE(nodes.contains(found_id));
        ASSERT_GE(nodes.at(found_id)->get_cell().get_number_dendrites_for(searched_signal_type), 0);

        const auto [target_rank, target_id] = found_target.value();

        ASSERT_EQ(target_rank, mpiPP::MPIRank::root_rank());
        ASSERT_NE(neuron_id, target_id);
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronNoChoiceDistributed) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto root_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);

    auto root = OctreeNode<additional_cell_attributes>{};
    root.set_level(0);
    root.set_rank(mpiPP::MPIRank(1));
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(0));
    root.set_cell_number_excitatory_dendrites(2);
    root.set_cell_number_inhibitory_dendrites(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (auto it = 0; it < 1000; it++) {
        const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto& position = SimulationFactory::get_random_position(mt);

        const auto first_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, { mpiPP::MPIRank::root_rank(), NeuronID(0) }, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(first_target_opt.has_value());

        const auto [first_rank, first_id] = first_target_opt.value();
        ASSERT_EQ(first_rank, mpiPP::MPIRank(1));
        ASSERT_EQ(first_id, NeuronID(0));

        const auto second_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, { mpiPP::MPIRank::root_rank(), NeuronID(1) }, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(second_target_opt.has_value());

        const auto [second_rank, second_id] = second_target_opt.value();
        ASSERT_EQ(second_rank, mpiPP::MPIRank(1));
        ASSERT_EQ(second_id, NeuronID(0));
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronFullChoiceDistributed) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto branching_level = SimulationFactory::get_small_positive_refinement_level(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);
    OctreeAdapter::mark_node_as_distributed(&root, branching_level);
    const auto& nodes = OctreeAdapter::find_nodes(&root);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto searching_id = RankNeuronId{ mpiPP::MPIRank::root_rank(), neuron_id };

        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        auto position = SimulationFactory::get_random_position(mt);

        if (nodes.contains(searching_id)) {
            position = nodes.at(searching_id)->get_cell().get_neuron_position().value();
        }

        const auto found_target = BarnesHutBase<additional_cell_attributes>::find_target_neuron(kernel, node_cache, searching_id, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

        ASSERT_TRUE(found_target.has_value());

        const auto found_id = found_target.value();
        ASSERT_NE(searching_id, found_id);

        ASSERT_TRUE(nodes.contains(found_id));
        ASSERT_GE(nodes.at(found_id)->get_cell().get_number_dendrites_for(searched_signal_type), 0);
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronsException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto neuron_id = NeuronID(1000000);
    const auto position = RelearnTypes::position_type{ 0.0 };

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto too_small_acceptance_criterion = RandomFactory::get_random_double(RelearnTypes::acceptance_criterion_type{ -1000 }, RelearnTypes::acceptance_criterion_type{ 0 }, mt);
    const auto too_large_acceptance_criterion = RandomFactory::get_random_double(Constants::bh_max_theta + utility::cast<RelearnTypes::acceptance_criterion_type>(eps), RelearnTypes::acceptance_criterion_type{ 10000 }, mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neurons(kernel, node_cache, { mpiPP::MPIRank::root_rank(), neuron_id }, position, 1, nullptr, ElementType::Dendrite, searched_signal_type, Constants::bh_default_theta);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neurons(kernel, node_cache, { mpiPP::MPIRank::root_rank(), neuron_id }, position, 1, &root, ElementType::Dendrite, searched_signal_type, 0.0);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neurons(kernel, node_cache, { mpiPP::MPIRank::root_rank(), neuron_id }, position, 1, &root, ElementType::Dendrite, searched_signal_type, Constants::bh_max_theta + utility::cast<RelearnTypes::acceptance_criterion_type>(eps));, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neurons(kernel, node_cache, { mpiPP::MPIRank::root_rank(), neuron_id }, position, 1, &root, ElementType::Dendrite, searched_signal_type, too_small_acceptance_criterion);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neurons(kernel, node_cache, { mpiPP::MPIRank::root_rank(), neuron_id }, position, 1, &root, ElementType::Dendrite, searched_signal_type, too_large_acceptance_criterion);, RelearnException);
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronsNoChoice) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto root_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);

    auto root = OctreeNode<additional_cell_attributes>{};
    root.set_level(0);
    root.set_rank(mpiPP::MPIRank::root_rank());
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(0));
    root.set_cell_number_excitatory_dendrites(2);
    root.set_cell_number_inhibitory_dendrites(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (auto it = 0; it < 1000; it++) {
        const auto number_vacant_elements = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
        const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto& position = SimulationFactory::get_random_position(mt);

        const auto first_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons(kernel, node_cache, { mpiPP::MPIRank::root_rank(), NeuronID(0) }, position, number_vacant_elements,
                                                                                                  &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

        ASSERT_TRUE(first_targets.empty());

        const auto second_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons(kernel, node_cache, { mpiPP::MPIRank::root_rank(), NeuronID(1) }, position, number_vacant_elements,
                                                                                                   &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

        ASSERT_EQ(second_targets.size(), number_vacant_elements);

        for (RelearnTypes::counter_type i = 0; i < number_vacant_elements; i++) {
            const auto& [rank, creation_request] = second_targets[i];
            ASSERT_EQ(rank, mpiPP::MPIRank::root_rank());

            const auto& [target_id, source_id, signal_type] = creation_request;

            ASSERT_EQ(target_id, NeuronID(0));
            ASSERT_EQ(source_id, NeuronID(1));
            ASSERT_EQ(signal_type, searched_signal_type);
        }
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronsOneChoice) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto root_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeNode<additional_cell_attributes>{};
    root.set_level(0);
    root.set_rank(mpiPP::MPIRank::root_rank());
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(2));

    const auto first_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);
    std::ignore = root.insert(first_position, NeuronID(0), memory_holder);

    const auto second_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);
    std::ignore = root.insert(second_position, NeuronID(1), memory_holder);

    auto* first_node = OctreeAdapter::find_node<additional_cell_attributes>({ mpiPP::MPIRank::root_rank(), NeuronID(0) }, &root);
    auto* second_node = OctreeAdapter::find_node<additional_cell_attributes>({ mpiPP::MPIRank::root_rank(), NeuronID(1) }, &root);

    first_node->set_cell_number_excitatory_dendrites(1);
    first_node->set_cell_number_inhibitory_dendrites(1);
    second_node->set_cell_number_excitatory_dendrites(2);
    second_node->set_cell_number_inhibitory_dendrites(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto number_vacant_elements_1 = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
    const auto number_vacant_elements_2 = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
    const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    const auto first_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons(kernel, node_cache, { mpiPP::MPIRank::root_rank(), NeuronID(0) }, first_position, number_vacant_elements_1,
                                                                                              &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

    ASSERT_EQ(first_targets.size(), number_vacant_elements_1);

    for (RelearnTypes::counter_type i = 0; i < number_vacant_elements_1; i++) {
        const auto& [rank, creation_request] = first_targets[i];
        ASSERT_EQ(rank, mpiPP::MPIRank::root_rank());

        const auto& [target_id, source_id, signal_type] = creation_request;

        ASSERT_EQ(target_id, NeuronID(1));
        ASSERT_EQ(source_id, NeuronID(0));
        ASSERT_EQ(signal_type, searched_signal_type);
    }

    const auto second_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons(kernel, node_cache, { mpiPP::MPIRank::root_rank(), NeuronID(1) }, second_position, number_vacant_elements_2,
                                                                                               &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

    ASSERT_EQ(second_targets.size(), number_vacant_elements_2);

    for (RelearnTypes::counter_type i = 0; i < number_vacant_elements_2; i++) {
        const auto& [rank, creation_request] = second_targets[i];
        ASSERT_EQ(rank, mpiPP::MPIRank::root_rank());

        const auto& [target_id, source_id, signal_type] = creation_request;

        ASSERT_EQ(target_id, NeuronID(0));
        ASSERT_EQ(source_id, NeuronID(1));
        ASSERT_EQ(signal_type, searched_signal_type);
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronsFullChoice) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);
    const auto& nodes = OctreeAdapter::find_nodes(&root);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto searching_id = RankNeuronId{ mpiPP::MPIRank::root_rank(), neuron_id };

        const auto number_vacant_elements = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        auto position = SimulationFactory::get_random_position(mt);

        if (nodes.contains(searching_id)) {
            position = nodes.at(searching_id)->get_cell().get_neuron_position().value();
        }

        const auto found_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons(kernel, node_cache, searching_id, position, number_vacant_elements, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

        ASSERT_EQ(found_targets.size(), number_vacant_elements);

        for (RelearnTypes::counter_type i = 0; i < number_vacant_elements; i++) {
            const auto& [rank, creation_request] = found_targets[i];
            ASSERT_EQ(rank, mpiPP::MPIRank::root_rank());

            const auto& [target_id, source_id, signal_type] = creation_request;

            ASSERT_NE(target_id, neuron_id);
            ASSERT_EQ(source_id, neuron_id);
            ASSERT_EQ(signal_type, searched_signal_type);
        }
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronsNoChoiceDistributed) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto root_position = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);

    auto root = OctreeNode<additional_cell_attributes>{};
    root.set_level(0);
    root.set_rank(mpiPP::MPIRank(1));
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(0));
    root.set_cell_number_excitatory_dendrites(2);
    root.set_cell_number_inhibitory_dendrites(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (auto it = 0; it < 1000; it++) {
        const auto number_vacant_elements = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
        const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        const auto& position = SimulationFactory::get_random_position(mt);

        const auto first_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons(kernel, node_cache, { mpiPP::MPIRank::root_rank(), NeuronID(0) }, position, number_vacant_elements,
                                                                                                  &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

        ASSERT_EQ(first_targets.size(), number_vacant_elements);

        for (RelearnTypes::counter_type i = 0; i < number_vacant_elements; i++) {
            const auto& [rank, creation_request] = first_targets[i];
            ASSERT_EQ(rank, mpiPP::MPIRank(1));

            const auto& [target_id, source_id, signal_type] = creation_request;

            ASSERT_TRUE((target_id == NeuronID(0) || target_id == NeuronID(1)));
            ASSERT_EQ(source_id, NeuronID(0));
            ASSERT_EQ(signal_type, searched_signal_type);
        }

        const auto second_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons(kernel, node_cache, { mpiPP::MPIRank::root_rank(), NeuronID(1) }, position, number_vacant_elements,
                                                                                                   &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

        ASSERT_EQ(second_targets.size(), number_vacant_elements);

        for (RelearnTypes::counter_type i = 0; i < number_vacant_elements; i++) {
            const auto& [rank, creation_request] = second_targets[i];
            ASSERT_EQ(rank, mpiPP::MPIRank(1));

            const auto& [target_id, source_id, signal_type] = creation_request;

            ASSERT_TRUE((target_id == NeuronID(0) || target_id == NeuronID(1)));
            ASSERT_EQ(source_id, NeuronID(1));
            ASSERT_EQ(signal_type, searched_signal_type);
        }
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronsFullChoiceDistributed) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto branching_level = SimulationFactory::get_small_positive_refinement_level(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);
    OctreeAdapter::mark_node_as_distributed(&root, branching_level);
    const auto& nodes = OctreeAdapter::find_nodes(&root);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto searching_id = RankNeuronId{ mpiPP::MPIRank::root_rank(), neuron_id };

        const auto number_vacant_elements = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        auto position = SimulationFactory::get_random_position(mt);

        if (nodes.contains(searching_id)) {
            position = nodes.at(searching_id)->get_cell().get_neuron_position().value();
        }

        const auto found_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons(kernel, node_cache, searching_id, position, number_vacant_elements,
                                                                                                  &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

        ASSERT_EQ(found_targets.size(), number_vacant_elements);

        for (RelearnTypes::counter_type i = 0; i < number_vacant_elements; i++) {
            const auto& [rank, creation_request] = found_targets[i];
            const auto& [target_id, source_id, signal_type] = creation_request;

            ASSERT_EQ(source_id, neuron_id);
            ASSERT_EQ(signal_type, searched_signal_type);

            const auto found_id = RankNeuronId(rank, target_id);

            ASSERT_NE(found_id, searching_id);
            ASSERT_TRUE(nodes.contains(found_id));
            ASSERT_GE(nodes.at(found_id)->get_cell().get_number_dendrites_for(searched_signal_type), 0);
        }
    }
}

TEST_F(BarnesHutBaseTest, testConvertTargetNodeException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto neuron_id = NeuronID(1000000);
    const auto position = RelearnTypes::position_type{ 0.0 };

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto branching_level = SimulationFactory::get_small_positive_refinement_level(mt);

    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::convert_target_node({ mpiPP::MPIRank::root_rank(), neuron_id }, position, nullptr, searched_signal_type, branching_level), RelearnException);
}

TEST_F(BarnesHutBaseTest, testConvertTargetNodeLeaf) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto source = NeuronID(1000000);
    const auto target = NeuronID(1000001);

    const auto source_position = RelearnTypes::position_type{ utility::cast<RelearnTypes::space_type>(0.2) };
    const auto target_position = RelearnTypes::position_type{ utility::cast<RelearnTypes::space_type>(0.5) };

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto branching_level = SimulationFactory::get_small_positive_refinement_level(mt);
    const auto target_level = SimulationFactory::get_small_positive_refinement_level(mt);

    auto target_node = OctreeNode<additional_cell_attributes>{};
    target_node.set_cell_neuron_id(target);
    target_node.set_cell_size(RelearnTypes::position_type{ -1.0 }, RelearnTypes::position_type{ 1.0 });
    target_node.set_level(target_level);
    target_node.set_cell_neuron_position(target_position);
    target_node.set_rank(mpiPP::MPIRank::root_rank());

    for (const auto mpi_rank : mpiPP::MPIRankRange::range(1000)) {
        const auto rni = RankNeuronId{ mpi_rank, source };

        const auto& val = BarnesHutBase<additional_cell_attributes>::convert_target_node(rni, source_position, &target_node, searched_signal_type, branching_level);

        ASSERT_TRUE(val.has_value());

        const auto& [found_rank, distant_neuron_request] = val.value();

        ASSERT_EQ(found_rank, mpiPP::MPIRank::root_rank());

        const auto& [_source_id, _source_position, _target_identifier, _target_neuron_type, _searched_signal_type] = distant_neuron_request;

        ASSERT_EQ(_source_id, source);
        ASSERT_EQ(_source_position, source_position);
        ASSERT_EQ(_target_identifier, target.get_neuron_id());
        ASSERT_EQ(_target_neuron_type, DistantNeuronRequest::TargetNeuronType::Leaf);
        ASSERT_EQ(_searched_signal_type, searched_signal_type);
    }
}

TEST_F(BarnesHutBaseTest, testConvertTargetNodeTooHigh) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto source = NeuronID(1000000);

    const auto source_position = RelearnTypes::position_type{ utility::cast<RelearnTypes::space_type>(0.2) };

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto target_level = SimulationFactory::get_small_positive_refinement_level(mt);
    const auto branching_level = static_cast<RelearnTypes::level_type>(target_level + 1);

    auto target_node = OctreeNode<additional_cell_attributes>{};
    target_node.set_cell_size(RelearnTypes::position_type{ -1.0 }, RelearnTypes::position_type{ 1.0 });
    target_node.set_level(target_level);
    target_node.set_rank(mpiPP::MPIRank::root_rank());
    target_node.set_cell_neuron_id(NeuronID(0));
    target_node.set_cell_neuron_position(RelearnTypes::position_type{ 0.0 });

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    std::ignore = target_node.insert(RelearnTypes::position_type{ utility::cast<RelearnTypes::space_type>(0.3) }, NeuronID(1), memory_holder);
    std::ignore = target_node.insert(RelearnTypes::position_type{ utility::cast<RelearnTypes::space_type>(0.5) }, NeuronID(2), memory_holder);
    std::ignore = target_node.insert(RelearnTypes::position_type{ utility::cast<RelearnTypes::space_type>(0.7) }, NeuronID(3), memory_holder);

    const auto& val_0 = BarnesHutBase<additional_cell_attributes>::convert_target_node({ mpiPP::MPIRank(0), source }, source_position, &target_node, searched_signal_type, branching_level);
    ASSERT_FALSE(val_0.has_value());

    const auto& val_1 = BarnesHutBase<additional_cell_attributes>::convert_target_node({ mpiPP::MPIRank(1), source }, source_position, &target_node, searched_signal_type, branching_level);
    ASSERT_FALSE(val_1.has_value());
}

TEST_F(BarnesHutBaseTest, testConvertTargetNodeVirtual) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto source = NeuronID(1000000);

    const auto source_position = RelearnTypes::position_type{ utility::cast<RelearnTypes::space_type>(0.2) };

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto branching_level = SimulationFactory::get_small_positive_refinement_level(mt);
    const auto target_level = static_cast<RelearnTypes::level_type>(branching_level + 1);

    auto target_node = OctreeNode<additional_cell_attributes>{};
    target_node.set_cell_size(RelearnTypes::position_type{ -1.0 }, RelearnTypes::position_type{ 1.0 });
    target_node.set_level(target_level);
    target_node.set_cell_neuron_id(NeuronID(0));
    target_node.set_cell_neuron_position(RelearnTypes::position_type{ 0.0 });
    target_node.set_rank(mpiPP::MPIRank::root_rank());

    std::ignore = target_node.insert(RelearnTypes::position_type{ utility::cast<RelearnTypes::space_type>(0.3) }, NeuronID(1), memory_holder);
    std::ignore = target_node.insert(RelearnTypes::position_type{ utility::cast<RelearnTypes::space_type>(0.5) }, NeuronID(2), memory_holder);
    std::ignore = target_node.insert(RelearnTypes::position_type{ utility::cast<RelearnTypes::space_type>(0.7) }, NeuronID(3), memory_holder);

    target_node.set_cell_neuron_id(NeuronID(true, 10101010));

    for (const auto mpi_rank : mpiPP::MPIRankRange::range(1000)) {
        const auto rni = RankNeuronId{ mpi_rank, source };

        const auto& val = BarnesHutBase<additional_cell_attributes>::convert_target_node(rni, source_position, &target_node, searched_signal_type, branching_level);

        ASSERT_TRUE(val.has_value());

        const auto& [found_rank, distant_neuron_request] = val.value();

        ASSERT_EQ(found_rank, mpiPP::MPIRank::root_rank());

        const auto& [_source_id, _source_position, _target_identifier, _target_neuron_type, _searched_signal_type] = distant_neuron_request;

        ASSERT_EQ(_source_id, source);
        ASSERT_EQ(_source_position, source_position);
        ASSERT_EQ(_target_identifier, 10101010);
        ASSERT_EQ(_target_neuron_type, DistantNeuronRequest::TargetNeuronType::VirtualNode);
        ASSERT_EQ(_searched_signal_type, searched_signal_type);
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronLocationAwareException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto neuron_id = NeuronID(1000000);
    const auto position = RelearnTypes::position_type{ 0.0 };

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto branching_level = SimulationFactory::get_small_positive_refinement_level(mt);

    const auto too_small_acceptance_criterion = RandomFactory::get_random_double(RelearnTypes::acceptance_criterion_type{ -1000 }, RelearnTypes::acceptance_criterion_type{ 0 }, mt);
    const auto too_large_acceptance_criterion = RandomFactory::get_random_double(Constants::bh_max_theta + utility::cast<RelearnTypes::acceptance_criterion_type>(eps), RelearnTypes::acceptance_criterion_type{ 10000 }, mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);

    const auto source = RankNeuronId{ mpiPP::MPIRank::root_rank(), neuron_id };

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neuron_location_aware(kernel, node_cache, source, position, nullptr, ElementType::Dendrite, searched_signal_type, branching_level, Constants::bh_default_theta);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neuron_location_aware(kernel, node_cache, source, position, &root, ElementType::Dendrite, searched_signal_type, branching_level, 0.0);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neuron_location_aware(kernel, node_cache, source, position, &root, ElementType::Dendrite, searched_signal_type, branching_level, Constants::bh_max_theta + utility::cast<RelearnTypes::acceptance_criterion_type>(eps));, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neuron_location_aware(kernel, node_cache, source, position, &root, ElementType::Dendrite, searched_signal_type, branching_level, too_small_acceptance_criterion);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neuron_location_aware(kernel, node_cache, source, position, &root, ElementType::Dendrite, searched_signal_type, branching_level, too_large_acceptance_criterion);, RelearnException);
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronLocationAwareNoDendrites) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto branching_level = SimulationFactory::get_small_positive_refinement_level(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_tree_no_dendrites<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);
    const auto& nodes = OctreeAdapter::find_nodes(&root);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto searching_id = RankNeuronId{ mpiPP::MPIRank::root_rank(), neuron_id };

        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        auto position = SimulationFactory::get_random_position(mt);

        if (nodes.contains(searching_id)) {
            position = nodes.at(searching_id)->get_cell().get_neuron_position().value();
        }

        const auto found_target = BarnesHutBase<additional_cell_attributes>::find_target_neuron_location_aware(kernel, node_cache, searching_id, position, &root, ElementType::Dendrite, searched_signal_type, branching_level, acceptance_criterion);

        ASSERT_FALSE(found_target.has_value());
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronLocationAwareFullChoice) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto branching_level = SimulationFactory::get_small_positive_refinement_level(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);

    const auto& nodes = OctreeAdapter::find_nodes(&root);
    const auto& rma_dict = OctreeAdapter::find_child_offsets(&root);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto searching_id = RankNeuronId{ mpiPP::MPIRank::root_rank(), neuron_id };

        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        auto position = SimulationFactory::get_random_position(mt);

        if (nodes.contains(searching_id)) {
            position = nodes.at(searching_id)->get_cell().get_neuron_position().value();
        }

        const auto found_target = BarnesHutBase<additional_cell_attributes>::find_target_neuron_location_aware(kernel, node_cache, searching_id, position, &root, ElementType::Dendrite, searched_signal_type, branching_level, acceptance_criterion);

        ASSERT_TRUE(found_target.has_value());

        const auto& [found_rank, distant_neuron_request] = found_target.value();

        const auto& [_source_id, _source_position, _target_identifier, _target_neuron_type, _searched_signal_type] = distant_neuron_request;

        ASSERT_EQ(found_rank, mpiPP::MPIRank::root_rank());

        ASSERT_EQ(_searched_signal_type, searched_signal_type);
        ASSERT_EQ(_source_id, neuron_id);
        ASSERT_EQ(_source_position, position);

        if (_target_neuron_type == DistantNeuronRequest::TargetNeuronType::Leaf) {
            const auto found_id = NeuronID(_target_identifier);

            ASSERT_TRUE(nodes.contains({ found_rank, found_id }));

            const auto* found_node = nodes.at({ found_rank, found_id });
            ASSERT_GE(found_node->get_cell().get_number_dendrites_for(searched_signal_type), 0);
        }

        if (_target_neuron_type == DistantNeuronRequest::TargetNeuronType::VirtualNode) {
            ASSERT_TRUE(rma_dict.contains(_target_identifier));

            auto* expected_node = rma_dict.at(_target_identifier);

            for (auto child_idx = static_cast<unsigned char>(0); child_idx < Constants::number_oct; child_idx++) {
                if (expected_node->get_child(child_idx) != nullptr) {
                    auto* ptr1 = expected_node->get_child(child_idx);
                    // memory_holder is backed by a deque (SemiStableVector), so consecutive offsets
                    // are not necessarily contiguous in memory once a chunk boundary is crossed --
                    // look each child up by its own offset rather than doing pointer arithmetic.
                    auto* ptr2 = memory_holder->get_node_from_offset(_target_identifier + child_idx);

                    ASSERT_EQ(ptr1, ptr2);
                }
            }
        }
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronsLocationAwareException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto neuron_id = NeuronID(1000000);
    const auto position = RelearnTypes::position_type{ 0.0 };

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto branching_level = SimulationFactory::get_small_positive_refinement_level(mt);

    const auto too_small_acceptance_criterion = RandomFactory::get_random_double(RelearnTypes::acceptance_criterion_type{ -1000 }, RelearnTypes::acceptance_criterion_type{ 0 }, mt);
    const auto too_large_acceptance_criterion = RandomFactory::get_random_double(Constants::bh_max_theta + utility::cast<RelearnTypes::acceptance_criterion_type>(eps), RelearnTypes::acceptance_criterion_type{ 10000 }, mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);

    const auto source = RankNeuronId{ mpiPP::MPIRank::root_rank(), neuron_id };

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neurons_location_aware(kernel, node_cache, source, position, 1, nullptr, ElementType::Dendrite, searched_signal_type, branching_level, Constants::bh_default_theta);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neurons_location_aware(kernel, node_cache, source, position, 1, &root, ElementType::Dendrite, searched_signal_type, branching_level, 0.0);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neurons_location_aware(kernel, node_cache, source, position, 1, &root, ElementType::Dendrite, searched_signal_type, branching_level, Constants::bh_max_theta + utility::cast<RelearnTypes::acceptance_criterion_type>(eps));, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neurons_location_aware(kernel, node_cache, source, position, 1, &root, ElementType::Dendrite, searched_signal_type, branching_level, too_small_acceptance_criterion);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BarnesHutBase<additional_cell_attributes>::find_target_neurons_location_aware(kernel, node_cache, source, position, 1, &root, ElementType::Dendrite, searched_signal_type, branching_level, too_large_acceptance_criterion);, RelearnException);
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronsLocationAwareFullChoice) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto branching_level = SimulationFactory::get_small_positive_refinement_level(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);
    const auto& nodes = OctreeAdapter::find_nodes(&root);
    const auto& rma_dict = OctreeAdapter::find_child_offsets(&root);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto searching_id = RankNeuronId{ mpiPP::MPIRank::root_rank(), neuron_id };

        const auto number_vacant_elements = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        auto position = SimulationFactory::get_random_position(mt);

        if (nodes.contains(searching_id)) {
            position = nodes.at(searching_id)->get_cell().get_neuron_position().value();
        }

        const auto found_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons_location_aware(kernel, node_cache, searching_id, position, number_vacant_elements, &root, ElementType::Dendrite, searched_signal_type, branching_level, acceptance_criterion);

        ASSERT_EQ(found_targets.size(), number_vacant_elements);

        for (RelearnTypes::counter_type i = 0; i < number_vacant_elements; i++) {
            const auto& [rank, distant_creation_request] = found_targets[i];
            ASSERT_EQ(rank, mpiPP::MPIRank::root_rank());

            const auto& source_id = distant_creation_request.get_source_id();
            const auto& source_position = distant_creation_request.get_source_position();
            const auto& signal_type = distant_creation_request.get_signal_type();

            ASSERT_EQ(source_id, neuron_id);
            ASSERT_EQ(source_position, position);
            ASSERT_EQ(signal_type, searched_signal_type);

            const auto& target_type = distant_creation_request.get_target_neuron_type();

            if (target_type == DistantNeuronRequest::TargetNeuronType::Leaf) {
                const auto& leaf_id = distant_creation_request.get_leaf_node_id();
                ASSERT_LT(leaf_id, number_neurons);

                ASSERT_NE(NeuronID(leaf_id), neuron_id);

                const auto rni = RankNeuronId(mpiPP::MPIRank::root_rank(), NeuronID(leaf_id));

                ASSERT_TRUE(nodes.contains(rni));
                const auto* node = nodes.at(rni);
                ASSERT_GT(node->get_cell().get_number_dendrites_for(signal_type), 0);
            } else {
                const auto& rma_offset = distant_creation_request.get_rma_offset();

                ASSERT_TRUE(rma_dict.contains(rma_offset));
                const auto* node = rma_dict.at(rma_offset);

                ASSERT_GT(node->get_cell().get_number_dendrites_for(signal_type), 0);
            }
        }
    }
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronsLocationAwareFullChoiceDistributed) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto branching_level = SimulationFactory::get_small_positive_refinement_level(mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<additional_cell_attributes>();

    auto root = OctreeFactory::get_standard_tree<additional_cell_attributes>(number_neurons, memory_holder, minimum, maximum, mt);
    OctreeAdapter::mark_node_as_distributed(&root, branching_level);

    const auto& nodes = OctreeAdapter::find_nodes(&root);
    const auto& rma_dict = OctreeAdapter::find_child_offsets(&root);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    auto node_cache = NodeCache<additional_cell_attributes>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto searching_id = RankNeuronId{ mpiPP::MPIRank::root_rank(), neuron_id };

        const auto number_vacant_elements = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 10, mt);
        const auto acceptance_criterion = RandomFactory::get_random_double(utility::cast<RelearnTypes::acceptance_criterion_type>(eps), Constants::bh_max_theta, mt);
        auto position = SimulationFactory::get_random_position(mt);

        if (nodes.contains(searching_id)) {
            position = nodes.at(searching_id)->get_cell().get_neuron_position().value();
        }

        const auto found_targets = BarnesHutBase<additional_cell_attributes>::find_target_neurons_location_aware(kernel, node_cache, searching_id, position, number_vacant_elements, &root, ElementType::Dendrite, searched_signal_type, branching_level, acceptance_criterion);

        ASSERT_EQ(found_targets.size(), number_vacant_elements);

        for (RelearnTypes::counter_type i = 0; i < number_vacant_elements; i++) {
            const auto& [rank, distant_creation_request] = found_targets[i];

            const auto& source_id = distant_creation_request.get_source_id();
            const auto& source_position = distant_creation_request.get_source_position();
            const auto& signal_type = distant_creation_request.get_signal_type();

            ASSERT_EQ(source_id, neuron_id);
            ASSERT_EQ(source_position, position);
            ASSERT_EQ(signal_type, searched_signal_type);

            const auto& target_type = distant_creation_request.get_target_neuron_type();

            if (target_type == DistantNeuronRequest::TargetNeuronType::Leaf) {
                const auto& leaf_id = distant_creation_request.get_leaf_node_id();
                ASSERT_LT(leaf_id, number_neurons);

                const auto rni = RankNeuronId(rank, NeuronID(leaf_id));

                ASSERT_NE(rni, searching_id);

                ASSERT_TRUE(nodes.contains(rni));
                auto* node = nodes.at(rni);
                ASSERT_GT(node->get_cell().get_number_dendrites_for(signal_type), 0);
            } else {
                const auto& rma_offset = distant_creation_request.get_rma_offset();

                ASSERT_TRUE(rma_dict.contains(rma_offset));
                auto* node = rma_dict.at(rma_offset);

                ASSERT_GT(node->get_cell().get_number_dendrites_for(signal_type), 0);
                ASSERT_GE(node->get_level(), branching_level);
            }
        }
    }
}
