/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_octree_node.h"

#include "Config.h"

#include "algorithm/Algorithms.h"
#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/BarnesHutInternal/BarnesHutInvertedCell.h"
#include "algorithm/Cells.h"
#include "algorithm/FMMInternal/FastMultipoleMethodCell.h"
#include "algorithm/Internal/octree/Cell.h"
#include "algorithm/Internal/octree/NodeCache.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "algorithm/Internal/octree/OctreeNodeHelper.h"
#include "algorithm/NaiveInternal/NaiveCell.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/MemoryHolder.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include "adapter/octree/OctreeAdapter.h"

#include "factory/memory_holder/memory_holder_factory.h"
#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neurons/neurons_factory.h"
#include "factory/octree/octree_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <cpp-utility/data-structure/Stack.hpp>
#include <cpp-utility/ranges/Functional.hpp>

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/core/MPIRankRange.h>

#include <range/v3/algorithm/sort.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stack>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using test_types = ::testing::Types<BarnesHutCell, BarnesHutInvertedCell, FastMultipoleMethodCell, NaiveCell>;
TYPED_TEST_SUITE(OctreeNodeTest, test_types);

TYPED_TEST(OctreeNodeTest, testReset) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    auto node = OctreeNode<AdditionalCellAttributes>{};

    ASSERT_FALSE(node.is_parent());
    ASSERT_TRUE(node.get_mpi_rank() == mpiPP::MPIRank::uninitialized_rank());
    ASSERT_TRUE(node.get_children().size() == Constants::number_oct);

    const auto& children = node.get_children();

    for (auto i = static_cast<unsigned char>(0); i < Constants::number_oct; i++) {
        ASSERT_TRUE(node.get_child(i) == nullptr);
        const auto& const_node = node;
        ASSERT_TRUE(const_node.get_child(i) == nullptr);
        ASSERT_TRUE(children[i] == nullptr);
    }

    node.set_parent();

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(this->mt);
    const auto rank = MPIRankFactory::get_random_mpi_rank(number_ranks, this->mt);

    node.set_rank(rank);

    auto other_nodes = std::array<OctreeNode<AdditionalCellAttributes>, Constants::number_oct>{};
    for (auto i = static_cast<unsigned char>(0); i < Constants::number_oct; i++) {
        node.set_child(&(other_nodes[i]), i);
    }

    node.reset();

    ASSERT_FALSE(node.is_parent());
    ASSERT_TRUE(node.get_mpi_rank() == mpiPP::MPIRank::uninitialized_rank());
    ASSERT_TRUE(node.get_children().size() == Constants::number_oct);

    const auto& new_children = node.get_children();

    for (auto i = static_cast<unsigned char>(0); i < Constants::number_oct; i++) {
        ASSERT_TRUE(node.get_child(i) == nullptr);
        const auto& const_node = node;
        ASSERT_TRUE(const_node.get_child(i) == nullptr);
        ASSERT_TRUE(new_children[i] == nullptr);
    }
}

TYPED_TEST(OctreeNodeTest, testSetterGetter) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    auto node = OctreeNode<AdditionalCellAttributes>{};

    node.set_parent();

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(this->mt);
    const auto rank = MPIRankFactory::get_random_mpi_rank(number_ranks, this->mt);
    const auto level = SimulationFactory::get_small_refinement_level(this->mt);

    node.set_rank(rank);
    node.set_level(level);

    auto other_nodes = std::array<OctreeNode<AdditionalCellAttributes>, Constants::number_oct>{};
    for (auto i = static_cast<unsigned char>(0); i < Constants::number_oct; i++) {
        node.set_child(&(other_nodes[i]), i);
    }

    ASSERT_TRUE(node.is_parent());
    ASSERT_TRUE(node.get_mpi_rank() == rank);
    ASSERT_TRUE(node.get_children().size() == Constants::number_oct);
    ASSERT_EQ(node.get_level(), level);

    const auto& children = node.get_children();

    for (auto i = static_cast<unsigned char>(0); i < Constants::number_oct; i++) {
        ASSERT_TRUE(node.get_child(i) == &(other_nodes[i]));
        const auto& const_node = node;
        ASSERT_TRUE(const_node.get_child(i) == &(other_nodes[i]));
        ASSERT_TRUE(children[i] == &(other_nodes[i]));
    }

    const auto ub = std::numeric_limits<unsigned char>::max();

    for (auto i = Constants::number_oct; i <= ub; i++) {
        const auto octant = static_cast<unsigned char>(i);

        ASSERT_THROW_NO_PRINT(node.set_child(nullptr, octant), RelearnException);
        ASSERT_THROW_NO_PRINT(node.set_child(&node, octant), RelearnException);
        ASSERT_THROW_NO_PRINT(std::ignore = node.get_child(octant), RelearnException);
        const auto& const_node = node;
        ASSERT_THROW_NO_PRINT(std::ignore = const_node.get_child(octant), RelearnException);
    }
}

TYPED_TEST(OctreeNodeTest, testLocal) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    auto node = OctreeNode<AdditionalCellAttributes>{};
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    for (const auto rank : mpiPP::MPIRankRange::range(1000)) {
        node.set_rank(rank);

        if (rank == my_rank) {
            ASSERT_TRUE(node.is_actual_id());
        } else {
            ASSERT_FALSE(node.is_actual_id());
        }
    }
}

TYPED_TEST(OctreeNodeTest, testInsert) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<AdditionalCellAttributes>();

    auto node = OctreeNode<AdditionalCellAttributes>{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    const auto& [cell_min, cell_max] = node.get_size();

    ASSERT_EQ(cell_min, min);
    ASSERT_EQ(cell_max, max);

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(this->mt);

    auto neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, this->mt);

    for (const auto& [pos, id] : neurons_to_place) {
        node.insert(pos, id, memory_holder);
    }

    auto placed_neurons = OctreeAdapter::template extract_neurons<AdditionalCellAttributes>(&node);

    ranges::sort(neurons_to_place, std::greater{}, utility::element<1>);
    ranges::sort(placed_neurons, std::greater{}, utility::element<1>);

    ASSERT_EQ(neurons_to_place.size(), placed_neurons.size());

    for (auto i = 0U; i < neurons_to_place.size(); i++) {
        const auto& expected_neuron = neurons_to_place[i];
        const auto& found_neuron = placed_neurons[i];

        ASSERT_EQ(expected_neuron, found_neuron);
    }
}

TYPED_TEST(OctreeNodeTest, testInsertByHand) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto min = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto max = RelearnTypes::position_type{ 100.0, 100.0, 100.0 };

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<AdditionalCellAttributes>();

    auto node = OctreeNode<AdditionalCellAttributes>{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(RelearnTypes::position_type{ 25.0, 25.0, 25.0 });

    const auto& [cell_min, cell_max] = node.get_size();

    ASSERT_EQ(cell_min, min);
    ASSERT_EQ(cell_max, max);

    std::ignore = node.insert(RelearnTypes::position_type{ 25.0, 25.0, 75.0 }, NeuronID::virtual_id(), memory_holder);
    std::ignore = node.insert(RelearnTypes::position_type{ 25.0, 75.0, 25.0 }, NeuronID::virtual_id(), memory_holder);
    std::ignore = node.insert(RelearnTypes::position_type{ 75.0, 25.0, 25.0 }, NeuronID::virtual_id(), memory_holder);
    std::ignore = node.insert(RelearnTypes::position_type{ 25.0, 75.0, 75.0 }, NeuronID::virtual_id(), memory_holder);
    std::ignore = node.insert(RelearnTypes::position_type{ 75.0, 25.0, 75.0 }, NeuronID::virtual_id(), memory_holder);
    std::ignore = node.insert(RelearnTypes::position_type{ 75.0, 75.0, 25.0 }, NeuronID::virtual_id(), memory_holder);
    std::ignore = node.insert(RelearnTypes::position_type{ 75.0, 75.0, 75.0 }, NeuronID::virtual_id(), memory_holder);

    ASSERT_TRUE(node.is_parent());
    ASSERT_FALSE(node.is_leaf());
    ASSERT_EQ(node.get_level(), 0);

    for (auto child_id = static_cast<unsigned char>(0); child_id < Constants::number_oct; child_id++) {
        auto* child = node.get_child(child_id);
        ASSERT_NE(nullptr, child);

        ASSERT_FALSE(child->is_parent());
        ASSERT_TRUE(child->is_leaf());

        for (auto i = static_cast<unsigned char>(0); i < Constants::number_oct; i++) {
            ASSERT_EQ(nullptr, child->get_child(i));
        }
    }

    std::ignore = node.insert(RelearnTypes::position_type{ 24.0, 24.0, 24.0 }, NeuronID(11), memory_holder);
    std::ignore = node.insert(RelearnTypes::position_type{ 24.0, 24.0, 76.0 }, NeuronID(22), memory_holder);
    std::ignore = node.insert(RelearnTypes::position_type{ 24.0, 76.0, 24.0 }, NeuronID(33), memory_holder);
    std::ignore = node.insert(RelearnTypes::position_type{ 76.0, 24.0, 24.0 }, NeuronID(44), memory_holder);
    std::ignore = node.insert(RelearnTypes::position_type{ 24.0, 76.0, 76.0 }, NeuronID(55), memory_holder);
    std::ignore = node.insert(RelearnTypes::position_type{ 76.0, 24.0, 76.0 }, NeuronID(66), memory_holder);
    std::ignore = node.insert(RelearnTypes::position_type{ 76.0, 76.0, 24.0 }, NeuronID(77), memory_holder);
    std::ignore = node.insert(RelearnTypes::position_type{ 76.0, 76.0, 76.0 }, NeuronID(88), memory_holder);

    ASSERT_TRUE(node.is_parent());
    ASSERT_FALSE(node.is_leaf());
    ASSERT_EQ(node.get_level(), 0);

    ASSERT_EQ(cell_min, min);
    ASSERT_EQ(cell_max, max);

    for (auto child_id = static_cast<unsigned char>(0); child_id < Constants::number_oct; child_id++) {
        auto* child = node.get_child(child_id);
        ASSERT_NE(nullptr, child);

        ASSERT_FALSE(child->is_parent());
        ASSERT_TRUE(child->is_leaf());

        for (auto i = static_cast<unsigned char>(0); i < Constants::number_oct; i++) {
            ASSERT_EQ(nullptr, child->get_child(i));
        }
    }
}

TYPED_TEST(OctreeNodeTest, testContains) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto neuron_id = NeuronIdFactory::get_random_neuron_id(100, this->mt);
    const auto mpi_rank = MPIRankFactory::get_random_mpi_rank(20, this->mt);

    const auto rni = RankNeuronId{ mpi_rank, neuron_id };

    using AdditionalCellAttributes = TypeParam;

    for (const auto id_iterator : NeuronIDRange::range(100)) {
        for (const auto rank_iterator : mpiPP::MPIRankRange::range(20)) {
            auto node = OctreeNode<AdditionalCellAttributes>{};
            node.set_cell_neuron_id(id_iterator);
            node.set_rank(rank_iterator);

            const auto flag = node.contains(rni);

            if (node.is_leaf() && node.get_mpi_rank() == mpi_rank && node.get_cell_neuron_id() == neuron_id) {
                ASSERT_TRUE(flag);
            } else {
                ASSERT_FALSE(flag);
            }

            node.set_parent();

            ASSERT_FALSE(node.contains(rni));
        }
    }
}

TYPED_TEST(OctreeNodeTest, testLevel) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);
    const auto level = SimulationFactory::get_small_refinement_level(this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<AdditionalCellAttributes>();

    auto node = OctreeNode<AdditionalCellAttributes>{};
    node.set_level(level);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, this->mt);

    for (const auto& [pos, id] : neurons_to_place) {
        node.insert(pos, id, memory_holder);
    }

    ASSERT_EQ(node.get_level(), level);
    auto stack = utility::Stack<std::pair<const OctreeNode<AdditionalCellAttributes>*, const OctreeNode<AdditionalCellAttributes>*>>{};
    for (const auto* child : node.get_children()) {
        if (child != nullptr) {
            stack.emplace_back(&node, child);
        }
    }

    while (!stack.empty()) {
        const auto& [parent, child] = stack.pop_back();
        const auto parent_level = parent->get_level();
        const auto child_level = child->get_level();
        ASSERT_EQ(parent_level + 1, child_level);
        if (child->is_parent()) {
            for (const auto* new_child : child->get_children()) {
                if (new_child != nullptr) {
                    stack.emplace_back(child, new_child);
                }
            }
        }
    }
}

TYPED_TEST(OctreeNodeTest, testUpdateNode) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<AdditionalCellAttributes>();

    auto node = OctreeNode<AdditionalCellAttributes>{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    const auto midpoint = (max - min) / 2.0;
    const auto [mid_x, mid_y, mid_z] = midpoint;

    node.insert(SimulationFactory::get_random_position_in_box(min, min + midpoint, this->mt), NeuronID(1), memory_holder);
    node.insert(SimulationFactory::get_random_position_in_box(min, min + midpoint, this->mt) + RelearnTypes::position_type(mid_x, 0, 0), NeuronID(2), memory_holder);
    node.insert(SimulationFactory::get_random_position_in_box(min, min + midpoint, this->mt) + RelearnTypes::position_type(0, mid_y, 0), NeuronID(3), memory_holder);
    node.insert(SimulationFactory::get_random_position_in_box(min, min + midpoint, this->mt) + RelearnTypes::position_type(0, 0, mid_z), NeuronID(4), memory_holder);
    node.insert(SimulationFactory::get_random_position_in_box(min, min + midpoint, this->mt) + RelearnTypes::position_type(0, mid_y, mid_z), NeuronID(5), memory_holder);
    node.insert(SimulationFactory::get_random_position_in_box(min, min + midpoint, this->mt) + RelearnTypes::position_type(mid_x, 0, mid_z), NeuronID(6), memory_holder);
    node.insert(SimulationFactory::get_random_position_in_box(min, min + midpoint, this->mt) + RelearnTypes::position_type(mid_x, mid_y, 0), NeuronID(7), memory_holder);
    node.insert(SimulationFactory::get_random_position_in_box(min, min + midpoint, this->mt) + RelearnTypes::position_type(mid_x, mid_y, mid_z), NeuronID(8), memory_holder);

    auto golden_number_excitatory_dendrites = 0U;
    auto golden_number_inhibitory_dendrites = 0U;
    auto golden_number_excitatory_axons = 0U;
    auto golden_number_inhibitory_axons = 0U;

    auto golden_position_excitatory_dendrites = RelearnTypes::position_type{ 0 };
    auto golden_position_inhibitory_dendrites = RelearnTypes::position_type{ 0 };
    auto golden_position_excitatory_axons = RelearnTypes::position_type{ 0 };
    auto golden_position_inhibitory_axons = RelearnTypes::position_type{ 0 };

    if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
        for (auto* child : node.get_children()) {
            const auto vacant_elements = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
            golden_number_excitatory_dendrites += vacant_elements;
            child->set_cell_number_excitatory_dendrites(vacant_elements);

            golden_position_excitatory_dendrites += child->get_cell().get_neuron_position().value() * static_cast<RelearnTypes::space_type>(vacant_elements);
        }
    }

    if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
        for (auto* child : node.get_children()) {
            const auto vacant_elements = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
            golden_number_inhibitory_dendrites += vacant_elements;
            child->set_cell_number_inhibitory_dendrites(vacant_elements);

            golden_position_inhibitory_dendrites += child->get_cell().get_neuron_position().value() * static_cast<RelearnTypes::space_type>(vacant_elements);
        }
    }

    if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
        for (auto* child : node.get_children()) {
            const auto vacant_elements = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
            golden_number_excitatory_axons += vacant_elements;
            child->set_cell_number_excitatory_axons(vacant_elements);

            golden_position_excitatory_axons += child->get_cell().get_neuron_position().value() * static_cast<RelearnTypes::space_type>(vacant_elements);
        }
    }

    if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
        for (auto* child : node.get_children()) {
            const auto vacant_elements = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
            golden_number_inhibitory_axons += vacant_elements;
            child->set_cell_number_inhibitory_axons(vacant_elements);

            golden_position_inhibitory_axons += child->get_cell().get_neuron_position().value() * static_cast<RelearnTypes::space_type>(vacant_elements);
        }
    }

    OctreeNodeUpdater<AdditionalCellAttributes>::update_node(&node);

    const auto& cell = node.get_cell();

    if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
        ASSERT_EQ(cell.get_number_excitatory_dendrites(), golden_number_excitatory_dendrites);

        auto position = cell.get_excitatory_dendrites_position().value();
        auto scaled_position = position * static_cast<RelearnTypes::space_type>(golden_number_excitatory_dendrites);

        auto difference = scaled_position - golden_position_excitatory_dendrites;
        auto norm = difference.calculate_2_norm();

        // The node stores the summed position divided by the number of elements, so scaling it back up
        // recovers the sum only to the resolution space_type has at that magnitude.
        ASSERT_NEAR(norm, 0.0, this->template tolerance_for<RelearnTypes::space_type>(golden_position_excitatory_dendrites.calculate_2_norm()));
    }

    if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
        ASSERT_EQ(cell.get_number_inhibitory_dendrites(), golden_number_inhibitory_dendrites);

        auto position = cell.get_inhibitory_dendrites_position().value();
        auto scaled_position = position * static_cast<RelearnTypes::space_type>(golden_number_inhibitory_dendrites);

        auto difference = scaled_position - golden_position_inhibitory_dendrites;
        auto norm = difference.calculate_2_norm();

        // The node stores the summed position divided by the number of elements, so scaling it back up
        // recovers the sum only to the resolution space_type has at that magnitude.
        ASSERT_NEAR(norm, 0.0, this->template tolerance_for<RelearnTypes::space_type>(golden_position_inhibitory_dendrites.calculate_2_norm()));
    }

    if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
        ASSERT_EQ(cell.get_number_excitatory_axons(), golden_number_excitatory_axons);

        auto position = cell.get_excitatory_axons_position().value();
        auto scaled_position = position * static_cast<RelearnTypes::space_type>(golden_number_excitatory_axons);

        auto difference = scaled_position - golden_position_excitatory_axons;
        auto norm = difference.calculate_2_norm();

        // The node stores the summed position divided by the number of elements, so scaling it back up
        // recovers the sum only to the resolution space_type has at that magnitude.
        ASSERT_NEAR(norm, 0.0, this->template tolerance_for<RelearnTypes::space_type>(golden_position_excitatory_axons.calculate_2_norm()));
    }

    if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
        ASSERT_EQ(cell.get_number_inhibitory_axons(), golden_number_inhibitory_axons);

        auto position = cell.get_inhibitory_axons_position().value();
        auto scaled_position = position * static_cast<RelearnTypes::space_type>(golden_number_inhibitory_axons);

        auto difference = scaled_position - golden_position_inhibitory_axons;
        auto norm = difference.calculate_2_norm();

        // The node stores the summed position divided by the number of elements, so scaling it back up
        // recovers the sum only to the resolution space_type has at that magnitude.
        ASSERT_NEAR(norm, 0.0, this->template tolerance_for<RelearnTypes::space_type>(golden_position_inhibitory_axons.calculate_2_norm()));
    }
}

TYPED_TEST(OctreeNodeTest, testUpdateTree) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<AdditionalCellAttributes>();

    auto node = OctreeNode<AdditionalCellAttributes>{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, this->mt);

    for (const auto& [pos, id] : neurons_to_place) {
        node.insert(pos, id, memory_holder);
    }

    auto stack = std::stack<OctreeNode<AdditionalCellAttributes>*>{};
    stack.push(&node);

    auto vacant_excitatory_dendrites = std::unordered_map<OctreeNode<AdditionalCellAttributes>*, typename OctreeNode<AdditionalCellAttributes>::counter_type>{};
    auto vacant_inhibitory_dendrites = std::unordered_map<OctreeNode<AdditionalCellAttributes>*, typename OctreeNode<AdditionalCellAttributes>::counter_type>{};
    auto vacant_excitatory_axons = std::unordered_map<OctreeNode<AdditionalCellAttributes>*, typename OctreeNode<AdditionalCellAttributes>::counter_type>{};
    auto vacant_inhibitory_axons = std::unordered_map<OctreeNode<AdditionalCellAttributes>*, typename OctreeNode<AdditionalCellAttributes>::counter_type>{};

    auto position_excitatory_dendrites = std::unordered_map<OctreeNode<AdditionalCellAttributes>*, RelearnTypes::position_type>{};
    auto position_inhibitory_dendrites = std::unordered_map<OctreeNode<AdditionalCellAttributes>*, RelearnTypes::position_type>{};
    auto position_excitatory_axons = std::unordered_map<OctreeNode<AdditionalCellAttributes>*, RelearnTypes::position_type>{};
    auto position_inhibitory_axons = std::unordered_map<OctreeNode<AdditionalCellAttributes>*, RelearnTypes::position_type>{};

    while (!stack.empty()) {
        auto* current = stack.top();
        stack.pop();

        if (current->is_leaf()) {
            if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                const auto vacant_elements = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_excitatory_dendrites(vacant_elements);

                vacant_excitatory_dendrites[current] = vacant_elements;
                position_excitatory_dendrites[current] = current->get_cell().get_neuron_position().value();
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                const auto vacant_elements = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_inhibitory_dendrites(vacant_elements);

                vacant_inhibitory_dendrites[current] = vacant_elements;
                position_inhibitory_dendrites[current] = current->get_cell().get_neuron_position().value();
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                const auto vacant_elements = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_excitatory_axons(vacant_elements);

                vacant_excitatory_axons[current] = vacant_elements;
                position_excitatory_axons[current] = current->get_cell().get_neuron_position().value();
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                const auto vacant_elements = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_inhibitory_axons(vacant_elements);

                vacant_inhibitory_axons[current] = vacant_elements;
                position_inhibitory_axons[current] = current->get_cell().get_neuron_position().value();
            }

            continue;
        }

        for (auto* child : current->get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }
    }

    OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&node);
    stack.push(&node);

    while (!stack.empty()) {
        auto* current = stack.top();
        stack.pop();

        const auto& cell = current->get_cell();

        if (current->is_leaf()) {
            if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                auto expected_vacant_elements = vacant_excitatory_dendrites[current];
                auto expected_position = position_excitatory_dendrites[current];

                auto vacant_elements = cell.get_number_excitatory_dendrites();
                ASSERT_EQ(expected_vacant_elements, vacant_elements);

                if (vacant_elements != 0) {
                    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
                    auto position = cell.get_excitatory_dendrites_position().value();

                    ASSERT_EQ(expected_position, position);
                }
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                auto expected_vacant_elements = vacant_inhibitory_dendrites[current];
                auto expected_position = position_inhibitory_dendrites[current];

                auto vacant_elements = cell.get_number_inhibitory_dendrites();
                ASSERT_EQ(expected_vacant_elements, vacant_elements);

                if (vacant_elements != 0) {
                    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
                    auto position = cell.get_inhibitory_dendrites_position().value();

                    ASSERT_EQ(expected_position, position);
                }
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                auto expected_vacant_elements = vacant_excitatory_axons[current];
                auto expected_position = position_excitatory_axons[current];

                auto vacant_elements = cell.get_number_excitatory_axons();
                ASSERT_EQ(expected_vacant_elements, vacant_elements);

                if (vacant_elements != 0) {
                    ASSERT_TRUE(cell.get_excitatory_axons_position().has_value());
                    auto position = cell.get_excitatory_axons_position().value();

                    ASSERT_EQ(expected_position, position);
                }
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                auto expected_vacant_elements = vacant_inhibitory_axons[current];
                auto expected_position = position_inhibitory_axons[current];

                auto vacant_elements = cell.get_number_inhibitory_axons();
                ASSERT_EQ(expected_vacant_elements, vacant_elements);

                if (vacant_elements != 0) {
                    ASSERT_TRUE(cell.get_inhibitory_axons_position().has_value());
                    auto position = cell.get_inhibitory_axons_position().value();

                    ASSERT_EQ(expected_position, position);
                }
            }

            continue;
        }

        auto golden_number_excitatory_dendrites = 0U;
        auto golden_number_inhibitory_dendrites = 0U;
        auto golden_number_excitatory_axons = 0U;
        auto golden_number_inhibitory_axons = 0U;

        auto golden_position_excitatory_dendrites = RelearnTypes::position_type{ 0 };
        auto golden_position_inhibitory_dendrites = RelearnTypes::position_type{ 0 };
        auto golden_position_excitatory_axons = RelearnTypes::position_type{ 0 };
        auto golden_position_inhibitory_axons = RelearnTypes::position_type{ 0 };

        for (auto* child : current->get_children()) {
            if (child != nullptr) {
                stack.push(child);

                const auto& child_cell = child->get_cell();

                if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                    auto vacant_elements = child_cell.get_number_excitatory_dendrites();
                    if (vacant_elements != 0) {
                        ASSERT_TRUE(child_cell.get_excitatory_dendrites_position().has_value());
                        auto position = child_cell.get_excitatory_dendrites_position().value();

                        golden_number_excitatory_dendrites += vacant_elements;
                        golden_position_excitatory_dendrites += position * static_cast<RelearnTypes::space_type>(vacant_elements);
                    }
                }

                if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                    auto vacant_elements = child_cell.get_number_inhibitory_dendrites();
                    if (vacant_elements != 0) {
                        ASSERT_TRUE(child_cell.get_inhibitory_dendrites_position().has_value());
                        auto position = child_cell.get_inhibitory_dendrites_position().value();

                        golden_number_inhibitory_dendrites += vacant_elements;
                        golden_position_inhibitory_dendrites += position * static_cast<RelearnTypes::space_type>(vacant_elements);
                    }
                }

                if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                    auto vacant_elements = child_cell.get_number_excitatory_axons();
                    if (vacant_elements != 0) {
                        ASSERT_TRUE(child_cell.get_excitatory_axons_position().has_value());
                        auto position = child_cell.get_excitatory_axons_position().value();

                        golden_number_excitatory_axons += vacant_elements;
                        golden_position_excitatory_axons += position * static_cast<RelearnTypes::space_type>(vacant_elements);
                    }
                }

                if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                    auto vacant_elements = child_cell.get_number_inhibitory_axons();
                    if (vacant_elements != 0) {
                        ASSERT_TRUE(child_cell.get_inhibitory_axons_position().has_value());
                        auto position = child_cell.get_inhibitory_axons_position().value();

                        golden_number_inhibitory_axons += vacant_elements;
                        golden_position_inhibitory_axons += position * static_cast<RelearnTypes::space_type>(vacant_elements);
                    }
                }
            }
        }

        if (golden_number_excitatory_dendrites != 0) {
            golden_position_excitatory_dendrites /= static_cast<RelearnTypes::space_type>(golden_number_excitatory_dendrites);
        }

        if (golden_number_inhibitory_dendrites != 0) {
            golden_position_inhibitory_dendrites /= static_cast<RelearnTypes::space_type>(golden_number_inhibitory_dendrites);
        }

        if (golden_number_excitatory_axons != 0) {
            golden_position_excitatory_axons /= static_cast<RelearnTypes::space_type>(golden_number_excitatory_axons);
        }

        if (golden_number_inhibitory_axons != 0) {
            golden_position_inhibitory_axons /= static_cast<RelearnTypes::space_type>(golden_number_inhibitory_axons);
        }

        if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
            auto vacant_elements = cell.get_number_excitatory_dendrites();
            ASSERT_EQ(vacant_elements, golden_number_excitatory_dendrites);

            if (vacant_elements != 0) {
                ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
                auto position = cell.get_excitatory_dendrites_position().value();
                ASSERT_EQ(position, golden_position_excitatory_dendrites);
            } else {
                ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
            }
        }

        if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
            auto vacant_elements = cell.get_number_inhibitory_dendrites();
            ASSERT_EQ(vacant_elements, golden_number_inhibitory_dendrites);

            if (vacant_elements != 0) {

                ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
                auto position = cell.get_inhibitory_dendrites_position().value();
                ASSERT_EQ(position, golden_position_inhibitory_dendrites);
            } else {
                ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
            }
        }

        if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
            auto vacant_elements = cell.get_number_excitatory_axons();
            ASSERT_EQ(vacant_elements, golden_number_excitatory_axons);

            if (vacant_elements != 0) {
                ASSERT_TRUE(cell.get_excitatory_axons_position().has_value());
                auto position = cell.get_excitatory_axons_position().value();
                ASSERT_EQ(position, golden_position_excitatory_axons);
            } else {
                ASSERT_FALSE(cell.get_excitatory_axons_position().has_value());
            }
        }

        if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
            auto vacant_elements = cell.get_number_inhibitory_axons();
            ASSERT_EQ(vacant_elements, golden_number_inhibitory_axons);

            if (vacant_elements != 0) {
                ASSERT_TRUE(cell.get_inhibitory_axons_position().has_value());
                auto position = cell.get_inhibitory_axons_position().value();
                ASSERT_EQ(position, golden_position_inhibitory_axons);
            } else {
                ASSERT_FALSE(cell.get_inhibitory_axons_position().has_value());
            }
        }
    }
}

TYPED_TEST(OctreeNodeTest, testMemoryLayout) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(this->mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<AdditionalCellAttributes>();

    auto root = OctreeFactory::get_standard_tree<AdditionalCellAttributes>(number_neurons, memory_holder, minimum, maximum, this->mt);

    auto stack = std::stack<std::pair<OctreeNode<AdditionalCellAttributes>*, OctreeNode<AdditionalCellAttributes>*>>{};
    stack.emplace(&root, nullptr);

    auto touched_rmas = std::vector<std::uint64_t>{};

    while (!stack.empty()) {
        auto [current_node, parent] = stack.top();
        stack.pop();

        if (current_node->is_leaf()) {
            continue;
        }

        for (auto* child : current_node->get_children()) {
            if (child != nullptr) {
                stack.emplace(child, current_node);
            }
        }

        const auto saved_neuron_id = current_node->get_cell_neuron_id();
        ASSERT_TRUE(saved_neuron_id.is_virtual());
        const auto saved_rma_offset = saved_neuron_id.get_rma_offset();

        touched_rmas.emplace_back(saved_rma_offset);

        if (parent == nullptr) {
            ASSERT_EQ(&root, current_node);
            ASSERT_EQ(saved_rma_offset, 0);

            const auto mh_offset = memory_holder->get_offset_from_parent(current_node);
            ASSERT_EQ(mh_offset, 0);

            for (auto child_id = static_cast<unsigned char>(0); child_id < Constants::number_oct; child_id++) {
                auto* child = current_node->get_child(child_id);
                if (child == nullptr) {
                    continue;
                }

                // memory_holder is backed by a deque (SemiStableVector), so consecutive offsets are
                // not necessarily contiguous in memory once a chunk boundary is crossed -- look each
                // child up by its own offset rather than doing pointer arithmetic from offset 0.
                auto* expected_ptr = memory_holder->get_node_from_offset(child_id);
                ASSERT_EQ(expected_ptr, child);
            }

            continue;
        }

        const auto mh_offset = memory_holder->get_offset_from_parent(current_node);

        ASSERT_EQ(mh_offset, saved_rma_offset);

        for (auto child_id = static_cast<unsigned char>(0); child_id < Constants::number_oct; child_id++) {
            auto* child = current_node->get_child(child_id);
            if (child == nullptr) {
                continue;
            }

            // See the comment above: look each child up by its own offset instead of doing pointer
            // arithmetic from the parent's base offset, since the underlying deque is not guaranteed
            // to store the whole 8-node block contiguously.
            auto* expected_ptr = memory_holder->get_node_from_offset(saved_rma_offset + child_id);
            ASSERT_EQ(expected_ptr, child);
        }
    }

    std::ranges::sort(touched_rmas);

    for (auto idx = 0U; idx < touched_rmas.size(); idx++) {
        ASSERT_EQ(idx * Constants::number_oct, touched_rmas[idx]);
    }
}

TYPED_TEST(OctreeNodeTest, testNodeExtractor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<AdditionalCellAttributes>();

    auto node = OctreeNode<AdditionalCellAttributes>{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, this->mt);

    for (const auto& [pos, id] : neurons_to_place) {
        node.insert(pos, id, memory_holder);
    }

    auto stack = std::stack<OctreeNode<AdditionalCellAttributes>*>{};
    stack.push(&node);

    auto excitatory_dendrites = std::vector<std::pair<RelearnTypes::position_type, typename OctreeNode<AdditionalCellAttributes>::counter_type>>{};
    auto inhibitory_dendrites = std::vector<std::pair<RelearnTypes::position_type, typename OctreeNode<AdditionalCellAttributes>::counter_type>>{};
    auto excitatory_axons = std::vector<std::pair<RelearnTypes::position_type, typename OctreeNode<AdditionalCellAttributes>::counter_type>>{};
    auto inhibitory_axons = std::vector<std::pair<RelearnTypes::position_type, typename OctreeNode<AdditionalCellAttributes>::counter_type>>{};

    while (!stack.empty()) {
        auto* current = stack.top();
        stack.pop();

        if (current->is_leaf()) {
            if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                const auto vacant_elements = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_excitatory_dendrites(vacant_elements);

                if (vacant_elements != 0) {
                    excitatory_dendrites.emplace_back(current->get_cell().get_neuron_position().value(), vacant_elements);
                }
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                const auto vacant_elements = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_inhibitory_dendrites(vacant_elements);

                if (vacant_elements != 0) {
                    inhibitory_dendrites.emplace_back(current->get_cell().get_neuron_position().value(), vacant_elements);
                }
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                const auto vacant_elements = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_excitatory_axons(vacant_elements);

                if (vacant_elements != 0) {
                    excitatory_axons.emplace_back(current->get_cell().get_neuron_position().value(), vacant_elements);
                }
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                const auto vacant_elements = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
                current->set_cell_number_inhibitory_axons(vacant_elements);

                if (vacant_elements != 0) {
                    inhibitory_axons.emplace_back(current->get_cell().get_neuron_position().value(), vacant_elements);
                }
            }

            continue;
        }

        for (auto* child : current->get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }
    }

    OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&node);

    ranges::sort(excitatory_dendrites);
    ranges::sort(inhibitory_dendrites);
    ranges::sort(excitatory_axons);
    ranges::sort(inhibitory_axons);

    using TT = OctreeNodeExtractor<AdditionalCellAttributes>;

    auto node_cache = NodeCache<AdditionalCellAttributes>{};
    node_cache.set_is_already_downloaded();

    if (!AdditionalCellAttributes::has_excitatory_dendrite) {
        ASSERT_THROW_NO_PRINT(std::ignore = TT::get_all_positions_for(&node, node_cache, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    } else {
        auto nodes = TT::get_all_positions_for(&node, node_cache, ElementType::Dendrite, SignalType::Excitatory);
        ranges::sort(nodes);

        ASSERT_EQ(nodes, excitatory_dendrites);
    }

    if (!AdditionalCellAttributes::has_inhibitory_dendrite) {
        ASSERT_THROW_NO_PRINT(std::ignore = TT::get_all_positions_for(&node, node_cache, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);
    } else {
        auto nodes = TT::get_all_positions_for(&node, node_cache, ElementType::Dendrite, SignalType::Inhibitory);
        ranges::sort(nodes);

        ASSERT_EQ(nodes, inhibitory_dendrites);
    }

    if (!AdditionalCellAttributes::has_excitatory_axon) {
        ASSERT_THROW_NO_PRINT(std::ignore = TT::get_all_positions_for(&node, node_cache, ElementType::Axon, SignalType::Excitatory), RelearnException);
    } else {
        auto nodes = TT::get_all_positions_for(&node, node_cache, ElementType::Axon, SignalType::Excitatory);
        ranges::sort(nodes);

        ASSERT_EQ(nodes, excitatory_axons);
    }

    if (!AdditionalCellAttributes::has_inhibitory_axon) {
        ASSERT_THROW_NO_PRINT(std::ignore = TT::get_all_positions_for(&node, node_cache, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    } else {
        auto nodes = TT::get_all_positions_for(&node, node_cache, ElementType::Axon, SignalType::Inhibitory);
        ranges::sort(nodes);

        ASSERT_EQ(nodes, inhibitory_axons);
    }
}

TYPED_TEST(OctreeNodeTest, testPrint) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    const auto node = OctreeNode<AdditionalCellAttributes>{};

    auto ss = std::stringstream{};

    ASSERT_NO_THROW(ss << node);
}

TYPED_TEST(OctreeNodeTest, testSetNumberElements) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);

    auto node = OctreeNode<AdditionalCellAttributes>{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    const auto& cell = node.get_cell();

    if constexpr (AdditionalCellAttributes::has_excitatory_dendrite && AdditionalCellAttributes::has_inhibitory_dendrite) {
        const auto vacant_excitatory = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
        const auto vacant_inhibitry = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);

        node.set_cell_number_dendrites(vacant_excitatory, vacant_inhibitry);
        node.set_cell_number_dendrites(0, 0);
        node.set_cell_number_dendrites(3, 3);
        node.set_cell_number_dendrites(vacant_excitatory, vacant_inhibitry);

        ASSERT_EQ(cell.get_number_excitatory_dendrites(), vacant_excitatory);
        ASSERT_EQ(cell.get_number_inhibitory_dendrites(), vacant_inhibitry);
    }

    if constexpr (AdditionalCellAttributes::has_excitatory_axon && AdditionalCellAttributes::has_inhibitory_axon) {
        const auto vacant_excitatory = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);
        const auto vacant_inhibitry = RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 10, this->mt);

        node.set_cell_number_axons(vacant_excitatory, vacant_inhibitry);
        node.set_cell_number_axons(0, 0);
        node.set_cell_number_axons(3, 3);
        node.set_cell_number_axons(vacant_excitatory, vacant_inhibitry);

        ASSERT_EQ(cell.get_number_excitatory_axons(), vacant_excitatory);
        ASSERT_EQ(cell.get_number_inhibitory_axons(), vacant_inhibitry);
    }
}
