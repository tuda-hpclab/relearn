/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_octree.h"

#include "Types.h"

#include "algorithm/Algorithms.h"
#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/BarnesHutInternal/BarnesHutInvertedCell.h"
#include "algorithm/Cells.h"
#include "algorithm/FMMInternal/FastMultipoleMethodCell.h"
#include "algorithm/Internal/octree/Octree.h"
#include "algorithm/NaiveInternal/NaiveCell.h"
#include "gtest/gtest.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "structure/Morton.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include "cpp-utility/ranges/Functional.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "adapter/octree/OctreeAdapter.h"

#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neurons/neurons_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <gtest/gtest.h>

#include <range/v3/algorithm/sort.hpp>
#include <range/v3/view/map.hpp>

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stack>
#include <utility>
#include <vector>

using test_types = ::testing::Types<BarnesHutCell, BarnesHutInvertedCell, FastMultipoleMethodCell, NaiveCell>;
TYPED_TEST_SUITE(OctreeTest, test_types);

TYPED_TEST(OctreeTest, testConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationFactory::get_small_refinement_level(this->mt);
    const auto morton = std::make_shared<Morton>(level_of_branch_nodes);

    auto octree = Octree<TypeParam>({ min, max }, morton);

    ASSERT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);

    const auto& [retrieved_min, retrieved_max] = octree.get_simulation_box();
    ASSERT_EQ(retrieved_min, min);
    ASSERT_EQ(retrieved_max, max);

    const auto virtual_neurons = OctreeAdapter::extract_virtual_neurons(octree.get_root());

    auto level_to_count = std::map<std::size_t, std::size_t>{};

    for (const auto& id : virtual_neurons | ranges::views::values) {
        level_to_count[id]++;
    }

    ASSERT_EQ(level_to_count.size(), level_of_branch_nodes + 1);

    for (auto level = 0U; level <= level_of_branch_nodes; level++) {
        const auto expected_elements = std::pow(8U, level);

        if (level == level_of_branch_nodes) {
            ASSERT_EQ(octree.get_num_local_trees(), expected_elements);
        }

        ASSERT_EQ(level_to_count[level], expected_elements);
    }
}

TYPED_TEST(OctreeTest, testConstructorExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationFactory::get_small_refinement_level(this->mt);
    const auto morton = std::make_shared<Morton>(level_of_branch_nodes);

    ASSERT_THROW_NO_PRINT(const Octree<TypeParam> octree({ max, min }, morton), RelearnException);
}

TYPED_TEST(OctreeTest, testInsertNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationFactory::get_small_refinement_level(this->mt);
    const auto morton = std::make_shared<Morton>(level_of_branch_nodes);

    auto octree = Octree<TypeParam>({ min, max }, morton);

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(this->mt);

    auto neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, this->mt);

    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    auto placed_neurons = OctreeAdapter::template extract_neurons_tree<TypeParam>(octree);

    ASSERT_EQ(neurons_to_place.size(), placed_neurons.size());

    ranges::sort(neurons_to_place, std::greater{}, utility::element<1>);
    ranges::sort(placed_neurons, std::greater{}, utility::element<1>);

    for (auto i = 0U; i < neurons_to_place.size(); i++) {
        const auto& expected_neuron = neurons_to_place[i];
        const auto& found_neuron = placed_neurons[i];

        ASSERT_EQ(expected_neuron, found_neuron);
    }
}

TYPED_TEST(OctreeTest, testInsertNeuronsExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationFactory::get_small_refinement_level(this->mt);
    const auto morton = std::make_shared<Morton>(level_of_branch_nodes);

    auto octree = Octree<TypeParam>({ min, max }, morton);

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, this->mt);

    for (const auto& [position, id] : neurons_to_place) {
        const auto pos_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const auto pos_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const auto pos_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        const auto pos_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const auto pos_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const auto pos_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        ASSERT_THROW_NO_PRINT(octree.insert(position, NeuronID::uninitialized_id()), RelearnException);

        ASSERT_THROW_NO_PRINT(octree.insert(pos_invalid_x_max, id), RelearnException);
        ASSERT_THROW_NO_PRINT(octree.insert(pos_invalid_y_max, id), RelearnException);
        ASSERT_THROW_NO_PRINT(octree.insert(pos_invalid_z_max, id), RelearnException);

        ASSERT_THROW_NO_PRINT(octree.insert(pos_invalid_x_min, id), RelearnException);
        ASSERT_THROW_NO_PRINT(octree.insert(pos_invalid_y_min, id), RelearnException);
        ASSERT_THROW_NO_PRINT(octree.insert(pos_invalid_z_min, id), RelearnException);
    }
}

TYPED_TEST(OctreeTest, testStructure) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationFactory::get_small_refinement_level(this->mt);
    const auto morton = std::make_shared<Morton>(level_of_branch_nodes);

    auto octree = Octree<TypeParam>({ min, max }, morton);

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, this->mt);

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();
    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    auto* root = octree.get_root();

    auto octree_nodes = std::stack<std::pair<OctreeNode<AdditionalCellAttributes>*, std::size_t>>{};
    octree_nodes.emplace(root, std::size_t{ 0 });

    while (!octree_nodes.empty()) {
        const auto [current_node, level] = octree_nodes.top();

        octree_nodes.pop();
        ASSERT_EQ(level, current_node->get_level());
        ASSERT_TRUE(current_node->get_mpi_rank() == my_rank);

        if (current_node->is_parent()) {
            const auto& childs = current_node->get_children();
            auto one_child_exists = false;

            for (auto i = 0U; i < 8U; i++) {
                const auto child = childs[i];
                if (child != nullptr) {
                    octree_nodes.emplace(child, level + 1);

                    const auto& subcell_size = child->get_cell().get_size();
                    const auto& expected_subcell_size = current_node->get_cell().get_size_for_octant(static_cast<unsigned char>(i));

                    ASSERT_EQ(expected_subcell_size, subcell_size);

                    one_child_exists = true;
                }
            }

            ASSERT_TRUE(one_child_exists);
        } else {
            const auto& cell = current_node->get_cell();
            const auto& opt_position = cell.get_neuron_position();

            ASSERT_TRUE(opt_position.has_value());

            const auto& position = opt_position.value();

            const auto& cell_size = cell.get_size();
            const auto& [cell_min, cell_max] = cell_size;

            ASSERT_LE(cell_min.get_x(), position.get_x());
            ASSERT_LE(cell_min.get_y(), position.get_y());
            ASSERT_LE(cell_min.get_z(), position.get_z());

            ASSERT_LE(position.get_x(), cell_max.get_x());
            ASSERT_LE(position.get_y(), cell_max.get_y());
            ASSERT_LE(position.get_z(), cell_max.get_z());

            const auto neuron_id = cell.get_neuron_id();

            if (!neuron_id.is_initialized()) {
                ASSERT_LE(neuron_id, NeuronID{ number_neurons });
            }
        }
    }
}

TYPED_TEST(OctreeTest, testMemoryStructure) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationFactory::get_small_refinement_level(this->mt);
    const auto morton = std::make_shared<Morton>(level_of_branch_nodes);

    auto octree = Octree<TypeParam>({ min, max }, morton);

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, this->mt);

    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    auto octree_nodes = std::stack<OctreeNode<AdditionalCellAttributes>*>{};

    while (!octree_nodes.empty()) {
        const auto* current_node = octree_nodes.top();
        octree_nodes.pop();

        if (current_node->is_leaf()) {
            continue;
        }

        const auto& children = current_node->get_children();

        OctreeNode<AdditionalCellAttributes>* child_pointer = nullptr;
        int child_id = -1;

        for (auto i = 0U; i < 8U; i++) {
            const auto child = children[i];
            if (child == nullptr) {
                continue;
            }

            octree_nodes.emplace(child);

            if (child_pointer == nullptr) {
                child_pointer = child;
                child_id = static_cast<int>(i);
            }

            auto ptr = child_pointer + i - child_id;
            ASSERT_EQ(ptr, child);
            ASSERT_EQ(child, current_node->get_child(i));
        }
    }
}

TYPED_TEST(OctreeTest, testMemoryFootprint) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationFactory::get_small_refinement_level(this->mt);
    const auto morton = std::make_shared<Morton>(level_of_branch_nodes);

    auto octree = Octree<TypeParam>({ min, max }, morton);

    auto footprint = std::make_unique<utility::MemoryFootprint>(10);

    octree.record_memory_footprint(footprint);

    const auto& footprint_description = footprint->get_descriptions();
    ASSERT_EQ(footprint_description.size(), 2);

    ASSERT_TRUE(footprint_description.contains("Octree"));
    ASSERT_GE(footprint_description.at("Octree"), sizeof(Octree<AdditionalCellAttributes>));

    ASSERT_TRUE(footprint_description.contains("OctreeNode"));
    ASSERT_GE(footprint_description.at("OctreeNode"), sizeof(OctreeNode<AdditionalCellAttributes>));
}

TYPED_TEST(OctreeTest, testBranchNodes) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationFactory::get_small_refinement_level(this->mt);
    const auto morton = std::make_shared<Morton>(level_of_branch_nodes);

    auto octree = Octree<TypeParam>({ min, max }, morton);

    auto expected_number_elements = 1U;
    for (auto i = 0U; i < level_of_branch_nodes; i++) {
        expected_number_elements *= 8U;
    }

    const auto& branch_nodes = octree.get_local_branch_nodes();
    ASSERT_EQ(branch_nodes.size(), expected_number_elements);

    const auto& const_octree = octree;
    const auto& const_branch_nodes = const_octree.get_local_branch_nodes();
    ASSERT_EQ(const_branch_nodes.size(), expected_number_elements);

    for (auto i = 0U; i < expected_number_elements; i++) {
        ASSERT_EQ(branch_nodes[i], const_branch_nodes[i]);

        auto* branch_node_ptr = branch_nodes[i];
        ASSERT_NE(branch_node_ptr, nullptr);

        ASSERT_EQ(branch_node_ptr->get_level(), level_of_branch_nodes);
        ASSERT_EQ(octree.get_branch_node_pointer(i), branch_node_ptr);
    }

    for (auto i = expected_number_elements; i < 4 * expected_number_elements; i++) {
        ASSERT_THROW_NO_PRINT(std::ignore = octree.get_branch_node_pointer(i), RelearnException);
    }
}

TYPED_TEST(OctreeTest, testBranchNodesPatheticNonlocal) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationFactory::get_small_refinement_level(this->mt);
    const auto morton = std::make_shared<Morton>(level_of_branch_nodes);

    auto octree = Octree<TypeParam>({ min, max }, morton);

    auto local_branch_nodes_expected = std::vector<OctreeNode<AdditionalCellAttributes>*>{};

    const auto& branch_nodes = octree.get_local_branch_nodes();
    for (auto* branch_node : branch_nodes) {
        const auto is_actual_id = RandomFactory::get_random_bool(this->mt);
        if (is_actual_id) {
            local_branch_nodes_expected.push_back(branch_node);
        } else {
            branch_node->set_rank(mpiPP::MPIRank(2));
        }
    }

    const auto& now_branch_nodes = octree.get_local_branch_nodes();
    ASSERT_EQ(now_branch_nodes.size(), local_branch_nodes_expected.size());
    for (auto i = 0U; i < now_branch_nodes.size(); i++) {
        ASSERT_EQ(now_branch_nodes[i], local_branch_nodes_expected[i]);
    }

    const auto& const_octree = octree;

    const auto& now_const_branch_nodes = const_octree.get_local_branch_nodes();
    ASSERT_EQ(now_const_branch_nodes.size(), local_branch_nodes_expected.size());
    for (auto i = 0U; i < now_const_branch_nodes.size(); i++) {
        ASSERT_EQ(now_const_branch_nodes[i], local_branch_nodes_expected[i]);
    }
}

TYPED_TEST(OctreeTest, testLeafNodes) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationFactory::get_small_refinement_level(this->mt);
    const auto morton = std::make_shared<Morton>(level_of_branch_nodes);

    auto octree = Octree<TypeParam>({ min, max }, morton);

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, this->mt);

    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    octree.initializes_leaf_nodes(number_neurons);

    const auto& leaf_nodes = octree.get_leaf_nodes();

    ASSERT_EQ(leaf_nodes.size(), number_neurons);

    for (const auto& [position, id] : neurons_to_place) {
        auto id_id = id.get_neuron_id();
        auto* leaf_node = leaf_nodes[id_id];

        ASSERT_NE(leaf_node, nullptr);
        ASSERT_NO_THROW(std::ignore = leaf_node->get_cell().get_neuron_position());
        ASSERT_TRUE(leaf_node->get_cell().get_neuron_position().has_value());
        ASSERT_EQ(leaf_node->get_cell().get_neuron_position().value(), position);
    }
}

TYPED_TEST(OctreeTest, testLeafNodesException1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationFactory::get_small_refinement_level(this->mt);
    const auto morton = std::make_shared<Morton>(level_of_branch_nodes);

    auto octree = Octree<TypeParam>({ min, max }, morton);

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(this->mt);

    auto neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, this->mt);

    for (auto i = 0U; i < 9; i++) {
        const auto bad_id = RandomFactory::get_random_integer<std::size_t>(std::size_t{ 0 }, number_neurons - 1, this->mt);
        auto& [pos, id] = neurons_to_place[bad_id];
        id = NeuronID(number_neurons + bad_id);
    }

    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    ASSERT_THROW_NO_PRINT(octree.initializes_leaf_nodes(number_neurons), RelearnException);
}

TYPED_TEST(OctreeTest, testLeafNodesException2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationFactory::get_small_refinement_level(this->mt);
    const auto morton = std::make_shared<Morton>(level_of_branch_nodes);

    auto octree = Octree<TypeParam>({ min, max }, morton);

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(this->mt) + 20;

    auto neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, this->mt);

    const auto random_ids = RandomFactory::sample_from_integer_range<std::size_t>(0, number_neurons - 1, 10, this->mt);
    for (auto bad_id : random_ids) {
        if (bad_id == number_neurons - bad_id) {
            bad_id--;
        }
        auto& [pos, id] = neurons_to_place[bad_id];
        id = NeuronID(number_neurons - bad_id);
    }

    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    ASSERT_THROW_NO_PRINT(octree.initializes_leaf_nodes(number_neurons), RelearnException);
}
