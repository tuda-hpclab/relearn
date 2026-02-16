/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_node_cache.h"

#include "Config.h"

#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/Internal/octree/NodeCache.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "util/MemoryHolder.h"
#include "util/RelearnException.h"

#include "mpi-wrapper/MPICommunication.h"
#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "adapter/octree/OctreeAdapter.h"

#include "factory/memory_holder/memory_holder_factory.h"
#include "factory/octree/octree_factory.h"

#include <span>

using test_types = ::testing::Types<BarnesHutCell>;
TYPED_TEST_SUITE(NodeCacheTest, test_types);

TYPED_TEST(NodeCacheTest, testGetChildrenThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    auto node_cache = NodeCache<AdditionalCellAttributes>{};
    node_cache.set_is_already_downloaded();

    auto span = std::span{ this->rma_window->get_pointer(), Constants::mpi_alloc_mem };

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(span);

    const auto min = Vec3d{ 0.0, 0.0, 0.0 };
    const auto max = Vec3d{ 1.0, 1.0, 1.0 };

    auto tree = OctreeFactory::get_standard_tree<AdditionalCellAttributes>(100, memory_holder, min, max, this->mt);

    auto leaf_nodes = OctreeAdapter::extract_leaf_nodes(&tree);

    ASSERT_THROW_NO_PRINT(std::ignore = node_cache.get_children(nullptr), RelearnException);

    for (auto* node : leaf_nodes) {
        ASSERT_THROW_NO_PRINT(std::ignore = node_cache.get_children(node), RelearnException);
    }

    ASSERT_EQ(node_cache.get_memory_size(), 0);
    ASSERT_EQ(node_cache.get_cache_size(), 0);

    node_cache.clear();

    ASSERT_EQ(node_cache.get_memory_size(), 0);
    ASSERT_EQ(node_cache.get_cache_size(), 0);
}

TYPED_TEST(NodeCacheTest, testGetChildren) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    auto node_cache = NodeCache<AdditionalCellAttributes>{};
    node_cache.set_is_already_downloaded();

    auto span = std::span{ this->rma_window->get_pointer(), Constants::mpi_alloc_mem };

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(span);

    const auto min = Vec3d{ 0.0, 0.0, 0.0 };
    const auto max = Vec3d{ 1.0, 1.0, 1.0 };

    auto tree = OctreeFactory::get_standard_tree<AdditionalCellAttributes>(100, memory_holder, min, max, this->mt);

    auto inner_nodes = OctreeAdapter::extract_inner_nodes(&tree);

    for (auto* node : inner_nodes) {
        auto copy_of_node_ptr = std::array<OctreeNode<AdditionalCellAttributes>*, Constants::number_oct>{};
        copy_of_node_ptr = node->get_children();

        auto copy_of_nodes = std::array<OctreeNode<AdditionalCellAttributes>, Constants::number_oct>{};
        for (auto i = 0U; i < Constants::number_oct; ++i) {
            if (copy_of_node_ptr[i] == nullptr) {
                continue;
            }

            copy_of_nodes[i] = *copy_of_node_ptr[i];
        }

        auto downloaded_nodes = node_cache.get_children(node);

        ASSERT_EQ(downloaded_nodes, copy_of_node_ptr);

        for (auto i = 0U; i < Constants::number_oct; ++i) {
            if (copy_of_node_ptr[i] == nullptr) {
                continue;
            }

            ASSERT_EQ(downloaded_nodes[i]->get_mpi_rank(), copy_of_nodes[i].get_mpi_rank());
            ASSERT_EQ(downloaded_nodes[i]->is_parent(), copy_of_nodes[i].is_parent());
            ASSERT_EQ(downloaded_nodes[i]->is_leaf(), copy_of_nodes[i].is_leaf());
            ASSERT_EQ(downloaded_nodes[i]->get_children(), copy_of_nodes[i].get_children());
            ASSERT_EQ(downloaded_nodes[i]->is_actual_id(), copy_of_nodes[i].is_actual_id());
            ASSERT_EQ(downloaded_nodes[i]->is_actual_id(), true);
            ASSERT_EQ(downloaded_nodes[i]->get_level(), copy_of_nodes[i].get_level());

            const auto& downloaded_cell = downloaded_nodes[i]->get_cell();
            const auto& copied_cell = copy_of_nodes[i].get_cell();

            ASSERT_EQ(downloaded_cell.get_neuron_id(), copied_cell.get_neuron_id());
            ASSERT_EQ(downloaded_cell.get_size(), copied_cell.get_size());

            if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                ASSERT_EQ(downloaded_cell.get_number_excitatory_dendrites(), copied_cell.get_number_excitatory_dendrites());
                ASSERT_EQ(downloaded_cell.get_excitatory_dendrites_position(), copied_cell.get_excitatory_dendrites_position());
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                ASSERT_EQ(downloaded_cell.get_number_inhibitory_dendrites(), copied_cell.get_number_inhibitory_dendrites());
                ASSERT_EQ(downloaded_cell.get_inhibitory_dendrites_position(), copied_cell.get_inhibitory_dendrites_position());
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                ASSERT_EQ(downloaded_cell.get_number_excitatory_axons(), copied_cell.get_number_excitatory_axons());
                ASSERT_EQ(downloaded_cell.get_excitatory_axons_position(), copied_cell.get_excitatory_axons_position());
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                ASSERT_EQ(downloaded_cell.get_number_inhibitory_axons(), copied_cell.get_number_inhibitory_axons());
                ASSERT_EQ(downloaded_cell.get_inhibitory_axons_position(), copied_cell.get_inhibitory_axons_position());
            }
        }
    }

    ASSERT_EQ(node_cache.get_memory_size(), 0);
    ASSERT_EQ(node_cache.get_cache_size(), 0);

    node_cache.clear();

    ASSERT_EQ(node_cache.get_memory_size(), 0);
    ASSERT_EQ(node_cache.get_cache_size(), 0);
}

TYPED_TEST(NodeCacheTest, testGetOwnChildrenTwoRanks) {
    if (mpiPP::MPIInfo::get_number_ranks() != 2) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 2 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    auto node_cache = NodeCache<AdditionalCellAttributes>{};
    node_cache.set_is_already_downloaded();

    auto span = std::span{ this->rma_window->get_pointer(), Constants::mpi_alloc_mem };

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(span);

    const auto min = Vec3d{ 0.0, 0.0, 0.0 };
    const auto max = Vec3d{ 1.0, 1.0, 1.0 };

    auto tree = OctreeFactory::get_standard_tree<AdditionalCellAttributes>(100, memory_holder, min, max, this->mt);

    auto inner_nodes = OctreeAdapter::extract_inner_nodes(&tree);

    for (auto* node : inner_nodes) {
        auto copy_of_node_ptr = std::array<OctreeNode<AdditionalCellAttributes>*, Constants::number_oct>{};
        copy_of_node_ptr = node->get_children();

        auto copy_of_nodes = std::array<OctreeNode<AdditionalCellAttributes>, Constants::number_oct>{};
        for (auto i = 0U; i < Constants::number_oct; ++i) {
            if (copy_of_node_ptr[i] == nullptr) {
                continue;
            }

            copy_of_nodes[i] = *copy_of_node_ptr[i];
        }

        auto downloaded_nodes = node_cache.get_children(node);

        ASSERT_EQ(downloaded_nodes, copy_of_node_ptr);

        for (auto i = 0U; i < Constants::number_oct; ++i) {
            if (copy_of_node_ptr[i] == nullptr) {
                continue;
            }

            ASSERT_EQ(downloaded_nodes[i]->get_mpi_rank(), copy_of_nodes[i].get_mpi_rank());
            ASSERT_EQ(downloaded_nodes[i]->is_parent(), copy_of_nodes[i].is_parent());
            ASSERT_EQ(downloaded_nodes[i]->is_leaf(), copy_of_nodes[i].is_leaf());
            ASSERT_EQ(downloaded_nodes[i]->get_children(), copy_of_nodes[i].get_children());
            ASSERT_EQ(downloaded_nodes[i]->is_actual_id(), copy_of_nodes[i].is_actual_id());
            ASSERT_EQ(downloaded_nodes[i]->is_actual_id(), true);
            ASSERT_EQ(downloaded_nodes[i]->get_level(), copy_of_nodes[i].get_level());

            const auto& downloaded_cell = downloaded_nodes[i]->get_cell();
            const auto& copied_cell = copy_of_nodes[i].get_cell();

            ASSERT_EQ(downloaded_cell.get_neuron_id(), copied_cell.get_neuron_id());
            ASSERT_EQ(downloaded_cell.get_size(), copied_cell.get_size());

            if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                ASSERT_EQ(downloaded_cell.get_number_excitatory_dendrites(), copied_cell.get_number_excitatory_dendrites());
                ASSERT_EQ(downloaded_cell.get_excitatory_dendrites_position(), copied_cell.get_excitatory_dendrites_position());
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                ASSERT_EQ(downloaded_cell.get_number_inhibitory_dendrites(), copied_cell.get_number_inhibitory_dendrites());
                ASSERT_EQ(downloaded_cell.get_inhibitory_dendrites_position(), copied_cell.get_inhibitory_dendrites_position());
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                ASSERT_EQ(downloaded_cell.get_number_excitatory_axons(), copied_cell.get_number_excitatory_axons());
                ASSERT_EQ(downloaded_cell.get_excitatory_axons_position(), copied_cell.get_excitatory_axons_position());
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                ASSERT_EQ(downloaded_cell.get_number_inhibitory_axons(), copied_cell.get_number_inhibitory_axons());
                ASSERT_EQ(downloaded_cell.get_inhibitory_axons_position(), copied_cell.get_inhibitory_axons_position());
            }
        }
    }

    ASSERT_EQ(node_cache.get_memory_size(), 0);
    ASSERT_EQ(node_cache.get_cache_size(), 0);

    node_cache.clear();

    ASSERT_EQ(node_cache.get_memory_size(), 0);
    ASSERT_EQ(node_cache.get_cache_size(), 0);
}

TYPED_TEST(NodeCacheTest, testGetForeignChildrenTwoRanks) {
    if (mpiPP::MPIInfo::get_number_ranks() != 2) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 2 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    auto node_cache = NodeCache<AdditionalCellAttributes>{};
    node_cache.set_is_already_downloaded();

    auto [memory_holder, cells] = MemoryHolderFactory::get_memory_holder<AdditionalCellAttributes>();

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();
    const auto other_rank = my_rank == mpiPP::MPIRank::root_rank() ? mpiPP::MPIRank(1) : mpiPP::MPIRank(0);
    const auto am_i_root = my_rank == mpiPP::MPIRank::root_rank();

    const auto min = Vec3d{ 0.0, 0.0, 0.0 };
    const auto max = Vec3d{ 1.0, 1.0, 1.0 };

    auto tree = OctreeFactory::get_standard_tree<AdditionalCellAttributes>(2, memory_holder, min, max, this->mt);
    auto other_tree = OctreeNode<AdditionalCellAttributes>{};

    // Exchange tree roots
    if (am_i_root) {
        other_tree = mpiPP::MPICommunication::receive<OctreeNode<AdditionalCellAttributes>>(other_rank);
        mpiPP::MPICommunication::send(tree, other_rank);
    } else {
        mpiPP::MPICommunication::send(tree, other_rank);
        other_tree = mpiPP::MPICommunication::receive<OctreeNode<AdditionalCellAttributes>>(other_rank);
    }

    if (am_i_root) {
        auto flags = std::array<bool, Constants::number_oct>{};
        for (auto i = 0U; i < Constants::number_oct; ++i) {
            flags[i] = other_tree.get_child(i) != nullptr;
        }

        auto children = std::array<OctreeNode<AdditionalCellAttributes>, Constants::number_oct>{};
        for (auto i = 0U; i < Constants::number_oct; ++i) {
            if (flags[i]) {
                children[i] = mpiPP::MPICommunication::receive<OctreeNode<AdditionalCellAttributes>>(other_rank);
            }
        }

        auto rma_children = node_cache.get_children(&other_tree);

        for (auto i = 0U; i < Constants::number_oct; ++i) {
            if (rma_children[i] == nullptr) {
                ASSERT_FALSE(flags[i]);
                continue;
            }
            ASSERT_TRUE(flags[i]);

            ASSERT_EQ(rma_children[i]->get_mpi_rank(), children[i].get_mpi_rank());
            ASSERT_EQ(rma_children[i]->is_parent(), children[i].is_parent());
            ASSERT_EQ(rma_children[i]->is_leaf(), children[i].is_leaf());
            ASSERT_EQ(rma_children[i]->get_children(), children[i].get_children());
            ASSERT_EQ(rma_children[i]->is_actual_id(), children[i].is_actual_id());
        }

    } else {
        auto flags = std::array<bool, Constants::number_oct>{};
        for (auto i = 0U; i < Constants::number_oct; ++i) {
            flags[i] = tree.get_child(i) != nullptr;
        }

        for (auto i = 0U; i < Constants::number_oct; ++i) {
            if (flags[i]) {
                mpiPP::MPICommunication::send<OctreeNode<AdditionalCellAttributes>>((*(tree.get_child(i))), other_rank);
            }
        }
    }

    const auto check_node = [other_rank, &node_cache](auto* node) {
        if (!node->is_actual_id()) {
            auto flags = std::array<bool, Constants::number_oct>{};
            for (auto i = 0U; i < Constants::number_oct; ++i) {
                flags[i] = node->get_child(i) != nullptr;
            }

            auto children = std::array<OctreeNode<AdditionalCellAttributes>, Constants::number_oct>{};
            for (auto i = 0U; i < Constants::number_oct; ++i) {
                if (flags[i]) {
                    children[i] = mpiPP::MPICommunication::receive<OctreeNode<AdditionalCellAttributes>>(other_rank);
                }
            }

            auto rma_children = node_cache.get_children(node);

            for (auto i = 0U; i < Constants::number_oct; ++i) {
                if (rma_children[i] == nullptr) {
                    EXPECT_FALSE(flags[i]);
                    continue;
                }
                EXPECT_TRUE(flags[i]);

                EXPECT_EQ(rma_children[i]->get_mpi_rank(), children[i].get_mpi_rank());
                EXPECT_EQ(rma_children[i]->is_parent(), children[i].is_parent());
                EXPECT_EQ(rma_children[i]->is_leaf(), children[i].is_leaf());
                EXPECT_EQ(rma_children[i]->get_children(), children[i].get_children());
                EXPECT_EQ(rma_children[i]->is_actual_id(), children[i].is_actual_id());
            }

            return rma_children;
        } else {
            auto flags = std::array<bool, Constants::number_oct>{};
            for (auto i = 0U; i < Constants::number_oct; ++i) {
                flags[i] = node->get_child(i) != nullptr;
            }

            for (auto i = 0U; i < Constants::number_oct; ++i) {
                if (flags[i]) {
                    mpiPP::MPICommunication::send<OctreeNode<AdditionalCellAttributes>>((*(node->get_child(i))), other_rank);
                }
            }

            return node->get_children();
        }
    };

    const auto check_tree = [check_node](auto* node) {
        auto stack = std::stack<OctreeNode<AdditionalCellAttributes>*>{};
        stack.push(node);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto children = check_node(current);

            for (auto* child : children) {
                if (child != nullptr && !child->is_leaf()) {
                    stack.push(child);
                }
            }
        }
    };

    check_tree(&tree);

    // check_tree(&other_tree);

    // ASSERT_EQ(node_cache.get_memory_size(), 0);
    // ASSERT_EQ(node_cache.get_cache_size(), 0);

    node_cache.clear();

    ASSERT_EQ(node_cache.get_memory_size(), 0);
    ASSERT_EQ(node_cache.get_cache_size(), 0);

    mpiPP::MPISynchronization::barrier();
}

TYPED_TEST(NodeCacheTest, testGetForeignChildrenTwoRanksManual1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 2) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 2 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    auto node_cache = NodeCache<AdditionalCellAttributes>{};
    node_cache.set_is_already_downloaded();

    // This is for clang, which cannot capture a structured binding
    auto tuple = MemoryHolderFactory::get_memory_holder<AdditionalCellAttributes>();
    auto& memory_holder = std::get<0>(tuple);

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();
    const auto other_rank = my_rank == mpiPP::MPIRank::root_rank() ? mpiPP::MPIRank(1) : mpiPP::MPIRank(0);
    const auto am_i_root = my_rank == mpiPP::MPIRank::root_rank();

    const auto get_manual_tree = [my_rank, &memory_holder]() {
        auto root = OctreeNode<AdditionalCellAttributes>{};
        root.set_level(0);
        root.set_rank(my_rank);

        root.set_cell_neuron_id(NeuronID(0));
        root.set_cell_size(Vec3d{ 0.0, 0.0, 0.0 }, Vec3d{ 1.0, 1.0, 1.0 });
        root.set_cell_neuron_position(Vec3d{ 0.25, 0.35, 0.45 });

        root.insert(Vec3d{ 0.25, 0.77, 0.45 }, NeuronID{ 1 }, memory_holder);

        auto stack = std::stack<OctreeNode<AdditionalCellAttributes>*>{};
        stack.push(&root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                current->set_cell_number_excitatory_dendrites(2);
                current->set_cell_number_inhibitory_dendrites(1);

                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&root);

        return root;
    };

    auto tree = get_manual_tree();
    auto other_tree = OctreeNode<AdditionalCellAttributes>{};

    // Exchange tree roots
    if (am_i_root) {
        other_tree = mpiPP::MPICommunication::receive<OctreeNode<AdditionalCellAttributes>>(other_rank);
        mpiPP::MPICommunication::send(tree, other_rank);
    } else {
        mpiPP::MPICommunication::send(tree, other_rank);
        other_tree = mpiPP::MPICommunication::receive<OctreeNode<AdditionalCellAttributes>>(other_rank);
    }

    const auto children = tree.get_children();
    const auto other_children = node_cache.get_children(&other_tree);

    for (auto i = 0U; i < Constants::number_oct; i++) {
        if (children[i] == nullptr) {
            ASSERT_EQ(other_children[i], nullptr);
            continue;
        }

        ASSERT_NE(other_children[i], nullptr);

        ASSERT_EQ(other_rank, other_children[i]->get_mpi_rank());
        ASSERT_EQ(children[i]->is_parent(), other_children[i]->is_parent());
        ASSERT_EQ(children[i]->is_leaf(), other_children[i]->is_leaf());

        const auto& child_cell = children[i]->get_cell();
        const auto& other_child_cell = other_children[i]->get_cell();

        ASSERT_EQ(child_cell.get_neuron_id(), other_child_cell.get_neuron_id());
        ASSERT_EQ(child_cell.get_size(), other_child_cell.get_size());
        ASSERT_EQ(child_cell.get_number_excitatory_dendrites(), other_child_cell.get_number_excitatory_dendrites());
        ASSERT_EQ(child_cell.get_number_inhibitory_dendrites(), other_child_cell.get_number_inhibitory_dendrites());
        ASSERT_EQ(child_cell.get_excitatory_dendrites_position(), other_child_cell.get_excitatory_dendrites_position());
        ASSERT_EQ(child_cell.get_inhibitory_dendrites_position(), other_child_cell.get_inhibitory_dendrites_position());
    }

    node_cache.clear();

    ASSERT_EQ(node_cache.get_memory_size(), 0);
    ASSERT_EQ(node_cache.get_cache_size(), 0);

    mpiPP::MPISynchronization::barrier();
}

TYPED_TEST(NodeCacheTest, testGetForeignChildrenTwoRanksManual2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 2) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 2 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    auto node_cache = NodeCache<AdditionalCellAttributes>{};
    node_cache.set_is_already_downloaded();

    // This is for clang, which cannot capture a structured binding
    auto tuple = MemoryHolderFactory::get_memory_holder<AdditionalCellAttributes>();
    auto& memory_holder = std::get<0>(tuple);

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();
    const auto other_rank = my_rank == mpiPP::MPIRank::root_rank() ? mpiPP::MPIRank(1) : mpiPP::MPIRank(0);
    const auto am_i_root = my_rank == mpiPP::MPIRank::root_rank();

    const auto get_manual_tree = [my_rank, &memory_holder]() {
        auto root = OctreeNode<AdditionalCellAttributes>{};
        root.set_level(0);
        root.set_rank(my_rank);

        root.set_cell_neuron_id(NeuronID(0));
        root.set_cell_size(Vec3d{ 0.0, 0.0, 0.0 }, Vec3d{ 1.0, 1.0, 1.0 });
        root.set_cell_neuron_position(Vec3d{ 0.25, 0.35, 0.45 });

        root.insert(Vec3d{ 0.25, 0.77, 0.45 }, NeuronID{ 1 }, memory_holder);
        root.insert(Vec3d{ 0.25, 0.35, 0.87 }, NeuronID{ 2 }, memory_holder);
        root.insert(Vec3d{ 0.25, 0.77, 0.87 }, NeuronID{ 3 }, memory_holder);
        root.insert(Vec3d{ 0.67, 0.35, 0.45 }, NeuronID{ 4 }, memory_holder);
        root.insert(Vec3d{ 0.67, 0.77, 0.45 }, NeuronID{ 5 }, memory_holder);
        root.insert(Vec3d{ 0.67, 0.35, 0.87 }, NeuronID{ 7 }, memory_holder);
        root.insert(Vec3d{ 0.67, 0.77, 0.87 }, NeuronID{ 6 }, memory_holder);

        auto stack = std::stack<OctreeNode<AdditionalCellAttributes>*>{};
        stack.push(&root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                current->set_cell_number_excitatory_dendrites(2);
                current->set_cell_number_inhibitory_dendrites(1);

                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&root);

        return root;
    };

    auto tree = get_manual_tree();
    auto other_tree = OctreeNode<AdditionalCellAttributes>{};

    // Exchange tree roots
    if (am_i_root) {
        other_tree = mpiPP::MPICommunication::receive<OctreeNode<AdditionalCellAttributes>>(other_rank);
        mpiPP::MPICommunication::send(tree, other_rank);
    } else {
        mpiPP::MPICommunication::send(tree, other_rank);
        other_tree = mpiPP::MPICommunication::receive<OctreeNode<AdditionalCellAttributes>>(other_rank);
    }

    const auto children = tree.get_children();
    const auto other_children = node_cache.get_children(&other_tree);

    for (auto i = 0U; i < Constants::number_oct; i++) {
        if (children[i] == nullptr) {
            ASSERT_EQ(other_children[i], nullptr);
            continue;
        }

        ASSERT_NE(other_children[i], nullptr);

        ASSERT_EQ(other_rank, other_children[i]->get_mpi_rank());
        ASSERT_EQ(children[i]->is_parent(), other_children[i]->is_parent());
        ASSERT_EQ(children[i]->is_leaf(), other_children[i]->is_leaf());

        const auto& child_cell = children[i]->get_cell();
        const auto& other_child_cell = other_children[i]->get_cell();

        ASSERT_EQ(child_cell.get_neuron_id(), other_child_cell.get_neuron_id());
        ASSERT_EQ(child_cell.get_size(), other_child_cell.get_size());
        ASSERT_EQ(child_cell.get_number_excitatory_dendrites(), other_child_cell.get_number_excitatory_dendrites());
        ASSERT_EQ(child_cell.get_number_inhibitory_dendrites(), other_child_cell.get_number_inhibitory_dendrites());
        ASSERT_EQ(child_cell.get_excitatory_dendrites_position(), other_child_cell.get_excitatory_dendrites_position());
        ASSERT_EQ(child_cell.get_inhibitory_dendrites_position(), other_child_cell.get_inhibitory_dendrites_position());
    }

    node_cache.clear();

    ASSERT_EQ(node_cache.get_memory_size(), 0);
    ASSERT_EQ(node_cache.get_cache_size(), 0);

    mpiPP::MPISynchronization::barrier();
}

TYPED_TEST(NodeCacheTest, testGetForeignChildrenTwoRanksManual3) {
    if (mpiPP::MPIInfo::get_number_ranks() != 2) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 2 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    auto node_cache = NodeCache<AdditionalCellAttributes>{};
    node_cache.set_is_already_downloaded();

    // This is for clang, which cannot capture a structured binding
    auto tuple = MemoryHolderFactory::get_memory_holder<AdditionalCellAttributes>();
    auto& memory_holder = std::get<0>(tuple);

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();
    const auto other_rank = my_rank == mpiPP::MPIRank::root_rank() ? mpiPP::MPIRank(1) : mpiPP::MPIRank(0);
    const auto am_i_root = my_rank == mpiPP::MPIRank::root_rank();

    const auto get_manual_tree = [my_rank, &memory_holder]() {
        auto root = OctreeNode<AdditionalCellAttributes>{};
        root.set_level(0);
        root.set_rank(my_rank);

        root.set_cell_neuron_id(NeuronID(0));
        root.set_cell_size(Vec3d{ 0.0, 0.0, 0.0 }, Vec3d{ 1.0, 1.0, 1.0 });
        root.set_cell_neuron_position(Vec3d{ 0.24, 0.17, 0.09 });

        const auto x_values = { 0.24, 0.42, 0.67, 0.93 };
        const auto y_values = { 0.17, 0.35, 0.54, 0.77 };
        const auto z_values = { 0.09, 0.45, 0.71, 0.87 };

        auto counter = 0U;

        const auto counters_to_skip = { 0U };

        for (const auto& [x, y, z] : ranges::views::zip(x_values, y_values, z_values)) {
            if (std::find(counters_to_skip.begin(), counters_to_skip.end(), counter) != counters_to_skip.end()) {
                counter++;
                continue;
            }

            root.insert(Vec3d{ x, y, z }, NeuronID{ counter }, memory_holder);
            counter++;
        }

        auto stack = std::stack<OctreeNode<AdditionalCellAttributes>*>{};
        stack.push(&root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                current->set_cell_number_excitatory_dendrites(2);
                current->set_cell_number_inhibitory_dendrites(1);

                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&root);

        return root;
    };

    auto tree = get_manual_tree();
    auto other_tree = OctreeNode<AdditionalCellAttributes>{};

    // Exchange tree roots
    if (am_i_root) {
        other_tree = mpiPP::MPICommunication::receive<OctreeNode<AdditionalCellAttributes>>(other_rank);
        mpiPP::MPICommunication::send(tree, other_rank);
    } else {
        mpiPP::MPICommunication::send(tree, other_rank);
        other_tree = mpiPP::MPICommunication::receive<OctreeNode<AdditionalCellAttributes>>(other_rank);
    }

    const auto check_children = [other_rank, &node_cache](auto* local_node, auto* remote_node, auto* child_buffer) {
        if (local_node->is_leaf()) {
            return;
        }

        const auto children = local_node->get_children();
        const auto other_children = node_cache.get_children(remote_node);

        for (auto i = 0U; i < Constants::number_oct; i++) {
            if (children[i] == nullptr) {
                ASSERT_EQ(other_children[i], nullptr);
                continue;
            }

            ASSERT_NE(other_children[i], nullptr);

            ASSERT_EQ(other_rank, other_children[i]->get_mpi_rank());
            ASSERT_EQ(children[i]->is_parent(), other_children[i]->is_parent());
            ASSERT_EQ(children[i]->is_leaf(), other_children[i]->is_leaf());

            const auto& child_cell = children[i]->get_cell();
            const auto& other_child_cell = other_children[i]->get_cell();

            ASSERT_EQ(child_cell.get_neuron_id(), other_child_cell.get_neuron_id());
            ASSERT_EQ(child_cell.get_size(), other_child_cell.get_size());
            ASSERT_EQ(child_cell.get_number_excitatory_dendrites(), other_child_cell.get_number_excitatory_dendrites());
            ASSERT_EQ(child_cell.get_number_inhibitory_dendrites(), other_child_cell.get_number_inhibitory_dendrites());
            ASSERT_EQ(child_cell.get_excitatory_dendrites_position(), other_child_cell.get_excitatory_dendrites_position());
            ASSERT_EQ(child_cell.get_inhibitory_dendrites_position(), other_child_cell.get_inhibitory_dendrites_position());
        }

        *child_buffer = other_children;
    };

    const auto check_tree = [&check_children](auto* local_node, auto* remote_node) {
        using pair_type = std::pair<OctreeNode<AdditionalCellAttributes>*, OctreeNode<AdditionalCellAttributes>*>;

        auto stack = std::stack<pair_type>{};
        stack.emplace(local_node, remote_node);

        while (!stack.empty()) {
            const auto [local_current, remote_current] = stack.top();
            stack.pop();

            auto child_buffer = std::array<OctreeNode<AdditionalCellAttributes>*, Constants::number_oct>{};

            check_children(local_current, remote_current, &child_buffer);

            for (auto i = 0U; i < Constants::number_oct; i++) {
                if (local_current->get_child(i) == nullptr) {
                    continue;
                }

                stack.emplace(local_current->get_child(i), child_buffer[i]);
            }
        }
    };

    check_tree(&tree, &other_tree);

    node_cache.clear();

    ASSERT_EQ(node_cache.get_memory_size(), 0);
    ASSERT_EQ(node_cache.get_cache_size(), 0);

    mpiPP::MPISynchronization::barrier();
}
