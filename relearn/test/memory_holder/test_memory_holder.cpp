/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_memory_holder.h"

#include "Config.h"
#include "RelearnTest.hpp"

#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/BarnesHutInternal/BarnesHutInvertedCell.h"
#include "algorithm/FMMInternal/FastMultipoleMethodCell.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "algorithm/NaiveInternal/NaiveCell.h"
#include "gtest/gtest.h"
#include "util/MemoryHolder.h"
#include "util/RelearnException.h"
#include "util/shuffle/shuffle.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/random/random_factory.h"

#include <gtest/gtest.h>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

using test_types = ::testing::Types<BarnesHutCell, BarnesHutInvertedCell, FastMultipoleMethodCell, NaiveCell>;
TYPED_TEST_SUITE(MemoryHolderTest, test_types);

TYPED_TEST(MemoryHolderTest, testInit) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    constexpr auto cells_size = 1024UL * 1024UL;
    auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
    cells.resize(cells_size);

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(cells);

    auto memory = std::vector<OctreeNode<AdditionalCellAttributes>>(1024, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory = std::span<OctreeNode<AdditionalCellAttributes>>(memory);

    memory_holder.init(span_memory);
    ASSERT_EQ(memory_holder.get_size(), 1024);
    ASSERT_EQ(memory_holder.get_current_memory().size(), span_memory.size());
    ASSERT_EQ(memory_holder.get_current_memory().data(), span_memory.data());
    ASSERT_EQ(memory_holder.get_current_filling(), 0);

    memory_holder.make_all_available();
    ASSERT_EQ(memory_holder.get_size(), 1024);
    ASSERT_EQ(memory_holder.get_current_memory().size(), span_memory.size());
    ASSERT_EQ(memory_holder.get_current_memory().data(), span_memory.data());
    ASSERT_EQ(memory_holder.get_current_filling(), 0);

    auto memory2 = std::vector<OctreeNode<AdditionalCellAttributes>>(6 * 1024, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory2 = std::span<OctreeNode<AdditionalCellAttributes>>(memory2);

    memory_holder.init(span_memory2);
    ASSERT_EQ(memory_holder.get_size(), 6 * 1024);
    ASSERT_EQ(memory_holder.get_current_memory().size(), span_memory2.size());
    ASSERT_EQ(memory_holder.get_current_memory().data(), span_memory2.data());
    ASSERT_EQ(memory_holder.get_current_filling(), 0);
}

TYPED_TEST(MemoryHolderTest, testGetAvailableException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    constexpr auto cells_size = 1024UL * 1024UL;
    auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
    cells.resize(cells_size);

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(cells);

    auto memory = std::vector<OctreeNode<AdditionalCellAttributes>>(1024, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory = std::span<OctreeNode<AdditionalCellAttributes>>(memory);

    memory_holder.init(span_memory);

    for (auto i = 0U; i < Constants::number_oct; i++) {
        ASSERT_THROW_NO_PRINT(std::ignore = memory_holder.get_available(nullptr, i), RelearnException);
    }

    auto root = OctreeNode<AdditionalCellAttributes>{};

    for (auto i = Constants::number_oct; i < Constants::number_oct * 1000; i++) {
        ASSERT_THROW_NO_PRINT(std::ignore = memory_holder.get_available(&root, i), RelearnException);
    }

    auto memory2 = std::vector<OctreeNode<AdditionalCellAttributes>>(Constants::number_oct, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory2 = std::span<OctreeNode<AdditionalCellAttributes>>(memory2);

    memory_holder.init(span_memory2);

    auto arr = std::array<OctreeNode<AdditionalCellAttributes>*, Constants::number_oct>{};
    for (auto i = 0U; i < Constants::number_oct; i++) {
        arr[i] = memory_holder.get_available(&root, i);
    }

    for (auto& node : memory) {
        for (auto i = 0U; i < Constants::number_oct; i++) {
            ASSERT_THROW_NO_PRINT(std::ignore = memory_holder.get_available(&node, i), RelearnException);
        }
    }
}

TYPED_TEST(MemoryHolderTest, testGetAvailable) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    constexpr auto cells_size = 1024UL * 1024UL;
    auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
    cells.resize(cells_size);

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(cells);

    auto memory = std::vector<OctreeNode<AdditionalCellAttributes>>(9 * Constants::number_oct, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory = std::span<OctreeNode<AdditionalCellAttributes>>(memory);

    memory_holder.init(span_memory);

    auto root = OctreeNode<AdditionalCellAttributes>{};

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto* ptr = memory_holder.get_available(&root, i);
        auto dist = std::distance(span_memory.data(), ptr);

        ASSERT_EQ(dist, i);

        root.set_child(ptr, i);
    }

    for (auto child_idx = 0U; child_idx < Constants::number_oct; child_idx++) {
        auto* child = root.get_child(child_idx);

        for (auto i = 0U; i < Constants::number_oct; i++) {
            auto* ptr = memory_holder.get_available(child, i);
            auto dist = std::distance(span_memory.data(), ptr);

            ASSERT_EQ(dist, Constants::number_oct + child_idx * Constants::number_oct + i);

            child->set_child(ptr, i);
        }
    }

    auto memory2 = std::vector<OctreeNode<AdditionalCellAttributes>>(9 * Constants::number_oct, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory2 = std::span<OctreeNode<AdditionalCellAttributes>>(memory2);

    memory_holder.init(span_memory2);

    auto root2 = OctreeNode<AdditionalCellAttributes>{};

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto* ptr = memory_holder.get_available(&root2, i);
        auto dist = std::distance(span_memory2.data(), ptr);

        ASSERT_EQ(dist, i);

        root2.set_child(ptr, i);
    }

    for (auto child_idx = 0U; child_idx < Constants::number_oct; child_idx++) {
        auto* child = root2.get_child(child_idx);

        for (auto i = 0U; i < Constants::number_oct; i++) {
            auto* ptr = memory_holder.get_available(child, i);
            auto dist = std::distance(span_memory2.data(), ptr);

            ASSERT_EQ(dist, Constants::number_oct + child_idx * Constants::number_oct + i);

            child->set_child(ptr, i);
        }
    }
}

TYPED_TEST(MemoryHolderTest, testGetAvailableDisorganized) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    constexpr auto cells_size = 1024UL * 1024UL;
    auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
    cells.resize(cells_size);

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(cells);

    auto memory = std::vector<OctreeNode<AdditionalCellAttributes>>(9 * Constants::number_oct, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory = std::span<OctreeNode<AdditionalCellAttributes>>(memory);

    memory_holder.init(span_memory);

    auto root = OctreeNode<AdditionalCellAttributes>{};

    const auto indices = ranges::views::iota(0U, Constants::number_oct) | ranges::to_vector | actions::shuffle(this->mt);

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto child_index = indices[i];

        auto* ptr = memory_holder.get_available(&root, child_index);
        auto dist = std::distance(span_memory.data(), ptr);

        ASSERT_EQ(dist, child_index);

        root.set_child(ptr, child_index);
    }

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto child_index = indices[i];

        auto* child = root.get_child(child_index);

        const auto indices_child = ranges::views::iota(0U, Constants::number_oct) | ranges::to_vector | actions::shuffle(this->mt);

        for (auto j = 0U; j < Constants::number_oct; j++) {
            auto child_child_index = indices_child[j];

            auto* ptr = memory_holder.get_available(child, child_child_index);
            auto dist = std::distance(span_memory.data(), ptr);

            ASSERT_EQ(dist, Constants::number_oct + i * Constants::number_oct + child_child_index);

            child->set_child(ptr, child_child_index);
        }
    }
}

TYPED_TEST(MemoryHolderTest, testGetAvailableFull) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    constexpr auto cells_size = 1024UL * 1024UL;
    auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
    cells.resize(cells_size);

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(cells);

    const auto number_objects = RandomFactory::get_random_integer(Constants::number_oct, Constants::number_oct * 1024, this->mt);

    auto memory = std::vector<OctreeNode<AdditionalCellAttributes>>(number_objects, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory = std::span<OctreeNode<AdditionalCellAttributes>>(memory);

    memory_holder.init(span_memory);

    auto parents = std::vector<OctreeNode<AdditionalCellAttributes>>(1024, OctreeNode<AdditionalCellAttributes>{});

    for (auto i = 0U; i < 1024U; i++) {
        const auto current_filling_expected = i * Constants::number_oct;

        ASSERT_EQ(current_filling_expected, memory_holder.get_current_filling());
        ASSERT_EQ(number_objects, memory_holder.get_size());

        for (auto child_idx = 0U; child_idx < Constants::number_oct; child_idx++) {
            if (current_filling_expected + Constants::number_oct > number_objects) {
                ASSERT_THROW_NO_PRINT(std::ignore = memory_holder.get_available(&parents[i], child_idx), RelearnException);
            } else {
                ASSERT_NO_THROW(std::ignore = memory_holder.get_available(&parents[i], child_idx));
            }
        }

        ASSERT_EQ(current_filling_expected + Constants::number_oct, memory_holder.get_current_filling());
    }
}

TYPED_TEST(MemoryHolderTest, testMakeAvailable) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    constexpr auto cells_size = 1024UL * 1024UL;
    auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
    cells.resize(cells_size);

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(cells);

    const auto number_objects = RandomFactory::get_random_integer(Constants::number_oct, Constants::number_oct * 1024, this->mt);
    const auto number_requesting_objects = RandomFactory::get_random_integer(1U, 1024U, this->mt);

    auto memory = std::vector<OctreeNode<AdditionalCellAttributes>>(number_objects, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory = std::span<OctreeNode<AdditionalCellAttributes>>(memory);

    memory_holder.init(span_memory);

    auto parents = std::vector<OctreeNode<AdditionalCellAttributes>>(number_requesting_objects, OctreeNode<AdditionalCellAttributes>{});

    for (auto i = 0U; i < number_requesting_objects; i++) {
        if (i * Constants::number_oct + Constants::number_oct > number_objects) {
            ASSERT_THROW_NO_PRINT(std::ignore = memory_holder.get_available(&parents[i], 0), RelearnException);
            continue;
        }

        std::ignore = memory_holder.get_available(&parents[i], 0);
    }

    memory_holder.make_all_available();

    ASSERT_EQ(memory_holder.get_current_filling(), 0);
    ASSERT_EQ(memory_holder.get_size(), number_objects);

    for (auto i = 0U; i < number_requesting_objects; i++) {
        ASSERT_THROW_NO_PRINT(std::ignore = memory_holder.get_offset_from_parent(&parents[i]), RelearnException);
    }
}

TYPED_TEST(MemoryHolderTest, testGetOffsetException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    constexpr auto cells_size = 1024UL * 1024UL;
    auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
    cells.resize(cells_size);

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(cells);

    auto memory = std::vector<OctreeNode<AdditionalCellAttributes>>(1024, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory = std::span<OctreeNode<AdditionalCellAttributes>>(memory);

    memory_holder.init(span_memory);

    ASSERT_THROW_NO_PRINT(std::ignore = memory_holder.get_offset_from_parent(nullptr), RelearnException);

    for (auto i = 0U; i < 1024U; i++) {
        auto* ptr = &memory[i];
        ASSERT_THROW_NO_PRINT(std::ignore = memory_holder.get_offset_from_parent(ptr), RelearnException);
    }

    auto other_memory = std::vector<OctreeNode<AdditionalCellAttributes>>(1024, OctreeNode<AdditionalCellAttributes>{});

    for (auto i = 0U; i < 1024U; i++) {
        auto* ptr = &other_memory[i];
        ASSERT_THROW_NO_PRINT(std::ignore = memory_holder.get_offset_from_parent(ptr), RelearnException);
    }
}

TYPED_TEST(MemoryHolderTest, testGetOffset) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    constexpr auto cells_size = 1024UL * 1024UL;
    auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
    cells.resize(cells_size);

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(cells);

    auto memory = std::vector<OctreeNode<AdditionalCellAttributes>>(1024, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory = std::span<OctreeNode<AdditionalCellAttributes>>(memory);

    memory_holder.init(span_memory);

    auto root = OctreeNode<AdditionalCellAttributes>{};

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto* ptr = memory_holder.get_available(&root, i);
        auto dist = std::distance(span_memory.data(), ptr);

        ASSERT_EQ(dist, i);

        root.set_child(ptr, i);
    }

    for (auto child_idx = 0U; child_idx < Constants::number_oct; child_idx++) {
        auto* child = root.get_child(child_idx);

        for (auto i = 0U; i < Constants::number_oct; i++) {
            auto* ptr = memory_holder.get_available(child, i);
            auto dist = std::distance(span_memory.data(), ptr);

            ASSERT_EQ(dist, Constants::number_oct + child_idx * Constants::number_oct + i);

            child->set_child(ptr, i);
        }
    }

    ASSERT_EQ(0, memory_holder.get_offset_from_parent(&root));

    for (auto child_idx = 0U; child_idx < Constants::number_oct; child_idx++) {
        auto* child = root.get_child(child_idx);

        const auto offset_index = Constants::number_oct + (child_idx * Constants::number_oct);

        ASSERT_EQ(offset_index, memory_holder.get_offset_from_parent(child));
    }
}

TYPED_TEST(MemoryHolderTest, testGetOffsetDisorganized) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    constexpr auto cells_size = 1024UL * 1024UL;
    auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
    cells.resize(cells_size);

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(cells);

    auto memory = std::vector<OctreeNode<AdditionalCellAttributes>>(9 * Constants::number_oct, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory = std::span<OctreeNode<AdditionalCellAttributes>>(memory);

    memory_holder.init(span_memory);

    auto root = OctreeNode<AdditionalCellAttributes>{};

    const auto indices = ranges::views::iota(0U, Constants::number_oct) | ranges::to_vector | actions::shuffle(this->mt);

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto child_index = indices[i];
        auto* ptr = memory_holder.get_available(&root, child_index);
        root.set_child(ptr, child_index);
    }

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto child_index = indices[i];

        auto* child = root.get_child(child_index);

        const auto indices_child = ranges::views::iota(0U, Constants::number_oct) | ranges::to_vector | actions::shuffle(this->mt);

        for (auto j = 0U; j < Constants::number_oct; j++) {
            auto child_child_index = indices_child[j];
            auto* ptr = memory_holder.get_available(child, child_child_index);
            child->set_child(ptr, child_child_index);
        }
    }

    ASSERT_EQ(0, memory_holder.get_offset_from_parent(&root));

    for (auto i = 0U; i < Constants::number_oct; i++) {
        auto child_index = indices[i];
        const auto offset = memory_holder.get_offset_from_parent(root.get_child(child_index));
        const auto expected_offset = (i + 1) * Constants::number_oct;

        ASSERT_EQ(expected_offset, offset);
    }
}

TYPED_TEST(MemoryHolderTest, testGetNodeFromOffsetException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    constexpr auto cells_size = 1024UL * 1024UL;
    auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
    cells.resize(cells_size);

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(cells);

    auto memory = std::vector<OctreeNode<AdditionalCellAttributes>>(1024, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory = std::span<OctreeNode<AdditionalCellAttributes>>(memory);

    memory_holder.init(span_memory);

    for (auto i = std::uint16_t{ 0 }; i < 1024 * 10ULL; i++) {
        ASSERT_THROW_NO_PRINT(std::ignore = memory_holder.get_node_from_offset(i), RelearnException);
    }
}

TYPED_TEST(MemoryHolderTest, testGetNodeFromOffset) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    constexpr auto cells_size = 1024UL * 1024UL;
    auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
    cells.resize(cells_size);

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(cells);

    const auto number_objects = RandomFactory::get_random_integer(Constants::number_oct, Constants::number_oct * 1024, this->mt);
    const auto number_requesting_objects = number_objects / Constants::number_oct;

    auto memory = std::vector<OctreeNode<AdditionalCellAttributes>>(number_objects, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory = std::span<OctreeNode<AdditionalCellAttributes>>(memory);

    memory_holder.init(span_memory);

    auto parents = std::vector<OctreeNode<AdditionalCellAttributes>>(number_requesting_objects, OctreeNode<AdditionalCellAttributes>{});

    for (auto i = 0U; i < number_requesting_objects; i++) {
        auto* ptr = &parents[i];
        ASSERT_NO_THROW(std::ignore = memory_holder.get_available(ptr, 0));
    }

    for (auto offset = 0ULL; offset < number_objects; offset++) {
        if (offset >= number_requesting_objects * Constants::number_oct) {
            ASSERT_THROW_NO_PRINT(std::ignore = memory_holder.get_node_from_offset(offset), RelearnException);
            continue;
        }

        auto* ptr = memory_holder.get_node_from_offset(offset);
        auto dist = std::distance(span_memory.data(), ptr);

        ASSERT_EQ(dist, offset);
    }
}

TYPED_TEST(MemoryHolderTest, testGetParentFromOffsetException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;

    constexpr auto cells_size = 1024UL * 1024UL;
    auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
    cells.resize(cells_size);

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(cells);

    auto memory = std::vector<OctreeNode<AdditionalCellAttributes>>(1024, OctreeNode<AdditionalCellAttributes>{});
    const auto span_memory = std::span<OctreeNode<AdditionalCellAttributes>>(memory);

    memory_holder.init(span_memory);

    for (auto i = std::uint16_t{ 0 }; i < 1024 * 10ULL; i++) {
        ASSERT_THROW_NO_PRINT(std::ignore = memory_holder.get_parent_from_offset(i), RelearnException);
    }
}

TYPED_TEST(MemoryHolderTest, testGetParentFromOffset) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using AdditionalCellAttributes = TypeParam;
    using Node = OctreeNode<AdditionalCellAttributes>;

    constexpr auto cells_size = 1024UL * 1024UL;
    auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
    cells.resize(cells_size);

    auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
    memory_holder.init(cells);

    auto memory = std::vector<Node>(128 * Constants::number_oct, Node{});
    const auto span_memory = std::span<Node>(memory);

    memory_holder.init(span_memory);

    auto nodes = std::vector<Node>(128, Node{});
    auto relations = std::vector<std::pair<Node*, Node*>>{};

    for (auto& node : nodes) {
        for (auto i = 0U; i < Constants::number_oct; i++) {
            auto* ptr = memory_holder.get_available(&node, i);
            relations.emplace_back(&node, ptr);
        }
    }

    for (const auto& [parent, child] : relations) {
        auto offset = static_cast<std::uint64_t>(std::distance(memory.data(), child));
        if (offset % Constants::number_oct != 0) {
            ASSERT_THROW_NO_PRINT(std::ignore = memory_holder.get_parent_from_offset(offset), RelearnException);
            continue;
        }

        auto saved_parent = memory_holder.get_parent_from_offset(offset);
        ASSERT_EQ(parent, saved_parent);
    }
}
