/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_relearn_allocator.h"

#include "util/RelearnAllocator.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

TEST_F(RelearnAllocatorTest, testStaticAttributes) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    static_assert(std::is_same_v<RelearnAllocator<bool>::is_always_equal, std::true_type>);
    static_assert(std::is_same_v<RelearnAllocator<int>::is_always_equal, std::true_type>);
    static_assert(std::is_same_v<RelearnAllocator<double>::is_always_equal, std::true_type>);
    static_assert(std::is_same_v<RelearnAllocator<unsigned int>::is_always_equal, std::true_type>);
    static_assert(std::is_same_v<RelearnAllocator<std::tuple<bool, int, double, int>>::is_always_equal, std::true_type>);
}

TEST_F(RelearnAllocatorTest, testConstructors) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto alloc1 = RelearnAllocator<bool>{};
    const auto alloc2 = alloc1;

    const auto alloc3 = RelearnAllocator<int>{};
    const auto alloc4 = alloc3;
    auto alloc5 = RelearnAllocator<int>{ alloc1 };
    alloc5 = RelearnAllocator<int>{ alloc2 };
}

TEST_F(RelearnAllocatorTest, testAllocateCycle) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto alloc = RelearnAllocator<int>{};

    for (auto i = std::size_t{ 0 }; i < std::size_t{ 100 }; i++) {
        auto* ptr = alloc.allocate(i);
        static_assert(std::is_same_v<decltype(ptr), int*>);

        for (auto j = std::size_t{ 0 }; j < i; j++) {
            ASSERT_NO_THROW(ptr[j] = -1);
        }

        alloc.deallocate(ptr, i);
    }
}

TEST_F(RelearnAllocatorTest, testUniqueAddresses) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    constexpr auto number_pointers = std::size_t{ 1024 };
    constexpr auto number_elements = std::size_t{ 2048 };

    auto alloc = RelearnAllocator<double>{};
    auto pointers = std::vector<double*>{ number_pointers, nullptr };
    pointers.reserve(number_pointers);

    for (auto i = std::size_t{ 0 }; i < number_pointers; i++) {
        auto* ptr = alloc.allocate(number_elements);
        static_assert(std::is_same_v<decltype(ptr), double*>);

        for (auto j = std::size_t{ 0 }; j < number_elements; j++) {
            ASSERT_NO_THROW(ptr[j] = -1.5);
        }

        pointers[i] = ptr;
    }

    std::ranges::sort(pointers);

    for (auto i = std::size_t{ 1 }; i < number_pointers; i++) {
        auto* smaller_pointer = pointers[i - 1];
        auto* larger_pointer = pointers[i];

        auto diff = larger_pointer - smaller_pointer;

        ASSERT_GE(diff, number_elements);
    }

    for (auto i = std::size_t{ 0 }; i < number_pointers; i++) {
        ASSERT_NO_THROW(alloc.deallocate(pointers[i], number_elements));
    }
}

TEST_F(RelearnAllocatorTest, testDeallocateDifferentInstance) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    constexpr auto number_pointers = std::size_t{ 1024 };
    constexpr auto number_elements = std::size_t{ 2048 };

    auto alloc = RelearnAllocator<double>{};
    auto pointers = std::vector<double*>{ number_pointers, nullptr };
    pointers.reserve(number_pointers);

    for (auto i = std::size_t{ 0 }; i < number_pointers; i++) {
        auto* ptr = alloc.allocate(number_elements);
        pointers[i] = ptr;
    }

    auto dealloc = alloc;

    for (auto i = std::size_t{ 0 }; i < number_pointers; i++) {
        ASSERT_NO_THROW(dealloc.deallocate(pointers[i], number_elements));
    }
}

TEST_F(RelearnAllocatorTest, testDeallocateDifferentInstanceTypes) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    constexpr auto number_pointers = std::size_t{ 1024 };
    constexpr auto number_elements = std::size_t{ 2048 };

    auto alloc = RelearnAllocator<double>{};
    auto pointers = std::vector<double*>{ number_pointers, nullptr };
    pointers.reserve(number_pointers);

    for (auto i = std::size_t{ 0 }; i < number_pointers; i++) {
        auto* ptr = alloc.allocate(number_elements);
        pointers[i] = ptr;
    }

    const auto dealloc_temp = RelearnAllocator<std::uint16_t>{ alloc };
    auto dealloc = RelearnAllocator<double>{ dealloc_temp };

    for (auto i = std::size_t{ 0 }; i < number_pointers; i++) {
        ASSERT_NO_THROW(dealloc.deallocate(pointers[i], number_elements));
    }
}
