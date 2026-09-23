/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#ifdef RELEARN_CUDA_ENABLED

#include "test_random_gpu.h"

#include "RelearnTest.hpp"

#include "cuda/random/RandomNumberHost.h"
#include "cuda/random/RandomNumberKeys.h"

#include <algorithm>
#include <set>

class RandomGPUTest : public RelearnTest { };

// RandomNumbers::register_random_numbers() maintains process-global device state with no
// reset (see RandomNumbersHost.cpp), shared with production code paths (Poisson/Normal input,
// synapse deletion) that may already have registered streams elsewhere in this test binary.
// These tests therefore only ever assert relative effects (deltas, monotonicity) of their own
// calls, never absolute handle values or memory totals.

TEST_F(RandomGPUTest, testRegisterRandomNumbersReturnsMonotonicHandlesAndGrowsMemoryUsage) {
    const auto mem_before = RandomNumbers::get_memory_usage();

    const auto handle1 = RandomNumbers::register_random_numbers(RandomNumberKey::BARNES_HUT, RandomNumberType::UNIFORM, 4, 42);
    const auto mem_after_first = RandomNumbers::get_memory_usage();
    ASSERT_GT(mem_after_first, mem_before);

    const auto handle2 = RandomNumbers::register_random_numbers(RandomNumberKey::BARNES_HUT, RandomNumberType::NORMAL, 8, 43);
    const auto mem_after_second = RandomNumbers::get_memory_usage();
    ASSERT_GT(mem_after_second, mem_after_first);

    ASSERT_EQ(handle2, handle1 + 1);
}

TEST_F(RandomGPUTest, testSampleKUniqueReturnsUniqueSortedIndicesInRange) {
    constexpr auto n = 50U;
    constexpr auto k = 10U;
    const auto key = RandomNumbers::register_random_numbers(RandomNumberKey::BARNES_HUT, RandomNumberType::UNIFORM, 1, 1234);

    for (auto trial = 0U; trial < 20U; ++trial) {
        const auto sample = device_sample_k_unique(key, k, n);
        ASSERT_EQ(sample.size(), k);

        for (const auto v : sample) {
            ASSERT_LT(v, n);
        }

        ASSERT_TRUE(std::is_sorted(sample.begin(), sample.end()));

        auto distinct = std::set<std::uint32_t>(sample.begin(), sample.end());
        ASSERT_EQ(distinct.size(), sample.size()) << "sample_k_unique must never repeat an index";
    }
}

TEST_F(RandomGPUTest, testUniformDrawsAreWithinRangeAndVaryPerThread) {
    constexpr auto num_draws = 256U;
    const auto key = RandomNumbers::register_random_numbers(RandomNumberKey::BARNES_HUT, RandomNumberType::UNIFORM, num_draws, 777);

    const auto values = device_draw_random_values(key, num_draws);
    ASSERT_EQ(values.size(), num_draws);

    for (const auto v : values) {
        ASSERT_GE(v, 0.0);
        ASSERT_LE(v, 1.0);
    }

    // Each thread owns an independently-seeded cuRAND state; a broken setup (e.g. every thread
    // sharing state or seed) would collapse these draws to a handful of distinct values.
    const auto distinct = std::set<double>(values.begin(), values.end());
    ASSERT_GT(distinct.size(), num_draws / 2);
}

TEST_F(RandomGPUTest, testNormalDrawsVaryPerThread) {
    constexpr auto num_draws = 256U;
    const auto key = RandomNumbers::register_random_numbers(RandomNumberKey::BARNES_HUT, RandomNumberType::NORMAL, num_draws, 778);

    const auto values = device_draw_random_values(key, num_draws);
    ASSERT_EQ(values.size(), num_draws);

    const auto distinct = std::set<double>(values.begin(), values.end());
    ASSERT_GT(distinct.size(), num_draws / 2);
}

#endif
