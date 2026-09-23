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

#include "test_bloom_filter.h"

#include "RelearnTest.hpp"

#include "cuda/network_graph/BloomFilter.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

class BloomFilterTest : public RelearnTest { };

// The Bloom filter contract is "never a false negative": any key that was inserted must always
// query as (at least) possibly present.
TEST_F(BloomFilterTest, testInsertedKeysAreNeverReportedAbsent) {
    constexpr auto max_edges = 200U;
    auto filter = BloomFilter(1, max_edges, 0.01F);

    auto inserted = std::vector<std::uint32_t>{};
    for (auto i = 0U; i < max_edges; ++i) {
        inserted.push_back(i * 7U + 3U);
    }
    device_bloom_insert(filter, 0, inserted);

    const auto results = device_bloom_query(filter, 0, inserted);
    for (auto i = 0U; i < inserted.size(); ++i) {
        ASSERT_TRUE(results[i]) << "key " << inserted[i] << " was inserted but bloom_query reported it absent (false negative)";
    }
}

// Keys inserted into one neuron's filter must not "leak" into another neuron's filter.
TEST_F(BloomFilterTest, testFiltersAreIndependentPerNeuron) {
    constexpr auto max_edges = 100U;
    auto filter = BloomFilter(2, max_edges, 0.001F);

    const auto keys_for_neuron_0 = std::vector<std::uint32_t>{ 11, 22, 33 };
    device_bloom_insert(filter, 0, keys_for_neuron_0);

    // A very low target FPR keeps the filter large, so an untouched neuron's filter (all-zero
    // bits) should report every one of these keys as absent.
    const auto results_for_neuron_1 = device_bloom_query(filter, 1, keys_for_neuron_0);
    for (const auto r : results_for_neuron_1) {
        ASSERT_FALSE(r);
    }
}

// The false-positive rate should land in the right ballpark of the requested target -- this is a
// statistical, not exact, sanity check: a badly mis-sized filter (wrong bit/hash-count formula)
// should still fail it by a wide margin.
TEST_F(BloomFilterTest, testFalsePositiveRateIsReasonablyCloseToTarget) {
    constexpr auto max_edges = 500U;
    constexpr auto target_fpr = 0.02F;
    auto filter = BloomFilter(1, max_edges, target_fpr);

    auto inserted_set = std::set<std::uint32_t>{};
    auto inserted = std::vector<std::uint32_t>{};
    for (auto i = 0U; i < max_edges; ++i) {
        const auto key = i * 101U + 17U;
        inserted.push_back(key);
        inserted_set.insert(key);
    }
    device_bloom_insert(filter, 0, inserted);

    auto probe_keys = std::vector<std::uint32_t>{};
    for (auto k = 1'000'000U; probe_keys.size() < 20000U; ++k) {
        if (inserted_set.find(k) == inserted_set.end()) {
            probe_keys.push_back(k);
        }
    }

    const auto results = device_bloom_query(filter, 0, probe_keys);
    const auto false_positives = std::count(results.begin(), results.end(), std::uint8_t{ 1 });
    const auto observed_fpr = static_cast<double>(false_positives) / static_cast<double>(probe_keys.size());

    ASSERT_LT(observed_fpr, static_cast<double>(target_fpr) * 5.0) << "observed false-positive rate " << observed_fpr << " is far above the target " << target_fpr;
}

// max_edges == 0 takes the degenerate "no capacity" branch in BloomFilter's constructor, leaving
// number_hash_functions at its default of 0; bloom_query's hash loop then never runs, so every
// query vacuously returns "might be present" rather than crashing on an empty filter.
TEST_F(BloomFilterTest, testZeroCapacityFilterAlwaysReportsPresent) {
    auto filter = BloomFilter(1, 0, 0.01F);

    const auto results = device_bloom_query(filter, 0, std::vector<std::uint32_t>{ 1, 2, 3, 42 });
    for (const auto r : results) {
        ASSERT_TRUE(r);
    }
}

#endif
