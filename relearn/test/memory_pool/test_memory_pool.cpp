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

#include "test_memory_pool.h"

#include "RelearnTest.hpp"

#include "cuda/memory/DeviceVecVec.h"
#include "cuda/memory/SharedBlockPool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

class MemoryPoolTest : public RelearnTest { };

using u32 = std::uint32_t;

// ── Single add per neuron ─────────────────────────────────────────────────────
// Thread i adds (i * multiplier) to neuron i. Each neuron has exactly one element,
// so get(i, 0) must return i * multiplier.

TEST_F(MemoryPoolTest, SingleAddPerNeuron) {
    constexpr u32 n_neurons = 8;
    constexpr u32 init_size = 4;
    constexpr u32 new_chunk_size = 4;
    constexpr u32 overflow = 0;
    constexpr u32 multiplier = 10;

    SharedBlockPool shared_pool(overflow, sizeof(Chunk<u32>) + new_chunk_size * sizeof(u32));
    const DynamicVecVec<u32> pool(n_neurons, init_size, &shared_pool);

    device_add_one_per_neuron(pool.get_device_view(), n_neurons, multiplier);

    // Each neuron has one element at index 0; neuron i holds value i * multiplier.
    std::vector<u32> neuron_ids(n_neurons);
    const std::vector<u32> elem_idxs(n_neurons, 0U);
    for (u32 i = 0; i < n_neurons; i++) {
        neuron_ids[i] = i;
    }
    const auto result = device_get_at(pool.get_device_view(), neuron_ids, elem_idxs);

    ASSERT_EQ(result.size(), n_neurons);
    for (u32 i = 0; i < n_neurons; i++) {
        EXPECT_EQ(result[i], i * multiplier) << "neuron " << i;
    }
}

// ── Multiple adds within initial chunk ───────────────────────────────────────
// Add init_size values to neuron 0; get(0, j) must return values[j].

TEST_F(MemoryPoolTest, MultipleAddsNoOverflow) {
    constexpr u32 n_neurons = 1;
    constexpr u32 init_size = 6;
    constexpr u32 new_chunk_size = 6;
    constexpr u32 overflow = 0;

    SharedBlockPool shared_pool(overflow, sizeof(Chunk<u32>) + new_chunk_size * sizeof(u32));
    const DynamicVecVec<u32> pool(n_neurons, init_size, &shared_pool);

    const std::vector<u32> values = { 7, 13, 42, 99, 1, 255 };
    device_add_multiple_to_neuron(pool.get_device_view(), 0, values);

    const std::vector<u32> neuron_ids(init_size, 0U);
    std::vector<u32> elem_idxs(init_size);
    for (u32 i = 0; i < init_size; i++) {
        elem_idxs[i] = i;
    }
    const auto result = device_get_at(pool.get_device_view(), neuron_ids, elem_idxs);

    ASSERT_EQ(result.size(), values.size());
    for (u32 i = 0; i < init_size; i++) {
        EXPECT_EQ(result[i], values[i]) << "element " << i;
    }
}

// ── Overflow into a new chunk ─────────────────────────────────────────────────
// Fill neuron 0's initial chunk (3 slots), then add one more value that triggers
// allocation of an overflow chunk. get(0, 3) must return the overflow value.

TEST_F(MemoryPoolTest, OverflowToNewChunk) {
    constexpr u32 n_neurons = 1;
    constexpr u32 init_size = 3;
    constexpr u32 new_chunk_size = 4;
    constexpr u32 overflow = 1; // enough room for at least one extra chunk

    SharedBlockPool shared_pool(overflow, sizeof(Chunk<u32>) + new_chunk_size * sizeof(u32));
    const DynamicVecVec<u32> pool(n_neurons, init_size, &shared_pool);

    // 4th add triggers overflow into a new chunk.
    const std::vector<u32> values = { 10, 20, 30, 40 };
    device_add_multiple_to_neuron(pool.get_device_view(), 0, values);

    const std::vector<u32> neuron_ids(4, 0U);
    const std::vector<u32> elem_idxs = { 0, 1, 2, 3 };
    const auto result = device_get_at(pool.get_device_view(), neuron_ids, elem_idxs);

    ASSERT_EQ(result.size(), 4U);
    EXPECT_EQ(result[0], 10U);
    EXPECT_EQ(result[1], 20U);
    EXPECT_EQ(result[2], 30U);
    EXPECT_EQ(result[3], 40U);
}

// ── Multiple neurons, multiple adds, no overflow ──────────────────────────────
// Each neuron n receives adds_per_neuron values n*100+j. Verify get(n, j).

TEST_F(MemoryPoolTest, MultipleNeuronsMultipleAdds) {
    constexpr u32 n_neurons = 4;
    constexpr u32 init_size = 8;
    constexpr u32 new_chunk_size = 8;
    constexpr u32 overflow = 0;
    constexpr u32 adds_per_neuron = 5; // < init_size, no overflow

    SharedBlockPool shared_pool(overflow, sizeof(Chunk<u32>) + new_chunk_size * sizeof(u32));
    const DynamicVecVec<u32> pool(n_neurons, init_size, &shared_pool);

    for (u32 n = 0; n < n_neurons; n++) {
        std::vector<u32> values(adds_per_neuron);
        for (u32 j = 0; j < adds_per_neuron; j++) {
            values[j] = n * 100 + j;
        }
        device_add_multiple_to_neuron(pool.get_device_view(), n, values);
    }

    std::vector<u32> neuron_ids;
    std::vector<u32> elem_idxs;
    std::vector<u32> expected;
    for (u32 n = 0; n < n_neurons; n++) {
        for (u32 j = 0; j < adds_per_neuron; j++) {
            neuron_ids.push_back(n);
            elem_idxs.push_back(j);
            expected.push_back(n * 100 + j);
        }
    }

    const auto result = device_get_at(pool.get_device_view(), neuron_ids, elem_idxs);

    ASSERT_EQ(result.size(), expected.size());
    for (std::size_t k = 0; k < expected.size(); k++) {
        EXPECT_EQ(result[k], expected[k]) << "neuron=" << neuron_ids[k] << " elem=" << elem_idxs[k];
    }
}

// ── add_from_map ──────────────────────────────────────────────────────────────
// Populate the pool from a host map, then verify via device cursor.

TEST_F(MemoryPoolTest, AddFromMapSingleElement) {
    constexpr u32 n_neurons = 4;
    constexpr u32 init_size = 8;
    constexpr u32 new_chunk_size = 8;
    constexpr u32 overflow_chunks = 0;

    SharedBlockPool shared_pool(overflow_chunks, sizeof(Chunk<u32>) + new_chunk_size * sizeof(u32));
    DynamicVecVec<u32> pool(n_neurons, init_size, &shared_pool);

    std::unordered_map<u32, std::vector<u32>> map;
    for (u32 i = 0; i < n_neurons; ++i) {
        map[i] = { i * 5U };
    }
    pool.add_from_map(map);

    for (u32 i = 0; i < n_neurons; ++i) {
        const auto result = device_cursor_collect(pool.get_device_view(), i, 16);
        ASSERT_EQ(result.size(), 1U) << "neuron " << i;
        EXPECT_EQ(result[0], i * 5U) << "neuron " << i;
    }
}

TEST_F(MemoryPoolTest, AddFromMapMultipleElementsWithOverflow) {
    constexpr u32 n_neurons = 2;
    constexpr u32 init_size = 3;
    constexpr u32 new_chunk_size = 4;
    constexpr u32 overflow_chunks = 4;

    SharedBlockPool shared_pool(overflow_chunks, sizeof(Chunk<u32>) + new_chunk_size * sizeof(u32));
    DynamicVecVec<u32> pool(n_neurons, init_size, &shared_pool);

    const std::vector<u32> v0 = { 10, 20, 30, 40, 50 }; // 3 in init chunk, 2 in overflow
    const std::vector<u32> v1 = { 100, 200 };
    pool.add_from_map({ { 0, v0 }, { 1, v1 } });

    const auto r0 = device_cursor_collect(pool.get_device_view(), 0, 16);
    ASSERT_EQ(r0.size(), v0.size());
    for (u32 i = 0; i < static_cast<u32>(v0.size()); ++i) {
        EXPECT_EQ(r0[i], v0[i]) << "neuron 0 element " << i;
    }

    const auto r1 = device_cursor_collect(pool.get_device_view(), 1, 16);
    ASSERT_EQ(r1.size(), v1.size());
    for (u32 i = 0; i < static_cast<u32>(v1.size()); ++i) {
        EXPECT_EQ(r1[i], v1[i]) << "neuron 1 element " << i;
    }
}

// ── Cursor: single element per neuron ────────────────────────────────────────
// Each neuron has exactly one element; cursor should yield it then end.

TEST_F(MemoryPoolTest, CursorSingleElement) {
    constexpr u32 n_neurons = 4;
    constexpr u32 init_size = 8;
    constexpr u32 new_chunk_size = 8;
    constexpr u32 overflow = 0;
    constexpr u32 multiplier = 7;

    SharedBlockPool shared_pool(overflow, sizeof(Chunk<u32>) + new_chunk_size * sizeof(u32));
    const DynamicVecVec<u32> pool(n_neurons, init_size, &shared_pool);
    device_add_one_per_neuron(pool.get_device_view(), n_neurons, multiplier);

    for (u32 i = 0; i < n_neurons; i++) {
        const auto result = device_cursor_collect(pool.get_device_view(), i, 16);
        ASSERT_EQ(result.size(), 1U) << "neuron " << i;
        EXPECT_EQ(result[0], i * multiplier) << "neuron " << i;
    }
}

// ── Cursor: multiple elements, no overflow ────────────────────────────────────
// Neuron 0 receives several values; cursor must yield them all in insertion order.

TEST_F(MemoryPoolTest, CursorMultipleElementsNoOverflow) {
    constexpr u32 n_neurons = 1;
    constexpr u32 init_size = 8;
    constexpr u32 new_chunk_size = 8;
    constexpr u32 overflow = 0;

    SharedBlockPool shared_pool(overflow, sizeof(Chunk<u32>) + new_chunk_size * sizeof(u32));
    const DynamicVecVec<u32> pool(n_neurons, init_size, &shared_pool);

    const std::vector<u32> values = { 3, 1, 4, 1, 5, 9 };
    device_add_multiple_to_neuron(pool.get_device_view(), 0, values);

    const auto result = device_cursor_collect(pool.get_device_view(), 0, 16);

    ASSERT_EQ(result.size(), values.size());
    for (u32 i = 0; i < static_cast<u32>(values.size()); i++) {
        EXPECT_EQ(result[i], values[i]) << "element " << i;
    }
}

// ── Cursor: overflow into a second chunk ─────────────────────────────────────
// Values span two chunks; cursor must cross the chunk boundary and yield all values.

TEST_F(MemoryPoolTest, CursorOverflowChunkBoundary) {
    constexpr u32 n_neurons = 1;
    constexpr u32 init_size = 3;
    constexpr u32 new_chunk_size = 4;
    constexpr u32 overflow = 8;

    SharedBlockPool shared_pool(overflow, sizeof(Chunk<u32>) + new_chunk_size * sizeof(u32));
    const DynamicVecVec<u32> pool(n_neurons, init_size, &shared_pool);

    // 3 in first chunk, 2 in overflow chunk
    const std::vector<u32> values = { 10, 20, 30, 40, 50 };
    device_add_multiple_to_neuron(pool.get_device_view(), 0, values);

    const auto result = device_cursor_collect(pool.get_device_view(), 0, 16);

    ASSERT_EQ(result.size(), values.size());
    for (u32 i = 0; i < static_cast<u32>(values.size()); i++) {
        EXPECT_EQ(result[i], values[i]) << "element " << i;
    }
}

// ── Cursor: multiple neurons independent ─────────────────────────────────────
// Each neuron has a different number of elements; cursor on each must be independent.

TEST_F(MemoryPoolTest, CursorMultipleNeuronsIndependent) {
    constexpr u32 n_neurons = 3;
    constexpr u32 init_size = 10;
    constexpr u32 new_chunk_size = 10;
    constexpr u32 overflow = 0;

    SharedBlockPool shared_pool(overflow, sizeof(Chunk<u32>) + new_chunk_size * sizeof(u32));
    const DynamicVecVec<u32> pool(n_neurons, init_size, &shared_pool);

    const std::vector<std::vector<u32>> neuron_values = {
        { 1, 2, 3 },
        { 100, 200 },
        { 42 },
    };
    for (u32 n = 0; n < n_neurons; n++) {
        device_add_multiple_to_neuron(pool.get_device_view(), n, neuron_values[n]);
    }

    for (u32 n = 0; n < n_neurons; n++) {
        const auto result = device_cursor_collect(pool.get_device_view(), n, 16);
        ASSERT_EQ(result.size(), neuron_values[n].size()) << "neuron " << n;
        for (u32 j = 0; j < static_cast<u32>(neuron_values[n].size()); j++) {
            EXPECT_EQ(result[j], neuron_values[n][j]) << "neuron " << n << " element " << j;
        }
    }
}

// ── SharedBlockPool: concurrent acquire/release must never alias a block ─────
// Regression test for a race in DeviceSharedBlockPool::acquire_block/release_block where a
// block's index could be published (via the free counter/stack) before its payload write
// landed, or where a block being released could be handed out again while still in use --
// either way, two threads could end up holding the same physical block simultaneously.
// Hammers a pool with many concurrent acquire/release cycles and checks that no thread ever
// acquires a block another thread is still holding.
//
// num_blocks must be >= num_threads. This test used to run 256 threads over only 8 blocks
// (one block per warp on typical 32-wide hardware) to also stress the "acquire blocks on a
// momentarily-empty free stack" path -- but that makes a thread's acquire_block() spin-wait
// on a release from another thread, and on GPUs without Independent Thread Scheduling
// (pre-Volta, compute capability < 7.0; this project's CI runs on compute_61) a warp executes
// divergent branches with strict lockstep reconvergence: a lane that already acquired a block
// cannot execute its release_block() call (which lexically follows the spin loop) until every
// lane in its warp has exited the loop, but the still-spinning lanes can only exit once that
// exact release happens. With num_blocks == num_warps, every warp hits this simultaneously,
// deadlocking the whole kernel (observed as a CI job hanging until the 1h timeout). Sizing
// the pool so every thread can always get a block on its first attempt removes the spin-wait
// (and thus this cross-lane dependency) entirely, while still fully exercising the lock-free
// stack's concurrent push/pop/CAS/ABA-tag logic every iteration (each thread frees its block
// before the next iteration's acquire, so blocks are still recycled through the shared stack
// under real concurrent contention -- just never scarce enough to make anyone wait).
// timed_out_threads is kept as a defensive safety net in case a future regression reintroduces
// unbounded contention; it should never fire so long as num_blocks >= num_threads holds.

TEST_F(MemoryPoolTest, SharedBlockPoolConcurrentAcquireReleaseNoAliasing) {
    constexpr u32 num_threads = 256;
    constexpr u32 num_blocks = num_threads;
    constexpr u32 num_iterations = 500;
    const std::size_t block_bytes = sizeof(Chunk<std::uint32_t>) + 8;

    const SharedBlockPool pool(num_blocks, block_bytes);

    const auto result = device_stress_pool(pool.get_device_pool(), num_blocks, num_threads, num_iterations);
    ASSERT_EQ(result.timed_out_threads, 0U) << "acquire_block failed to make progress within the per-attempt "
                                               "timeout for "
                                            << result.timed_out_threads << " thread-iterations "
                                                                           "-- either a liveness bug in acquire_block/release_block, or "
                                                                           "this GPU is pathologically slow/contended";
    EXPECT_EQ(result.violations, 0U) << "acquire_block handed out a block that another thread still held";

    auto drained = device_drain_pool(pool.get_device_pool(), num_blocks + 1);
    std::sort(drained.begin(), drained.end());
    ASSERT_EQ(drained.size(), num_blocks) << "blocks lost or duplicated after the stress run";
    for (u32 i = 0; i < num_blocks; ++i) {
        EXPECT_EQ(drained[i], i) << "expected block id " << i << " to be present exactly once";
    }
}

// ── copy_to_host<bool>: neurons spanning multiple chunks ─────────────────────
// Regression test for a bug in DynamicVecVec<bool>::copy_to_host() where the bit-unpacking
// loop read each chunk's length via h_chunk_bits[offset + i] while `offset` itself already
// advanced by one per iteration -- for any neuron with 2+ chunks, every chunk after the
// first had its length read from the WRONG (next) slot in the flat per-chunk array, i.e.
// borrowed from a different neuron's chunk length instead of its own. Neurons 1 and 2 below
// each span a main chunk plus one overflow chunk of a distinctly different length, so a
// borrowed (wrong) length is unambiguously different from the correct one.

TEST_F(MemoryPoolTest, CopyToHostBoolMultiChunkNeurons) {
    constexpr u32 n_neurons = 4;
    constexpr u32 init_size_bits = 32; // main chunk holds exactly one 32-bit word
    constexpr u32 overflow_chunk_bits = 32;
    constexpr u32 overflow_chunks = 8;
    // block_bytes must be a multiple of alignof(Chunk<T>) (8) -- see round_up_to_alignment's
    // doc comment in MemoryPool.cu; DynamicVecVec's own private-pool constructor does this
    // rounding automatically, but here we construct SharedBlockPool directly.
    const std::size_t raw_block_bytes = sizeof(Chunk<std::uint32_t>) + overflow_chunk_bits / 8;
    const std::size_t block_bytes = ((raw_block_bytes + 7) / 8) * 8;

    SharedBlockPool shared_pool(overflow_chunks, block_bytes);
    DynamicVecVec<bool> pool(n_neurons, init_size_bits, &shared_pool);

    auto make_pattern = [](std::size_t n, std::size_t seed) {
        std::vector<bool> v(n);
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = ((i + seed) % 3) != 0;
        }
        return v;
    };

    const auto v0 = make_pattern(5, 0);  // fits entirely in the main chunk
    const auto v1 = make_pattern(50, 1); // 32 (main) + 18 (overflow) -- 2 chunks
    const auto v2 = make_pattern(39, 2); // 32 (main) +  7 (overflow) -- 2 chunks, different length
    const auto v3 = make_pattern(3, 3);  // fits entirely in the main chunk

    pool.add_from_map({ { 0, v0 }, { 1, v1 }, { 2, v2 }, { 3, v3 } });

    const auto result = pool.copy_to_host();
    ASSERT_EQ(result.size(), n_neurons);

    const std::vector<std::vector<bool>> expected = { v0, v1, v2, v3 };
    for (u32 n = 0; n < n_neurons; ++n) {
        ASSERT_EQ(result[n].size(), expected[n].size()) << "neuron " << n;
        for (std::size_t i = 0; i < expected[n].size(); ++i) {
            EXPECT_EQ(result[n][i], expected[n][i]) << "neuron " << n << " bit " << i;
        }
    }
}

// ── SharedBlockPool::ensure_capacity ─────────────────────────────────────────

// When there's already enough headroom, ensure_capacity must be a no-op: no segment appended,
// footprint unchanged.
TEST_F(MemoryPoolTest, EnsureCapacityNoOpWhenHeadroomSufficient) {
    constexpr u32 num_blocks = 8;
    const std::size_t block_bytes = sizeof(Chunk<u32>) + 8;

    SharedBlockPool pool(num_blocks, block_bytes);
    const auto footprint_before = pool.get_gpu_memory_footprint();

    pool.ensure_capacity(4); // <= the 8 free blocks already available

    EXPECT_EQ(pool.get_gpu_memory_footprint(), footprint_before);
    EXPECT_EQ(pool.get_block_usage().total_blocks, num_blocks);
}

// Once the arena is fully bump-allocated, ensure_capacity must append a new segment sized to
// cover the requested demand, and the new blocks must be servable via the bump allocator.
TEST_F(MemoryPoolTest, EnsureCapacityGrowsWhenInsufficient) {
    constexpr u32 num_blocks = 4;
    const std::size_t block_bytes = sizeof(Chunk<u32>) + 8;

    SharedBlockPool pool(num_blocks, block_bytes);
    const auto footprint_before = pool.get_gpu_memory_footprint();

    // Exhaust the initial segment's bump cursor.
    const auto initial_ids = device_drain_pool(pool.get_device_pool(), num_blocks + 1);
    ASSERT_EQ(initial_ids.size(), num_blocks);

    pool.ensure_capacity(6);

    // deficit (6) > growth floor (num_blocks/2 == 2), so growth == the deficit exactly.
    EXPECT_EQ(pool.get_block_usage().total_blocks, num_blocks + 6);
    EXPECT_EQ(pool.get_gpu_memory_footprint() - footprint_before, 6U * block_bytes);

    // The new blocks (ids num_blocks..num_blocks+5) must actually be servable via the bump
    // allocator, in that id range, not just reflected in the bookkeeping totals above.
    const auto grown_ids = device_drain_pool(pool.get_device_pool(), 100);
    std::vector<u32> expected_ids(6);
    for (u32 i = 0; i < 6; ++i) {
        expected_ids[i] = num_blocks + i;
    }
    EXPECT_EQ(grown_ids, expected_ids);
}

// A second ensure_capacity call must add its own additional segment on top of the first,
// accumulating rather than replacing it.
TEST_F(MemoryPoolTest, EnsureCapacityAccumulatesAcrossMultipleGrowths) {
    constexpr u32 num_blocks = 2;
    const std::size_t block_bytes = sizeof(Chunk<u32>) + 8;

    SharedBlockPool pool(num_blocks, block_bytes);
    const auto footprint_initial = pool.get_gpu_memory_footprint();

    // First growth: exhaust, then ask for more than the +50% floor would give.
    ASSERT_EQ(device_drain_pool(pool.get_device_pool(), num_blocks + 1).size(), num_blocks);
    pool.ensure_capacity(5);
    ASSERT_EQ(pool.get_block_usage().total_blocks, num_blocks + 5); // 2 + 5 == 7

    // Second growth: exhaust the now-larger pool, ask for more again.
    ASSERT_EQ(device_drain_pool(pool.get_device_pool(), 100).size(), 5U);
    pool.ensure_capacity(3);
    EXPECT_EQ(pool.get_block_usage().total_blocks, num_blocks + 5 + 3); // 7 + 3 == 10

    const auto footprint_after = pool.get_gpu_memory_footprint();
    EXPECT_EQ(footprint_after - footprint_initial, static_cast<std::uint64_t>(5 + 3) * block_bytes);
}

// Blocks served out of a segment appended by ensure_capacity must be fully usable through the
// normal DynamicVecVec add/get path, not just addressable at the raw pool level.
TEST_F(MemoryPoolTest, DynamicVecVecUsesBlocksFromGrownSegment) {
    constexpr u32 n_neurons = 1;
    constexpr u32 init_size = 2;
    constexpr u32 new_chunk_size = 2;
    // Only one overflow block up front -- not nearly enough for the 5 overflow chunks this test
    // needs, so every chunk past the first overflow comes from a segment ensure_capacity appends.
    constexpr u32 initial_overflow_blocks = 1;
    const std::size_t block_bytes = sizeof(Chunk<u32>) + new_chunk_size * sizeof(u32);

    SharedBlockPool shared_pool(initial_overflow_blocks, block_bytes);
    // Grow up front for 5 overflow chunks, exactly like ExchangeAlgorithm's
    // ensure_capacity(3 * number_requests) call does before a kernel launch. No blocks have been
    // acquired yet, so this guarantees at least 5 free blocks total, not "5 more than initial".
    shared_pool.ensure_capacity(5);
    ASSERT_GE(shared_pool.get_block_usage().total_blocks, 5U);

    DynamicVecVec<u32> pool(n_neurons, init_size, &shared_pool);

    // 2 in the main chunk, then 2 more per overflow chunk -- 12 values need 1 (initial) + 4
    // (from the grown segment) overflow chunks, i.e. every overflow chunk after the first is
    // necessarily served from the grown segment.
    std::vector<u32> values(12);
    for (u32 i = 0; i < values.size(); ++i) {
        values[i] = 100 + i;
    }
    device_add_multiple_to_neuron(pool.get_device_view(), 0, values);

    const auto result = device_cursor_collect(pool.get_device_view(), 0, 32);
    ASSERT_EQ(result.size(), values.size());
    for (u32 i = 0; i < static_cast<u32>(values.size()); ++i) {
        EXPECT_EQ(result[i], values[i]) << "element " << i;
    }
}

// A block released after being acquired from a grown segment must round-trip through the
// freelist correctly: release_block/acquire_block resolve block ids to addresses via block_ptr,
// which must handle ids belonging to a non-initial segment just as well as the first one.
TEST_F(MemoryPoolTest, ReleasedBlockFromGrownSegmentIsReacquired) {
    constexpr u32 num_blocks = 2;
    const std::size_t block_bytes = sizeof(Chunk<u32>) + 8;

    SharedBlockPool pool(num_blocks, block_bytes);
    ASSERT_EQ(device_drain_pool(pool.get_device_pool(), num_blocks + 1).size(), num_blocks);

    pool.ensure_capacity(3);
    ASSERT_EQ(pool.get_block_usage().total_blocks, num_blocks + 3);

    // Drain the grown segment's blocks (ids num_blocks..num_blocks+2), then release the middle
    // one and confirm it -- and only it -- comes back out.
    const auto grown_ids = device_drain_pool(pool.get_device_pool(), 100);
    ASSERT_EQ(grown_ids.size(), 3U);

    const auto released_id = grown_ids[1];
    device_release_block(pool.get_device_pool(), released_id);

    const auto reacquired = device_drain_pool(pool.get_device_pool(), 100);
    ASSERT_EQ(reacquired.size(), 1U);
    EXPECT_EQ(reacquired[0], released_id);
}

// ensure_capacity must stop growing once DeviceSharedBlockPool::MAX_SEGMENTS is reached, quietly
// leaving the pool at its current size, rather than overflow the fixed-size segment table or
// throw/crash -- this is a best-effort call (see its doc comment). The constant is duplicated
// here (see SharedBlockPool.cuh) since test_memory_pool.cpp is a plain .cpp and deliberately
// never includes the .cuh (DeviceSharedBlockPool has __device__-qualified inline methods); if
// that constant ever changes, update MAX_SEGMENTS below to match.
TEST_F(MemoryPoolTest, EnsureCapacityStopsGrowingPastMaxSegments) {
    constexpr u32 max_segments = 32;
    constexpr u32 num_blocks = 1;
    const std::size_t block_bytes = sizeof(Chunk<u32>) + 8;

    SharedBlockPool pool(num_blocks, block_bytes);

    // Segment 0 already exists; each successful ensure_capacity call below appends exactly one
    // more segment (regardless of how many blocks it grows by), so max_segments - 1 successful
    // calls fill the table exactly (1 + (max_segments - 1) == max_segments segments). Blocks are
    // never released here, so current_alloc only ever advances -- draining with a generous bound
    // each time consumes exactly whatever new blocks the last growth added, regardless of how
    // many prior iterations already drained.
    for (u32 i = 1; i < max_segments; ++i) {
        device_drain_pool(pool.get_device_pool(), pool.get_block_usage().total_blocks + 1);
        ASSERT_EQ(pool.get_block_usage().used_blocks, pool.get_block_usage().total_blocks) << "iteration " << i;
        pool.ensure_capacity(1);
    }

    // The table is now full: one more required growth must be a silent no-op instead of
    // overflowing it (or throwing).
    device_drain_pool(pool.get_device_pool(), pool.get_block_usage().total_blocks + 1);
    ASSERT_EQ(pool.get_block_usage().used_blocks, pool.get_block_usage().total_blocks);
    const auto total_before = pool.get_block_usage().total_blocks;
    const auto footprint_before = pool.get_gpu_memory_footprint();

    pool.ensure_capacity(1);

    EXPECT_EQ(pool.get_block_usage().total_blocks, total_before);
    EXPECT_EQ(pool.get_gpu_memory_footprint(), footprint_before);
}

// ensure_capacity is a headroom reservation, not a hard cap -- if actual demand exceeds what was
// reserved, the pre-existing acquire_block()-returns-nullptr safety net must still apply: the
// over-demand add() must fail gracefully (silently dropping that value, exactly like running out
// of an un-grown pool always has), not crash or corrupt state.
TEST_F(MemoryPoolTest, AddBeyondEnsuredCapacityFailsGracefully) {
    constexpr u32 n_neurons = 1;
    constexpr u32 init_size = 1;      // main chunk holds exactly 1 element
    constexpr u32 new_chunk_size = 1; // each overflow chunk holds exactly 1 element
    constexpr u32 initial_overflow_blocks = 0;

    SharedBlockPool shared_pool(initial_overflow_blocks, sizeof(Chunk<u32>) + new_chunk_size * sizeof(u32));
    shared_pool.ensure_capacity(1); // reserves headroom for exactly one overflow chunk

    DynamicVecVec<u32> pool(n_neurons, init_size, &shared_pool);

    // value[0] -> main chunk. value[1] -> the one reserved overflow chunk (succeeds).
    // value[2] -> needs a second overflow chunk, which was never reserved -- must fail and be
    // dropped, not crash.
    const std::vector<u32> values = { 10, 20, 30 };
    device_add_multiple_to_neuron(pool.get_device_view(), 0, values);

    const auto result = device_cursor_collect(pool.get_device_view(), 0, 16);
    ASSERT_EQ(result.size(), 2U) << "the third add should have failed and been dropped";
    EXPECT_EQ(result[0], 10U);
    EXPECT_EQ(result[1], 20U);
}

#endif // RELEARN_CUDA_ENABLED
