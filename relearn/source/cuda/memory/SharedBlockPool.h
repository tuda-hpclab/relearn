#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cpp-utility/MemoryFootprint.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct DeviceSharedBlockPool;

/**
 * Used = ever bump-allocated minus returned to freelist.
 */
struct BlockUsageStats {
    std::uint32_t used_blocks;
    std::uint32_t total_blocks;
};

/**
 * Stats over all currently used blocks.
 * blocks_at_max = count of used blocks where filled == size.
 * avg_fill_fraction = mean of filled/size over used blocks (0 if none used).
 */
struct BlockFillStats {
    std::uint32_t blocks_at_max;
    float avg_fill_fraction;
};

/**
 * Type-erased shared overflow pool shared by multiple DynamicVecVec instances.
 *
 * All DynamicVecVec instances that share this pool draw overflow-chunk data from one or more
 * GPU byte buffers ("segments"). block_bytes must be divisible by sizeof(T) for every
 * participating T. The initial cudaMalloc is sized from a heuristic and can turn out too small
 * over a long run; ensure_capacity() appends further segments on demand rather than failing.
 * Segments are never freed individually -- the destructor frees all of them once, at the end of
 * the simulation.
 */
class SharedBlockPool {
public:
    /**
     * @brief Allocates the shared block pool on the device.
     * @param num_blocks  Number of fixed-size blocks to allocate.
     * @param block_bytes Size of each block in bytes.
     */
    SharedBlockPool(std::size_t num_blocks, std::size_t block_bytes);
    ~SharedBlockPool();
    SharedBlockPool(const SharedBlockPool&) = delete;
    SharedBlockPool& operator=(const SharedBlockPool&) = delete;
    SharedBlockPool(SharedBlockPool&&) = delete;
    SharedBlockPool& operator=(SharedBlockPool&&) = delete;

    /**
     * @brief Returns the device-side pool descriptor passed to kernels.
     */
    [[nodiscard]] DeviceSharedBlockPool* get_device_pool() const { return device_pool; }

    /**
     * @brief Best-effort attempt to make at least @p additional_blocks_needed free blocks
     * available, appending another cudaMalloc'd segment first if not. Call this on the host,
     * before launching a kernel, with that kernel's exact worst-case block demand -- it must
     * never be called concurrently with a kernel that reads the device pool descriptor.
     *
     * Headroom is computed from the bump cursor only (total_blocks - current_alloc), ignoring
     * blocks sitting in the freelist: cheap (a few bytes copied back, no kernel launch) at the
     * cost of occasionally growing a bit earlier than strictly necessary. Appended segments are
     * never freed early -- see the class comment.
     *
     * This never throws or crashes: if the segment table is full (DeviceSharedBlockPool::
     * MAX_SEGMENTS) or the device is genuinely out of memory, it shrinks the requested growth
     * and retries, down to doing nothing at all, rather than fail the call. In that case the pool
     * simply stops growing and reverts to its original fixed-size behavior -- callers that need a
     * hard guarantee still have acquire_block()'s nullptr-on-exhaustion path as the last resort.
     */
    void ensure_capacity(std::size_t additional_blocks_needed);

    /**
     * @brief Returns the size of each block in bytes.
     */
    [[nodiscard]] std::size_t get_block_bytes() const { return block_bytes_; }

    /**
     * @brief Returns the total GPU memory (in bytes) consumed by this pool.
     */
    [[nodiscard]] std::uint64_t get_gpu_memory_footprint() const;

    /**
     * @brief Returns used/total block counts.
     * Used = ever bump-allocated minus returned to freelist.
     */
    [[nodiscard]] BlockUsageStats get_block_usage() const;

    /**
     * @brief Returns fill stats for all currently used blocks.
     */
    [[nodiscard]] BlockFillStats get_block_fill_stats() const;

    /**
     * @brief Records per-field memory usage into @p footprint with the given name prefix.
     * @param footprint Target memory footprint tracker.
     * @param prefix    String prefix for field names.
     */
    void record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint,
                                const std::string& prefix) const;

    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint,
                                 const std::string& prefix) const;

private:
    // One entry per cudaMalloc'd segment (index 0 is the initial allocation from the ctor,
    // further entries are appended by ensure_capacity). Kept only for freeing in the destructor
    // and for the memory-footprint total; the device-visible copy lives in *device_pool.
    std::vector<std::uint8_t*> segments;
#ifdef RELEARN_CUDA_ENABLED
    std::size_t num_blocks_{};
#endif

    DeviceSharedBlockPool* device_pool{};
    std::size_t block_bytes_{};
};
