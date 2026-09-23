/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SharedBlockPool.h"

#include "cuda/memory/SharedBlockPool.cuh"
#include "thrust/device_ptr.h"
#include "thrust/reduce.h"
#include "util/Timers.h"
#include "util/Util.cuh"

#include <cpp-utility/Cast.hpp>

#include <algorithm>

// ---- Statistics kernels ----

// Counts free blocks by walking the intrusive Treiber-stack chain from `head` (see
// DeviceSharedBlockPool). Single-threaded -- only used by the infrequent host-side stats path,
// never in a hot loop, so a serial walk is fine.
__global__ void count_free_blocks_kernel(const DeviceSharedBlockPool* pool, std::uint32_t* out_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    auto top = static_cast<std::uint32_t>(pool->head & 0xFFFFFFFFull);
    std::uint32_t count = 0;
    while (top != DeviceSharedBlockPool::EMPTY_BLOCK_ID) {
        ++count;
        const auto* next_ptr = reinterpret_cast<const std::uint32_t*>(pool->block_ptr(top));
        top = *next_ptr;
    }
    *out_count = count;
}

// Same walk as above, but also marks every visited block id in `is_free`.
__global__ void mark_free_blocks_kernel(
    const DeviceSharedBlockPool* pool, std::uint8_t* is_free, std::uint32_t* out_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    auto top = static_cast<std::uint32_t>(pool->head & 0xFFFFFFFFull);
    std::uint32_t count = 0;
    while (top != DeviceSharedBlockPool::EMPTY_BLOCK_ID) {
        is_free[top] = 1;
        ++count;
        const auto* next_ptr = reinterpret_cast<const std::uint32_t*>(pool->block_ptr(top));
        top = *next_ptr;
    }
    *out_count = count;
}

// For each allocated block, read fill stats from the embedded Chunk header.
// Chunk<T> layout (constant across all T on 64-bit): T* begin (8 B), uint32_t chunk_id (4 B),
// uint32_t filled (4 B, offset 12), uint32_t size (4 B, offset 16).
__global__ void get_block_fill_stats_kernel(
    const DeviceSharedBlockPool* pool,
    const std::uint8_t* is_free, std::uint32_t n_alloc,
    float* fill_fractions, std::uint32_t* at_max_flags) {
    const auto bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= n_alloc) {
        return;
    }
    if (is_free[bid]) {
        fill_fractions[bid] = 0.f;
        at_max_flags[bid] = 0;
        return;
    }

    const auto* blk = pool->block_ptr(bid);
    const auto filled = *reinterpret_cast<const std::uint32_t*>(blk + 12);
    const auto size = *reinterpret_cast<const std::uint32_t*>(blk + 16);
    fill_fractions[bid] = size > 0 ? static_cast<float>(filled) / static_cast<float>(size) : 0.f;
    at_max_flags[bid] = (size > 0 && filled == size) ? 1U : 0U;
}

// ---- DeviceSharedBlockPool ----

__device__ std::uint8_t* DeviceSharedBlockPool::acquire_block() {
    // Try recycled blocks first: pop the Treiber stack rooted at `head`.
    auto old_head = head;
    for (;;) {
        const auto top = static_cast<std::uint32_t>(old_head & 0xFFFFFFFFull);
        if (top == EMPTY_BLOCK_ID)
            break; // stack empty -> fall through to bump-allocate

        const auto tag = static_cast<std::uint32_t>(old_head >> 32);
        // Reading `next` here can race with another thread also popping `top`; that's fine --
        // the CAS below is the sole arbiter. If `head` changed underneath us (someone else got
        // `top`, or the ABA tag no longer matches), the CAS fails and this speculative `next`
        // is simply discarded on retry.
        const auto* next_ptr = reinterpret_cast<const std::uint32_t*>(block_ptr(top));
        const auto next = *next_ptr;
        const auto new_head = (static_cast<unsigned long long>(tag + 1U) << 32) | next;

        const auto prev = atomicCAS(&head, old_head, new_head);
        if (prev == old_head) {
            return block_ptr(top);
        }
        old_head = prev;
    }
    // Bump-allocate a fresh block. Uses a bounded CAS loop rather than a blind atomicAdd:
    // every thread that observes the stack empty falls through to here, which under heavy
    // contention (many threads, few blocks) happens constantly and is expected, not a leak --
    // an unconditional atomicAdd would keep incrementing current_alloc forever in that case,
    // and being a uint32_t, enough failed attempts (unsigned overflow wraps, both in C++ and
    // atomicAdd) eventually wrap it back below total_blocks, handing out an already-circulating
    // block a second time.
    auto slot = current_alloc;
    for (;;) {
        if (slot >= total_blocks)
            return nullptr;
        const auto prev = atomicCAS(&current_alloc, slot, slot + 1U);
        if (prev == slot)
            return block_ptr(slot);
        slot = prev;
    }
}

__device__ void DeviceSharedBlockPool::release_block(std::uint32_t block_id) {
    // Push block_id onto the Treiber stack rooted at `head`, storing the "next" link inside
    // the block's own (now-unused) memory instead of a separate freelist array.
    auto* next_ptr = reinterpret_cast<std::uint32_t*>(block_ptr(block_id));
    auto old_head = head;
    for (;;) {
        const auto old_top = static_cast<std::uint32_t>(old_head & 0xFFFFFFFFull);
        const auto tag = static_cast<std::uint32_t>(old_head >> 32);
        *next_ptr = old_top;
        // Ensure the next-link write above is visible to any thread that subsequently observes
        // the CAS below succeeding -- atomics in CUDA don't imply ordering for other addresses,
        // so without this fence a concurrent acquire_block could dereference `next_ptr` before
        // this write has landed.
        __threadfence();
        const auto new_head = (static_cast<unsigned long long>(tag + 1U) << 32) | block_id;
        const auto prev = atomicCAS(&head, old_head, new_head);
        if (prev == old_head)
            return;
        old_head = prev;
    }
}

// ---- SharedBlockPool (host) ----

SharedBlockPool::SharedBlockPool(std::size_t num_blocks, std::size_t block_bytes)
    : num_blocks_(num_blocks)
    , block_bytes_(block_bytes) {

    std::uint8_t* first_segment{};
    cudaMalloc_bridge(reinterpret_cast<void**>(&first_segment), num_blocks * block_bytes);
    segments.push_back(first_segment);

    cudaMalloc_bridge(reinterpret_cast<void**>(&device_pool), sizeof(DeviceSharedBlockPool));
    DeviceSharedBlockPool h_pool{};
    h_pool.segment_data[0] = first_segment;
    h_pool.segment_block_offset[0] = 0U;
    h_pool.segment_count = 1U;
    h_pool.head = DeviceSharedBlockPool::EMPTY_BLOCK_ID;
    h_pool.current_alloc = 0U;
    h_pool.total_blocks = static_cast<std::uint32_t>(num_blocks);
    h_pool.block_bytes = static_cast<std::uint32_t>(block_bytes);
    cudaMemcpy_to_device_bridge(device_pool, &h_pool, sizeof(DeviceSharedBlockPool));
}

SharedBlockPool::~SharedBlockPool() {
    for (auto* segment : segments) {
        cudaFree_bridge(segment);
    }
    cudaFree_bridge(device_pool);
}

void SharedBlockPool::ensure_capacity(std::size_t additional_blocks_needed) {
    DeviceSharedBlockPool h{};
    cudaMemcpy_to_host_bridge(&h, device_pool, sizeof(DeviceSharedBlockPool));

    const auto free_headroom = static_cast<std::size_t>(h.total_blocks - h.current_alloc);
    if (free_headroom >= additional_blocks_needed) {
        return;
    }

    // Out of growth segments, or genuinely out of device memory: this is a best-effort call, not
    // a hard guarantee, so give up quietly rather than throwing/crashing here. The pool simply
    // stops growing and behaves like the original fixed-size pool from here on -- callers still
    // relying on acquire_block()'s pre-existing nullptr-on-exhaustion path as the last resort.
    if (h.segment_count >= DeviceSharedBlockPool::MAX_SEGMENTS) {
        return;
    }

    // Grow by at least the deficit, but never by less than +50% of the current size, so repeated
    // small deficits don't each trigger their own tiny cudaMalloc.
    const auto deficit = additional_blocks_needed - free_headroom;
    const auto growth_floor = std::max<std::size_t>(1, static_cast<std::size_t>(h.total_blocks) / 2);
    auto growth_blocks = std::max(deficit, growth_floor);

    // If the device doesn't have room for the desired growth, keep halving and retrying rather
    // than propagating the allocation failure -- grow by as much as actually fits, down to
    // nothing, instead of crashing on a request that turned out to be too greedy.
    std::uint8_t* new_segment{};
    while (growth_blocks > 0) {
        try {
            cudaMalloc_bridge(reinterpret_cast<void**>(&new_segment), growth_blocks * block_bytes_);
            break;
        } catch (const std::runtime_error&) {
            cudaResetLastError_bridge();
            new_segment = nullptr;
            growth_blocks /= 2;
        }
    }
    if (new_segment == nullptr) {
        return;
    }

    segments.push_back(new_segment);
    num_blocks_ += growth_blocks;

    h.segment_data[h.segment_count] = new_segment;
    h.segment_block_offset[h.segment_count] = h.total_blocks;
    h.segment_count += 1U;
    h.total_blocks += static_cast<std::uint32_t>(growth_blocks);

    cudaMemcpy_to_device_bridge(device_pool, &h, sizeof(DeviceSharedBlockPool));
}

std::uint64_t SharedBlockPool::get_gpu_memory_footprint() const {
    return num_blocks_ * block_bytes_
           + sizeof(DeviceSharedBlockPool);
}

BlockUsageStats SharedBlockPool::get_block_usage() const {
    DeviceSharedBlockPool h{};
    cudaMemcpy_to_host_bridge(&h, device_pool, sizeof(DeviceSharedBlockPool));
    if (h.current_alloc == 0)
        return { 0U, h.total_blocks };

    DeviceArray<std::uint32_t> d_free_count(1);
    Timers::start(TimerRegion::CUDA_SHARED_BLOCK_POOL_COUNT_FREE_KERNEL);
    count_free_blocks_kernel<<<1, 1>>>(device_pool, d_free_count.device_ptr());
    cudaDeviceSynchronize();
    Timers::stop_and_add(TimerRegion::CUDA_SHARED_BLOCK_POOL_COUNT_FREE_KERNEL);
    const auto n_free = d_free_count.get_device_data()[0];

    const auto used = h.current_alloc > n_free ? h.current_alloc - n_free : 0U;
    return { used, h.total_blocks };
}

BlockFillStats SharedBlockPool::get_block_fill_stats() const {
    DeviceSharedBlockPool h{};
    cudaMemcpy_to_host_bridge(&h, device_pool, sizeof(DeviceSharedBlockPool));

    const auto n_alloc = h.current_alloc;

    if (n_alloc == 0)
        return { 0U, 0.f };

    std::uint8_t* d_is_free{};
    cudaMalloc_bridge(reinterpret_cast<void**>(&d_is_free), n_alloc * sizeof(std::uint8_t));
    cudaMemset_bridge(d_is_free, 0, n_alloc * sizeof(std::uint8_t));

    DeviceArray<std::uint32_t> d_free_count(1);
    Timers::start(TimerRegion::CUDA_SHARED_BLOCK_POOL_MARK_FREE_KERNEL);
    mark_free_blocks_kernel<<<1, 1>>>(device_pool, d_is_free, d_free_count.device_ptr());
    cudaDeviceSynchronize();
    Timers::stop_and_add(TimerRegion::CUDA_SHARED_BLOCK_POOL_MARK_FREE_KERNEL);
    const auto n_free = d_free_count.get_device_data()[0];
    const auto used = n_alloc > n_free ? n_alloc - n_free : 0U;

    DeviceArray<float> d_fill(n_alloc);
    DeviceArray<std::uint32_t> d_at_max(n_alloc);
    Timers::start(TimerRegion::CUDA_SHARED_BLOCK_POOL_FILL_STATS_KERNEL);
    get_block_fill_stats_kernel<<<(n_alloc + 255U) / 256U, 256>>>(
        device_pool, d_is_free, n_alloc,
        d_fill.device_ptr(), d_at_max.device_ptr());
    cudaDeviceSynchronize();
    Timers::stop_and_add(TimerRegion::CUDA_SHARED_BLOCK_POOL_FILL_STATS_KERNEL);

    const auto sum_at_max = thrust::reduce(
        thrust::device_pointer_cast(d_at_max.device_ptr()),
        thrust::device_pointer_cast(d_at_max.device_ptr() + n_alloc));
    const auto sum_fill = thrust::reduce(
        thrust::device_pointer_cast(d_fill.device_ptr()),
        thrust::device_pointer_cast(d_fill.device_ptr() + n_alloc));

    cudaFree_bridge(d_is_free);

    const auto avg_fill = used > 0 ? sum_fill / static_cast<float>(used) : 0.f;
    return { static_cast<std::uint32_t>(sum_at_max), avg_fill };
}

void SharedBlockPool::record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint,
                                             const std::string& prefix) const {
    const auto [used, total] = get_block_usage();
    const auto [at_max, avg_fill] = get_block_fill_stats();
    footprint->emplace(prefix + ": used blocks", static_cast<std::uint64_t>(used * 100));
    footprint->emplace(prefix + ": total blocks", static_cast<std::uint64_t>(total * 100));
    footprint->emplace(prefix + ": utilization", total > 0 ? static_cast<std::uint64_t>(used) * 100ULL / total : 0ULL);
    footprint->emplace(prefix + ": blocks at max", static_cast<std::uint64_t>(at_max * 100));
    footprint->emplace(prefix + ": avg fill used blocks", static_cast<std::uint64_t>(avg_fill * 100.f));
}
void SharedBlockPool::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix) const {
    footprint->emplace(prefix, get_gpu_memory_footprint());
}
