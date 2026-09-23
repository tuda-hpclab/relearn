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

#include "util/Util.cuh"

#include <cuda.h>

// Type-erased shared block pool — device side.
// Gives out raw byte blocks of a fixed size from a shared GPU buffer. Recycled blocks are
// managed with a lock-free (Treiber) stack: `head` packs a 32-bit block id together with a
// 32-bit ABA-guard tag into one atomically-CAS'd 64-bit word, and each free block's own
// (otherwise unused, since it's not backing a live Chunk) memory stores the "next" link --
// no separate freelist array is needed. See acquire_block/release_block for the fence/CAS
// protocol that makes a block's "next" write visible before it's reachable via `head`.
//
// The arena backing block ids is not one contiguous allocation: SharedBlockPool::ensure_capacity
// can append additional cudaMalloc'd segments (see SharedBlockPool.cu) whenever the host knows a
// kernel launch's worst-case demand would otherwise exhaust it. Segment bookkeeping
// (segment_data/segment_block_offset/segment_count) is written ONLY by the host, and only
// between kernel launches (never while a kernel that might read it is in flight), so no
// atomics/fences are needed for it here -- current_alloc and head remain the only fields
// mutated concurrently by device threads.
struct DeviceSharedBlockPool {
    static constexpr std::uint32_t EMPTY_BLOCK_ID = 0xFFFFFFFFu;
    // Generous fixed cap on the number of growth segments. With the ~1.5x-per-growth strategy
    // used by ensure_capacity, this supports enormous total growth well beyond any realistic run.
    static constexpr std::uint32_t MAX_SEGMENTS = 32;

    std::uint8_t* segment_data[MAX_SEGMENTS]{};         // base pointer of each segment
    std::uint32_t segment_block_offset[MAX_SEGMENTS]{}; // first block id served by each segment
    std::uint32_t segment_count{};

    unsigned long long head{ EMPTY_BLOCK_ID }; // atomic: (tag << 32) | block_id
    std::uint32_t current_alloc{};             // atomic: bump cursor for fresh blocks
    std::uint32_t total_blocks{};
    std::uint32_t block_bytes{};

    __device__ std::uint8_t* acquire_block();
    __device__ void release_block(std::uint32_t block_id);

    // Resolves a block id to its address, searching segments back-to-front (most blocks are
    // recently allocated, so the last segment is the common case).
    __device__ std::uint8_t* block_ptr(std::uint32_t block_id) const {
        for (std::uint32_t i = segment_count; i-- > 0;) {
            if (block_id >= segment_block_offset[i]) {
                return segment_data[i] + static_cast<std::uint64_t>(block_id - segment_block_offset[i]) * block_bytes;
            }
        }
        return nullptr;
    }

    __device__ std::uint32_t ptr_to_block_id(const std::uint8_t* ptr) const {
        for (std::uint32_t i = 0; i < segment_count; ++i) {
            const auto seg_blocks = (i + 1U < segment_count) ? (segment_block_offset[i + 1U] - segment_block_offset[i]) : (total_blocks - segment_block_offset[i]);
            auto* base = segment_data[i];
            auto* end = base + static_cast<std::uint64_t>(seg_blocks) * block_bytes;
            if (ptr >= base && ptr < end) {
                return segment_block_offset[i] + static_cast<std::uint32_t>((ptr - base) / block_bytes);
            }
        }
        return EMPTY_BLOCK_ID;
    }
};
