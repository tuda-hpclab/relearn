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

#include "cuda/util/Util.cuh"

#include <cstdint>

// CAS-based uint16 atomics for block_count. CUDA lacks a native 16-bit atomicAdd/Sub;
// we emulate them by operating on the 4-byte aligned word that contains the target half-word.
//
// Unlike the earlier uint8 design, these counts must be EXACT: find_delete_locate_sorted_warp_kernel
// uses them as real partial sums to walk directly to the block containing a drawn deletion index,
// not just as a zero/nonzero filter. A wrapped or saturated value would silently corrupt that walk.
// uint16_t (max 65535) comfortably exceeds any plausible per-(block,target) edge count — a block
// spans at most a few thousand source neurons — so overflow should never happen in practice; the
// checks below turn it into a loud, immediate abort instead of silent corruption if it ever does.
__device__ inline void block_count_add(std::uint16_t* ptr) {
    auto* base = reinterpret_cast<std::uint32_t*>(reinterpret_cast<uintptr_t>(ptr) & ~3ULL);
    const std::uint32_t shift = (reinterpret_cast<uintptr_t>(ptr) & 2U) << 3U;
    const std::uint32_t mask = 0xFFFFU << shift;
    std::uint32_t old_val = *base, assumed;
    do {
        assumed = old_val;
        const std::uint32_t half = (assumed & mask) >> shift;
        RELEARN_DEVICE_CUDA_CHECK(half != 0xFFFFU, "block_count_add: overflow (block filter counter exceeds uint16_t)");
        const std::uint32_t new_val = (assumed & ~mask) | ((half + 1U) << shift);
        old_val = atomicCAS(base, assumed, new_val);
    } while (old_val != assumed);
}

__device__ inline void block_count_sub(std::uint16_t* ptr) {
    auto* base = reinterpret_cast<std::uint32_t*>(reinterpret_cast<uintptr_t>(ptr) & ~3ULL);
    const std::uint32_t shift = (reinterpret_cast<uintptr_t>(ptr) & 2U) << 3U;
    const std::uint32_t mask = 0xFFFFU << shift;
    std::uint32_t old_val = *base, assumed;
    do {
        assumed = old_val;
        const std::uint32_t half = (assumed & mask) >> shift;
        RELEARN_DEVICE_CUDA_CHECK(half != 0U, "block_count_sub: underflow (block filter counter already zero)");
        const std::uint32_t new_val = (assumed & ~mask) | ((half - 1U) << shift);
        old_val = atomicCAS(base, assumed, new_val);
    } while (old_val != assumed);
}
