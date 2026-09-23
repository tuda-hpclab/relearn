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

#include "cpp-utility/MemoryFootprint.hpp"
#include "cuda/util/Util.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

// SharedBlockPool is only implemented in SharedBlockPool.cu (compiled into relearn_gpu, which is
// only built when RELEARN_CUDA_ENABLED). This file is always compiled, so on a CPU-only build
// these stubs are the only definitions available -- matching the NOT_SUPPORTED convention used
// for the CUDA bridge functions in CudaBridgeFunctions.cpp.
#ifndef RELEARN_CUDA_ENABLED

SharedBlockPool::SharedBlockPool(std::size_t, std::size_t){ CUDA_NOT_SUPPORTED }

SharedBlockPool::~SharedBlockPool() = default;

void SharedBlockPool::ensure_capacity(std::size_t){ CUDA_NOT_SUPPORTED }

std::uint64_t SharedBlockPool::get_gpu_memory_footprint() const { CUDA_NOT_SUPPORTED }

BlockUsageStats SharedBlockPool::get_block_usage() const { CUDA_NOT_SUPPORTED }

BlockFillStats SharedBlockPool::get_block_fill_stats() const { CUDA_NOT_SUPPORTED }

void SharedBlockPool::record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>&,
                                             const std::string&) const { CUDA_NOT_SUPPORTED }

void SharedBlockPool::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>&,
                                              const std::string&) const { CUDA_NOT_SUPPORTED }

#endif
