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
#include "cuda/util/Util.h"
#include "cuda/wrapper/StreamWrapper.cuh"
#include "util/Timers.h"

template <typename T>
__global__ void set_memory_kernel(T* data, std::size_t size, T value) {
    const auto thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id >= size) {
        return;
    }

    data[thread_id] = value;
}

static std::size_t max_memory_usage{};

std::size_t get_gpu_max_memory_used() {
    return max_memory_usage;
}

void set_memory_entry(std::uint32_t* data, std::size_t size, std::uint32_t value) {
    Timers::start(TimerRegion::CUDA_SET_MEMORY_KERNEL);
    const auto& [blocks, threads] = get_grid_ands_block_size(size, set_memory_kernel<std::uint32_t>);
    set_memory_kernel<<<blocks, threads>>>(data, size, value);
    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_SET_MEMORY_KERNEL);
}
