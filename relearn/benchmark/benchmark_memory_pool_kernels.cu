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

#include "benchmark_memory_pool.h"

#include "cuda/memory/DeviceVecVec.cuh"
#include "cuda/memory/SharedBlockPool.cuh"

#include <cstdint>

namespace {

constexpr unsigned BLOCK = 256;

unsigned grid(unsigned n) {
    return (n + BLOCK - 1) / BLOCK;
}

__global__ void k_bench_acquire_release(DeviceSharedBlockPool* pool, std::uint32_t num_threads, std::uint32_t iterations) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads)
        return;

    // Bounded retry, NOT an unconditional "spin until success": once the pool is oversubscribed
    // (num_threads > pool capacity), a warp can end up with some lanes that already acquired a
    // block -- about to fall through to release_block() below -- and other lanes still failing
    // acquire_block() and looping. On lockstep-reconvergence hardware, the warp only reconverges
    // (letting the successful lanes reach release_block()) once every lane has exited this loop;
    // an unbounded `while (nullptr)` therefore deadlocks, since the release those lanes are
    // waiting on can never run before reconvergence, and reconvergence can never happen without
    // it. Bounding the attempt count guarantees every lane exits in finite time either way.
    constexpr std::uint32_t max_attempts = 100000U;

    for (std::uint32_t it = 0; it < iterations; ++it) {
        std::uint8_t* raw = nullptr;
        for (std::uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
            raw = pool->acquire_block();
            if (raw != nullptr)
                break;
        }
        if (raw != nullptr) {
            pool->release_block(pool->ptr_to_block_id(raw));
        }
    }
}

__global__ void k_bench_add_elements(DynamicVecVecView<std::uint32_t>* view, std::uint32_t n, std::uint32_t elements_per_neuron) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    for (std::uint32_t i = 0; i < elements_per_neuron; ++i) {
        std::uint32_t val = i;
        (void)view->add(tid, std::move(val));
    }
}

} // namespace

void device_bench_acquire_release(DeviceSharedBlockPool* d_pool, std::uint32_t num_threads, std::uint32_t iterations) {
    k_bench_acquire_release<<<grid(num_threads), BLOCK>>>(d_pool, num_threads, iterations);
    cudaDeviceSynchronize();
}

void device_bench_add_elements(DynamicVecVecView<std::uint32_t>* d_view, std::uint32_t n_neurons, std::uint32_t elements_per_neuron) {
    k_bench_add_elements<<<grid(n_neurons), BLOCK>>>(d_view, n_neurons, elements_per_neuron);
    cudaDeviceSynchronize();
}

#endif // RELEARN_CUDA_ENABLED
