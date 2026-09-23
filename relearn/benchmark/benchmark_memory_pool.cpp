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

#include "main.h"

#include "cuda/memory/DeviceVecVec.h"
#include "cuda/memory/SharedBlockPool.h"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>

namespace {

// Fixed at a size much smaller than the thread counts benchmarked below, so the higher end of
// the sweep genuinely exercises acquire/release under contention rather than every thread
// getting its own block on the first try.
constexpr std::size_t pool_num_blocks = 64;
constexpr std::size_t pool_block_bytes = 256;

constexpr auto acquire_release_iterations = std::uint32_t{ 50 };

// Repeatedly acquires and releases a block from a small shared pool -- the allocator's raw
// throughput, isolated from DynamicVecVec's chunk-growth logic. num_threads sweeps from well
// below to well above pool_num_blocks to show how throughput degrades under contention.
void BM_SharedBlockPool_AcquireRelease(benchmark::State& state) {
    const auto num_threads = static_cast<std::uint32_t>(state.range(0));

    auto pool = std::make_unique<SharedBlockPool>(pool_num_blocks, pool_block_bytes);

    for (auto _ : state) {
        device_bench_acquire_release(pool->get_device_pool(), num_threads, acquire_release_iterations);
    }
}

// One thread per neuron, each sequentially appending elements_per_neuron values. init_size is
// kept small relative to elements_per_neuron so most neurons overflow their main chunk at least
// once, forcing repeated SharedBlockPool::acquire_block calls through DynamicVecVec's own growth
// path -- the allocator as actually used, rather than in isolation.
constexpr std::size_t dynamic_vec_vec_init_size = 4;

void BM_DynamicVecVec_Add(benchmark::State& state) {
    const auto number_neurons = static_cast<std::uint32_t>(state.range(0));
    const auto elements_per_neuron = static_cast<std::uint32_t>(state.range(1));

    auto pool = std::make_unique<SharedBlockPool>(
        static_cast<std::size_t>(number_neurons) * elements_per_neuron / dynamic_vec_vec_init_size + 16,
        pool_block_bytes);

    for (auto _ : state) {
        state.PauseTiming();
        auto vec = DynamicVecVec<std::uint32_t>(number_neurons, dynamic_vec_vec_init_size, pool.get());
        state.ResumeTiming();

        device_bench_add_elements(vec.get_device_view(), number_neurons, elements_per_neuron);
    }
}

} // namespace

BENCHMARK(BM_SharedBlockPool_AcquireRelease)
    ->Unit(benchmark::kMillisecond)
    ->Arg(32)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Iterations(medium_number_iterations);

BENCHMARK(BM_DynamicVecVec_Add)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 4 })
    ->Args({ small_number_neurons, 32 })
    ->Iterations(medium_number_iterations);

#endif // RELEARN_CUDA_ENABLED
