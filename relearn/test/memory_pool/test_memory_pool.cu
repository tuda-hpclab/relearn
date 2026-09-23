/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_memory_pool.h"

#include "cuda/memory/DeviceVecVec.h"
#include "cuda/memory/SharedBlockPool.h"
#include "cuda/memory/DeviceVecVec.cuh"
#include "cuda/memory/SharedBlockPool.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

using u32 = std::uint32_t;
using View = DynamicVecVecView<u32>;

// ── Kernels ───────────────────────────────────────────────────────────────────

// Thread i adds (i * multiplier) to neuron i.
__global__ void k_add_one_per_neuron(View* view, u32 n, u32 multiplier) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    u32 val = tid * multiplier;
    (void)view->add(tid, std::move(val));
}

// Thread 0 adds all values in d_values sequentially to neuron_id.
__global__ void k_add_multiple(View* view, u32 neuron_id, const u32* d_values, u32 n) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    for (u32 i = 0; i < n; i++) {
        u32 val = d_values[i];
        (void)view->add(neuron_id, std::move(val));
    }
}

// Each thread reads view->get(neuron_ids[tid], elem_idxs[tid]) into d_out[tid].
__global__ void k_get(const View* view, const u32* neuron_ids, const u32* elem_idxs, u32* d_out, u32 n) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    d_out[tid] = view->get(neuron_ids[tid], elem_idxs[tid]);
}

// Thread 0 drains all elements of neuron_id into d_out via cursor; writes count to d_count.
__global__ void k_cursor_collect(View* view, u32 neuron_id, u32* d_out, u32* d_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    auto it = view->begin(neuron_id);
    const auto sentinel = view->end(neuron_id);
    u32 count = 0;
    while (!(it == sentinel)) {
        d_out[count++] = it.next();
    }
    *d_count = count;
}

// Each thread repeatedly acquires a block, marks it as owned by its own thread id (checking
// no one else already owns it), holds it briefly, releases ownership, then returns it to the
// pool. See test_memory_pool.h for what `violations` measures.
//
// Each acquire attempt is bounded by timeout_cycles (an SM clock-cycle count read via clock64();
// safe to compare directly since all threads here are in one block and thus share one SM's
// clock). timeout_cycles is chosen generously enough that this trivial amount of work (a few
// hundred thousand CAS operations total) should never come close to it on any real GPU -- if a
// thread does hit it, that's a real problem (a liveness bug, or a GPU pathologically starved of
// time to run this kernel), and bailing out here converts what would otherwise be an unbounded
// hang into a fast, clearly-diagnosed test failure.
__global__ void k_stress_pool(DeviceSharedBlockPool* pool, u32* owner, u32 iterations, u32* violations, u32* timeouts, long long timeout_cycles) {
    constexpr u32 UNOWNED = 0xFFFFFFFFu;
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (u32 it = 0; it < iterations; ++it) {
        std::uint8_t* raw = nullptr;
        const long long attempt_start = clock64();
        while ((raw = pool->acquire_block()) == nullptr) {
            if (clock64() - attempt_start > timeout_cycles) {
                atomicAdd(timeouts, 1u);
                return; // this thread never held a block this iteration, so nothing to release
            }
        }
        const auto block_id = pool->ptr_to_block_id(raw);

        const auto prev_owner = atomicExch(&owner[block_id], tid);
        if (prev_owner != UNOWNED) {
            atomicAdd(violations, 1u);
        }

        int spin = 0;
        volatile int* spin_counter = &spin;
        while (*spin_counter < 64) {
            *spin_counter = *spin_counter + 1;
        }

        atomicExch(&owner[block_id], UNOWNED);
        pool->release_block(block_id);
    }
}

// Single-threaded: repeatedly acquires blocks until the pool reports empty (no other thread
// running concurrently), recording each block id. Used to verify, after a stress run, that the
// pool's at-rest free list contains exactly the right set of ids -- not derived from
// current_alloc/free-count bookkeeping, which contention can inflate independent of any actual
// leak (see test_memory_pool.cpp for why that metric isn't used directly).
__global__ void k_drain_pool(DeviceSharedBlockPool* pool, u32* out_ids, u32* out_count, u32 max_to_drain) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    u32 count = 0;
    while (count < max_to_drain) {
        auto* raw = pool->acquire_block();
        if (raw == nullptr)
            break;
        out_ids[count++] = pool->ptr_to_block_id(raw);
    }
    *out_count = count;
}

// Releases one specific block id back to the pool's free list.
__global__ void k_release_block(DeviceSharedBlockPool* pool, u32 block_id) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    pool->release_block(block_id);
}

// ── Bridge helpers ────────────────────────────────────────────────────────────

static constexpr unsigned BLOCK = 256;

static unsigned grid(unsigned n) {
    return (n + BLOCK - 1) / BLOCK;
}

template <typename T>
static T* push(const std::vector<T>& h) {
    T* d{};
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <typename T>
static std::vector<T> pull(const T* d, std::size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

// ── Bridge functions ──────────────────────────────────────────────────────────

void device_add_one_per_neuron(View* d_view, u32 n, u32 multiplier) {
    k_add_one_per_neuron<<<grid(n), BLOCK>>>(d_view, n, multiplier);
    cudaDeviceSynchronize();
}

void device_add_multiple_to_neuron(View* d_view, u32 neuron_id,
                                   const std::vector<u32>& values) {
    u32* d_values = push(values);
    k_add_multiple<<<1, 1>>>(d_view, neuron_id, d_values, static_cast<u32>(values.size()));
    cudaDeviceSynchronize();
    cudaFree(d_values);
}

std::vector<u32> device_cursor_collect(View* d_view, u32 neuron_id, u32 max_elems) {
    u32* d_out{};
    u32* d_count{};
    cudaMalloc(&d_out, max_elems * sizeof(u32));
    cudaMalloc(&d_count, sizeof(u32));

    k_cursor_collect<<<1, 1>>>(d_view, neuron_id, d_out, d_count);
    cudaDeviceSynchronize();

    u32 count{};
    cudaMemcpy(&count, d_count, sizeof(u32), cudaMemcpyDeviceToHost);

    auto result = pull(d_out, count);
    cudaFree(d_out);
    cudaFree(d_count);
    return result;
}

std::vector<u32> device_get_at(View* d_view,
                               const std::vector<u32>& neuron_ids,
                               const std::vector<u32>& elem_idxs) {
    const auto n = static_cast<u32>(neuron_ids.size());
    u32* d_neuron_ids = push(neuron_ids);
    u32* d_elem_idxs = push(elem_idxs);
    u32* d_out{};
    cudaMalloc(&d_out, n * sizeof(u32));

    k_get<<<grid(n), BLOCK>>>(d_view, d_neuron_ids, d_elem_idxs, d_out, n);
    cudaDeviceSynchronize();

    auto result = pull(d_out, n);
    cudaFree(d_neuron_ids);
    cudaFree(d_elem_idxs);
    cudaFree(d_out);
    return result;
}

StressPoolResult device_stress_pool(DeviceSharedBlockPool* d_pool, u32 num_blocks, u32 num_threads, u32 iterations) {
    u32* d_owner{};
    cudaMalloc(&d_owner, num_blocks * sizeof(u32));
    cudaMemset(d_owner, 0xFF, num_blocks * sizeof(u32));

    u32* d_violations{};
    cudaMalloc(&d_violations, sizeof(u32));
    cudaMemset(d_violations, 0, sizeof(u32));

    u32* d_timeouts{};
    cudaMalloc(&d_timeouts, sizeof(u32));
    cudaMemset(d_timeouts, 0, sizeof(u32));

    // Generous even on a very slow/heavily time-sliced GPU: at a conservative 500 MHz SM clock,
    // this is 20 seconds per acquire attempt -- for reference, the whole stress run's total
    // useful work is a few hundred thousand CAS operations, normally well under a second.
    constexpr long long timeout_cycles = 10'000'000'000LL;

    k_stress_pool<<<1, num_threads>>>(d_pool, d_owner, iterations, d_violations, d_timeouts, timeout_cycles);
    cudaDeviceSynchronize();

    StressPoolResult result{};
    cudaMemcpy(&result.violations, d_violations, sizeof(u32), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.timed_out_threads, d_timeouts, sizeof(u32), cudaMemcpyDeviceToHost);

    cudaFree(d_owner);
    cudaFree(d_violations);
    cudaFree(d_timeouts);
    return result;
}

void device_release_block(DeviceSharedBlockPool* d_pool, u32 block_id) {
    k_release_block<<<1, 1>>>(d_pool, block_id);
    cudaDeviceSynchronize();
}

std::vector<u32> device_drain_pool(DeviceSharedBlockPool* d_pool, u32 max_to_drain) {
    u32* d_ids{};
    cudaMalloc(&d_ids, max_to_drain * sizeof(u32));
    u32* d_count{};
    cudaMalloc(&d_count, sizeof(u32));

    k_drain_pool<<<1, 1>>>(d_pool, d_ids, d_count, max_to_drain);
    cudaDeviceSynchronize();

    u32 count{};
    cudaMemcpy(&count, d_count, sizeof(u32), cudaMemcpyDeviceToHost);
    auto result = pull(d_ids, count);

    cudaFree(d_ids);
    cudaFree(d_count);
    return result;
}
