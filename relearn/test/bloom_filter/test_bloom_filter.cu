/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_bloom_filter.h"

#ifdef RELEARN_CUDA_ENABLED

#include "cuda/network_graph/BloomFilter.cuh"
#include "cuda/network_graph/BloomFilter.h"
#include "cuda/util/Util.cuh"

#include <cuda_runtime.h>

namespace {

__global__ void k_bloom_insert(BloomFilterView view, std::uint32_t neuron_id, const std::uint32_t* keys, std::uint32_t num_keys) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_keys) {
        return;
    }
    // Mirrors the atomicOr insertion pattern used by rebuild_bloom_kernel / OnlyOutgoingView::add_synapse.
    const auto key = keys[tid];
    auto* f = view.bits + static_cast<std::uint64_t>(neuron_id) * view.words_per_filter;
    const auto mask = view.words_per_filter * 32u - 1u;
    for (auto h = 0u; h < view.number_hash_functions; ++h) {
        const auto bit_pos = view.bloom_hash(key, h) & mask;
        atomicOr(&f[bit_pos >> 5u], 1u << (bit_pos & 31u));
    }
}

__global__ void k_bloom_query(BloomFilterView view, std::uint32_t neuron_id, const std::uint32_t* keys, std::uint8_t* out, std::uint32_t num_keys) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_keys) {
        return;
    }
    out[tid] = view.bloom_query(neuron_id, keys[tid]) ? std::uint8_t{ 1 } : std::uint8_t{ 0 };
}

} // namespace

void device_bloom_insert(BloomFilter& filter, std::uint32_t neuron_id, const std::vector<std::uint32_t>& keys_to_insert) {
    if (keys_to_insert.empty()) {
        return;
    }

    const auto view = filter.get_gpu_view();
    const auto n = static_cast<std::uint32_t>(keys_to_insert.size());

    std::uint32_t* d_keys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, sizeof(std::uint32_t) * n));
    CUDA_CHECK(cudaMemcpy(d_keys, keys_to_insert.data(), sizeof(std::uint32_t) * n, cudaMemcpyHostToDevice));

    constexpr std::uint32_t block_size = 128U;
    const auto grid_size = (n + block_size - 1U) / block_size;
    k_bloom_insert<<<grid_size, block_size>>>(view, neuron_id, d_keys, n);
    cudaDeviceSynchronize();
    kernelErrCheck();

    CUDA_CHECK(cudaFree(d_keys));
}

std::vector<std::uint8_t> device_bloom_query(BloomFilter& filter, std::uint32_t neuron_id, const std::vector<std::uint32_t>& keys_to_query) {
    const auto view = filter.get_gpu_view();
    const auto n = static_cast<std::uint32_t>(keys_to_query.size());

    std::uint32_t* d_keys = nullptr;
    std::uint8_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, sizeof(std::uint32_t) * n));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(std::uint8_t) * n));
    CUDA_CHECK(cudaMemcpy(d_keys, keys_to_query.data(), sizeof(std::uint32_t) * n, cudaMemcpyHostToDevice));

    constexpr std::uint32_t block_size = 128U;
    const auto grid_size = (n + block_size - 1U) / block_size;
    k_bloom_query<<<grid_size, block_size>>>(view, neuron_id, d_keys, d_out, n);
    cudaDeviceSynchronize();
    kernelErrCheck();

    auto h_out = std::vector<std::uint8_t>(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(std::uint8_t) * n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_out));

    return h_out;
}

#endif
