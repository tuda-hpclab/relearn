/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_random_gpu.h"

#ifdef RELEARN_CUDA_ENABLED

#include "cuda/random/RandomNumber.cuh"
#include "cuda/util/Util.cuh"

#include <cuda_runtime.h>

namespace {

__global__ void k_sample_k_unique(std::uint32_t* out, std::uint32_t k, std::uint32_t n, std::uint32_t key) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    RandomNumbers::sample_k_unique(out, k, n, /*neuron_id=*/0, key);
}

__global__ void k_draw_random_values(double* out, std::uint32_t num_draws, std::uint32_t key) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_draws) {
        return;
    }
    out[tid] = RandomNumbers::get_random_value(tid, key);
}

} // namespace

std::vector<std::uint32_t> device_sample_k_unique(std::uint32_t key, std::uint32_t k, std::uint32_t n) {
    std::uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(std::uint32_t) * k));

    k_sample_k_unique<<<1, 1>>>(d_out, k, n, key);
    cudaDeviceSynchronize();
    kernelErrCheck();

    auto h_out = std::vector<std::uint32_t>(k);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(std::uint32_t) * k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    return h_out;
}

std::vector<double> device_draw_random_values(std::uint32_t key, std::uint32_t num_draws) {
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * num_draws));

    constexpr std::uint32_t block_size = 128U;
    const auto grid_size = (num_draws + block_size - 1U) / block_size;
    k_draw_random_values<<<grid_size, block_size>>>(d_out, num_draws, key);
    cudaDeviceSynchronize();
    kernelErrCheck();

    auto h_out = std::vector<double>(num_draws);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(double) * num_draws, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    return h_out;
}

#endif
