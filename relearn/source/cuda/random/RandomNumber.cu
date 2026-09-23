/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "CudaConfig.h"
#include "RandomNumber.cuh"
#include "RandomNumberKeys.h"

#include "util/Timers.h"
#include "util/Util.cuh"

#include <curand_kernel.h>

#include <cstdio>

__device__ RandomNumbers::RandomNumbersConfig* d_configs{};

__host__ void init_random_configs(RandomNumbers::RandomNumbersConfig* _d_configs) {
    CUDA_CHECK(
        cudaMemcpyToSymbol(d_configs, &_d_configs, sizeof(RandomNumbers::RandomNumbersConfig*)));
    cudaDeviceSynchronize();
    kernelErrCheck();
}

__global__ void curand_setup_kernel(const CudaConfig::number_neurons_type num_neurons, curandState* state, const std::uint64_t seed) {
    const auto neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_id >= num_neurons) {
        return;
    }

    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(static_cast<unsigned long long>(seed), neuron_id, 0, &(state[neuron_id]));
}

void curand_setup_entry(const CudaConfig::number_neurons_type num_neurons, void* state, const std::uint64_t seed) {
    Timers::start(TimerRegion::CUDA_CURAND_ENTRY_KERNEL);

    const auto& [blocks, threads] = get_grid_ands_block_size(num_neurons, curand_setup_kernel);
    curand_setup_kernel<<<blocks, threads>>>(num_neurons, static_cast<curandState*>(state), seed);

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_CURAND_ENTRY_KERNEL);
}

__device__ double RandomNumbers::get_curand(const std::uint64_t neuron_id, const std::uint32_t random_key) {
    curandState* local_state = &(reinterpret_cast<curandState*>(d_configs[random_key].d_curand_states)[neuron_id]);
    if (d_configs[random_key].random_type == RandomNumberType::UNIFORM) {
        return curand_uniform_double(local_state);
    }
    if (d_configs[random_key].random_type == RandomNumberType::NORMAL) {
        return curand_normal_double(local_state);
    }
    __trap();
    return 0;
}

__device__ double RandomNumbers::get_random_value(const std::uint64_t neuron_id, const std::uint32_t key) {
    return get_curand(neuron_id, key);
}

__device__ void RandomNumbers::sample_k_unique(std::uint32_t* out, std::uint32_t k, std::uint32_t n,
                                               std::uint64_t neuron_id, std::uint32_t key) {
    for (std::uint32_t j = 0; j < k; ++j) {
        std::uint32_t r = get_random_value_int(n - j, neuron_id, key);
        // Map r to the r-th unchosen index by shifting past already-chosen
        // entries (out[0..j-1] is kept sorted ascending).
        for (std::uint32_t i = 0; i < j; ++i) {
            if (out[i] <= r)
                ++r;
            else
                break;
        }
        // Insertion-sort r into out[0..j]
        std::uint32_t pos = j;
        while (pos > 0 && out[pos - 1] > r) {
            out[pos] = out[pos - 1];
            --pos;
        }
        out[pos] = r;
    }
}

__device__ float RandomNumbers::get_stateless_random_number(const std::uint64_t thread_id, const std::uint64_t seed, const std::uint64_t state_idx) {

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, thread_id, state_idx, &rng);

    const auto x = curand_uniform(&rng);
    return x;
}
