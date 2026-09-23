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

#include "RandomNumberKeys.h"
#include "cuda_runtime.h"

#include <curand_kernel.h>

#include <cassert>

namespace RandomNumbers {

__device__ double get_curand(std::uint64_t neuron_id, std::uint32_t key);

__device__ double get_random_value(std::uint64_t neuron_id, std::uint32_t key);

__device__ float get_stateless_random_number(std::uint64_t thread_id, std::uint64_t seed, std::uint64_t state_idx);

template <typename T>
__device__ T get_random_value_int(const T max_exclusive, const std::uint64_t neuron_id, const std::uint32_t key) {
    const auto x = get_random_value(neuron_id, key);
    const auto k = static_cast<T>(x * max_exclusive);
    return k;
}

// Fills out[0..k-1] with k unique indices sampled without replacement from [0, n).
// Indices are stored in ascending order.  Exactly k calls to get_random_value_int,
// no retries.  out must point to at least k writable uint32_t slots.
__device__ void sample_k_unique(std::uint32_t* out, std::uint32_t k, std::uint32_t n,
                                std::uint64_t neuron_id, std::uint32_t key);
}; // namespace RandomNumbers