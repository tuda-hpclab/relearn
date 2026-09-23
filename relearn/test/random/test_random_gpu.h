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

#ifdef RELEARN_CUDA_ENABLED

#include <cstdint>
#include <vector>

// Launches a single thread that calls RandomNumbers::sample_k_unique(out, k, n, neuron_id=0, key)
// against the cuRAND state registered under `key`, and returns the k sampled values.
std::vector<std::uint32_t> device_sample_k_unique(std::uint32_t key, std::uint32_t k, std::uint32_t n);

// Launches num_draws threads (one cuRAND state per thread) and returns num_draws independently
// drawn values from RandomNumbers::get_random_value for the given key.
std::vector<double> device_draw_random_values(std::uint32_t key, std::uint32_t num_draws);

#endif
