/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

#include <cstdint>

struct DeviceSharedBlockPool;

template <typename T>
class DynamicVecVecView;

// Launches num_threads threads (one block, so occupancy doesn't hide contention effects),
// each repeatedly acquiring a block from d_pool and immediately releasing it, iterations times.
// Synchronizes internally before returning.
void device_bench_acquire_release(DeviceSharedBlockPool* d_pool, std::uint32_t num_threads, std::uint32_t iterations);

// Launches one thread per neuron; thread i sequentially adds elements_per_neuron values to
// neuron i, exercising DynamicVecVec's overflow-chunk growth (and therefore the shared pool's
// acquire path) whenever a neuron's current chunk fills up. Synchronizes internally.
void device_bench_add_elements(DynamicVecVecView<std::uint32_t>* d_view, std::uint32_t n_neurons, std::uint32_t elements_per_neuron);
