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

#include "cuda/CudaConfig.h"

/**
 * @brief Copies the array of per-stream random-number configs to device constant memory.
 * @param _d_configs Device pointer to the config array (one entry per RandomNumberKey).
 */
void init_random_configs(RandomNumbers::RandomNumbersConfig* _d_configs);

/**
 * @brief Initializes cuRAND states for all neurons using the given seed.
 * @param num_neurons Number of cuRAND states to initialize (one per neuron).
 * @param state       Device pointer to the cuRAND state array.
 * @param seed        Base seed; each thread derives a unique sequence from it.
 */
void curand_setup_entry(CudaConfig::number_neurons_type num_neurons, void* state, std::uint64_t seed);
