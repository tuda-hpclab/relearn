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

#include <cstddef>
#include <cstdint>

/**
 * Identifies the consumer of a pre-drawn random-number stream.
 */
enum RandomNumberKey {
    POISSON_INPUT = 0, ///< Poisson-distributed input currents.
    NORMAL_INPUT = 1,  ///< Normally-distributed input currents.
    DELETE = 2,        ///< Random synapse-deletion draws.
    BARNES_HUT = 3,    ///< Target-selection draws in the Barnes-Hut algorithm.
};

/**
 * Distribution family for a random-number stream.
 */
enum RandomNumberType {
    UNIFORM, ///< Uniform distribution on [0, 1).
    NORMAL   ///< Standard normal distribution.
};

constexpr int max_number_random_keys = 100;

namespace RandomNumbers {

/**
 * Per-stream configuration placed in device memory and used by kernels to draw random numbers.
 */
struct RandomNumbersConfig {
    bool d_use_pre_drawn_cpu{};       ///< If true, use pre-drawn CPU values instead of cuRAND.
    double* d_pre_drawn{};            ///< Device pointer to pre-drawn values.
    std::size_t* d_pre_drawn_state{}; ///< Current consumption offset into d_pre_drawn.
    void* d_curand_states{};          ///< cuRAND state array (one state per neuron).
    std::uint64_t number_neurons{};   ///< Number of neurons (and cuRAND states) allocated.
    RandomNumberType random_type{};   ///< Distribution to generate.
    std::size_t number_values{};      ///< Number of values stored in d_pre_drawn.
};

} // namespace RandomNumbers
