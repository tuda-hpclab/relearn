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

namespace RandomNumbers {

/**
 * @brief Registers a random-number stream, allocating device state and pre-drawn buffers as needed.
 * @param key            Identifies the consumer (e.g. POISSON_INPUT, DELETE).
 * @param type           Distribution family (UNIFORM or NORMAL).
 * @param number_neurons Number of neurons; determines the cuRAND state array size.
 * @param seed           RNG seed.
 * @return Opaque handle index used to retrieve values from this stream.
 */
[[nodiscard]] std::uint32_t
register_random_numbers(RandomNumberKey key, RandomNumberType type,
                        std::uint64_t number_neurons, std::uint64_t seed);

/**
 * @brief Returns total device memory (in bytes) currently allocated for all registered random-number streams.
 */
[[nodiscard]] std::size_t get_memory_usage();

/**
 * @brief Frees all device state allocated by register_random_numbers() calls so far and resets
 *      the registration count back to zero.
 *
 * register_random_numbers() hands out a fixed number of slots (max_number_random_keys) for the
 * entire process lifetime and never reclaims them on its own -- calling this between independent
 * uses (e.g. between tests, so each test starts with a clean slate) avoids exhausting that pool.
 */
void reset();

}; // namespace RandomNumbers
