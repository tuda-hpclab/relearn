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

#include "BloomFilter.h"

struct BloomFilterView {
    std::uint32_t bit_per_edge{};
    std::uint32_t number_hash_functions{};
    std::uint32_t words_per_filter{};
    std::uint32_t* bits{};

    // Murmur3-inspired finalizer seeded by round index i.
    __device__ inline uint32_t bloom_hash(uint32_t key, uint32_t i) {
        uint32_t x = key + i * 0x9e3779b9u;
        x ^= x >> 16u;
        x *= 0x45d9f3bu;
        x ^= x >> 16u;
        return x;
    }

    // Returns true if key MIGHT be in the filter stored for neuron_id.
    // Never false-negative: if key was inserted, this always returns true.
    __device__ inline bool bloom_query(uint32_t neuron_id, uint32_t key) {
        const uint32_t* f = bits + static_cast<uint64_t>(neuron_id) * words_per_filter;
        const uint32_t mask = words_per_filter * 32u - 1u;
        for (uint32_t i = 0u; i < number_hash_functions; ++i) {
            const uint32_t bit_pos = bloom_hash(key, i) & mask;
            if (!(f[bit_pos >> 5u] & (1u << (bit_pos & 31u))))
                return false;
        }
        return true;
    }
};