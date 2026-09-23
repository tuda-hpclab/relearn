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

#include "cuda/memory/LazySyncedArray.h"

#include <cpp-utility/Cast.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>

struct BloomFilterView;

/**
 * Host-side owner of a per-neuron Bloom filter array backed by GPU memory.
 *
 * Each neuron gets its own fixed-size Bloom filter bitset.  The filter size and number of hash
 * functions are computed from the expected maximum edge count and a desired false-positive rate.
 */
class BloomFilter {
public:
    /**
     * @brief Constructs Bloom filters for @p _number_neurons neurons.
     * @param _number_neurons Number of neurons, each getting their own filter.
     * @param max_edges       Expected maximum number of edges per neuron (determines filter size).
     * @param target_fpr      Desired false-positive rate (e.g. 0.01 for 1%).
     */
    BloomFilter(std::uint32_t _number_neurons, uint32_t max_edges, float target_fpr) {
        if (max_edges == 0U) {
            words_per_filter = 1;
            return;
        }

        // Optimale Bits/Element: log₂(1/ε) / ln(2)
        const float bits_per_element = std::log2(1.0F / target_fpr) / std::numbers::ln2_v<float>;
        bit_per_edge = static_cast<std::uint32_t>(std::round(bits_per_element));

        // Optimale Hash-Anzahl: ln(2) · m/n
        number_hash_functions = std::max(1U, static_cast<uint32_t>(std::round(bits_per_element * std::numbers::ln2_v<float>)));

        // Aufrunden auf nächste Potenz von 2 (für schnelles Masking)
        const auto total_bits = static_cast<uint32_t>(std::round(static_cast<float>(max_edges) * bits_per_element));
        uint32_t n = 32U;
        while (n < total_bits) {
            n <<= 1U;
        }
        words_per_filter = std::min(n / 32U, 512U);

        bits.resize(static_cast<std::size_t>(_number_neurons) * words_per_filter, 0U);
    }

    /**
     * @brief Returns a lightweight GPU view struct suitable for passing to device kernels.
     *      Marks the underlying bit array device-modified, since kernels receiving this view
     *      (e.g. the bloom-rebuild kernel) may write into it via atomicOr.
     */
    [[nodiscard]] BloomFilterView get_gpu_view();

    /**
     * @brief Returns the total device memory (in bytes) used by all filter bitsets.
     */
    [[nodiscard]] uint64_t get_memory_footprint() const {
#ifdef RELEARN_CUDA_ENABLED
        return bits.get_memory_footprint();
#else
        return 0;
#endif
    }

    /**
     * @brief Fills all filter bits with @p i (use 0 to reset all filters).
     * @param i Value to fill into every word of every filter.
     */
    void fill([[maybe_unused]] unsigned i) {
#ifdef RELEARN_CUDA_ENABLED
        bits.fill(i);
#endif
    }

private:
    std::uint32_t bit_per_edge{};          ///< Bits allocated per expected edge.
    std::uint32_t number_hash_functions{}; ///< Number of independent hash probes per lookup.
    std::uint32_t words_per_filter{};      ///< Number of 32-bit words per neuron's filter.

    LazySyncedArray<std::uint32_t> bits{}; ///< Flat device array: number_neurons * words_per_filter words.
};
