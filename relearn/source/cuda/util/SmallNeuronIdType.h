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

#include "Macros.h"

/**
 * A 3-byte neuron-ID type that packs a uint32 into 24 bits to save device memory.
 * Neuron IDs fit in 24 bits in practice, so storing them as 3 bytes instead of 4
 * reduces the per-edge memory footprint inside the adjacency lists on the GPU.
 */
struct SmallNeuronIdType {
    uint8_t b0{};
    uint8_t b1{};
    uint8_t b2{};

#ifndef CUDA_COMPILER
    SmallNeuronIdType() = default;

    SmallNeuronIdType(uint32_t value)
        : b0(static_cast<uint8_t>(value & 0xFF))
        , b1(static_cast<uint8_t>((value >> 8) & 0xFF))
        , b2(static_cast<uint8_t>((value >> 16) & 0xFF)) {
    }

    operator uint32_t() const {
        return static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1) << 8) | (static_cast<uint32_t>(b2) << 16);
    }

#else

    __host__ __device__ SmallNeuronIdType() = default;

    __host__ __device__
    SmallNeuronIdType(uint32_t value)
        : b0(value & 0xFF)
        , b1((value >> 8) & 0xFF)
        , b2((value >> 16) & 0xFF) { }

    __host__ __device__
    operator uint32_t() const {
        return uint32_t(b0) | (uint32_t(b1) << 8) | (uint32_t(b2) << 16);
    }
#endif
};

static_assert(sizeof(SmallNeuronIdType) == 3);

// Largest neuron ID SmallNeuronIdType can address. Simulations with more neurons than this must
// fall back to a 4-byte (std::uint32_t) neuron-ID representation on the GPU -- see
// CudaConfig::use_wide_neuron_ids.
inline constexpr std::uint32_t max_small_neuron_id = (1U << 24) - 1;
