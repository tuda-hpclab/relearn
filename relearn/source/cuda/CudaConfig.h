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

#include "types/BasicTypes.h"

#include <cstdint>

/**
 * Central configuration class for all CUDA-accelerated simulation code.
 *
 * Provides type aliases mapping host simulation types to GPU-friendly names,
 * kernel launch geometry constants, and runtime-configurable algorithm flags.
 */
class CudaConfig {
public:
    // --- Type aliases mapping host simulation types to GPU-friendly names ---
    using static_synapse_weight = RelearnTypes::static_synapse_weight;
    using plastic_synapse_weight = RelearnTypes::plastic_synapse_weight;
    using number_neurons_type = std::uint32_t;
    using counter_type = RelearnTypes::counter_type;
    using synaptic_count_type = RelearnTypes::counter_type;
    using synaptic_grown_type = RelearnTypes::grown_type;
    using bh_index_type = std::uint32_t; ///< Index into the linearized Barnes-Hut octree.
    using gaussian_type = RelearnTypes::acceptance_criterion_type;
    using mpi_rank_type = int;
    using membrane_potential_type = RelearnTypes::activity_type;
    using input_type = RelearnTypes::activity_type;
    using calcium_type = RelearnTypes::calcium_type;

    // --- Kernel launch geometry ---
    static inline std::size_t expected_synapses_per_neuron = 1000; ///< Expected number of synapses per neuron; used to size GPU edge storage.

    inline static bool use_pre_drawn_cpu{ false }; ///< Use pre-drawn CPU random numbers instead of cuRAND.

    inline static int local_gpu_id; ///< CUDA device ID assigned to this MPI rank.

    /** Scaling function applied to synaptic input before accumulation. */
    enum scaling_function_enum {
        LINEAR,
        LOGARITHMIC,
        HYPERBOLIC_TANGENT
    };

    // Set once, before any GPUEdgesBase is constructed, from the actual (global) neuron count
    // (see Simulation::initialize()). When true, GPU edge storage packs neuron IDs into 4 bytes
    // (std::uint32_t) instead of the default 3 bytes (SmallNeuronIdType), since 3 bytes can only
    // address up to max_small_neuron_id (2^24 - 1) distinct neuron IDs.
    inline static bool use_wide_neuron_ids = false;

    // Each neuron's edge/synapse storage (DynamicVecVec, see GPUEdges/NetworkGraphGPU) starts as a
    // fixed-size "main chunk" (see non_overflow_size_factor/local_edges_ratio below); once a device
    // kernel's atomic-append fills it, the neuron overflows into additional fixed-size "overflow
    // chunks" pulled from a shared pool (SharedBlockPool) allocated once up front -- the pool
    // cannot grow during a kernel, so overflow_chunk_size_factor/number_overflow_chunks_factor must
    // together leave enough headroom for concurrent per-neuron overshoot, or appends silently fail
    // (see DynamicVecVecView::add's `return false` on pool exhaustion).
    //
    // Size (in elements) of a single overflow chunk, as a fraction of expected_synapses_per_neuron
    // (i.e. overflow chunks are smaller than the main chunk). Feeds SharedBlockPool's fixed
    // block size (see DeviceVecVec.cu, NetworkGraphGPU.cpp).
    inline static double overflow_chunk_size_factor = 0.2;
    // Multiplier for the total number of overflow chunks pre-allocated in the shared pool, relative
    // to the number of buffers (neurons x number of edge-storage instances sharing the pool). This
    // is the safety margin against pool exhaustion described above -- too small risks silently
    // dropped edges under concurrent overflow, too large wastes GPU memory.
    inline static double number_overflow_chunks_factor = 2.0;
    // Scales down the main (non-overflow) chunk's initial size, relative to
    // expected_synapses_per_neuron, specifically for local edges -- leaves room so local + distant
    // main-chunk allocations don't overrun the total edge budget before overflow chunks are needed.
    inline static double non_overflow_size_factor = 0.8;
    // Fraction of the main-chunk budget given to local edges vs. distant edges (which get
    // 1 - local_edges_ratio), i.e. how per-neuron main-chunk capacity is split between the two
    // edge categories before either needs to overflow.
    inline static double local_edges_ratio = 0.7;
};
