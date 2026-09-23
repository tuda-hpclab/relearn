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

#include "network_graph/NetworkHandle.h"
#include "cuda/CudaConfig.h"
#include "cuda/wrapper/EventWrapper.h"
#include "cuda/wrapper/StreamWrapper.h"

#include <optional>

/** Whether synapse weights are stored explicitly or treated as uniform 1. */
enum class WeightMode {
    Weighted,  ///< Each synapse carries its own weight value.
    Unweighted ///< All synapses have weight 1; no weight array is accessed.
};

/** How the set of fired neurons is encoded for lookup during input accumulation. */
enum class SpikeMode {
    LocalVector, ///< Explicit array of local fired-neuron IDs.
    Set,         ///< Unordered set of fired-neuron IDs.
    BinarySearch ///< Sorted array with binary search for each lookup.
};

/** Layout of the network graph on the GPU. */
enum class NetworkMode {
    Default, ///< Standard per-neuron adjacency lists.
    Sorted,  ///< Adjacency lists sorted by target MPI rank for coalesced access.
    Map      ///< Adjacency lists accelerated with per-rank GPU hash maps.
};

/** Which data structure carries the fired-neuron information into the input kernel. */
enum class FireInformation {
    NeuronIDs,   ///< Flat array of fired neuron IDs.
    LocalVector, ///< Vector of local fired-neuron IDs (same rank).
};

/** Whether the kernel iterates over all neurons or only over the fired spikes. */
enum class IterationMode {
    Neurons, ///< One thread per (target) neuron; iterates its incoming edges.
    Spikes   ///< One thread per spike; iterates the outgoing edges of the firing neuron.
};

/** Selects the kernel variant to use for a given input-computation call. */
struct LaunchConfig {
    WeightMode weight;
    SpikeMode spike;
    NetworkMode network;
    FireInformation fire_information;
    IterationMode iteration = IterationMode::Neurons;
};

/** Bundles the GPU handles needed by the selected kernel variant. */
struct LaunchHandles {
    const NetworkHandle network;  ///< GPU view of the network graph.
    const void* fire_information; ///< Pointer to the appropriate fire-information structure.
};

/**
 * @brief Dispatches the synaptic-input accumulation kernel for a range of neurons.
 *
 * Selects the specific kernel variant at runtime based on @p config, then launches it
 * asynchronously on @p stream_wrapper.
 *
 * @param first                 First neuron index to process (inclusive).
 * @param last                  Last neuron index to process (exclusive).
 * @param d_input               Device array of input currents; accumulated in place.
 * @param config                Selects the kernel variant (weight mode, spike encoding, etc.).
 * @param handles               GPU data handles required by the selected variant.
 * @param stream_wrapper        CUDA stream to launch on.
 * @param my_rank               MPI rank of this process.
 * @param local                 If true, process local (same-rank) synapses.
 * @param local_distant_helper  If true, process a helper pass for local-distant synapse blending.
 * @param synapse_conductance   Global conductance scale applied to all synapse weights.
 * @return An EventWrapper that fires when the kernel completes, or nullopt if nothing was launched.
 */
[[nodiscard]] std::optional<EventWrapper> launch(CudaConfig::number_neurons_type first,
                                                 CudaConfig::number_neurons_type last,
                                                 CudaConfig::input_type* d_input,
                                                 const LaunchConfig& config,
                                                 const LaunchHandles& handles,
                                                 const std::shared_ptr<StreamWrapper>& stream_wrapper,
                                                 CudaConfig::mpi_rank_type my_rank, bool local, bool local_distant_helper, CudaConfig::input_type synapse_conductance);

/**
 * @brief Releases device memory used by the set-based spike-activity accumulator.
 */
void release_synaptic_activity_set();
