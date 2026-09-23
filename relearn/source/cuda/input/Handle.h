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

#include "cuda/CudaConfig.h"

/** Abstract base for all fire-status communicator handles. */
struct FireStatusCommunicatorHandle {
    FireStatusCommunicatorHandle() = default;
    virtual ~FireStatusCommunicatorHandle() = default;

    FireStatusCommunicatorHandle(const FireStatusCommunicatorHandle&) = default;
    FireStatusCommunicatorHandle& operator=(const FireStatusCommunicatorHandle&) = default;
    FireStatusCommunicatorHandle(FireStatusCommunicatorHandle&&) = default;
    FireStatusCommunicatorHandle& operator=(FireStatusCommunicatorHandle&&) = default;
};

/** Fire-status handle backed by an uncompressed list of fired-neuron IDs received from remote ranks. */
struct FireStatusCommunicatorUncompressedHandle : FireStatusCommunicatorHandle {
    explicit FireStatusCommunicatorUncompressedHandle(const int* _incoming_displ,
                                                      const CudaConfig::number_neurons_type* _incoming_neuron_ids)
        : incoming_displ(_incoming_displ)
        , incoming_neuron_ids(_incoming_neuron_ids) {
    }

    const int* incoming_displ{ nullptr };                                  ///< Per-rank start offset into incoming_neuron_ids.
    const CudaConfig::number_neurons_type* incoming_neuron_ids{ nullptr }; ///< Flat array of fired-neuron IDs from all remote ranks.

    ~FireStatusCommunicatorUncompressedHandle() override = default;

    FireStatusCommunicatorUncompressedHandle(const FireStatusCommunicatorUncompressedHandle&) = default;
    FireStatusCommunicatorUncompressedHandle& operator=(const FireStatusCommunicatorUncompressedHandle&) = default;
    FireStatusCommunicatorUncompressedHandle(FireStatusCommunicatorUncompressedHandle&&) = default;
    FireStatusCommunicatorUncompressedHandle& operator=(FireStatusCommunicatorUncompressedHandle&&) = default;
};

enum class FiredStatus : char;

/** Fire-status handle for the local rank's own neurons (direct pointer into the fire-status array). */
struct FireStatusLocalVectorHandle : FireStatusCommunicatorHandle {
    explicit FireStatusLocalVectorHandle(const FiredStatus* _fired)
        : fired(_fired) {
    }

    const FiredStatus* fired; ///< Per-neuron fire status for this rank's neurons.

    ~FireStatusLocalVectorHandle() override = default;

    FireStatusLocalVectorHandle(const FireStatusLocalVectorHandle&) = default;
    FireStatusLocalVectorHandle& operator=(const FireStatusLocalVectorHandle&) = default;
    FireStatusLocalVectorHandle(FireStatusLocalVectorHandle&&) = default;
    FireStatusLocalVectorHandle& operator=(FireStatusLocalVectorHandle&&) = default;
};
