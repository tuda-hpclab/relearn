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

#include "cuda/CudaTypes.h"
#include "types/BasicTypes.h"

#include <cstdint>
#include <span>
#include <vector>

template <typename t>
class DeviceArray;

namespace NaiveCUDA_CU {

/**
 * One batch of axon-centric target-selection tasks: which neurons have vacant axons
 * (mapping), how many vacant dendrites each neuron currently offers (vacant_dendrites),
 * the pre-drawn Gaussian-kernel thresholds (picked_probabilities), and how many tasks
 * (== mapping.size() == picked_probabilities.size()) there are in total.
 */
struct NaiveTargetSelectionTask {
    std::span<const RelearnTypes::counter_type> mapping;
    std::span<const RelearnTypes::counter_type> vacant_dendrites;
    std::span<const double> picked_probabilities;
    std::uint64_t target_size;
};

/**
 * @brief Finds synapse candidates using a flat task list (axon-centric).
 *
 * Each entry in task.mapping represents one vacant axon; the kernel draws one target
 * dendrite per entry from all neurons with vacant dendrites.
 *
 * @param d_neuron_pos                         Device pointer to neuron positions.
 * @param neurons_count                        Total number of neurons.
 * @param task                                  The batch of tasks to process (mapping,
 *                                              vacant dendrites, pre-drawn probabilities, size).
 * @param h_found_target_dendrites             Output: target neuron indices per vacant axon.
 * @param squared_sigma_inv                    1 / (2σ²) for the Gaussian kernel.
 */
void find_target_neurons(const DeviceArray<SimpleVec3d>& d_neuron_pos, std::uint64_t neurons_count,
                         NaiveTargetSelectionTask task,
                         std::vector<std::uint64_t>& h_found_target_dendrites,
                         double squared_sigma_inv);

} // namespace NaiveCUDA_CU
