/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "CudaTypes.h"
#include "NaiveCUDA_CU.h"

#include "cuda/CudaTypes.h"
#include "cuda/algorithm/Kernel.cuh"
#include "cuda/util/Util.cuh"
#include "util/Timers.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <stdexcept>

namespace NaiveCUDA_CU {

__global__ void
find_target_neurons_kernel(const std::uint64_t number_neurons, const SimpleVec3d* neuron_pos,
                           const std::uint32_t* vacant_dendrites, const std::uint64_t number_nodes,
                           const std::uint64_t* vacant_axons_neuron_id_mapping,
                           const double* picked_probabilities,
                           std::uint64_t* found_target_neurons, const double squared_sigma_inv) {
    const auto thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id >= number_nodes) {
        return;
    }
    const auto i = thread_id;
    // for (auto i = thread_id; i < number_nodes; i += stride) {
    const auto neuron_id_with_vacant_axon = vacant_axons_neuron_id_mapping[i];
    if (neuron_id_with_vacant_axon != std::numeric_limits<std::uint64_t>::max()) {
        auto total_sum_probabilities = 0.0;
        const auto source_position = neuron_pos[neuron_id_with_vacant_axon];
        auto picked_index = 0;
        auto sum_probabilities = 0.0;

        for (auto j = 0ULL; j < number_neurons; j++) {
            const auto target_free_count = vacant_dendrites[j];
            const auto target_position = neuron_pos[j];

            total_sum_probabilities += calculate_attractiveness_to_connect(source_position, target_position,
                                                                           target_free_count, false, squared_sigma_inv);
        }

        // recalculate probabilities if all target neurons are too far away
        if (total_sum_probabilities == 0.0) {
            for (auto j = 0ULL; j < number_neurons; j++) {
                const auto target_free_count = vacant_dendrites[j];
                const auto target_position = neuron_pos[j];

                const auto pos_difference = calculate_2_norm(source_position, target_position);
                if (target_free_count > 0 && pos_difference > 0) {
                    total_sum_probabilities += target_free_count / pos_difference;
                }
            }

            // if there are still no available neurons
            if (total_sum_probabilities == 0.0) {
                // return autapse for this case
                // this must be filtered later!
                found_target_neurons[i] = neuron_id_with_vacant_axon;
                // continue;
                return;
            }

            const auto threshold = picked_probabilities[i] * total_sum_probabilities;

            for (auto k = 0ULL; k < number_neurons; k++) {
                auto prob = 0.0;
                const auto target_free_count = vacant_dendrites[k];
                const auto target_position = neuron_pos[k];

                const auto pos_difference = calculate_2_norm(source_position, target_position);

                if (target_free_count > 0 && pos_difference > 0) {
                    prob = target_free_count / pos_difference;
                    sum_probabilities += prob;
                }
                // update picked_index if prob > 0
                if (prob > 0.0) {
                    picked_index = k;
                }
                if (sum_probabilities >= threshold) {
                    break;
                }
            }
        } else {
            const auto threshold = picked_probabilities[i] * total_sum_probabilities;

            // with this threshold, we aggregate the probabilities again:
            for (auto k = 0; k < number_neurons; k++) {
                const auto target_free_count = vacant_dendrites[k];
                const auto target_position = neuron_pos[k];

                const auto prob = calculate_attractiveness_to_connect(source_position, target_position,
                                                                      target_free_count, false, squared_sigma_inv);
                sum_probabilities += prob;

                // update picked_index if prob > 0
                if (prob > 0.0) {
                    picked_index = k;
                }
                if (sum_probabilities >= threshold) {
                    break;
                }
            }
        }
        found_target_neurons[i] = picked_index;
    }
}

} // namespace NaiveCUDA_CU

__host__ void NaiveCUDA_CU::find_target_neurons(const DeviceArray<SimpleVec3d>& d_neuron_pos, const std::uint64_t neurons_count,
                                                const NaiveTargetSelectionTask task,
                                                std::vector<std::uint64_t>& h_found_target_dendrites,
                                                const double squared_sigma_inv) {
    const auto h_vacant_axons_neuron_id_mapping = task.mapping;
    const auto h_vacant_dendrites = task.vacant_dendrites;
    const auto target_size = task.target_size;
    const auto h_picked_probabilities = task.picked_probabilities;

    // positions not null
    const auto pos_ok = d_neuron_pos.device_ptr() != nullptr;
    // size check neuron count
    const auto dendrites_size_ok = h_vacant_dendrites.size() == neurons_count;
    // size check for targets
    const auto axons_size_ok = h_vacant_axons_neuron_id_mapping.size() == target_size;
    const auto target_size_ok = h_found_target_dendrites.size() == target_size;
    const auto randoms_size_ok = h_picked_probabilities.size() >= target_size;

    RELEARN_CUDA_CHECK(pos_ok, "NaiveCUDA_CU::find_target_neurons: Device pointer for positions array is null.");
    RELEARN_CUDA_CHECK(axons_size_ok, "NaiveCUDA_CU::find_target_neurons: Size of axons array is not equal to target size.");
    RELEARN_CUDA_CHECK(dendrites_size_ok, "NaiveCUDA_CU::find_target_neurons: Size of dendrites array is not equal to number of neurons.");
    RELEARN_CUDA_CHECK(target_size_ok, "NaiveCUDA_CU::find_target_neurons: Size of target_dendrites array is not equal to target size.");
    RELEARN_CUDA_CHECK(randoms_size_ok, "NaiveCUDA_CU::find_target_neurons: Size of random numbers array must be at least target size.");

    auto d_neuron_id_vacant_axons = thrust::device_vector<std::uint64_t>(h_vacant_axons_neuron_id_mapping.begin(), h_vacant_axons_neuron_id_mapping.end());

    // Allocate and copy free dendrites
    auto d_vacant_dendrites = thrust::device_vector<std::uint32_t>(h_vacant_dendrites.begin(), h_vacant_dendrites.end());

    // Allocate and copy picked probabilities
    auto d_picked_probabilities = thrust::device_vector<double>(h_picked_probabilities.begin(), h_picked_probabilities.end());

    // Allocate found_target_neurons on device and initialize to std::numeric_limits<std::uint64_t>::max()
    auto d_found_target_neurons = thrust::device_vector<std::uint64_t>(target_size, std::numeric_limits<std::uint64_t>::max());

    // Call the kernel using raw pointers from thrust device vectors
    const auto& [blocks, threads] = get_grid_ands_block_size(target_size,
                                                             find_target_neurons_kernel);
    Timers::start(TimerRegion::CUDA_NAIVE_FIND_TARGET_NEURONS_KERNEL);
    find_target_neurons_kernel<<<blocks, threads>>>(
        neurons_count,
        d_neuron_pos.device_ptr(),
        thrust::raw_pointer_cast(d_vacant_dendrites.data()),
        target_size,
        thrust::raw_pointer_cast(d_neuron_id_vacant_axons.data()),
        thrust::raw_pointer_cast(d_picked_probabilities.data()), thrust::raw_pointer_cast(d_found_target_neurons.data()), squared_sigma_inv);

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_NAIVE_FIND_TARGET_NEURONS_KERNEL);

    // Copy results back to the host
    thrust::copy(d_found_target_neurons.begin(), d_found_target_neurons.end(), h_found_target_dendrites.begin());

    cudaDeviceSynchronize();
    kernelErrCheck();
}
