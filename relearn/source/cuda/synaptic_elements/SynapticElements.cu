/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticElements.cuh"
#include "SynapticElements.h"
#include "SynapticElementsHandle.h"

#include "cuda/CudaBaseBridgeFunctions.h"
#include "cuda/wrapper/StreamWrapper.cuh"
#include "neurons/enums/UpdateStatus.h"
#include "util/NeuronsExtraInfoHandle.h"
#include "util/Timers.h"
#include "util/Util.cuh"

__device__ inline double d_gaussian_growth_curve(const double current, const double eta, const double epsilon) noexcept {
    if (eta == epsilon) {
        // This is a corner case when using decaying target calcium
        if (current == eta) {
            return 0.0;
        }
        return -1.0;
    }

    constexpr auto factor = 1.6651092223153955127063292897904020952611777045288814583336582344;
    // 1.6651092223153955127063292897904020952611777045288814583336582344... = (2 * sqrt(-log(0.5)))

    const auto xi = (eta + epsilon) / 2;
    const auto zeta = (eta - epsilon) / factor;

    const auto difference = current - xi;
    const auto quotient = difference / zeta;
    const auto product = quotient * quotient;

    const auto dz = (2 * std::exp(-product)) - 1;
    return dz;
}

__global__ void update_number_elements_delta_kernel(const CalciumHandleConst calcium_handle,
                                                    NeuronsExtraInfoGPUHandleConst info_handle,
                                                    SynapticElementsBaseCudaHandle handle,
                                                    const CudaConfig::synaptic_grown_type growth_rate) {
    const uint64_t neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_id >= info_handle.number_neurons) {
        return;
    }

    if (info_handle.disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }

    const auto target_calcium_value = calcium_handle.target_calcium[neuron_id];
    const auto current_calcium_value = calcium_handle.calcium[neuron_id];

    const auto clamped_target = max(target_calcium_value, handle.minimum_calcium[neuron_id]);

    const auto delta = d_gaussian_growth_curve(current_calcium_value, handle.minimum_calcium[neuron_id],
                                               clamped_target);
    const auto inc = delta * growth_rate;

    handle.delta_since_last_update[neuron_id] += inc;
}

__global__ void commit_synaptic_elements_kernel(SynapticElementsBaseCudaHandle handle,
                                                CudaConfig::synaptic_count_type* d_to_delete_elements,
                                                NeuronsExtraInfoGPUHandleConst info_handle) {
    const uint64_t neuron_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (neuron_id >= info_handle.number_neurons) {
        return;
    }

    if (info_handle.disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }

    const auto current_count = handle.grown_elements[neuron_id];
    const auto current_connected_count_integral = handle.connected_elements[neuron_id];
    const auto retract_ratio = handle.vacant_retract_ratio[neuron_id];
    const auto current_delta = handle.delta_since_last_update[neuron_id];
    const auto current_connected_count = static_cast<CudaConfig::synaptic_grown_type>(current_connected_count_integral);
    const auto current_vacant = current_count - current_connected_count;

    RELEARN_DEVICE_CUDA_CHECK(current_count >= 0.0, "commit_synaptic_elements_kernel: current_count is negative: %f", current_count);
    RELEARN_DEVICE_CUDA_CHECK(current_connected_count >= 0.0, "commit_synaptic_elements_kernel: current_connected_count is negative: %f", current_connected_count);
    RELEARN_DEVICE_CUDA_CHECK(current_vacant >= 0.0, "commit_synaptic_elements_kernel: current_vacant is negative: vacant=%f neuron=%lu count=%f connected=%f", current_vacant, neuron_id, current_count, current_connected_count);

    if (const auto new_vacant = current_vacant + current_delta; new_vacant >= 0.0) {
        // The vacant portion after caring for the delta
        // No deletion of bound synaptic elements required, connected_elements stays the same

        const auto new_vacant_after_retract = (1 - retract_ratio) * new_vacant;
        const auto new_count = new_vacant_after_retract + current_connected_count;
        RELEARN_DEVICE_CUDA_CHECK(new_count >= current_connected_count, "commit_synaptic_elements_kernel: new count is smaller than connected count");

        handle.grown_elements[neuron_id] = new_count;
        handle.delta_since_last_update[neuron_id] = 0.0;
        handle.vacant_elements[neuron_id] = static_cast<CudaConfig::synaptic_count_type>(new_vacant_after_retract);
        // connected_elements does not need to change

        d_to_delete_elements[neuron_id] = 0U;
        return;
    }

    if (current_count + current_delta <= 0.0) {
        // More bound elements should be deleted than are available.
        // Now, neither vacant (see if branch above) nor bound elements are left.

        handle.grown_elements[neuron_id] = 0.0;
        handle.delta_since_last_update[neuron_id] = 0.0;
        handle.vacant_elements[neuron_id] = 0U;
        handle.connected_elements[neuron_id] = 0;

        d_to_delete_elements[neuron_id] = current_connected_count_integral;
        return;
    }

    const auto new_count = current_count + current_delta;
    const auto new_connected_count = std::floor(new_count);
    const auto num_vacant = new_count - new_connected_count;

    const auto retracted_new_count = ((1 - retract_ratio) * num_vacant) + new_connected_count;

    RELEARN_DEVICE_CUDA_CHECK(num_vacant >= 0.0, "commit_synaptic_elements_kernel: num_vacant is negative");
    RELEARN_DEVICE_CUDA_CHECK(num_vacant < 1.0, "commit_synaptic_elements_kernel: num_vacant is larger than 1.0");

    handle.grown_elements[neuron_id] = retracted_new_count;
    handle.delta_since_last_update[neuron_id] = 0.0;
    handle.vacant_elements[neuron_id] = 0U;
    handle.connected_elements[neuron_id] = static_cast<CudaConfig::synaptic_count_type>(new_connected_count);

    const auto deleted_counts = current_connected_count - new_connected_count;

    RELEARN_DEVICE_CUDA_CHECK(deleted_counts >= 0.0, "commit_synaptic_elements_kernel: deleted was negative");
    const auto num_delete_connected = static_cast<CudaConfig::synaptic_count_type>(deleted_counts);
    d_to_delete_elements[neuron_id] = num_delete_connected;
}

void commit_synaptic_elements_entry(SynapticElementsBaseCudaHandle handle,
                                    CudaConfig::synaptic_count_type* d_to_delete_elements,
                                    NeuronsExtraInfoGPUHandleConst info_handle, const std::shared_ptr<StreamWrapper>& stream) {
    Timers::start(TimerRegion::CUDA_COMMIT_SYNAPTIC_ELEMENTS_KERNEL);
    const auto& [blocks, threads] = get_grid_ands_block_size(info_handle.number_neurons, commit_synaptic_elements_kernel);

    commit_synaptic_elements_kernel<<<blocks, threads, 0, get_cuda_stream_from_wrapper(*stream)>>>(handle,
                                                                                                   d_to_delete_elements, info_handle);

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_COMMIT_SYNAPTIC_ELEMENTS_KERNEL);
}

void update_number_elements_kernel_entry(const CalciumHandleConst calcium_handle,
                                         SynapticElementsBaseCudaHandle handle,
                                         NeuronsExtraInfoGPUHandleConst info_handle,
                                         const CudaConfig::synaptic_grown_type growth_rate,
                                         const std::shared_ptr<StreamWrapper>& stream) {

    const auto& [blocks, threads] = get_grid_ands_block_size(info_handle.number_neurons, update_number_elements_delta_kernel);

    // Called once per element type (axons, excitatory/inhibitory dendrites), each on its own stream,
    // with a single cudaDeviceSynchronize_bridge() only after all three have been launched (see
    // SynapticElements::update_number_elements) -- so time the actual kernel via CUDA events instead
    // of Timers::start/stop, which would only capture launch overhead and not overlap correctly.
    auto* gpu_timer = cuda_start_gpu_timer(TimerRegion::CUDA_UPDATE_NUMBER_ELEMENTS_KERNEL, *stream);
    update_number_elements_delta_kernel<<<blocks, threads, 0, get_cuda_stream_from_wrapper(*stream)>>>(
        calcium_handle, info_handle,
        handle, growth_rate);
    cuda_stop_gpu_timer(gpu_timer, *stream);
}

__device__ void connect_elements(const CudaConfig::synaptic_count_type newly_connected, const CudaConfig::number_neurons_type neuron_id, CudaConfig::synaptic_count_type* d_connected_elements, CudaConfig::synaptic_count_type* d_vacant_elements, const CudaConfig::number_neurons_type size) {
    RELEARN_DEVICE_CUDA_CHECK(neuron_id < size, "connect_elements: neuron_id is too large: %u", neuron_id);

    const auto vacant = d_vacant_elements[neuron_id];
    RELEARN_DEVICE_CUDA_CHECK(vacant >= newly_connected, "connect_elements: Not enough vacant elements for neuron %u: %u vs %u", neuron_id, newly_connected, vacant);

    d_connected_elements[neuron_id] += newly_connected;
    d_vacant_elements[neuron_id] -= newly_connected;
}

__device__ void disconnect_elements(const CudaConfig::synaptic_count_type newly_disconnected, const CudaConfig::number_neurons_type neuron_id, CudaConfig::synaptic_count_type* d_connected_elements, CudaConfig::synaptic_count_type* d_vacant_elements, const CudaConfig::number_neurons_type size) {
    RELEARN_DEVICE_CUDA_CHECK(neuron_id < size, "disconnect_elements: neuron_id is too large: %u", neuron_id);
    RELEARN_DEVICE_CUDA_CHECK(d_connected_elements[neuron_id] >= newly_disconnected, "disconnect_elements: %u connected_elements %u < newly_disconnected %u", neuron_id, d_connected_elements[neuron_id], newly_disconnected);
    d_connected_elements[neuron_id] -= newly_disconnected;
    d_vacant_elements[neuron_id] += newly_disconnected;
}
