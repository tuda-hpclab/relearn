/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Calcium.h"

#include "cuda/util/Util.cuh"
#include "util/Timers.h"

__global__ void update_current_calcium_kernel(const NeuronsExtraInfoGPUHandleConst info_handle,
                                              const FiredStatus* const d_fired, const CalciumHandle calcium_handle,
                                              const unsigned int h, const CudaConfig::calcium_type tau_C, const CudaConfig::calcium_type beta) {
    const auto neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= info_handle.number_neurons) {
        return;
    }

    if (info_handle.disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }

    const auto scale = (1.0 / static_cast<CudaConfig::calcium_type>(h));
    const auto tau_C_inverse = -1.0 / tau_C;

    auto c = calcium_handle.calcium[neuron_id];
    for (auto integration_steps = 0U; integration_steps < h; ++integration_steps) {
        if (d_fired[neuron_id] == FiredStatus::Inactive) {
            c += scale * (c * tau_C_inverse);
        } else {
            c += scale * (c * tau_C_inverse + beta);
        }
    }

    calcium_handle.calcium[neuron_id] = c;
}

__global__ void update_target_calcium_absolute_decay_kernel(const NeuronsExtraInfoGPUHandleConst info_handle,
                                                            const CalciumHandle calcium_handle, const CudaConfig::calcium_type decay_amount) {
    const auto neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= info_handle.number_neurons) {
        return;
    }

    if (info_handle.disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }

    calcium_handle.target_calcium[neuron_id] -= decay_amount;
}

__global__ void update_target_calcium_relative_decay_kernel(const NeuronsExtraInfoGPUHandleConst info_handle,
                                                            const CalciumHandle calcium_handle, const CudaConfig::calcium_type decay_amount) {
    const auto neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= info_handle.number_neurons) {
        return;
    }

    if (info_handle.disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }

    calcium_handle.target_calcium[neuron_id] *= decay_amount;
}

void update_current_calcium_entry(const NeuronsExtraInfoGPUHandleConst info_handle,
                                  const FiredStatus* d_fired, const CalciumHandle calcium_handle, const unsigned int h,
                                  const CudaConfig::calcium_type tau_C, const CudaConfig::calcium_type beta) {

    Timers::start(TimerRegion::CUDA_UPDATE_CURRENT_CALCIUM_KERNEL);
    const auto& [blocks, threads] = get_grid_ands_block_size(info_handle.number_neurons, update_current_calcium_kernel);

    update_current_calcium_kernel<<<blocks, threads>>>(info_handle, d_fired, calcium_handle,
                                                       h, tau_C,
                                                       beta);

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_UPDATE_CURRENT_CALCIUM_KERNEL);
}

void update_target_calcium_absolute_decay_entry(const NeuronsExtraInfoGPUHandleConst info_handle,
                                                const CalciumHandle calcium_handle,
                                                const CudaConfig::calcium_type decay_amount) {
    Timers::start(TimerRegion::CUDA_UPDATE_TARGET_CALCIUM_DECAY_KERNEL);
    const auto& [blocks, threads] = get_grid_ands_block_size(info_handle.number_neurons, update_target_calcium_absolute_decay_kernel);

    update_target_calcium_absolute_decay_kernel<<<blocks, threads>>>(
        info_handle, calcium_handle, decay_amount);

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_UPDATE_TARGET_CALCIUM_DECAY_KERNEL);
}

void update_target_calcium_relative_decay_entry(const NeuronsExtraInfoGPUHandleConst info_handle,
                                                const CalciumHandle calcium_handle,
                                                const CudaConfig::calcium_type decay_amount) {
    Timers::start(TimerRegion::CUDA_UPDATE_TARGET_CALCIUM_RELATIVE_DECAY_KERNEL);
    const auto& [blocks, threads] = get_grid_ands_block_size(info_handle.number_neurons, update_target_calcium_relative_decay_kernel);

    update_target_calcium_relative_decay_kernel<<<blocks, threads>>>(
        info_handle, calcium_handle, decay_amount);

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_UPDATE_TARGET_CALCIUM_RELATIVE_DECAY_KERNEL);
}