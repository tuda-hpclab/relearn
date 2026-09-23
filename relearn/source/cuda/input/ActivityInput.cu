/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cuda/CudaBaseBridgeFunctions.h"
#include "cuda/CudaConfig.h"
#include "cuda/input/ActivityInput.h"
#include "cuda/random/RandomNumber.cuh"
#include "cuda/util/Util.cuh"
#include "cuda/wrapper/StreamWrapper.cuh"
#include "util/BinarySearch.cuh"
#include "util/Timers.h"

__device__ inline bool contains(const int* d_incoming_displ,
                                const CudaConfig::number_neurons_type* d_incoming_neuron_ids, const int mpi_rank, const CudaConfig::number_neurons_type neuron_id) {
    const auto begin = d_incoming_displ[mpi_rank];
    const auto end = d_incoming_displ[mpi_rank + 1];
    RELEARN_DEVICE_CUDA_CHECK(end >= begin, "contains: end >= begin not fulfilled");
    const auto size = end - begin;
    if (size == 0) {
        return false;
    }

    return binary_search(begin, end, d_incoming_neuron_ids, neuron_id) != std::numeric_limits<std::size_t>::max();
}

__global__ void update_combined_activity_input_range_kernel(const CudaConfig::number_neurons_type first, const CudaConfig::number_neurons_type last, const std::size_t sub_input_size, CudaConfig::input_type* d_input, CudaConfig::input_type** d_sub_input_ptrs) {
    const uint64_t neuron_id = first + (blockIdx.x * blockDim.x) + threadIdx.x;
    if (neuron_id >= last) {
        return;
    }

    d_input[neuron_id] = 0;

    for (int i = 0; i < sub_input_size; i++) {
        d_input[neuron_id] += d_sub_input_ptrs[i][neuron_id];
    }
}

__global__ void update_constant_activity_input_range_kernel(const CudaConfig::number_neurons_type first, const CudaConfig::number_neurons_type last, const UpdateStatus* d_disable_flags, CudaConfig::input_type* d_input, const CudaConfig::input_type base_input) {
    const uint64_t neuron_id = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= last) {
        return;
    }

    d_input[neuron_id] = d_disable_flags[neuron_id] == UpdateStatus::Disabled ? 0.0 : base_input;
}

__global__ void update_normal_activity_input_range_kernel(const CudaConfig::number_neurons_type first, const CudaConfig::number_neurons_type last,
                                                          const UpdateStatus* d_disable_flags, CudaConfig::input_type* d_input,
                                                          const CudaConfig::input_type mean, const CudaConfig::input_type stddev, const std::uint32_t random_key) {
    const uint64_t neuron_id = first + blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_id >= last) {
        return;
    }

    d_input[neuron_id] = 0.0;
    if (d_disable_flags[neuron_id] != UpdateStatus::Disabled) {
        const auto random = RandomNumbers::get_random_value(neuron_id, random_key) * stddev + mean;
        d_input[neuron_id] = random;
    }
}

void update_combined_activity_input_range_entry(CudaConfig::number_neurons_type first, CudaConfig::number_neurons_type last,
                                                const std::size_t sub_input_size, CudaConfig::input_type* d_input,
                                                CudaConfig::input_type** d_sub_input_ptrs, const std::shared_ptr<StreamWrapper>& stream) {
    const auto& [blocks, threads] = get_grid_ands_block_size(last - first + 1, update_combined_activity_input_range_kernel);

    // This kernel runs on its own stream without a following CPU sync, so a plain Timers::start/stop
    // around the (non-blocking) launch would only measure launch overhead -- use CUDA events instead
    // to capture actual device execution time without forcing a synchronization here.
    auto* gpu_timer = cuda_start_gpu_timer(TimerRegion::CUDA_UPDATE_COMBINED_ACTIVITY_KERNEL, *stream);
    update_combined_activity_input_range_kernel<<<blocks, threads, 0, get_cuda_stream_from_wrapper(*stream)>>>(
        first, last, sub_input_size, d_input, d_sub_input_ptrs);
    cuda_stop_gpu_timer(gpu_timer, *stream);
}

void update_constant_activity_input_range_entry(CudaConfig::number_neurons_type first, CudaConfig::number_neurons_type last,
                                                const UpdateStatus* d_disable_flags, CudaConfig::input_type* d_input,
                                                const CudaConfig::input_type base_input, const std::shared_ptr<StreamWrapper>& stream) {
    const auto& [blocks, threads] = get_grid_ands_block_size(last - first + 1, update_constant_activity_input_range_kernel);

    // Async launch on its own stream, no following CPU sync -- measure via CUDA events instead of
    // Timers::start/stop, which would only capture launch overhead here.
    auto* gpu_timer = cuda_start_gpu_timer(TimerRegion::CUDA_UPDATE_CONSTANT_ACTIVITY_KERNEL, *stream);
    update_constant_activity_input_range_kernel<<<blocks, threads, 0, get_cuda_stream_from_wrapper(*stream)>>>(first, last, d_disable_flags,
                                                                                                               d_input, base_input);
    cuda_stop_gpu_timer(gpu_timer, *stream);
}

void update_normal_activity_input_range_entry(CudaConfig::number_neurons_type first, CudaConfig::number_neurons_type last,
                                              const UpdateStatus* d_disable_flags, CudaConfig::input_type* d_input, CudaConfig::input_type mean,
                                              CudaConfig::input_type stddev, const std::uint32_t random_key, const std::shared_ptr<StreamWrapper>& stream) {
    const auto& [blocks, threads] = get_grid_ands_block_size(last - first + 1, update_normal_activity_input_range_kernel);

    // Async launch on its own stream, no following CPU sync -- measure via CUDA events instead of
    // Timers::start/stop, which would only capture launch overhead here.
    auto* gpu_timer = cuda_start_gpu_timer(TimerRegion::CUDA_UPDATE_NORMAL_ACTIVITY_KERNEL, *stream);
    update_normal_activity_input_range_kernel<<<blocks, threads, 0, get_cuda_stream_from_wrapper(*stream)>>>(first, last, d_disable_flags,
                                                                                                             d_input, mean, stddev, random_key);
    cuda_stop_gpu_timer(gpu_timer, *stream);
}