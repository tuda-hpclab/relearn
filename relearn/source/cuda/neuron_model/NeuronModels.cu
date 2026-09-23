/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cuda/neuron_model//NeuronModels.h"
#include "cuda/random/RandomNumber.cuh"
#include "cuda/util/Util.cuh"
#include "util/Timers.h"

__global__ void update_activity_izhikevich_kernel(const uint64_t size, const UpdateStatus* d_disable_flags,
                                                  const unsigned h,
                                                  const CudaConfig::input_type* d_input, CudaConfig::membrane_potential_type* d_x, CudaConfig::membrane_potential_type* d_u,
                                                  FiredStatus* d_fired, CudaConfig::counter_type** d_fired_recorder,
                                                  const int d_fire_recorder_size, const IzhikevichKernelParams params) {
    const auto neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= size) {
        return;
    }

    if (d_disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }

    const auto scale = 1.0 / h;

    const auto input = d_input[neuron_id];
    auto x_val = d_x[neuron_id];
    auto u_val = d_u[neuron_id];

    auto has_spiked = FiredStatus::Inactive;

    for (auto integration_steps = 0U; integration_steps < h; ++integration_steps) {
        const auto x_increase = (params.k1 * x_val * x_val) + (params.k2 * x_val) + params.k3 - u_val + input;
        const auto u_increase = params.a * (params.b * x_val - u_val);

        x_val += x_increase * scale;
        u_val += u_increase * scale;

        const auto spiked = x_val >= params.V_spike;
        if (spiked) {
            x_val = params.c;
            u_val += params.d;
            has_spiked = FiredStatus::Fired;
            break;
        }
    }

    d_u[neuron_id] = u_val;
    d_x[neuron_id] = x_val;
    d_fired[neuron_id] = has_spiked;

    if (has_spiked != FiredStatus::Fired) {
        return;
    }

    for (int i = 0; i < d_fire_recorder_size; i++) {
        d_fired_recorder[i][neuron_id]++;
    }
}

__global__ void update_activity_aeif_kernel(const uint64_t size, const UpdateStatus* d_disable_flags,
                                            const unsigned h,
                                            CudaConfig::input_type* const d_input, CudaConfig::membrane_potential_type* const d_x, CudaConfig::membrane_potential_type* const d_w,
                                            FiredStatus* const d_fired, CudaConfig::counter_type* const* const d_fired_recorder,
                                            const int d_fire_recorder_size, const AeifKernelParams params) {
    const auto neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= size) {
        return;
    }

    if (d_disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }

    const auto scale = 1.0 / h;
    const auto d_T_inverse = 1.0 / params.d_T;
    const auto tau_w_inverse = 1.0 / params.tau_w;
    const auto C_inverse = 1.0 / params.C;

    const auto input = d_input[neuron_id];

    auto x_val = d_x[neuron_id];
    auto w_val = d_w[neuron_id];

    auto has_spiked = FiredStatus::Inactive;

    for (auto integration_steps = 0U; integration_steps < h; ++integration_steps) {
        const auto linear_part = -params.g_L * (x_val - params.E_L);
        const auto exp_part = params.g_L * params.d_T * std::exp((x_val - params.V_T) * d_T_inverse);
        const auto x_increase = (linear_part + exp_part - w_val + input) * C_inverse;
        const auto w_increase = (params.a * (x_val - params.E_L) - w_val) * tau_w_inverse;

        x_val += x_increase * scale;
        w_val += w_increase * scale;

        if (x_val >= params.V_spike) {
            x_val = params.E_L;
            w_val += params.b;
            has_spiked = FiredStatus::Fired;
            break;
        }
    }

    d_w[neuron_id] = w_val;
    d_x[neuron_id] = x_val;
    d_fired[neuron_id] = has_spiked;

    if (has_spiked != FiredStatus::Fired) {
        return;
    }

    for (int i = 0; i < d_fire_recorder_size; i++) {
        d_fired_recorder[i][neuron_id]++;
    }
}

__global__ void update_activity_fitzhughnagumo_kernel(const uint64_t size, const UpdateStatus* d_disable_flags,
                                                      const unsigned h,
                                                      const CudaConfig::input_type* d_input, CudaConfig::membrane_potential_type* d_x, CudaConfig::membrane_potential_type* d_w,
                                                      FiredStatus* d_fired, CudaConfig::counter_type** d_fired_recorder,
                                                      const int d_fire_recorder_size, const FitzHughNagumoKernelParams params) {
    const auto neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= size) {
        return;
    }

    if (d_disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }

    const auto scale = 1.0 / h;

    const auto input = d_input[neuron_id];

    auto x_val = d_x[neuron_id];
    auto w_val = d_w[neuron_id];

    for (auto integration_steps = 0U; integration_steps < h; ++integration_steps) {
        const auto x_increase = x_val - (x_val * x_val * x_val * (1.0 / 2.0)) - w_val + input;
        const auto w_increase = params.phi * (x_val + params.a - params.b * w_val);

        x_val += x_increase * scale;
        w_val += w_increase * scale;
    }

    const auto spiked = w_val > x_val - x_val * x_val * x_val * (1.0 / 2.0) && x_val > 1.0;

    d_w[neuron_id] = w_val;
    d_x[neuron_id] = x_val;
    d_fired[neuron_id] = spiked ? FiredStatus::Fired : FiredStatus::Inactive;

    if (!spiked) {
        return;
    }

    for (int i = 0; i < d_fire_recorder_size; i++) {
        d_fired_recorder[i][neuron_id]++;
    }
}

__global__ void update_activity_poisson_kernel(const uint64_t size, const UpdateStatus* const d_disable_flags,
                                               const unsigned h,
                                               const CudaConfig::input_type* const d_input, CudaConfig::membrane_potential_type* const d_x, unsigned int* const d_refractory_time,
                                               FiredStatus* const d_fired, CudaConfig::counter_type* const* const d_fired_recorder,
                                               const int d_fire_recorder_size, const PoissonKernelParams params, const std::uint32_t random_key) {
    const auto neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= size) {
        return;
    }

    if (d_disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }

    const auto scale = 1.0 / h;
    const auto tau_x_inverse = 1.0 / params.tau_x;

    const auto input = d_input[neuron_id];

    auto x_val = d_x[neuron_id];

    for (auto integration_steps = 0U; integration_steps < h; integration_steps++) {
        x_val += ((params.x_0 - x_val) * tau_x_inverse + input) * scale;
    }

    d_x[neuron_id] = x_val;

    if (d_refractory_time[neuron_id] == 0) {
        const auto threshold = RandomNumbers::get_random_value(neuron_id, random_key);
        if (x_val >= threshold) {
            d_fired[neuron_id] = FiredStatus::Fired;
            for (int i = 0; i < d_fire_recorder_size; i++) {
                d_fired_recorder[i][neuron_id]++;
            }
            d_refractory_time[neuron_id] = params.refractory_period;
        } else {
            d_fired[neuron_id] = FiredStatus::Inactive;
        }
    } else {
        d_fired[neuron_id] = FiredStatus::Inactive;
        d_refractory_time[neuron_id] -= 1;
    }
}

void update_activity_izhikevich_entry(const NeuronExtraInfoHandle extra_info, const unsigned h,
                                      const models::izhikevich::Parameters<RelearnTypes::activity_type>& parameters,
                                      const CudaConfig::input_type* d_input, IzhikevichDeviceState state,
                                      FiredRecorderHandle recorder) {
    Timers::start(TimerRegion::CUDA_UPDATE_ACTIVITY_IZHIKEVICH_KERNEL);
    const IzhikevichKernelParams params{
        parameters.get_k1(), parameters.get_k2(), parameters.get_k3(),
        parameters.get_a(), parameters.get_b(), parameters.get_c(), parameters.get_d(),
        parameters.get_V_spike()
    };

    const auto& [blocks, threads] = get_grid_ands_block_size(extra_info.num_neurons, update_activity_izhikevich_kernel);

    update_activity_izhikevich_kernel<<<blocks, threads>>>(
        extra_info.num_neurons, extra_info.d_disable_flags, h, d_input, state.d_x, state.d_u,
        recorder.d_fired, recorder.d_fired_recorder, recorder.d_fire_recorder_size,
        params);

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_UPDATE_ACTIVITY_IZHIKEVICH_KERNEL);
}

void update_activity_aeif_entry(const NeuronExtraInfoHandle extra_info, const unsigned h,
                                const models::aeif::Parameters<RelearnTypes::activity_type>& parameters,
                                CudaConfig::input_type* const d_input, AeifDeviceState state,
                                FiredRecorderHandle recorder) {
    Timers::start(TimerRegion::CUDA_UPDATE_ACTIVITY_AEIF_KERNEL);
    const AeifKernelParams params{
        parameters.get_d_T(), parameters.get_tau_w(), parameters.get_C(),
        parameters.get_g_L(), parameters.get_E_L(), parameters.get_V_T(),
        parameters.get_a(), parameters.get_b(), parameters.get_V_spike()
    };

    const auto& [blocks, threads] = get_grid_ands_block_size(extra_info.num_neurons, update_activity_aeif_kernel);

    update_activity_aeif_kernel<<<blocks, threads>>>(
        extra_info.num_neurons, extra_info.d_disable_flags, h, d_input, state.d_x, state.d_w,
        recorder.d_fired, recorder.d_fired_recorder, recorder.d_fire_recorder_size,
        params);

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_UPDATE_ACTIVITY_AEIF_KERNEL);
}

void update_activity_fitzhughnagumo_entry(const NeuronExtraInfoHandle extra_info, const unsigned h,
                                          const models::fitzhughnagumo::Parameters<RelearnTypes::activity_type>& parameters,
                                          const CudaConfig::input_type* d_input, FitzHughNagumoDeviceState state,
                                          FiredRecorderHandle recorder) {
    Timers::start(TimerRegion::CUDA_UPDATE_ACTIVITY_FHN_KERNEL);

    const FitzHughNagumoKernelParams params{ parameters.get_a(), parameters.get_b(), parameters.get_phi() };

    const auto& [blocks, threads] = get_grid_ands_block_size(extra_info.num_neurons, update_activity_fitzhughnagumo_kernel);

    update_activity_fitzhughnagumo_kernel<<<blocks, threads>>>(
        extra_info.num_neurons, extra_info.d_disable_flags, h, d_input, state.d_x, state.d_w,
        recorder.d_fired, recorder.d_fired_recorder, recorder.d_fire_recorder_size, params);

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_UPDATE_ACTIVITY_FHN_KERNEL);
}

void update_activity_poisson_entry(const NeuronExtraInfoHandle extra_info, const unsigned h,
                                   const models::poisson::Parameters<RelearnTypes::activity_type>& parameters,
                                   const CudaConfig::input_type* d_input, PoissonDeviceState state,
                                   FiredRecorderHandle recorder, const std::uint32_t random_key) {
    Timers::start(TimerRegion::CUDA_UPDATE_ACTIVITY_POISSON_KERNEL);
    const PoissonKernelParams params{ parameters.get_tau_x(), parameters.get_x_0(), parameters.get_refractory_period() };

    const auto& [blocks, threads] = get_grid_ands_block_size(extra_info.num_neurons, update_activity_poisson_kernel);

    update_activity_poisson_kernel<<<blocks, threads>>>(
        extra_info.num_neurons, extra_info.d_disable_flags, h, d_input, state.d_x, state.d_refractory_time,
        recorder.d_fired, recorder.d_fired_recorder, recorder.d_fire_recorder_size,
        params, random_key);

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_UPDATE_ACTIVITY_POISSON_KERNEL);
}