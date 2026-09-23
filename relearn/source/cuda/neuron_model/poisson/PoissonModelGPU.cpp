/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "PoissonModelGPU.h"

#include "cuda/neuron_model/NeuronModels.h"
#include "cuda/random/RandomNumberHost.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "neurons/models/poisson/Calculation.h"
#include "neurons/models/poisson/Parameters.h"
#include "types/BasicTypes.h"

#include <cpp-utility/MemoryFootprint.hpp>

using models::PoissonModelGPU;

void PoissonModelGPU::init(const number_neurons_type number_neurons) {
    PoissonModelBase::init(number_neurons);

    random_key = RandomNumbers::register_random_numbers(RandomNumberKey::POISSON_INPUT, RandomNumberType::UNIFORM, get_number_neurons(), Config::random_seed);

    init_neurons(0, number_neurons);
}

void PoissonModelGPU::create_neurons(const number_neurons_type creation_count) {
    const auto old_size = NeuronModel::get_number_neurons();
    PoissonModelBase::create_neurons(creation_count);
    init_neurons(old_size, creation_count);
}

void PoissonModelGPU::update_activity() {
    const number_neurons_type num_neurons = get_number_neurons();

    const auto h = get_h();

    const auto* d_disable_flags = get_extra_infos()->get_gpu_handle().disable_flags;
    const auto* d_input = get_d_input_const();

    update_activity_poisson_entry(
        NeuronExtraInfoHandle{ static_cast<CudaConfig::number_neurons_type>(num_neurons), d_disable_flags },
        h, parameters, d_input,
        PoissonDeviceState{ get_d_x(), refractory_time.get_device_ptr() },
        get_fired_status_recorder()->get_d_fired_handle(),
        random_key);

    if (Config::calculate_fire_history) {
        const auto disable_flags = get_extra_infos()->get_disable_flags();
#pragma omp parallel for shared(disable_flags, num_neurons) default(none)
        for (number_neurons_type neuron_id = 0; neuron_id < num_neurons; ++neuron_id) {
            const auto neuron_id_cast = static_cast<unsigned int>(neuron_id);
            if (disable_flags[neuron_id_cast] == UpdateStatus::Disabled) {
                continue;
            }
            // update_fire_history(neuron_id_cast);
        }
    }
}

void PoissonModelGPU::update_activity_benchmark() {
    const number_neurons_type num_neurons = get_number_neurons();

    const auto h = get_h();

    const auto* d_disable_flags = get_extra_infos()->get_gpu_handle().disable_flags;
    const auto* d_input = get_d_input_const();

    update_activity_poisson_entry(
        NeuronExtraInfoHandle{ static_cast<CudaConfig::number_neurons_type>(num_neurons), d_disable_flags },
        h, parameters, d_input,
        PoissonDeviceState{ get_d_x(), refractory_time.get_device_ptr() },
        get_fired_status_recorder()->get_d_fired_handle(),
        random_key);

    if (Config::calculate_fire_history) {
        const auto disable_flags = get_extra_infos()->get_disable_flags();
#pragma omp parallel for shared(disable_flags, num_neurons) default(none)
        for (number_neurons_type neuron_id = 0; neuron_id < num_neurons; ++neuron_id) {
            const auto neuron_id_cast = static_cast<unsigned int>(neuron_id);
            if (disable_flags[neuron_id_cast] == UpdateStatus::Disabled) {
                continue;
            }
            // update_fire_history(neuron_id_cast);
        }
    }
}

void PoissonModelGPU::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    PoissonModelBase::record_memory_footprint(footprint);
    footprint->emplace("PoissonModel GPU", refractory_time.get_memory_footprint());
}
