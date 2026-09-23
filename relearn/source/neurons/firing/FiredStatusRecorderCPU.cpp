/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FiredStatusRecorderCPU.h"

#include "Config.h"

#include "neurons/enums/FiredStatus.h"
#include "neurons/helper/NeuronMonitor.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <range/v3/algorithm/fill.hpp>

#include <memory>
#include <span>

void FiredStatusRecorderCPU::init(const number_neurons_type number_neurons) {
    RelearnException::check(number_neurons > 0, "FiredStatusRecorderCPU::init: Number of neurons must be greater than 0.");
    RelearnException::check(number_local_neurons == 0,
                            "FiredStatusRecorderCPU::init: FiredStatusRecorder already initialized.");

    for (auto& recorder : fired_recorder) {
        recorder.resize(number_neurons, 0U);
    }

    fired.resize(number_neurons, FiredStatus::Inactive);

    number_local_neurons = number_neurons;
}

void FiredStatusRecorderCPU::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(creation_count > 0,
                            "FiredStatusRecorderCPU::create_neurons: Creation count must be greater than 0.");
    RelearnException::check(number_local_neurons > 0,
                            "FiredStatusRecorderCPU::create_neurons: FiredStatusRecorder not initialized.");

    const auto new_size = number_local_neurons + creation_count;

    for (auto& recorder : fired_recorder) {
        recorder.resize(new_size, 0U);
    }

    fired.resize(new_size, FiredStatus::Inactive);
    number_local_neurons = new_size;
}

void FiredStatusRecorderCPU::set_fired(const NeuronID neuron_id, const FiredStatus new_value) {
    const auto local_neuron_id = neuron_id.get_neuron_id();

    RelearnException::check(local_neuron_id < number_local_neurons,
                            "FiredStatusRecorderCPU::set_fired: Neuron ID out of bounds.");

    fired[local_neuron_id] = new_value;

    if (new_value != FiredStatus::Fired) {
        return;
    }

    for (auto& recorder : fired_recorder) {
        recorder[local_neuron_id]++;
    }
}

std::span<const FiredStatusRecorderCPU::counter_type>
FiredStatusRecorderCPU::get_fired_recorder(const FireRecorderPeriod fire_recorder_period) const noexcept {
    return fired_recorder[static_cast<std::size_t>(fire_recorder_period)];
}

void FiredStatusRecorderCPU::reset(const FireRecorderPeriod period) {
    ranges::fill(fired_recorder[static_cast<std::size_t>(period)], 0U);
}

void FiredStatusRecorderCPU::register_neuron_monitor(NeuronMonitor& monitor) {
    monitor.register_paramter("Hz", [this](const RelearnTypes::number_neurons_type neuron_id) {
        const auto& fire_hist = get_fired_recorder(FireRecorderPeriod::NeuronMonitor);
        const auto hz = static_cast<double>(fire_hist[neuron_id]) / Config::neuron_monitor_log_step;
        return utility::cast<float>(hz) * 1000.0F; }, []() { }, []() { });
}

void FiredStatusRecorderCPU::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this);

    auto recorder_footprint = 0ULL;
    for (const auto& recorder : fired_recorder) {
        recorder_footprint += recorder.capacity() * sizeof(unsigned int);
    }

    footprint->emplace("FiredStatusRecorder", my_footprint + recorder_footprint);
}
