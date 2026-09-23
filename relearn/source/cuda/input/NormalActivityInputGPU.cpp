/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NormalActivityInputGPU.h"

#include "cuda/input/ActivityInput.h"
#include "neurons/NeuronsExtraInfo.h"
#include "util/RelearnException.h"

std::vector<EventWrapper> NormalActivityInputGPU::update_input_range(step_type /*step*/, NeuronID first, NeuronID last, const std::shared_ptr<StreamWrapper>& stream) {
    RelearnException::check(stddev > activity_type{ 0 }, "NormalActivityInputGPU::update_input_range: stddev is {}", stddev);

    const auto extra_infos = get_extra_infos();
    const auto disable_flags = extra_infos->get_disable_flags();
    const auto number_neurons = get_number_neurons();
    RelearnException::check(disable_flags.size() == number_neurons,
                            "NormalActivityInputGPU::update_input_range: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

    // No Timers wrap here: this only enqueues async work on `stream` (no sync), so a host
    // Timers::start/stop around it would close long before the GPU kernel actually finishes and
    // could never correctly bound the CUDA-event-timed CUDA_UPDATE_NORMAL_ACTIVITY_KERNEL nested
    // inside update_normal_activity_input_range_entry. Only the outer CALC_ACTIVITY_INPUT (in
    // NeuronModel.cpp, where the device is actually synced and GPU timers resolved) can.
    const auto* d_disable_flags = extra_infos->get_gpu_handle().disable_flags;
    update_normal_activity_input_range_entry(static_cast<CudaConfig::number_neurons_type>(first.get_neuron_id()), static_cast<CudaConfig::number_neurons_type>(last.get_neuron_id()), d_disable_flags, _input.get_device_ptr(), mean, stddev, random_key, stream);

    std::vector<EventWrapper> events;
    events.emplace_back(stream);
    return events;
}
