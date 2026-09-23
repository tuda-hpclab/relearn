#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/input/ConstantActivityInputBase.h"
#include "cuda/calcium/Calcium.h"
#include "cuda/input/ActivityInput.h"
#include "neurons/NeuronsExtraInfo.h"
#include "util/RelearnException.h"

#include <memory>
#include <vector>

/**
 * GPU implementation of ConstantActivityInput: update_input_range enqueues the constant-input
 * kernel on the given stream and returns the events it queued.
 */
class ConstantActivityInputGPU : public ConstantActivityInputBase {
public:
    using ConstantActivityInputBase::ConstantActivityInputBase;

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
    std::vector<EventWrapper> update_input_range([[maybe_unused]] step_type step, NeuronID first, NeuronID last, const std::shared_ptr<StreamWrapper>& stream) override {
        const auto& extra_infos = get_extra_infos();
        const auto disable_flags = extra_infos->get_disable_flags();
        const auto number_neurons = get_number_neurons();
        RelearnException::check(disable_flags.size() == number_neurons,
                                "ConstantActivityInput::update_input_range: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        // No Timers wrap here: this only enqueues async work on `stream` (no sync), so a host
        // Timers::start/stop around it would close long before the GPU kernel actually finishes and
        // could never correctly bound the CUDA-event-timed CUDA_UPDATE_CONSTANT_ACTIVITY_KERNEL
        // nested inside update_constant_activity_input_range_entry. Only the outer CALC_ACTIVITY_INPUT
        // (in NeuronModel.cpp, where the device is actually synced and GPU timers resolved) can.
        const auto* const d_disable_flags = extra_infos->get_gpu_handle().disable_flags;
        update_constant_activity_input_range_entry(static_cast<CudaConfig::number_neurons_type>(first.get_neuron_id()), static_cast<CudaConfig::number_neurons_type>(last.get_neuron_id()), d_disable_flags, _input.get_device_ptr(), base_input, stream);
        return {};
    }

    using ActivityInput::update_input_range;
};
