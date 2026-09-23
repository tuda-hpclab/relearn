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

#include "ConstantActivityInputBase.h"

#include "neurons/NeuronsExtraInfo.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

/**
 * CPU implementation of ConstantActivityInput: update_input_range writes the constant input in
 * place on the host.
 */
class ConstantActivityInputCPU : public ConstantActivityInputBase {
public:
    using ConstantActivityInputBase::ConstantActivityInputBase;

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
    void update_input_range([[maybe_unused]] step_type step, NeuronID first, NeuronID last) override {
        Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_CONSTANT);

        const auto& extra_infos = get_extra_infos();
        const auto disable_flags = extra_infos->get_disable_flags();
        const auto number_neurons = get_number_neurons();
        RelearnException::check(disable_flags.size() == number_neurons,
                                "ConstantActivityInput::update_input_range: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        const auto input = get_input_internal();

        for (const auto neuron_id : NeuronIDRange::range(first, last)) {
            input[neuron_id.get_neuron_id()] = disable_flags[neuron_id.get_neuron_id()] == UpdateStatus::Disabled ? activity_type{ 0 } : base_input;
        }
        Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_CONSTANT);
    }
};
