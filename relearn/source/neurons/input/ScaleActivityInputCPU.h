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

#include "ScaleActivityInputBase.h"

#include "util/NeuronIDRange.h"
#include "util/Timers.h"

#include <functional>

/**
 * CPU implementation of ScaleActivityInput: scales the held input via an arbitrary
 * std::function, computed on the host.
 */
class ScaleActivityInputCPU : public ScaleActivityInputBase {
public:
    /**
     * @brief Constructs a new instance of type ScaleActivityInputCPU with 0 neurons and the passed values for all parameters
     * @param activity_input The activity input which should be scaled, not empty
     * @param scaling_function The function that scales the input, not empty
     * @exception Throws a RelearnException if any argument is empty
     */
    ScaleActivityInputCPU(const int _number_ranks, std::shared_ptr<ActivityInput> activity_input, std::function<activity_type(activity_type)> scaling_function)
        : ScaleActivityInputBase(_number_ranks, std::move(activity_input))
        , scaler(std::move(scaling_function)) {
        RelearnException::check(scaler != nullptr, "SynapticInputCalculator::SynapticInputCalculator: scaling_function was empty.");
    }

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
    void update_input_range([[maybe_unused]] const step_type step, const NeuronID first, const NeuronID last) override {
        other_input->update_input_range(step, first, last);

        Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_SCALE);
        const auto extra_infos = get_extra_infos();
        const auto disable_flags = extra_infos->get_disable_flags();
        const auto number_neurons = get_number_neurons();
        RelearnException::check(disable_flags.size() == number_neurons,
                                "ScaleActivityInput::update_input_range: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);
        const auto _other_input = other_input->get_input();
        const auto input = get_input_internal();

        for (const auto neuron_id : NeuronIDRange::range(first, last)) {
            input[neuron_id.get_neuron_id()] = scaler(_other_input[neuron_id.get_neuron_id()]);
        }

        Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_SCALE);
    }

private:
    std::function<activity_type(activity_type)> scaler{};
};
