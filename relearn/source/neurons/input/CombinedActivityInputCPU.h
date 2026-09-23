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

#include "CombinedActivityInputBase.h"

#include "util/NeuronIDRange.h"
#include "util/Timers.h"

#include <algorithm>
#include <memory>
#include <vector>

/**
 * CPU implementation of CombinedActivityInput: sums each sub-input's host-side input into this
 * input's own.
 */
class CombinedActivityInputCPU : public CombinedActivityInputBase {
public:
    explicit CombinedActivityInputCPU(const int _number_ranks, std::vector<std::shared_ptr<ActivityInput>> inputs)
        : CombinedActivityInputBase(_number_ranks, std::move(inputs)) { }

    CombinedActivityInputCPU(const CombinedActivityInputCPU&) = delete;
    CombinedActivityInputCPU& operator=(const CombinedActivityInputCPU&) = delete;

    CombinedActivityInputCPU(CombinedActivityInputCPU&&) = default;
    CombinedActivityInputCPU& operator=(CombinedActivityInputCPU&&) = default;

    ~CombinedActivityInputCPU() override = default;

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
    void update_input_range(step_type step, NeuronID first, NeuronID last) override {
        Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_COMBINE);
        const auto input = get_input_internal();
        std::ranges::fill(std::next(input.begin(), static_cast<std::int64_t>(first.get_neuron_id())), std::next(input.begin(), static_cast<std::int64_t>(last.get_neuron_id())), activity_type{ 0 });
        Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_COMBINE);

        for (const auto& sub_input : sub_inputs) {
            sub_input->update_input_range(step, first, last);
            Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_COMBINE);
            const auto inputs = sub_input->get_input();

            for (const auto neuron_id : NeuronIDRange::range(first, last)) {
                input[neuron_id.get_neuron_id()] += inputs[neuron_id.get_neuron_id()];
            }
            Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_COMBINE);
        }
    }
};
