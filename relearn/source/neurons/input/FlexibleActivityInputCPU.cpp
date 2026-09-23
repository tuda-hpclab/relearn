/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FlexibleActivityInputCPU.h"

#include "neurons/input/ActivityInput.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

void FlexibleActivityInputCPU::update_input_range(step_type step, NeuronID first, NeuronID last) {
    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_FLEXIBLE);
    const auto input = get_input_internal();
    std::ranges::fill(std::next(input.begin(), static_cast<std::int64_t>(first.get_neuron_id())), std::next(input.begin(), static_cast<std::int64_t>(last.get_neuron_id())), activity_type{ 0 });
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_FLEXIBLE);

    std::vector<std::vector<NeuronID>> index_to_neuron_ids{};
    index_to_neuron_ids.resize(potential_inputs.size());

    for (auto index = 0U; index < potential_inputs.size(); index++) {
        auto& potential_input = potential_inputs[index];
        const auto& neuron_id_ranges = chooser->get_neuron_id_ranges_for_input(step, index);

        for (const auto& [first_neuron_id, last_neuron_id] : neuron_id_ranges) {
            const auto adapted_first = std::max(first, first_neuron_id);
            const auto adapted_last = std::min(last, last_neuron_id);

            if (adapted_first <= adapted_last) {
                potential_input->update_input_range(step, first_neuron_id, last_neuron_id);
            }
        }
    }

    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_FLEXIBLE);
    for (const auto neuron_id : NeuronIDRange::range(first, last)) {
        const auto& indices = chooser->get_inputs_for_neuron_id(step, neuron_id);
        for (const auto index : indices) {
            RelearnException::check(index < potential_inputs.size(),
                                    "FlexibleActivityInput::update_input_range: Step {} NeuronID {}: Index {} is chosen, only {} present", step, neuron_id, index, potential_inputs.size());

            input[neuron_id.get_neuron_id()] += potential_inputs[index]->get_input(neuron_id);
        }
    }

    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_FLEXIBLE);
}
