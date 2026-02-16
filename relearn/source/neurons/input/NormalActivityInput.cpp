/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NormalActivityInput.h"

#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/input/ActivityInput.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <cstddef>

void NormalActivityInput::update_input_range(step_type /*step*/, NeuronID first, NeuronID last) {
    RelearnException::check(stddev > 0.0, "NormalActivityInput::update_input_range: stddev is {}", stddev);

    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_NORMAL);

    const auto extra_infos = get_extra_infos();
    const auto disable_flags = extra_infos->get_disable_flags();
    const auto number_neurons = get_number_neurons();
    RelearnException::check(disable_flags.size() == number_neurons,
                            "NormalActivityInput::update_input_range: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

    const auto input = get_input_internal();

    for (const auto neuron_id : NeuronID::range(first, last)) {
        input[neuron_id.get_neuron_id()] = disable_flags[neuron_id.get_neuron_id()] == UpdateStatus::Disabled ? 0.0 : RandomHolder::get_random_normal_double(RandomHolderKey::BackgroundActivity, mean, stddev);
    }

    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_NORMAL);
}

void NormalActivityInput::register_neuron_monitor(NeuronMonitor& monitor) {
    monitor.register_paramter("Normal Input", [this](const RelearnTypes::number_neurons_type neuron_id) {
        const auto input = get_input_internal();
        return static_cast<float>(input[neuron_id]);
    });
}

void FastNormalActivityInput::init(const number_neurons_type number_neurons) {
    ActivityInput::init(number_neurons);

    pre_drawn_values.resize(number_neurons * multiplier);

    for (auto neuron_id = std::size_t{ 0 }; neuron_id < number_neurons * multiplier; neuron_id++) {
        const auto input = RandomHolder::get_random_normal_double(RandomHolderKey::BackgroundActivity, mean, stddev);
        pre_drawn_values[neuron_id] = input;
    }
}

void FastNormalActivityInput::create_neurons(const number_neurons_type creation_count) {
    const auto previous_number_neurons = get_number_neurons();

    ActivityInput::create_neurons(creation_count);

    const auto now_number_neurons = get_number_neurons();
    pre_drawn_values.resize(now_number_neurons * multiplier);

    for (auto neuron_id = previous_number_neurons * multiplier; neuron_id < now_number_neurons * multiplier; neuron_id++) {
        const auto input = RandomHolder::get_random_normal_double(RandomHolderKey::BackgroundActivity, mean, stddev);
        pre_drawn_values[neuron_id] = input;
    }
}

void FastNormalActivityInput::update_input_range(step_type /*step*/, NeuronID /*first*/, NeuronID /*last*/) {
    RelearnException::check(stddev > 0.0, "FastNormalActivityInput::update_input_range: stddev is {}", stddev);

    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_FAST_NORMAL);

    const auto extra_infos = get_extra_infos();
    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_neurons = get_number_neurons();
    RelearnException::check(disable_flags.size() == number_neurons,
                            "FastNormalBackgroundActivityCalculator::update_input_range: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

    const auto min_offset = 0;
    const auto max_offset = pre_drawn_values.size() - number_neurons;

    const auto new_offset = RandomHolder::get_random_uniform_integer<std::size_t>(RandomHolderKey::BackgroundActivity, min_offset, max_offset);
    offset = new_offset;

    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_FAST_NORMAL);
}

void FastNormalActivityInput::register_neuron_monitor(NeuronMonitor& monitor) {
    monitor.register_paramter("Fast Normal Input", [this](const RelearnTypes::number_neurons_type neuron_id) {
        const auto input = pre_drawn_values[offset + neuron_id];
        return static_cast<float>(input);
    });
}
