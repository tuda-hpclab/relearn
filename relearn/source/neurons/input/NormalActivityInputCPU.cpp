/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NormalActivityInputCPU.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/UpdateStatus.h"
#include "util/NeuronIDRange.h"
#include "util/Random.h"
#include "util/RandomHolderKey.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

void NormalActivityInputCPU::update_input_range(step_type /*step*/, NeuronID first, NeuronID last) {
    RelearnException::check(stddev > activity_type{ 0 }, "NormalActivityInputCPU::update_input_range: stddev is {}", stddev);

    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_NORMAL);

    const auto extra_infos = get_extra_infos();
    const auto disable_flags = extra_infos->get_disable_flags();
    const auto number_neurons = get_number_neurons();
    RelearnException::check(disable_flags.size() == number_neurons,
                            "NormalActivityInputCPU::update_input_range: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

    const auto input = get_input_internal();

    for (const auto neuron_id : NeuronIDRange::range(first, last)) {
        input[neuron_id.get_neuron_id()] = disable_flags[neuron_id.get_neuron_id()] == UpdateStatus::Disabled ? activity_type{ 0 } : RandomHolder::get_random_normal_double(RandomHolderKey::BackgroundActivity, mean, stddev);
    }

    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_NORMAL);
}
