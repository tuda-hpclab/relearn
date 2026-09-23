/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticActivityInputCPU.h"

#include "neurons/NeuronsExtraInfo.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <algorithm>
#include <cstdint>
#include <iterator>

void SynapticActivityInputCPU::update_input_range(step_type /*step*/, NeuronID first, NeuronID last) {
    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_SYNAPSES);

    const auto& extra_infos = get_extra_infos();

    RelearnException::check(extra_infos != nullptr, "SynapticActivityInputCPU::update_input_range: extra_infos is empty");
    RelearnException::check(fired_status_comm != nullptr, "SynapticActivityInputCPU::update_input_range: fired_status_comm is empty");
    RelearnException::check(fired_status_comm->get_network_graph() != nullptr, "SynapticActivityInputCPU::update_input_range: network_graph of fired_status_comm is empty");

    const auto disable_flags = extra_infos->get_disable_flags();

    const auto num_local_neurons = get_number_neurons();

    RelearnException::check(num_local_neurons > 0,
                            "SynapticActivityInputCPU::update_input_range: There were no local neurons.");
    RelearnException::check(disable_flags.size() == num_local_neurons,
                            "SynapticActivityInputCPU::update_input_range: Size of disable_flags did not match number of local neurons: {} vs {}", disable_flags.size(), num_local_neurons);

    const auto fired_status = fired_status_comm->get_fired_status_recorder()->get_fired();
    RelearnException::check(fired_status.size() == num_local_neurons,
                            "SynapticActivityInputCPU::update_input_range: Size of fired_status did not match number of local neurons: {} vs {}", fired_status.size(), num_local_neurons);
    const auto input = get_input_internal();

    std::ranges::fill(std::next(input.begin(), static_cast<std::int64_t>(first.get_neuron_id())), std::next(input.begin(), static_cast<std::int64_t>(last.get_neuron_id())), activity_type{ 0 });

    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_SYNAPSES_LOCAL);
    update_local_input(fired_status, input, first, last);
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_SYNAPSES_LOCAL);

    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_SYNAPSES_DISTANT);
    update_distant_input(fired_status, input, first, last);
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_SYNAPSES_DISTANT);

    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_SYNAPSES);
}
