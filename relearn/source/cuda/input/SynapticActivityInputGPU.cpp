/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticActivityInputGPU.h"

#include "cuda/util/Util.h"
#include "neurons/NeuronsExtraInfo.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <neurons/NetworkGraph.h>

std::vector<EventWrapper> SynapticActivityInputGPU::update_input_range(step_type /*step*/, NeuronID first, NeuronID last, [[maybe_unused]] const std::shared_ptr<StreamWrapper>& stream) {
    const auto& extra_infos = get_extra_infos();

    RelearnException::check(extra_infos != nullptr, "SynapticActivityInputGPU::update_input_range: extra_infos is empty");
    RelearnException::check(fired_status_comm != nullptr, "SynapticActivityInputGPU::update_input_range: fired_status_comm is empty");
    RelearnException::check(fired_status_comm->get_network_graph() != nullptr, "SynapticActivityInputGPU::update_input_range: network_graph of fired_status_comm is empty");

    const auto disable_flags = extra_infos->get_disable_flags();

    const auto num_local_neurons = get_number_neurons();

    RelearnException::check(num_local_neurons > 0,
                            "SynapticActivityInputGPU::update_input_range: There were no local neurons.");
    RelearnException::check(disable_flags.size() == num_local_neurons,
                            "SynapticActivityInputGPU::update_input_range: Size of disable_flags did not match number of local neurons: {} vs {}", disable_flags.size(), num_local_neurons);

    const auto network_graph = fired_status_comm->get_network_graph();

    const auto* d_fired = fired_status_comm->get_fired_status_recorder()->get_d_fired_const();
    auto fire_status_handle = fired_status_comm->get_handle();

    // The kernel writes directly through the raw device pointer, bypassing LazySyncedArray's own
    // tracking, so requesting the non-const (auto-marking) pointer here is what flags the device
    // copy as modified for host()'s lazy sync -- the kernel call below doesn't do that itself.
    auto distant_event = update_distant_input(first.get_neuron_id(), last.get_neuron_id(), d_input_distant.get_device_ptr(), d_scales, fire_status_handle, distant_stream);

    auto local_event = update_local_input(first.get_neuron_id(), last.get_neuron_id(), d_input_local.get_device_ptr(), d_fired, d_scales, local_stream);

    std::vector<EventWrapper> events{};
    if (local_event.has_value()) {
        events.push_back(std::move(*local_event));
    }
    if (distant_event.has_value()) {
        events.push_back(std::move(*distant_event));
    }

    return events;
}

std::vector<LazySyncedArray<ActivityInput::activity_type>*> SynapticActivityInputGPU::get_input_arr() {
    return { &d_input_local, &d_input_distant };
}

std::vector<const LazySyncedArray<ActivityInput::activity_type>*> SynapticActivityInputGPU::get_input_arr_const() const {
    return { &d_input_local, &d_input_distant };
}
