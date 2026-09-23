/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "IzhikevichModelBase.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/input/ActivityInput.h"
#include "neurons/models/NeuronModel.h"
#include "neurons/models/izhikevich/Parameters.h"
#include "util/NeuronID.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>
#include <utility>

using models::IzhikevichModelBase;

IzhikevichModelBase::IzhikevichModelBase(
    const unsigned int h,
    std::shared_ptr<ActivityInput> activity_input,
    std::shared_ptr<FiredStatusCommunicator> fired_status_communicator,
    const models::izhikevich::Parameters<RelearnTypes::activity_type>& params)
    : NeuronModel{ h, std::move(activity_input), std::move(fired_status_communicator) }
    , parameters{ params } {
}

void IzhikevichModelBase::init(const number_neurons_type number_neurons) {
    NeuronModel::init(number_neurons);
    u.resize(number_neurons);
    init_neurons(0, number_neurons);
}

void IzhikevichModelBase::create_neurons(const number_neurons_type creation_count) {

    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    u.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void IzhikevichModelBase::register_neuron_monitor(NeuronMonitor& monitor) {
    NeuronModel::register_neuron_monitor(monitor);
    monitor.register_paramter("u", [this](const RelearnTypes::number_neurons_type neuron_id) { return utility::cast<float>(u[neuron_id]); }, []() { }, []() { });
}

void IzhikevichModelBase::init_neurons(const number_neurons_type start_id, const number_neurons_type end_id) {
    const auto c = parameters.get_c();
    for (auto neuron_id = start_id; neuron_id < end_id; ++neuron_id) {
        const auto id = NeuronID{ neuron_id };
        u[neuron_id] = 0.0;
        set_x(id, c);
    }
}

void IzhikevichModelBase::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this) - sizeof(NeuronModel)
                              + (u.capacity() * sizeof(activity_type));
    footprint->emplace("IzhikevichModel", my_footprint);

    NeuronModel::record_memory_footprint(footprint);
}
