/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FitzHughNagumoModelBase.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/input/ActivityInput.h"
#include "neurons/models/NeuronModel.h"
#include "neurons/models/fitzhughnagumo/Parameters.h"
#include "types/BasicTypes.h"
#include "util/RelearnException.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>
#include <utility>

using models::FitzHughNagumoModelBase;

FitzHughNagumoModelBase::FitzHughNagumoModelBase(
    const unsigned int h,
    std::shared_ptr<ActivityInput> activity_input,
    std::shared_ptr<FiredStatusCommunicator> fired_status_communicator,
    const models::fitzhughnagumo::Parameters<RelearnTypes::activity_type>& params)
    : NeuronModel{ h, std::move(activity_input), std::move(fired_status_communicator) }
    , parameters{ params } {
}

void FitzHughNagumoModelBase::init(const number_neurons_type number_neurons) {
    NeuronModel::init(number_neurons);
    w.resize(number_neurons);
    init_neurons(0, number_neurons);
}

void FitzHughNagumoModelBase::create_neurons(const number_neurons_type creation_count) {

    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    w.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void FitzHughNagumoModelBase::register_neuron_monitor(NeuronMonitor& monitor) {
    NeuronModel::register_neuron_monitor(monitor);
    monitor.register_paramter("w", [this](const RelearnTypes::number_neurons_type neuron_id) { return utility::cast<float>(w[neuron_id]); }, []() { }, []() { });
}

void FitzHughNagumoModelBase::init_neurons(const number_neurons_type start_id, const number_neurons_type end_id) {
    const auto init_w = parameters.get_init_w();
    const auto init_x = parameters.get_init_x();

    for (auto neuron_id = start_id; neuron_id < end_id; ++neuron_id) {
        const auto id = NeuronID{ neuron_id };
        w[neuron_id] = init_w;
        set_x(id, init_x);
    }
}

void FitzHughNagumoModelBase::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this) - sizeof(NeuronModel)
                              + (w.capacity() * sizeof(activity_type));
    footprint->emplace("FitzHughNagumoModel", my_footprint);

    NeuronModel::record_memory_footprint(footprint);
}
