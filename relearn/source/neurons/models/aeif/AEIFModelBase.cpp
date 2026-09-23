/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "AEIFModelBase.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/input/ActivityInput.h"
#include "neurons/models/NeuronModel.h"
#include "neurons/models/aeif/Parameters.h"
#include "util/NeuronID.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>
#include <utility>

using models::AEIFModelBase;

AEIFModelBase::AEIFModelBase(
    const unsigned int h,
    std::shared_ptr<ActivityInput> activity_input,
    std::shared_ptr<FiredStatusCommunicator> fired_status_communicator,
    const models::aeif::Parameters<RelearnTypes::activity_type>& params)
    : NeuronModel{ h, std::move(activity_input), std::move(fired_status_communicator) }
    , parameters{ params } {
}

void AEIFModelBase::init(const number_neurons_type number_neurons) {
    NeuronModel::init(number_neurons);
    w.resize(number_neurons);
    init_neurons(0, number_neurons);
}

void AEIFModelBase::create_neurons(const number_neurons_type creation_count) {

    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    w.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void AEIFModelBase::register_neuron_monitor(NeuronMonitor& monitor) {
    NeuronModel::register_neuron_monitor(monitor);
    monitor.register_paramter("w", [this](const RelearnTypes::number_neurons_type neuron_id) { return utility::cast<float>(w[neuron_id]); }, []() { }, []() { });
}

void AEIFModelBase::init_neurons(const number_neurons_type start_id, const number_neurons_type end_id) {
    const auto E_L = parameters.get_E_L();
    for (auto neuron_id = start_id; neuron_id < end_id; ++neuron_id) {
        const auto id = NeuronID{ neuron_id };

        w[neuron_id] = 0.0;
        set_x(id, E_L);
    }
}

void AEIFModelBase::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this) - sizeof(NeuronModel)
                              + (w.capacity() * sizeof(activity_type));
    footprint->emplace("AEIFModel", my_footprint);

    NeuronModel::record_memory_footprint(footprint);
}
