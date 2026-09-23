/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "PoissonModelBase.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/input/ActivityInput.h"
#include "neurons/models/NeuronModel.h"
#include "neurons/models/poisson/Parameters.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>
#include <utility>

using models::PoissonModelBase;

PoissonModelBase::PoissonModelBase(
    const unsigned int h,
    std::shared_ptr<ActivityInput> activity_input,                      // NOLINT(performance-unnecessary-value-param) - binds to NeuronModel's rvalue-ref ctor param below; const& would not compile
    std::shared_ptr<FiredStatusCommunicator> fired_status_communicator, // NOLINT(performance-unnecessary-value-param) - same as above
    const models::poisson::Parameters<RelearnTypes::activity_type, unsigned int>& params)
    : NeuronModel{ h, std::move(activity_input), std::move(fired_status_communicator) }
    , parameters{ params } {
}

void PoissonModelBase::init(const number_neurons_type number_neurons) {
    NeuronModel::init(number_neurons);
    refractory_time.resize(number_neurons, 0);
}

void PoissonModelBase::init_neurons([[maybe_unused]] const number_neurons_type start_id,
                                    [[maybe_unused]] const number_neurons_type end_id) {
}

void PoissonModelBase::create_neurons(const number_neurons_type creation_count) {

    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    refractory_time.resize(old_size + creation_count, 0);
}

void PoissonModelBase::register_neuron_monitor(NeuronMonitor& monitor) {
    NeuronModel::register_neuron_monitor(monitor);
    monitor.register_paramter("r", [this](const RelearnTypes::number_neurons_type neuron_id) { return refractory_time[neuron_id]; }, []() { }, []() { });
}

void PoissonModelBase::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this) - sizeof(NeuronModel)
                              + (refractory_time.capacity() * sizeof(unsigned int));
    footprint->emplace("PoissonModel", my_footprint);

    NeuronModel::record_memory_footprint(footprint);
}
