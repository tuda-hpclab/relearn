/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronModel.h"

#include "neurons/NetworkGraph.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/input/ActivityInput.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include <memory>
#include <span>
#include <utility>

NeuronModel::NeuronModel(const unsigned int h, std::shared_ptr<ActivityInput>&& activity_input, std::shared_ptr<FiredStatusCommunicator>&& fired_status_communicator)
    : precision_h(h)
    , act_input(std::move(activity_input))
    , fired_status_comm(std::move(fired_status_communicator)) {
    RelearnException::check(h > 0, "NeuronModel::NeuronModel: h is 0");
    RelearnException::check(this->act_input != nullptr, "NeuronModel::NeuronModel: The activity input is empty");
    RelearnException::check(this->fired_status_comm != nullptr, "NeuronModel::NeuronModel: The fired status communicator is empty");

    fired_status_recorder = std::make_shared<FiredStatusRecorder>();
    this->fired_status_comm->set_fired_status_recorder(fired_status_recorder);
}

void NeuronModel::init(number_neurons_type number_neurons) {
    RelearnException::check(number_local_neurons == 0, "NeuronModel::init: Was already initialized");
    RelearnException::check(number_neurons > 0, "NeuronModel::init: Must initialize with more than 0 neurons");

    number_local_neurons = number_neurons;

    x.resize(number_neurons, 0.0);

    RelearnException::check(act_input != nullptr, "NeuronModel::init: Activity input was not set");
    act_input->init(number_neurons);
    fired_status_comm->init(number_neurons);
    fired_status_recorder->init(number_neurons);
}

void NeuronModel::create_neurons(number_neurons_type creation_count) {
    RelearnException::check(number_local_neurons > 0, "NeuronModel::create_neurons: Was not initialized");
    RelearnException::check(creation_count > 0, "NeuronModel::create_neurons: Must create more than 0 neurons");

    const auto current_size = number_local_neurons;
    const auto new_size = current_size + creation_count;
    number_local_neurons = new_size;

    x.resize(new_size, 0.0);

    act_input->create_neurons(creation_count);
    fired_status_comm->create_neurons(creation_count);
    fired_status_recorder->create_neurons(creation_count);
}

void NeuronModel::register_neuron_monitor(NeuronMonitor& monitor) {
    act_input->register_neuron_monitor(monitor);
    fired_status_recorder->register_neuron_monitor(monitor);
    monitor.register_paramter("x", [this](const RelearnTypes::number_neurons_type neuron_id) {
        return static_cast<float>(x[neuron_id]);
    });
}

void NeuronModel::disable_neurons(const std::span<const NeuronID> neuron_ids) {
    fired_status_recorder->disable_neurons(neuron_ids);
}

void NeuronModel::enable_neurons(const std::span<const NeuronID> neuron_ids) {
    fired_status_recorder->enable_neurons(neuron_ids);
}

void NeuronModel::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this) + (x.capacity() * sizeof(double));
    footprint->emplace("NeuronModel", my_footprint);

    act_input->record_memory_footprint(footprint);
    fired_status_comm->record_memory_footprint(footprint);
    fired_status_recorder->record_memory_footprint(footprint);
}

void NeuronModel::set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
    const auto is_filled = new_extra_info != nullptr;
    RelearnException::check(is_filled, "NeuronModel::set_extra_infos: new_extra_info is empty");
    extra_infos = std::move(new_extra_info);

    act_input->set_extra_infos(extra_infos);
    fired_status_comm->set_extra_infos(extra_infos);
}

void NeuronModel::set_network_graph(std::shared_ptr<NetworkGraph> new_network_graph) {
    const auto is_filled = new_network_graph != nullptr;
    RelearnException::check(is_filled, "SynapticInputCalculator::set_network_graph: new_network_graph is empty");
    network_graph = std::move(new_network_graph);

    act_input->set_network_graph(network_graph);
    fired_status_comm->set_network_graph(network_graph);
}

bool NeuronModel::has_fired(const NeuronID neuron_id) const {
    return fired_status_recorder->has_fired(neuron_id);
}

std::span<const FiredStatus> NeuronModel::get_fired() const noexcept {
    return fired_status_recorder->get_fired();
}

std::span<const double> NeuronModel::get_input() const noexcept {
    return act_input->get_input();
}

void NeuronModel::update_electrical_activity(const step_type step) {
    Timers::start(TimerRegion::EXCHANGE_FIRED_STATUS);
    fired_status_comm->commit_local_fired_status(step);
    fired_status_comm->exchange_fired_status(step);
    Timers::stop_and_add(TimerRegion::EXCHANGE_FIRED_STATUS);

    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT);
    act_input->update_input(step);
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT);

    Timers::start(TimerRegion::CALC_ACTIVITY);
    update_activity();
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY);
}

void NeuronModel::update_electrical_activity_benchmark(const step_type step) {
    Timers::start(TimerRegion::EXCHANGE_FIRED_STATUS);
    fired_status_comm->commit_local_fired_status(step);
    fired_status_comm->exchange_fired_status(step);
    Timers::stop_and_add(TimerRegion::EXCHANGE_FIRED_STATUS);

    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT);
    act_input->update_input(step);
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT);

    Timers::start(TimerRegion::CALC_ACTIVITY);
    update_activity_benchmark();
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY);
}

void NeuronModel::notify_of_plasticity_change(const step_type step) {
    fired_status_comm->notify_of_plasticity_change(step);
}
