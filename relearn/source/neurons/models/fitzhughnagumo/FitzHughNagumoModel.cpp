/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FitzHughNagumoModel.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/input/ActivityInput.h"
#include "neurons/models/NeuronModel.h"
#include "neurons/models/fitzhughnagumo/Calculation.h"
#include "neurons/models/fitzhughnagumo/Parameters.h"
#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include <memory>
#include <utility>

using models::FitzHughNagumoModel;

FitzHughNagumoModel::FitzHughNagumoModel(
    const unsigned int h,
    std::shared_ptr<ActivityInput> activity_input,
    std::shared_ptr<FiredStatusCommunicator> fired_status_communicator,
    const models::fitzhughnagumo::Parameters<double>& params)
    : NeuronModel{ h, std::move(activity_input), std::move(fired_status_communicator) }
    , parameters{ params } {
}

void FitzHughNagumoModel::init(const number_neurons_type number_neurons) {
    NeuronModel::init(number_neurons);
    w.resize(number_neurons);
    init_neurons(0, number_neurons);
}

void FitzHughNagumoModel::create_neurons(const number_neurons_type creation_count) {
    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    w.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void FitzHughNagumoModel::register_neuron_monitor(NeuronMonitor& monitor) {
    NeuronModel::register_neuron_monitor(monitor);
    monitor.register_paramter("w", [this](const RelearnTypes::number_neurons_type neuron_id) {
        return static_cast<float>(w[neuron_id]);
    });
}

void FitzHughNagumoModel::init_neurons(const number_neurons_type start_id, const number_neurons_type end_id) {
    const auto init_w = parameters.get_init_w();
    const auto init_x = parameters.get_init_x();

    for (auto neuron_id = start_id; neuron_id < end_id; ++neuron_id) {
        const auto id = NeuronID{ neuron_id };
        w[neuron_id] = init_w;
        set_x(id, init_x);
    }
}

void models::FitzHughNagumoModel::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this) - sizeof(NeuronModel)
                              + (w.capacity() * sizeof(double));
    footprint->emplace("FitzHughNagumoModel", my_footprint);

    NeuronModel::record_memory_footprint(footprint);
}

void FitzHughNagumoModel::update_activity_benchmark() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = 1.0 / h;

    RelearnException::check(w.size() == number_neurons, "FitzHughNagumoModel::update_activity_benchmark: Size of w is too small");

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto a = parameters.get_a();
        const auto b = parameters.get_b();
        const auto phi = parameters.get_phi();

        const auto converted_id = NeuronID{ neuron_id };

        const auto input = get_input(converted_id);

        auto x_val = get_x(converted_id);
        auto w_val = w[neuron_id];

        for (auto integration_steps = 0U; integration_steps < h; ++integration_steps) {
            const auto x_increase = x_val - (x_val * x_val * x_val * (1.0 / 3.0)) - w_val + input;
            const auto w_increase = phi * (x_val + a - b * w_val);

            x_val += x_increase * scale;
            w_val += w_increase * scale;
        }

        const auto spiked = w_val > x_val - x_val * x_val * x_val * (1.0 / 3.0) && x_val > 1.0;

        if (spiked) {
            set_fired(converted_id, FiredStatus::Fired);
        } else {
            set_fired(converted_id, FiredStatus::Inactive);
        }

        set_x(converted_id, x_val);
        w[neuron_id] = w_val;
    }
}

void FitzHughNagumoModel::update_activity() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = 1.0 / h;

    RelearnException::check(w.size() == number_neurons, "FitzHughNagumoModel::update_activity: Size of w is too small");

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto a = parameters.get_a();
        const auto b = parameters.get_b();
        const auto phi = parameters.get_phi();

        const auto input = get_input(neuron_id);

        auto x_val = get_x(neuron_id);
        auto w_val = w[neuron_id];

        for (auto integration_steps = 0U; integration_steps < h; ++integration_steps) {
            const auto x_increase = x_val - (x_val * x_val * x_val * (1.0 / 3.0)) - w_val + input;
            const auto w_increase = phi * (x_val + a - b * w_val);

            x_val += x_increase * scale;
            w_val += w_increase * scale;
        }

        const auto spiked = w_val > x_val - x_val * x_val * x_val * (1.0 / 3.0) && x_val > 1.0;

        if (spiked) {
            set_fired(neuron_id, FiredStatus::Fired);
        } else {
            set_fired(neuron_id, FiredStatus::Inactive);
        }

        set_x(neuron_id, x_val);
        w[neuron_id] = w_val;
    }
}
