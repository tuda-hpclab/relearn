/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "IzhikevichModel.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/input/ActivityInput.h"
#include "neurons/models/NeuronModel.h"
#include "neurons/models/izhikevich/Calculation.h"
#include "neurons/models/izhikevich/Parameters.h"
#include "util/NeuronID.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include <memory>
#include <utility>

using models::IzhikevichModel;

IzhikevichModel::IzhikevichModel(
    const unsigned int h,
    std::shared_ptr<ActivityInput> activity_input,
    std::shared_ptr<FiredStatusCommunicator> fired_status_communicator,
    const models::izhikevich::Parameters<double>& params)
    : NeuronModel{ h, std::move(activity_input), std::move(fired_status_communicator) }
    , parameters{ params } {
}

void IzhikevichModel::init(const number_neurons_type number_neurons) {
    NeuronModel::init(number_neurons);
    u.resize(number_neurons);
    init_neurons(0, number_neurons);
}

void IzhikevichModel::create_neurons(const number_neurons_type creation_count) {
    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    u.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void IzhikevichModel::register_neuron_monitor(NeuronMonitor& monitor) {
    NeuronModel::register_neuron_monitor(monitor);
    monitor.register_paramter("u", [this](const RelearnTypes::number_neurons_type neuron_id) {
        return static_cast<float>(u[neuron_id]);
    });
}

void IzhikevichModel::init_neurons(const number_neurons_type start_id, const number_neurons_type end_id) {
    const auto c = parameters.get_c();
    for (auto neuron_id = start_id; neuron_id < end_id; ++neuron_id) {
        const auto id = NeuronID{ neuron_id };
        u[neuron_id] = 0.0;
        set_x(id, c);
    }
}

void IzhikevichModel::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this) - sizeof(NeuronModel)
                              + (u.capacity() * sizeof(double));
    footprint->emplace("IzhikevichModel", my_footprint);

    NeuronModel::record_memory_footprint(footprint);
}

void IzhikevichModel::update_activity_benchmark() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = 1.0 / h;

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto k1 = parameters.get_k1();
        const auto k2 = parameters.get_k2();
        const auto k3 = parameters.get_k3();
        const auto a = parameters.get_a();
        const auto b = parameters.get_b();
        const auto c = parameters.get_c();
        const auto d = parameters.get_d();
        const auto V_spike = parameters.get_V_spike();

        const auto converted_id = NeuronID{ neuron_id };

        const auto input = get_input(converted_id);

        auto x_val = get_x(converted_id);
        auto u_val = u[neuron_id];

        auto has_spiked = FiredStatus::Inactive;

        for (auto integration_steps = 0U; integration_steps < h; ++integration_steps) {
            const auto x_increase = (k1 * x_val * x_val) + (k2 * x_val) + k3 - u_val + input;
            const auto u_increase = a * (b * x_val - u_val);

            x_val += x_increase * scale;
            u_val += u_increase * scale;

            const auto spiked = x_val >= V_spike;
            if (spiked) {
                x_val = c;
                u_val += d;
                has_spiked = FiredStatus::Fired;
                break;
            }
        }

        set_fired(converted_id, has_spiked);
        set_x(converted_id, x_val);
        u[neuron_id] = u_val;
    }
}

void IzhikevichModel::update_activity() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = 1.0 / h;

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto k1 = parameters.get_k1();
        const auto k2 = parameters.get_k2();
        const auto k3 = parameters.get_k3();
        const auto a = parameters.get_a();
        const auto b = parameters.get_b();
        const auto c = parameters.get_c();
        const auto d = parameters.get_d();
        const auto V_spike = parameters.get_V_spike();

        const auto ab = a * b;

        const auto input = get_input(neuron_id);

        auto x_val = get_x(neuron_id);
        auto u_val = u[neuron_id];

        auto has_spiked = FiredStatus::Inactive;

        for (auto integration_steps = 0U; integration_steps < h; ++integration_steps) {
            const auto x_increase = (k1 * x_val * x_val) + (k2 * x_val) + k3 - u_val + input;
            const auto u_increase = (ab * x_val) - (a * u_val);

            x_val += x_increase * scale;
            u_val += u_increase * scale;

            const auto spiked = x_val >= V_spike;
            if (spiked) {
                x_val = c;
                u_val += d;
                has_spiked = FiredStatus::Fired;
                break;
            }
        }

        set_fired(neuron_id, has_spiked);
        set_x(neuron_id, x_val);
        u[neuron_id] = u_val;
    }
}
