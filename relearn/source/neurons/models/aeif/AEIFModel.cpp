/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "AEIFModel.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/input/ActivityInput.h"
#include "neurons/models/NeuronModel.h"
#include "neurons/models/aeif/Calculation.h"
#include "neurons/models/aeif/Parameters.h"
#include "util/NeuronID.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include <cmath>
#include <memory>
#include <utility>

using models::AEIFModel;

AEIFModel::AEIFModel(
    const unsigned int h,
    std::shared_ptr<ActivityInput> activity_input,
    std::shared_ptr<FiredStatusCommunicator> fired_status_communicator,
    const models::aeif::Parameters<double>& params)
    : NeuronModel{ h, std::move(activity_input), std::move(fired_status_communicator) }
    , parameters{ params } {
}

void AEIFModel::init(const number_neurons_type number_neurons) {
    NeuronModel::init(number_neurons);
    w.resize(number_neurons);
    init_neurons(0, number_neurons);
}

void AEIFModel::create_neurons(const number_neurons_type creation_count) {
    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    w.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void AEIFModel::register_neuron_monitor(NeuronMonitor& monitor) {
    NeuronModel::register_neuron_monitor(monitor);
    monitor.register_paramter("w", [this](const RelearnTypes::number_neurons_type neuron_id) {
        return static_cast<float>(w[neuron_id]);
    });
}

void AEIFModel::init_neurons(const number_neurons_type start_id, const number_neurons_type end_id) {
    const auto E_L = parameters.get_E_L();
    for (auto neuron_id = start_id; neuron_id < end_id; ++neuron_id) {
        const auto id = NeuronID{ neuron_id };

        w[neuron_id] = 0.0;
        set_x(id, E_L);
    }
}

void AEIFModel::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this) - sizeof(NeuronModel)
                              + (w.capacity() * sizeof(double));
    footprint->emplace("AEIFModel", my_footprint);

    NeuronModel::record_memory_footprint(footprint);
}

void AEIFModel::update_activity_benchmark() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = 1.0 / h;

    const auto d_T = parameters.get_d_T();
    const auto tau_w = parameters.get_tau_w();
    const auto C = parameters.get_C();

    const auto d_T_inverse = 1.0 / d_T;
    const auto tau_w_inverse = 1.0 / tau_w;
    const auto C_inverse = 1.0 / C;

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale, d_T, d_T_inverse, tau_w_inverse, C_inverse) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto g_L = parameters.get_g_L();
        const auto E_L = parameters.get_E_L();
        const auto V_T = parameters.get_V_T();
        const auto a = parameters.get_a();
        const auto b = parameters.get_b();
        const auto V_spike = parameters.get_V_spike();

        const auto converted_id = NeuronID{ neuron_id };

        const auto input = get_input(converted_id);

        auto x_val = get_x(converted_id);
        auto w_val = w[neuron_id];

        auto has_spiked = FiredStatus::Inactive;

        for (auto integration_steps = 0U; integration_steps < h; ++integration_steps) {
            const auto linear_part = -g_L * (x_val - E_L);
            const auto exp_part = g_L * d_T * std::exp((x_val - V_T) * d_T_inverse);
            const auto x_increase = (linear_part + exp_part - w_val + input) * C_inverse;
            const auto w_increase = (a * (x_val - E_L) - w_val) * tau_w_inverse;

            x_val += x_increase * scale;
            w_val += w_increase * scale;

            if (x_val >= V_spike) {
                x_val = E_L;
                w_val += b;
                has_spiked = FiredStatus::Fired;
                break;
            }
        }

        set_fired(converted_id, has_spiked);
        set_x(converted_id, x_val);
        w[neuron_id] = w_val;
    }
}

void AEIFModel::update_activity() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = 1.0 / h;

    const auto d_T = parameters.get_d_T();
    const auto tau_w = parameters.get_tau_w();
    const auto C = parameters.get_C();

    const auto d_T_inverse = 1.0 / d_T;
    const auto tau_w_inverse = 1.0 / tau_w;
    const auto C_inverse = 1.0 / C;

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale, d_T, d_T_inverse, tau_w_inverse, C_inverse) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto g_L = parameters.get_g_L();
        const auto E_L = parameters.get_E_L();
        const auto V_T = parameters.get_V_T();
        const auto a = parameters.get_a();
        const auto b = parameters.get_b();
        const auto V_spike = parameters.get_V_spike();

        const auto input = get_input(neuron_id);

        auto x_val = get_x(neuron_id);
        auto w_val = w[neuron_id];

        auto has_spiked = FiredStatus::Inactive;

        for (auto integration_steps = 0U; integration_steps < h; ++integration_steps) {
            const auto linear_part = -g_L * (x_val - E_L);
            const auto exp_part = g_L * d_T * std::exp((x_val - V_T) * d_T_inverse);
            const auto x_increase = (linear_part + exp_part - w_val + input) * C_inverse;
            const auto w_increase = (a * (x_val - E_L) - w_val) * tau_w_inverse;

            x_val += x_increase * scale;
            w_val += w_increase * scale;

            if (x_val >= V_spike) {
                x_val = E_L;
                w_val += b;
                has_spiked = FiredStatus::Fired;
                break;
            }
        }

        set_fired(neuron_id, has_spiked);
        set_x(neuron_id, x_val);
        w[neuron_id] = w_val;
    }
}
