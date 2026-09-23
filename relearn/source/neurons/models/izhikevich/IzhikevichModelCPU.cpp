/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "IzhikevichModelCPU.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/models/izhikevich/Calculation.h"
#include "neurons/models/izhikevich/Parameters.h"
#include "util/NeuronID.h"

using models::IzhikevichModelCPU;

void IzhikevichModelCPU::update_activity_benchmark() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = activity_type{ 1 } / static_cast<activity_type>(h);

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

void IzhikevichModelCPU::update_activity() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = activity_type{ 1 } / static_cast<activity_type>(h);

    const auto k1 = parameters.get_k1();
    const auto k2 = parameters.get_k2();
    const auto k3 = parameters.get_k3();
    const auto a = parameters.get_a();
    const auto b = parameters.get_b();
    const auto c = parameters.get_c();
    const auto d = parameters.get_d();
    const auto V_spike = parameters.get_V_spike();
    const auto ab = a * b;

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale, k1, k2, k3, a, b, c, d, V_spike, ab) schedule(guided) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) { // NOLINT(openmp-exception-escape) - accessor bound checks are invariants, already validated before entering the loop; rest is pure arithmetic
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto input = get_input(neuron_id);

        auto x_val = get_x(neuron_id);
        auto u_val = u[neuron_id];

        auto has_spiked = FiredStatus::Inactive;

        for (auto integration_steps = 0U; integration_steps < h; ++integration_steps) {
            const auto x_increase = (k1 * x_val * x_val) + (k2 * x_val) + k3 - u_val + static_cast<activity_type>(input);
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
