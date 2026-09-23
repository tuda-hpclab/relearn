/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "AEIFModelCPU.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/models/aeif/Calculation.h"
#include "neurons/models/aeif/Parameters.h"

#include <cmath>

using models::AEIFModelCPU;

void AEIFModelCPU::update_activity() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = activity_type{ 1 } / static_cast<activity_type>(h);

    const auto d_T = parameters.get_d_T();
    const auto tau_w = parameters.get_tau_w();
    const auto C = parameters.get_C();

    const auto d_T_inverse = activity_type{ 1 } / d_T;
    const auto tau_w_inverse = activity_type{ 1 } / tau_w;
    const auto C_inverse = activity_type{ 1 } / C;

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale, d_T, d_T_inverse, tau_w_inverse, C_inverse) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) { // NOLINT(openmp-exception-escape) - accessor bound checks are invariants (neuron_id < number_local_neurons by construction); rest is pure arithmetic
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
            const auto x_increase = (linear_part + exp_part - w_val + static_cast<activity_type>(input)) * C_inverse;
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

void AEIFModelCPU::update_activity_benchmark() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = activity_type{ 1 } / static_cast<activity_type>(h);

    const auto d_T = parameters.get_d_T();
    const auto tau_w = parameters.get_tau_w();
    const auto C = parameters.get_C();

    const auto d_T_inverse = activity_type{ 1 } / d_T;
    const auto tau_w_inverse = activity_type{ 1 } / tau_w;
    const auto C_inverse = activity_type{ 1 } / C;

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale, d_T, d_T_inverse, tau_w_inverse, C_inverse) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) { // NOLINT(openmp-exception-escape) - accessor bound checks are invariants (neuron_id < number_local_neurons by construction); rest is pure arithmetic
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
            const auto x_increase = (linear_part + exp_part - w_val + static_cast<activity_type>(input)) * C_inverse;
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
