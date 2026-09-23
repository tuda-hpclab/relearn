/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FitzHughNagumoModelCPU.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/models/fitzhughnagumo/Calculation.h"
#include "neurons/models/fitzhughnagumo/Parameters.h"
#include "util/RelearnException.h"

using models::FitzHughNagumoModelCPU;

void FitzHughNagumoModelCPU::update_activity_benchmark() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = activity_type{ 1 } / static_cast<activity_type>(h);

    RelearnException::check(w.size() == number_neurons, "FitzHughNagumoModel::update_activity_benchmark: Size of w is too small");

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
            const auto x_increase = x_val - (x_val * x_val * x_val * (activity_type{ 1 } / activity_type{ 3 })) - w_val + input;
            const auto w_increase = phi * (x_val + a - b * w_val);

            x_val += x_increase * scale;
            w_val += w_increase * scale;
        }

        const auto spiked = w_val > x_val - x_val * x_val * x_val * (activity_type{ 1 } / activity_type{ 3 }) && x_val > activity_type{ 1 };

        if (spiked) {
            set_fired(neuron_id, FiredStatus::Fired);
        } else {
            set_fired(neuron_id, FiredStatus::Inactive);
        }

        set_x(neuron_id, x_val);
        w[neuron_id] = w_val;
    }
}

void FitzHughNagumoModelCPU::update_activity() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = activity_type{ 1 } / static_cast<activity_type>(h);

    RelearnException::check(w.size() == number_neurons, "FitzHughNagumoModel::update_activity: Size of w is too small");

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) { // NOLINT(openmp-exception-escape) - accessor bound checks are invariants, already validated before entering the loop; rest is pure arithmetic
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
            const auto x_increase = x_val - (x_val * x_val * x_val * (activity_type{ 1 } / activity_type{ 3 })) - w_val + input;
            const auto w_increase = phi * (x_val + a - b * w_val);

            x_val += x_increase * scale;
            w_val += w_increase * scale;
        }

        const auto spiked = w_val > x_val - x_val * x_val * x_val * (activity_type{ 1 } / activity_type{ 3 }) && x_val > activity_type{ 1 };

        if (spiked) {
            set_fired(neuron_id, FiredStatus::Fired);
        } else {
            set_fired(neuron_id, FiredStatus::Inactive);
        }

        set_x(neuron_id, x_val);
        w[neuron_id] = w_val;
    }
}
