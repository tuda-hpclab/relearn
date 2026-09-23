/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "PoissonModelCPU.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/models/poisson/Calculation.h"
#include "neurons/models/poisson/Parameters.h"
#include "util/Random.h"
#include "util/RandomHolderKey.h"

using models::PoissonModelCPU;

void PoissonModelCPU::update_activity() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = activity_type{ 1 } / static_cast<activity_type>(h);

    const auto tau_x = parameters.get_tau_x();
    const auto tau_x_inverse = activity_type{ 1 } / tau_x;

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale, tau_x_inverse) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) { // NOLINT(openmp-exception-escape) - accessor bound checks are invariants; RandomHolder::get_random_uniform_double(0.0, 1.0) can never actually throw here
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto x_0 = parameters.get_x_0();
        const auto refractory_period = parameters.get_refractory_period();

        const auto input = get_input(neuron_id);

        auto x_val = get_x(neuron_id);

        for (auto integration_steps = 0U; integration_steps < h; integration_steps++) {
            x_val += ((x_0 - x_val) * tau_x_inverse + static_cast<activity_type>(input)) * scale;
        }

        if (refractory_time[neuron_id] == 0) {
            const auto threshold = RandomHolder::get_random_uniform_double(RandomHolderKey::PoissonModel, activity_type{ 0 }, activity_type{ 1 });
            const auto f = x_val >= threshold;
            if (f) {
                set_fired(neuron_id, FiredStatus::Fired);
                refractory_time[neuron_id] = refractory_period;
            } else {
                set_fired(neuron_id, FiredStatus::Inactive);
            }
        } else {
            set_fired(neuron_id, FiredStatus::Inactive);
            --refractory_time[neuron_id];
        }

        set_x(neuron_id, x_val);
    }
}

void PoissonModelCPU::update_activity_benchmark() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = activity_type{ 1 } / static_cast<activity_type>(h);

    const auto tau_x = parameters.get_tau_x();
    const auto tau_x_inverse = activity_type{ 1 } / tau_x;

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale, tau_x_inverse) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) { // NOLINT(openmp-exception-escape) - accessor bound checks are invariants; RandomHolder::get_random_uniform_double(0.0, 1.0) can never actually throw here
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto x_0 = parameters.get_x_0();
        const auto refractory_period = parameters.get_refractory_period();

        const auto input = get_input(neuron_id);

        auto x_val = get_x(neuron_id);

        for (auto integration_steps = 0U; integration_steps < h; integration_steps++) {
            x_val += ((x_0 - x_val) * tau_x_inverse + static_cast<activity_type>(input)) * scale;
        }

        if (refractory_time[neuron_id] == 0) {
            const auto threshold = RandomHolder::get_random_uniform_double(RandomHolderKey::PoissonModel, activity_type{ 0 }, activity_type{ 1 });
            const auto f = x_val >= threshold;
            if (f) {
                set_fired(neuron_id, FiredStatus::Fired);
                refractory_time[neuron_id] = refractory_period;
            } else {
                set_fired(neuron_id, FiredStatus::Inactive);
            }
        } else {
            set_fired(neuron_id, FiredStatus::Inactive);
            --refractory_time[neuron_id];
        }

        set_x(neuron_id, x_val);
    }
}
