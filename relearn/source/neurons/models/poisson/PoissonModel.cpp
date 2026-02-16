/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "PoissonModel.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/input/ActivityInput.h"
#include "neurons/models/NeuronModel.h"
#include "neurons/models/poisson/Calculation.h"
#include "neurons/models/poisson/Parameters.h"
#include "util/NeuronID.h"
#include "util/Random.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include <memory>
#include <utility>

using models::PoissonModel;

PoissonModel::PoissonModel(
    const unsigned int h,
    std::shared_ptr<ActivityInput> activity_input,
    std::shared_ptr<FiredStatusCommunicator> fired_status_communicator,
    const models::poisson::Parameters<double, unsigned int>& params)
    : NeuronModel{ h, std::move(activity_input), std::move(fired_status_communicator) }
    , parameters{ params } {
}

void PoissonModel::init(const number_neurons_type number_neurons) {
    NeuronModel::init(number_neurons);
    refractory_time.resize(number_neurons, 0);
}

void PoissonModel::init_neurons([[maybe_unused]] const number_neurons_type start_id, [[maybe_unused]] const number_neurons_type end_id) {
}

void PoissonModel::create_neurons(const number_neurons_type creation_count) {
    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    refractory_time.resize(old_size + creation_count, 0);
}

void PoissonModel::register_neuron_monitor(NeuronMonitor& monitor) {
    NeuronModel::register_neuron_monitor(monitor);
    monitor.register_paramter("r", [this](const RelearnTypes::number_neurons_type neuron_id) {
        return refractory_time[neuron_id];
    });
}

void models::PoissonModel::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this) - sizeof(NeuronModel)
                              + (refractory_time.capacity() * sizeof(unsigned int));
    footprint->emplace("PoissonModel", my_footprint);

    NeuronModel::record_memory_footprint(footprint);
}

void PoissonModel::update_activity_benchmark() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = 1.0 / h;

    const auto tau_x = parameters.get_tau_x();
    const auto tau_x_inverse = 1.0 / tau_x;

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale, tau_x_inverse) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto x_0 = parameters.get_x_0();
        const auto refractory_period = parameters.get_refractory_period();

        const auto converted_id = NeuronID{ neuron_id };

        const auto input = get_input(converted_id);

        auto x_val = get_x(converted_id);

        for (auto integration_steps = 0U; integration_steps < h; integration_steps++) {
            x_val += ((x_0 - x_val) * tau_x_inverse + input) * scale;
        }

        if (refractory_time[neuron_id] == 0) {
            const auto threshold = RandomHolder::get_random_uniform_double(RandomHolderKey::PoissonModel, 0.0, 1.0);
            const auto f = x_val >= threshold;
            if (f) {
                set_fired(converted_id, FiredStatus::Fired);
                refractory_time[neuron_id] = refractory_period;
            } else {
                set_fired(converted_id, FiredStatus::Inactive);
            }
        } else {
            set_fired(converted_id, FiredStatus::Inactive);
            --refractory_time[neuron_id];
        }

        set_x(converted_id, x_val);
    }
}

void PoissonModel::update_activity() {
    const auto number_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = 1.0 / h;

    const auto tau_x = parameters.get_tau_x();
    const auto tau_x_inverse = 1.0 / tau_x;

#pragma omp parallel for shared(disable_flags, number_neurons, h, scale, tau_x_inverse) default(none)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto x_0 = parameters.get_x_0();
        const auto refractory_period = parameters.get_refractory_period();

        const auto input = get_input(neuron_id);

        auto x_val = get_x(neuron_id);

        for (auto integration_steps = 0U; integration_steps < h; integration_steps++) {
            x_val += ((x_0 - x_val) * tau_x_inverse + input) * scale;
        }

        if (refractory_time[neuron_id] == 0) {
            const auto threshold = RandomHolder::get_random_uniform_double(RandomHolderKey::PoissonModel, 0.0, 1.0);
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
