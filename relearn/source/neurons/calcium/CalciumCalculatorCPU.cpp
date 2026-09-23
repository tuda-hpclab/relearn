/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "CalciumCalculatorCPU.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/UpdateStatus.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <limits>
#include <span>

void CalciumCalculatorCPU::update_calcium(const step_type step, const std::span<const FiredStatus> fired_status) {
    const auto info_size = extra_infos->get_size();
    const auto fired_size = fired_status.size();
    const auto calcium_size = calcium.size();
    const auto target_calcium_size = target_calcium.size();

    const auto all_same_size = info_size == fired_size && fired_size == calcium_size && calcium_size == target_calcium_size;
    RelearnException::check(all_same_size, "CalciumCalculatorCPU::update_calcium: The vectors had different sizes!");

    Timers::start(TimerRegion::UPDATE_CALCIUM);
    update_current_calcium(fired_status);
    Timers::stop_and_add(TimerRegion::UPDATE_CALCIUM);

    Timers::start(TimerRegion::UPDATE_TARGET_CALCIUM);
    update_target_calcium(step);
    Timers::stop_and_add(TimerRegion::UPDATE_TARGET_CALCIUM);
}

void CalciumCalculatorCPU::update_current_calcium(std::span<const FiredStatus> fired_status) noexcept {
    const auto scale = (calcium_type{ 1 } / static_cast<calcium_type>(h));
    const auto tau_C_inverse = calcium_type{ -1 } / tau_C;

    const auto disable_flags = extra_infos->get_disable_flags();

    auto minimum_id = NeuronID::uninitialized_id();
    auto minimum_ca = std::numeric_limits<calcium_type>::max();

    auto maximum_id = NeuronID::uninitialized_id();
    auto maximum_ca = -std::numeric_limits<calcium_type>::max();

#pragma omp parallel default(none) shared(disable_flags, fired_status, scale, tau_C_inverse, minimum_ca, maximum_ca, minimum_id, maximum_id)
    {
        auto thread_minimum_id = NeuronID::uninitialized_id();
        auto thread_minimum_ca = std::numeric_limits<calcium_type>::max();

        auto thread_maximum_id = NeuronID::uninitialized_id();
        auto thread_maximum_ca = -std::numeric_limits<calcium_type>::max();

#pragma omp for nowait
        for (auto neuron_id = 0UL; neuron_id < calcium.size(); ++neuron_id) {
            if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
                continue;
            }

            // Update calcium depending on the firing
            auto c = calcium[neuron_id];
            auto local_beta = beta * static_cast<calcium_type>(fired_status[neuron_id] == FiredStatus::Fired);

            for (auto integration_steps = 0U; integration_steps < h; ++integration_steps) {
                c += scale * (c * tau_C_inverse + local_beta);
            }

            calcium[neuron_id] = c;

            if (thread_minimum_ca > c) {
                thread_minimum_ca = c;
                thread_minimum_id = NeuronID(neuron_id);
            }

            if (thread_maximum_ca < c) {
                thread_maximum_ca = c;
                thread_maximum_id = NeuronID(neuron_id);
            }
        }

#pragma omp critical
        {
            if (minimum_ca > thread_minimum_ca) {
                minimum_ca = thread_minimum_ca; // NOLINT(clang-analyzer-deadcode.DeadStores) - shared across threads, read by the next thread entering this critical section
                minimum_id = thread_minimum_id;
            }

            if (maximum_ca < thread_maximum_ca) {
                maximum_ca = thread_maximum_ca; // NOLINT(clang-analyzer-deadcode.DeadStores) - shared across threads, read by the next thread entering this critical section
                maximum_id = thread_maximum_id;
            }
        }
    }

    current_minimum = minimum_id;
    current_maximum = maximum_id;
}
