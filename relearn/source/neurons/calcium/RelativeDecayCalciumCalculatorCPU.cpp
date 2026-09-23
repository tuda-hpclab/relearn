/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelativeDecayCalciumCalculatorCPU.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/UpdateStatus.h"

void RelativeDecayCalciumCalculatorCPU::update_target_calcium(const step_type step) noexcept {
    if (!decay_interval.hits_step(step)) {
        return;
    }

    const auto& extra_information = get_neuron_extra_info();
    const auto disable_flags = extra_information->get_disable_flags();
    const auto target_ca = get_target_calcium_internal();

#pragma omp parallel for default(none) shared(disable_flags, target_ca)
    for (auto neuron_id = 0UL; neuron_id < target_ca.size(); ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const auto new_target_calcium = target_ca[neuron_id] * decay_amount;
        target_ca[neuron_id] = new_target_calcium;
    }
}
