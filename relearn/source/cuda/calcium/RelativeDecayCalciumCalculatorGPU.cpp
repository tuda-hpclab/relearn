/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelativeDecayCalciumCalculatorGPU.h"

#include "cuda/calcium/Calcium.h"
#include "neurons/NeuronsExtraInfo.h"

void RelativeDecayCalciumCalculatorGPU::update_target_calcium(const step_type step) noexcept {
    if (!decay_interval.hits_step(step)) {
        return;
    }

    update_target_calcium_relative_decay_entry(get_neuron_extra_info()->get_gpu_handle(), CalciumHandle{ .target_calcium = get_d_target_calcium() }, decay_amount);
}
