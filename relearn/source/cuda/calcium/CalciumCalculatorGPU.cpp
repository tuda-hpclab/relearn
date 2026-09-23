/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "CalciumCalculatorGPU.h"

#include "cuda/calcium/Calcium.h"
#include "neurons/NeuronsExtraInfo.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>

CalciumCalculatorGPU::~CalciumCalculatorGPU() = default;

void CalciumCalculatorGPU::update_calcium(const step_type step, const FiredStatus* d_fired) {
    const auto info_size = extra_infos->get_size();
    const auto all_same_size = info_size > 0 && info_size == calcium.size() && calcium.size() == target_calcium.size();
    RelearnException::check(all_same_size, "CalciumCalculatorGPU::update_calcium: The vectors had different sizes!");

    Timers::start(TimerRegion::UPDATE_CALCIUM);
    update_current_calcium(d_fired);
    Timers::stop_and_add(TimerRegion::UPDATE_CALCIUM);

    Timers::start(TimerRegion::UPDATE_TARGET_CALCIUM);
    update_target_calcium(step);
    Timers::stop_and_add(TimerRegion::UPDATE_TARGET_CALCIUM);
}

void CalciumCalculatorGPU::update_current_calcium(const FiredStatus* d_fired) {
    update_current_calcium_entry(extra_infos->get_gpu_handle(), d_fired, CalciumHandle{ .calcium = get_d_calcium() }, h, tau_C, beta);
}

void CalciumCalculatorGPU::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    Base::record_memory_footprint(footprint);
    footprint->emplace("CalciumCalculator GPU", calcium.get_memory_footprint() + target_calcium.get_memory_footprint());
}
