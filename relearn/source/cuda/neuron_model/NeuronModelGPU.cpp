/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronModelGPU.h"

#include "cuda/util/Util.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "neurons/input/ActivityInput.h"
#include "util/Timers.h"

#include <cpp-utility/MemoryFootprint.hpp>

void NeuronModelGPU::update_electrical_activity(const step_type step) {

    Timers::start(TimerRegion::WAIT_FOR_EXCHANGE_FIRED_STATUS);
    fired_status_comm->wait_for_exchange_to_finish();
    Timers::stop_and_add(TimerRegion::WAIT_FOR_EXCHANGE_FIRED_STATUS);

    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT);
    act_input->update_input(step, act_input_stream);
    cudaDeviceSynchronize_bridge();
    // The device is already caught up above, so resolving the GPU-event timers queued by the
    // various (stream-async, unsynchronized) ActivityInput kernels here is free -- it never
    // introduces a synchronization point of its own.
    cuda_resolve_gpu_timers();
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT);

    Timers::start(TimerRegion::CALC_ACTIVITY);
    update_activity();
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY);

    Timers::start(TimerRegion::EXCHANGE_FIRED_STATUS);
    fired_status_comm->commit_local_fired_status(step);
    fired_status_comm->exchange_fired_status(step);
    Timers::stop_and_add(TimerRegion::EXCHANGE_FIRED_STATUS);
}

void NeuronModelGPU::update_electrical_activity_benchmark(const step_type step) {
    Timers::start(TimerRegion::EXCHANGE_FIRED_STATUS);
    fired_status_comm->commit_local_fired_status(step);
    fired_status_comm->exchange_fired_status(step);
    Timers::stop_and_add(TimerRegion::EXCHANGE_FIRED_STATUS);

    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT);
    act_input->update_input(step, act_input_stream);
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT);

    Timers::start(TimerRegion::CALC_ACTIVITY);
    update_activity_benchmark();
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY);
}

void NeuronModelGPU::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    Base::record_memory_footprint(footprint);
    footprint->emplace("NeuronModel GPU", x.get_memory_footprint());
}
