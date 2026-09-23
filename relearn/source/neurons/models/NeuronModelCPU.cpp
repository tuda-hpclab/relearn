/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronModelCPU.h"

#include "neurons/firing/FiredStatusCommunicator.h"
#include "neurons/input/ActivityInput.h"
#include "util/Timers.h"

void NeuronModelCPU::update_electrical_activity(const step_type step) {

    Timers::start(TimerRegion::WAIT_FOR_EXCHANGE_FIRED_STATUS);
    fired_status_comm->wait_for_exchange_to_finish();
    Timers::stop_and_add(TimerRegion::WAIT_FOR_EXCHANGE_FIRED_STATUS);

    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT);
    act_input->update_input(step);
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT);

    Timers::start(TimerRegion::CALC_ACTIVITY);
    update_activity();
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY);

    Timers::start(TimerRegion::EXCHANGE_FIRED_STATUS);
    fired_status_comm->commit_local_fired_status(step);
    fired_status_comm->exchange_fired_status(step);
    Timers::stop_and_add(TimerRegion::EXCHANGE_FIRED_STATUS);
}

void NeuronModelCPU::update_electrical_activity_benchmark(const step_type step) {
    Timers::start(TimerRegion::EXCHANGE_FIRED_STATUS);
    fired_status_comm->commit_local_fired_status(step);
    fired_status_comm->exchange_fired_status(step);
    Timers::stop_and_add(TimerRegion::EXCHANGE_FIRED_STATUS);

    Timers::start(TimerRegion::CALC_ACTIVITY_INPUT);
    act_input->update_input(step);
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT);

    Timers::start(TimerRegion::CALC_ACTIVITY);
    update_activity_benchmark();
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY);
}
