#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "util/Timers.h"

#include "factory/random/random_factory.h"

#include <random>

class TimersAdapter {
public:
    static TimerRegion get_random_timer_region(std::mt19937& mt) {
        constexpr auto min = 0;
        constexpr auto max = NUMBER_TIMERS - 1;

        const auto index = RandomFactory::get_random_integer<unsigned int>(min, max, mt);
        return static_cast<TimerRegion>(index);
    }
};
