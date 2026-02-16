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

#include "Types.h"

#include "cpp-utility/Interval.hpp"

#include <random>
#include <string>
#include <utility>
#include <vector>

class IntervalFactory {
public:
    static utility::Interval<RelearnTypes::step_type> generate_random_interval(std::mt19937& mt);

    static utility::Interval<RelearnTypes::step_type> get_random_interval(RelearnTypes::step_type num_steps, RelearnTypes::step_type frequency, std::mt19937& mt);

    static std::vector<utility::Interval<RelearnTypes::step_type>> get_random_non_overlapping_intervals(RelearnTypes::step_type num_intervals, RelearnTypes::step_type num_steps, std::mt19937& mt);

    static std::pair<utility::Interval<RelearnTypes::step_type>, std::string> generate_random_interval_description(std::mt19937& mt);
};
