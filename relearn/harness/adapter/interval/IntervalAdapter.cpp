/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "IntervalAdapter.h"

#include <cpp-utility/Interval.hpp>

#include <sstream>
#include <string>

std::string IntervalAdapter::codify_interval(const utility::Interval<std::uint32_t>& interval) {
    auto ss = std::stringstream{};
    ss << interval.begin << '-' << interval.end << ':' << interval.frequency;
    return ss.str();
}
