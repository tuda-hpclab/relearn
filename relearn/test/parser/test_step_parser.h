#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnTest.hpp"

#include "types/BasicTypes.h"

#include "factory/random/random_factory.h"

#include <cpp-utility/Interval.hpp>

#include <climits>
#include <string>
#include <utility>

class StepParserTest : public RelearnTest {
protected:
    utility::Interval<RelearnTypes::step_type> generate_random_interval() {
        using int_type = utility::Interval<RelearnTypes::step_type>::step_type;

        constexpr auto min = std::numeric_limits<int_type>::min();
        constexpr auto max = std::numeric_limits<int_type>::max();

        const auto begin = RandomFactory::get_random_integer<int_type>(min, max, mt);
        const auto end = RandomFactory::get_random_integer<int_type>(min, max, mt);
        const auto frequency = RandomFactory::get_random_integer<int_type>(min, max, mt);

        return utility::Interval<RelearnTypes::step_type>{ .begin = std::min(begin, end), .end = std::max(begin, end), .frequency = frequency };
    }

    static std::string codify_interval(const utility::Interval<RelearnTypes::step_type>& interval) {
        auto ss = std::stringstream{};
        ss << interval.begin << '-' << interval.end << ':' << interval.frequency;
        return ss.str();
    }

    std::pair<utility::Interval<RelearnTypes::step_type>, std::string> generate_random_interval_description() {
        auto interval = generate_random_interval();
        auto description = codify_interval(interval);
        return { interval, std::move(description) };
    }
};
