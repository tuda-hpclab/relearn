/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "interval_factory.h"

#include "types/BasicTypes.h"
#include "util/RelearnException.h"

#include "adapter/interval/IntervalAdapter.h"

#include "factory/random/random_factory.h"

#include <cpp-utility/Interval.hpp>

#include <range/v3/algorithm/find.hpp>
#include <range/v3/view/indices.hpp>

#include <algorithm>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

utility::Interval<RelearnTypes::step_type> IntervalFactory::generate_random_interval(std::mt19937& mt) {
    using int_type = utility::Interval<RelearnTypes::step_type>::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomFactory::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomFactory::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomFactory::get_random_integer<int_type>(min, max, mt);

    return utility::Interval<RelearnTypes::step_type>{ .begin = std::min(begin, end), .end = std::max(begin, end), .frequency = frequency };
}

utility::Interval<RelearnTypes::step_type> IntervalFactory::get_random_interval(RelearnTypes::step_type num_steps, RelearnTypes::step_type frequency, std::mt19937& mt) {
    const auto begin = RandomFactory::get_random_integer(0U, num_steps, mt);
    const auto end = RandomFactory::get_random_integer(begin, num_steps, mt);
    return utility::Interval<RelearnTypes::step_type>{ .begin = begin, .end = end, .frequency = frequency };
}

std::vector<utility::Interval<RelearnTypes::step_type>> IntervalFactory::get_random_non_overlapping_intervals(RelearnTypes::step_type num_intervals, RelearnTypes::step_type num_steps, std::mt19937& mt) {
    auto intervals = std::vector<utility::Interval<RelearnTypes::step_type>>{};
    const auto interval_max_size = num_steps / num_intervals / 10;

    for ([[maybe_unused]] const auto _ : ranges::views::indices(num_intervals)) {
        auto test_interval = std::vector<utility::Interval<RelearnTypes::step_type>>{};
        auto interval = utility::Interval<RelearnTypes::step_type>{};
        do {
            test_interval.clear();
            ranges::copy(intervals, std::back_inserter(test_interval));
            const auto begin = RandomFactory::get_random_integer(0U, num_steps - interval_max_size, mt);
            const auto end = RandomFactory::get_random_integer(begin, begin + interval_max_size, mt);
            interval = { .begin = begin, .end = end, .frequency = 1U };
            test_interval.emplace_back(interval);
        } while (utility::Interval<RelearnTypes::step_type>::check_intervals_for_intersection(test_interval));
        intervals.emplace_back(interval);
    }
    return intervals;
}

std::pair<utility::Interval<RelearnTypes::step_type>, std::string> IntervalFactory::generate_random_interval_description(std::mt19937& mt) {
    auto interval = generate_random_interval(mt);
    auto description = IntervalAdapter::codify_interval(interval);
    return { interval, std::move(description) };
}
