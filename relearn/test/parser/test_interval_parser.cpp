/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_interval_parser.h"

#include "io/parser/IntervalParser.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/interval/interval_factory.h"
#include "factory/random/random_factory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

TEST_F(IntervalParserTest, testParseInterval) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [golden_interval, description] = IntervalFactory::generate_random_interval_description(mt);

    const auto& opt_interval = IntervalParser::parse_interval(description);

    ASSERT_TRUE(opt_interval.has_value());

    const auto& interval = opt_interval.value();

    ASSERT_EQ(golden_interval.begin, interval.begin);
    ASSERT_EQ(golden_interval.end, interval.end);
    ASSERT_EQ(golden_interval.frequency, interval.frequency);
}

TEST_F(IntervalParserTest, testParseIntervalException1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& opt_interval = IntervalParser::parse_interval({});

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalParserTest, testParseIntervalException2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using int_type = utility::Interval<RelearnTypes::step_type>::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomFactory::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomFactory::get_random_integer<int_type>(min, max, mt);

    auto ss = std::stringstream{};
    ss << std::min(begin, end) << '-' << std::max(begin, end);

    const auto& description = ss.str();

    const auto& opt_interval = IntervalParser::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalParserTest, testParseIntervalException3) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using int_type = utility::Interval<RelearnTypes::step_type>::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomFactory::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomFactory::get_random_integer<int_type>(min, max, mt);

    auto ss = std::stringstream{};
    ss << begin << ':' << frequency;

    const auto& description = ss.str();

    const auto& opt_interval = IntervalParser::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalParserTest, testParseIntervalException4) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using int_type = utility::Interval<RelearnTypes::step_type>::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomFactory::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomFactory::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomFactory::get_random_integer<int_type>(min, max, mt);

    auto ss = std::stringstream{};
    ss << '-' << std::min(begin, end) << '-' << std::max(begin, end) << ':' << frequency;

    const auto& description = ss.str();

    const auto& opt_interval = IntervalParser::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalParserTest, testParseIntervalException5) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using int_type = utility::Interval<RelearnTypes::step_type>::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomFactory::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomFactory::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomFactory::get_random_integer<int_type>(min, max, mt);

    auto ss = std::stringstream{};
    ss << '-' << std::min(begin, end) << '-' << std::max(begin, end) << ':' << frequency << ':';

    const auto& description = ss.str();

    const auto& opt_interval = IntervalParser::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalParserTest, testParseIntervalException6) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using int_type = utility::Interval<RelearnTypes::step_type>::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomFactory::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomFactory::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomFactory::get_random_integer<int_type>(min, max, mt);

    auto ss = std::stringstream{};
    ss << std::max(begin, end) << '-' << std::min(begin, end) << ':' << frequency;

    const auto& description = ss.str();

    const auto& opt_interval = IntervalParser::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalParserTest, testParseIntervals1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto golden_intervals = std::vector<utility::Interval<RelearnTypes::step_type>>{};
    auto ss = std::stringstream{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalFactory::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        if (i != 9) {
            ss << ';';
        }
    }

    const auto& intervals = IntervalParser::parse_description_as_intervals(ss.str());

    for (auto i = 0U; i < 10U; i++) {
        ASSERT_EQ(golden_intervals[i], intervals[i]);
    }
}

TEST_F(IntervalParserTest, testParseInterval2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto golden_intervals = std::vector<utility::Interval<RelearnTypes::step_type>>{};
    auto ss = std::stringstream{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalFactory::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        ss << ';';
    }

    const auto& intervals = IntervalParser::parse_description_as_intervals(ss.str());

    ASSERT_EQ(intervals.size(), 10);
}

TEST_F(IntervalParserTest, testParseIntervalsException1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& intervals = IntervalParser::parse_description_as_intervals({});
    ASSERT_TRUE(intervals.empty());
}

TEST_F(IntervalParserTest, testParseIntervalsException2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& intervals = IntervalParser::parse_description_as_intervals("sgahkllkrduf,'�.;f�lsa�df::SAfd--dfasdjf45");
    ASSERT_TRUE(intervals.empty());
}

TEST_F(IntervalParserTest, testParseIntervalsException3) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto golden_intervals = std::vector<utility::Interval<RelearnTypes::step_type>>{};
    auto ss = std::stringstream{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalFactory::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        if (i != 9) {
            ss << ',';
        }
    }

    const auto& intervals = IntervalParser::parse_description_as_intervals(ss.str());

    ASSERT_TRUE(intervals.empty());
}

TEST_F(IntervalParserTest, testParseIntervalsException4) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto golden_intervals = std::vector<utility::Interval<RelearnTypes::step_type>>{};
    auto ss = std::stringstream{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalFactory::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        if (i != 9) {
            ss << ':';
        }
    }

    const auto& intervals = IntervalParser::parse_description_as_intervals(ss.str());

    ASSERT_TRUE(intervals.empty());
}

TEST_F(IntervalParserTest, testParseIntervalsException5) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto golden_intervals = std::vector<utility::Interval<RelearnTypes::step_type>>{};
    auto ss = std::stringstream{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalFactory::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        ss << ';';
    }

    ss << "136546543135";

    const auto& intervals = IntervalParser::parse_description_as_intervals(ss.str());

    ASSERT_EQ(intervals.size(), 10);
}
