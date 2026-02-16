/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_step_parser.h"

#include "Types.h"

#include "io/parser/StepParser.h"

#include "cpp-utility/Interval.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/random/random_factory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

TEST_F(StepParserTest, testGenerateFunction1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using int_type = utility::Interval<RelearnTypes::step_type>::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    auto function = StepParser::generate_step_check_function(std::vector<utility::Interval<RelearnTypes::step_type>>{});

    for (auto step = RelearnTypes::step_type{ 0 }; step < 10000; step++) {
        const auto result_1 = function(step);
        ASSERT_FALSE(result_1) << step;

        const auto random_step = RandomFactory::get_random_integer<int_type>(min, max, mt);
        const auto result_2 = function(random_step);
        ASSERT_FALSE(result_2) << random_step;
    }
}

TEST_F(StepParserTest, testGenerateFunction2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using int_type = utility::Interval<RelearnTypes::step_type>::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto i = utility::Interval<RelearnTypes::step_type>{ .begin = min, .end = max, .frequency = 1 };

    const auto function = StepParser::generate_step_check_function({ i });

    for (auto step = RelearnTypes::step_type{ 0 }; step < 10000; step++) {
        const auto result_1 = function(step);
        ASSERT_TRUE(result_1) << step;

        const auto random_step = RandomFactory::get_random_integer<int_type>(min, max, mt);
        const auto result_2 = function(random_step);
        ASSERT_TRUE(result_2) << random_step;
    }
}

TEST_F(StepParserTest, testGenerateFunction3) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using int_type = utility::Interval<RelearnTypes::step_type>::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto i1 = utility::Interval<RelearnTypes::step_type>{ .begin = 0, .end = 99, .frequency = 10 };
    const auto i2 = utility::Interval<RelearnTypes::step_type>{ .begin = 100, .end = 999, .frequency = 10 };
    const auto i3 = utility::Interval<RelearnTypes::step_type>{ .begin = 1000, .end = 2689, .frequency = 10 };
    const auto i4 = utility::Interval<RelearnTypes::step_type>{ .begin = 2690, .end = 10000, .frequency = 10 };

    const auto function = StepParser::generate_step_check_function({ i1, i2, i3, i4 });

    for (auto step = RelearnTypes::step_type{ 0 }; step < 20000; step++) {
        const auto result_1 = function(step);
        ASSERT_EQ(result_1, (step <= 10000 && step % 10 == 0)) << step;

        const auto random_step = RandomFactory::get_random_integer<int_type>(min, max, mt);
        const auto result_2 = function(random_step);
        ASSERT_EQ(result_2, (random_step <= 10000 && random_step % 10 == 0)) << random_step;
    }
}

TEST_F(StepParserTest, testGenerateFunction4) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using int_type = utility::Interval<RelearnTypes::step_type>::step_type;

    constexpr auto min = 10000;
    constexpr auto max = 90000;

    auto begin = RandomFactory::get_random_integer<int_type>(min, max, mt);
    auto end = RandomFactory::get_random_integer<int_type>(min, max, mt);

    const auto i1 = utility::Interval<RelearnTypes::step_type>{ .begin = 0, .end = 99, .frequency = 7 };
    const auto i2 = utility::Interval<RelearnTypes::step_type>{ .begin = 100, .end = 999, .frequency = 7 };
    const auto i3 = utility::Interval<RelearnTypes::step_type>{ .begin = 1000, .end = 2689, .frequency = 7 };
    const auto i4 = utility::Interval<RelearnTypes::step_type>{ .begin = 2690, .end = 9999, .frequency = 7 };
    const auto i5 = utility::Interval<RelearnTypes::step_type>{ .begin = std::min(begin, end), .end = std::max(begin, end), .frequency = 11 };

    auto ss = std::stringstream{};
    ss << codify_interval(i1) << ';';
    ss << codify_interval(i2) << ';';
    ss << codify_interval(i3) << ';';
    ss << codify_interval(i4) << ';';
    ss << codify_interval(i5);

    auto function_1 = StepParser::generate_step_check_function({ i1, i2, i3, i4, i5 });
    auto function_2 = StepParser::generate_step_check_function(ss.str());

    for (auto step = RelearnTypes::step_type{ 0 }; step < 90000; step++) {
        const auto result_1 = function_1(step);
        const auto result_2 = function_2(step);
        ASSERT_EQ(result_1, result_2) << step;
    }
}

TEST_F(StepParserTest, testGenerateFunction5) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using int_type = utility::Interval<RelearnTypes::step_type>::step_type;

    constexpr auto min = 10000;
    constexpr auto max = 90000;

    auto begin = RandomFactory::get_random_integer<int_type>(min, max, mt);
    auto end = RandomFactory::get_random_integer<int_type>(min, max, mt);

    auto i1 = utility::Interval<RelearnTypes::step_type>{ .begin = 0, .end = 99, .frequency = 7 };
    auto i2 = utility::Interval<RelearnTypes::step_type>{ .begin = 100, .end = 999, .frequency = 7 };
    auto i3 = utility::Interval<RelearnTypes::step_type>{ .begin = 400, .end = 2689, .frequency = 7 };
    auto i4 = utility::Interval<RelearnTypes::step_type>{ .begin = 2690, .end = 9999, .frequency = 7 };
    auto i5 = utility::Interval<RelearnTypes::step_type>{ .begin = std::min(begin, end), .end = std::max(begin, end), .frequency = 11 };

    auto ss = std::stringstream{};
    ss << codify_interval(i1) << ';';
    ss << codify_interval(i2) << ';';
    ss << codify_interval(i3) << ';';
    ss << codify_interval(i4) << ';';
    ss << codify_interval(i5);

    auto function_1 = StepParser::generate_step_check_function({ i1, i2, i3, i4, i5 });
    ASSERT_FALSE(function_1 != nullptr);
}
