/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_kernel.h"

#include "algorithm/Kernel/Linear.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <cpp-utility/Cast.hpp>

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <vector>

// The kernels are parameterized in attraction_type and evaluated at a space_type distance; the literals
// below are spelled in those types so that they need no conversion.
using attraction_type = RelearnTypes::attraction_type;
using space_type = RelearnTypes::space_type;

TEST_F(LinearKernelTest, testDefaultConstruction) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = LinearDistributionKernel{};

    ASSERT_EQ(kernel.get_cutoff(), LinearDistributionKernel::default_cutoff);
}

TEST_F(LinearKernelTest, testConstruction) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_NO_THROW(std::ignore = LinearDistributionKernel(utility::as<attraction_type>(0.2)));
    ASSERT_NO_THROW(std::ignore = LinearDistributionKernel(utility::as<attraction_type>(1.1)));
    ASSERT_NO_THROW(std::ignore = LinearDistributionKernel(utility::as<attraction_type>(841248.45)));
}

TEST_F(LinearKernelTest, testConstructionExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_THROW_NO_PRINT(std::ignore = LinearDistributionKernel(utility::as<attraction_type>(-0.001)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = LinearDistributionKernel(utility::as<attraction_type>(-1.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = LinearDistributionKernel(utility::as<attraction_type>(-1.1)), RelearnException);
}

TEST_F(LinearKernelTest, testGetter) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto kernel = LinearDistributionKernel{ utility::as<attraction_type>(841248.45) };
    ASSERT_EQ(kernel.get_cutoff(), utility::as<attraction_type>(841248.45));
}

TEST_F(LinearKernelTest, testDefaultProbability) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = LinearDistributionKernel{};

    auto source_positions = std::vector<RelearnTypes::position_type>{};
    auto target_positions = std::vector<RelearnTypes::position_type>{};

    source_positions.emplace_back(utility::as<space_type>(0.0), utility::as<space_type>(0.0), utility::as<space_type>(0.0));
    source_positions.emplace_back(utility::as<space_type>(5.63347), utility::as<space_type>(6.77419), utility::as<space_type>(6.68022));
    source_positions.emplace_back(utility::as<space_type>(0.434595), utility::as<space_type>(1.3763), utility::as<space_type>(9.85119));
    source_positions.emplace_back(utility::as<space_type>(9.71033), utility::as<space_type>(6.27692), utility::as<space_type>(5.29921));
    source_positions.emplace_back(utility::as<space_type>(8.00937), utility::as<space_type>(4.2314), utility::as<space_type>(4.9292));

    target_positions.emplace_back(utility::as<space_type>(0.0), utility::as<space_type>(0.0), utility::as<space_type>(0.0));
    target_positions.emplace_back(utility::as<space_type>(5.63347), utility::as<space_type>(6.77419), utility::as<space_type>(6.68022));
    target_positions.emplace_back(utility::as<space_type>(0.434595), utility::as<space_type>(1.3763), utility::as<space_type>(9.85119));
    target_positions.emplace_back(utility::as<space_type>(9.71033), utility::as<space_type>(6.27692), utility::as<space_type>(5.29921));
    target_positions.emplace_back(utility::as<space_type>(8.00937), utility::as<space_type>(4.2314), utility::as<space_type>(4.9292));

    for (const auto& source_position : source_positions) {
        for (const auto& target_position : target_positions) {
            const auto calculated_difference = (target_position - source_position).calculate_2_norm<space_type>();

            const auto probability_1 = kernel.get_probability(calculated_difference);
            const auto probability_2 = kernel.get_probability(source_position, target_position);

            ASSERT_NEAR(probability_1, probability_2, eps);
            ASSERT_NEAR(probability_1, 1.0, eps);
        }
    }
}

TEST_F(LinearKernelTest, testCutoffProbability) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = LinearDistributionKernel{ utility::as<attraction_type>(120.4) };

    const auto probability_1 = kernel.get_probability(utility::as<space_type>(120.4));
    const auto probability_2 = kernel.get_probability(utility::as<space_type>(120.5));
    const auto probability_3 = kernel.get_probability(utility::as<space_type>(0.0));
    const auto probability_4 = kernel.get_probability(utility::as<space_type>(60.2));
    const auto probability_5 = kernel.get_probability(utility::as<space_type>(30.1));
    const auto probability_6 = kernel.get_probability(utility::as<space_type>(90.3));
    const auto probability_7 = kernel.get_probability(utility::as<space_type>(150.5));

    ASSERT_NEAR(probability_1, 0.0, eps);
    ASSERT_NEAR(probability_2, 0.0, eps);
    ASSERT_NEAR(probability_3, 1.0, eps);
    ASSERT_NEAR(probability_4, 0.5, eps);
    ASSERT_NEAR(probability_5, 0.75, eps);
    ASSERT_NEAR(probability_6, 0.25, eps);
    ASSERT_NEAR(probability_7, 0.0, eps);
}

TEST_F(LinearKernelTest, testPositions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = LinearDistributionKernel{ utility::as<attraction_type>(120.4) };

    auto source_positions = std::vector<RelearnTypes::position_type>{};
    auto target_positions = std::vector<RelearnTypes::position_type>{};

    source_positions.emplace_back(utility::as<space_type>(0.0), utility::as<space_type>(0.0), utility::as<space_type>(0.0));
    source_positions.emplace_back(utility::as<space_type>(5.63347), utility::as<space_type>(6.77419), utility::as<space_type>(6.68022));
    source_positions.emplace_back(utility::as<space_type>(0.434595), utility::as<space_type>(1.3763), utility::as<space_type>(9.85119));
    source_positions.emplace_back(utility::as<space_type>(9.71033), utility::as<space_type>(6.27692), utility::as<space_type>(5.29921));
    source_positions.emplace_back(utility::as<space_type>(8.00937), utility::as<space_type>(4.2314), utility::as<space_type>(4.9292));

    target_positions.emplace_back(utility::as<space_type>(0.0), utility::as<space_type>(0.0), utility::as<space_type>(0.0));
    target_positions.emplace_back(utility::as<space_type>(5.63347), utility::as<space_type>(6.77419), utility::as<space_type>(6.68022));
    target_positions.emplace_back(utility::as<space_type>(0.434595), utility::as<space_type>(1.3763), utility::as<space_type>(9.85119));
    target_positions.emplace_back(utility::as<space_type>(9.71033), utility::as<space_type>(6.27692), utility::as<space_type>(5.29921));
    target_positions.emplace_back(utility::as<space_type>(8.00937), utility::as<space_type>(4.2314), utility::as<space_type>(4.9292));

    for (const auto& source_position : source_positions) {
        for (const auto& target_position : target_positions) {
            const auto calculated_difference = (target_position - source_position).calculate_2_norm<space_type>();

            const auto probability_1 = kernel.get_probability(calculated_difference);
            const auto probability_2 = kernel.get_probability(source_position, target_position);

            ASSERT_NEAR(probability_1, probability_2, eps);
        }
    }
}
