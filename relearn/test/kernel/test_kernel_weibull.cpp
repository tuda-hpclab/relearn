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

#include "algorithm/Kernel/Weibull.h"
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

TEST_F(WeibullKernelTest, testDefaultConstruction) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = WeibullDistributionKernel{};

    ASSERT_EQ(kernel.get_k(), WeibullDistributionKernel::default_k);
    ASSERT_EQ(kernel.get_b(), WeibullDistributionKernel::default_b);
}

TEST_F(WeibullKernelTest, testConstruction) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_NO_THROW(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(0.1), utility::as<attraction_type>(1.0)));
    ASSERT_NO_THROW(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(1.0), utility::as<attraction_type>(0.1)));
    ASSERT_NO_THROW(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(1.1), utility::as<attraction_type>(1.0)));

    ASSERT_NO_THROW(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(0.1), utility::as<attraction_type>(750.0)));
    ASSERT_NO_THROW(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(1.0), utility::as<attraction_type>(750.0)));
    ASSERT_NO_THROW(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(1.1), utility::as<attraction_type>(750.0)));
}

TEST_F(WeibullKernelTest, testConstructionExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(0.1), utility::as<attraction_type>(0.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(0.0), utility::as<attraction_type>(0.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(1.1), utility::as<attraction_type>(0.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(-5.0), utility::as<attraction_type>(0.0)), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(0.1), utility::as<attraction_type>(-2.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(0.0), utility::as<attraction_type>(-2.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(1.1), utility::as<attraction_type>(-2.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(-5.0), utility::as<attraction_type>(-2.0)), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(0.0), utility::as<attraction_type>(0.1)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(0.0), utility::as<attraction_type>(0.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(0.0), utility::as<attraction_type>(1.1)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(0.0), utility::as<attraction_type>(5.0)), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(-2.0), utility::as<attraction_type>(0.1)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(-2.0), utility::as<attraction_type>(-0.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(-2.0), utility::as<attraction_type>(1.1)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = WeibullDistributionKernel(utility::as<attraction_type>(-2.0), utility::as<attraction_type>(-5.0)), RelearnException);
}

TEST_F(WeibullKernelTest, testGetter) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto kernel = WeibullDistributionKernel{ utility::as<attraction_type>(8541.58), utility::as<attraction_type>(156478.587) };
    ASSERT_EQ(kernel.get_k(), utility::as<attraction_type>(8541.58));
    ASSERT_EQ(kernel.get_b(), utility::as<attraction_type>(156478.587));
}

TEST_F(WeibullKernelTest, testPositions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = WeibullDistributionKernel{ utility::as<attraction_type>(7.5), utility::as<attraction_type>(1.1) };

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

TEST_F(WeibullKernelTest, testDefaultProbabilities) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = WeibullDistributionKernel{};

    const auto probability_1 = kernel.get_probability(utility::as<space_type>(0.0));
    const auto probability_2 = kernel.get_probability(utility::as<space_type>(0.5));
    const auto probability_3 = kernel.get_probability(utility::as<space_type>(1.2));
    const auto probability_4 = kernel.get_probability(utility::as<space_type>(5.2));
    const auto probability_5 = kernel.get_probability(utility::as<space_type>(6.6));
    const auto probability_6 = kernel.get_probability(utility::as<space_type>(10.1));
    const auto probability_7 = kernel.get_probability(utility::as<space_type>(150.5));

    ASSERT_NEAR(probability_1, 1.0, eps);
    ASSERT_GT(probability_1, probability_2);
    ASSERT_GT(probability_2, probability_3);
    ASSERT_GT(probability_3, probability_4);
    ASSERT_GT(probability_4, probability_5);
    ASSERT_GT(probability_5, probability_6);
    ASSERT_GT(probability_6, probability_7);

    // The exponential tail of the default kernel underflows to exactly 0 once the distance exceeds ~103
    // in attraction_type = float, so the far probability can only be asserted to stay non-negative.
    ASSERT_GE(probability_7, utility::as<attraction_type>(0.0));
}

TEST_F(WeibullKernelTest, testProbabilities) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = WeibullDistributionKernel{ utility::as<attraction_type>(2.0), utility::as<attraction_type>(0.25) };

    const auto probability_1 = kernel.get_probability(utility::as<space_type>(0.0));
    const auto probability_2 = kernel.get_probability(utility::as<space_type>(0.5));
    const auto probability_3 = kernel.get_probability(utility::as<space_type>(1.2));
    const auto probability_4 = kernel.get_probability(utility::as<space_type>(5.2));
    const auto probability_5 = kernel.get_probability(utility::as<space_type>(6.6));
    const auto probability_6 = kernel.get_probability(utility::as<space_type>(10.1));
    const auto probability_7 = kernel.get_probability(utility::as<space_type>(150.5));

    ASSERT_NEAR(probability_1, 0.0, eps);
    ASSERT_NEAR(probability_2, 0.234853, eps);
    ASSERT_NEAR(probability_3, 0.418606, eps);
    ASSERT_NEAR(probability_4, 0.003014, eps);
    ASSERT_NEAR(probability_5, 0.0000615243, eps);
    ASSERT_NEAR(probability_6, 0.0, eps);
    ASSERT_NEAR(probability_7, 0.0, eps);
}
