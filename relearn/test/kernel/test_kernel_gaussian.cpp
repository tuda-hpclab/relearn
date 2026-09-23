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

#include "algorithm/Kernel/Gaussian.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <cpp-utility/Cast.hpp>

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <utility>
#include <vector>

// The kernels are parameterized in attraction_type and evaluated at a space_type distance; the literals
// below are spelled in those types so that they need no conversion.
using attraction_type = RelearnTypes::attraction_type;
using space_type = RelearnTypes::space_type;

TEST_F(GaussianKernelTest, testDefaultConstruction) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = GaussianDistributionKernel{};

    ASSERT_EQ(kernel.get_mu(), GaussianDistributionKernel::default_mu);
    ASSERT_EQ(kernel.get_sigma(), GaussianDistributionKernel::default_sigma);
}

TEST_F(GaussianKernelTest, testConstruction) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(0.1), utility::as<attraction_type>(1.0)));
    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(0.0), utility::as<attraction_type>(1.0)));
    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(1.1), utility::as<attraction_type>(1.0)));
    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(-5.0), utility::as<attraction_type>(1.0)));

    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(0.1), utility::as<attraction_type>(750.0)));
    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(0.0), utility::as<attraction_type>(750.0)));
    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(1.1), utility::as<attraction_type>(750.0)));
    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(-5.0), utility::as<attraction_type>(750.0)));
}

TEST_F(GaussianKernelTest, testConstructionExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(0.1), utility::as<attraction_type>(0.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(0.0), utility::as<attraction_type>(0.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(1.1), utility::as<attraction_type>(0.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(-5.0), utility::as<attraction_type>(0.0)), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(0.1), utility::as<attraction_type>(-2.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(0.0), utility::as<attraction_type>(-2.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(1.1), utility::as<attraction_type>(-2.0)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(utility::as<attraction_type>(-5.0), utility::as<attraction_type>(-2.0)), RelearnException);
}

TEST_F(GaussianKernelTest, testGetter) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto kernel = GaussianDistributionKernel{ utility::as<attraction_type>(8541.58), utility::as<attraction_type>(156478.587) };
    ASSERT_EQ(kernel.get_mu(), utility::as<attraction_type>(8541.58));
    ASSERT_EQ(kernel.get_sigma(), utility::as<attraction_type>(156478.587));
}

TEST_F(GaussianKernelTest, testIncreasingDistance) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto sigma = utility::as<attraction_type>(43.2211);
    const auto mu = attraction_type{ 0 };

    const auto kernel = GaussianDistributionKernel{ mu, sigma };

    const auto probability_1 = kernel.get_probability(utility::as<space_type>(0.0));
    const auto probability_2 = kernel.get_probability(utility::as<space_type>(1.0));
    const auto probability_3 = kernel.get_probability(utility::as<space_type>(2.0));
    const auto probability_4 = kernel.get_probability(utility::as<space_type>(3.0));
    const auto probability_5 = kernel.get_probability(utility::as<space_type>(4.0));

    ASSERT_EQ(probability_1, 1.0);
    ASSERT_GT(probability_1, probability_2);
    ASSERT_GT(probability_2, probability_3);
    ASSERT_GT(probability_3, probability_4);
    ASSERT_GT(probability_4, probability_5);
    ASSERT_GT(probability_5, 0.0);
}

TEST_F(GaussianKernelTest, testIncreasingSigma) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto sigma_1 = utility::as<attraction_type>(1.25);
    const auto sigma_2 = utility::as<attraction_type>(3.52);
    const auto sigma_3 = utility::as<attraction_type>(4.99);
    const auto sigma_4 = utility::as<attraction_type>(5.01);
    const auto sigma_5 = utility::as<attraction_type>(9.6);

    const auto mu = attraction_type{ 0 };

    const auto kernel_1 = GaussianDistributionKernel{ mu, sigma_1 };
    const auto kernel_2 = GaussianDistributionKernel{ mu, sigma_2 };
    const auto kernel_3 = GaussianDistributionKernel{ mu, sigma_3 };
    const auto kernel_4 = GaussianDistributionKernel{ mu, sigma_4 };
    const auto kernel_5 = GaussianDistributionKernel{ mu, sigma_5 };

    for (const auto value : { utility::as<space_type>(0.05), space_type{ 1 }, space_type{ 10 }, space_type{ 20 } }) {
        const auto probability_1 = kernel_1.get_probability(value);
        const auto probability_2 = kernel_2.get_probability(value);
        const auto probability_3 = kernel_3.get_probability(value);
        const auto probability_4 = kernel_4.get_probability(value);
        const auto probability_5 = kernel_5.get_probability(value);

        ASSERT_GT(1.0, probability_1);
        ASSERT_LT(probability_1, probability_2);
        ASSERT_LT(probability_2, probability_3);
        ASSERT_LT(probability_3, probability_4);
        ASSERT_LT(probability_4, probability_5);
        ASSERT_GT(probability_5, 0.0);
    }
}

TEST_F(GaussianKernelTest, testIncreasingMu) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto sigma = utility::as<attraction_type>(1.25);

    const auto mu_1 = utility::as<attraction_type>(-20.2);
    const auto mu_2 = utility::as<attraction_type>(-0.05);
    const auto mu_3 = utility::as<attraction_type>(0.0);
    const auto mu_4 = utility::as<attraction_type>(0.95);
    const auto mu_5 = utility::as<attraction_type>(12.01);

    const auto kernel_1 = GaussianDistributionKernel{ mu_1, sigma };
    const auto kernel_2 = GaussianDistributionKernel{ mu_2, sigma };
    const auto kernel_3 = GaussianDistributionKernel{ mu_3, sigma };
    const auto kernel_4 = GaussianDistributionKernel{ mu_4, sigma };
    const auto kernel_5 = GaussianDistributionKernel{ mu_5, sigma };

    for (const auto value : { utility::as<space_type>(0.05), space_type{ 1 }, space_type{ 10 }, space_type{ 20 } }) {
        const auto probability_1 = kernel_1.get_probability(value + mu_1);
        const auto probability_2 = kernel_2.get_probability(value + mu_2);
        const auto probability_3 = kernel_3.get_probability(value + mu_3);
        const auto probability_4 = kernel_4.get_probability(value + mu_4);
        const auto probability_5 = kernel_5.get_probability(value + mu_5);

        ASSERT_LE(probability_1, 1.0);
        ASSERT_NEAR(probability_2, probability_1, eps);
        ASSERT_NEAR(probability_3, probability_1, eps);
        ASSERT_NEAR(probability_4, probability_1, eps);
        ASSERT_NEAR(probability_5, probability_1, eps);
        ASSERT_GE(probability_1, 0.0);
    }
}

TEST_F(GaussianKernelTest, testPositions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto sigma = utility::as<attraction_type>(1.25);
    const auto mu = utility::as<attraction_type>(2.01);

    const auto kernel = GaussianDistributionKernel{ mu, sigma };

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
