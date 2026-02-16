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

#include "algorithm/Kernel/Gaussian.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "test_kernel.h"

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

    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(0.1, 1.0));
    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(0.0, 1.0));
    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(1.1, 1.0));
    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(-5.0, 1.0));

    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(0.1, 750.0));
    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(0.0, 750.0));
    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(1.1, 750.0));
    ASSERT_NO_THROW(std::ignore = GaussianDistributionKernel(-5.0, 750.0));
}

TEST_F(GaussianKernelTest, testConstructionExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(0.1, 0.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(0.0, 0.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(1.1, 0.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(-5.0, 0.0), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(0.1, -2.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(0.0, -2.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(1.1, -2.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GaussianDistributionKernel(-5.0, -2.0), RelearnException);
}

TEST_F(GaussianKernelTest, testGetter) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto kernel = GaussianDistributionKernel{ 8541.58, 156478.587 };
    ASSERT_EQ(kernel.get_mu(), 8541.58);
    ASSERT_EQ(kernel.get_sigma(), 156478.587);
}

TEST_F(GaussianKernelTest, testIncreasingDistance) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto sigma = 43.2211;
    const auto mu = 0.0;

    const auto kernel = GaussianDistributionKernel{ mu, sigma };

    const auto probability_1 = kernel.get_probability(0.0);
    const auto probability_2 = kernel.get_probability(1.0);
    const auto probability_3 = kernel.get_probability(2.0);
    const auto probability_4 = kernel.get_probability(3.0);
    const auto probability_5 = kernel.get_probability(4.0);

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

    const auto sigma_1 = 1.25;
    const auto sigma_2 = 3.52;
    const auto sigma_3 = 4.99;
    const auto sigma_4 = 5.01;
    const auto sigma_5 = 9.6;

    const auto mu = 0.0;

    const auto kernel_1 = GaussianDistributionKernel{ mu, sigma_1 };
    const auto kernel_2 = GaussianDistributionKernel{ mu, sigma_2 };
    const auto kernel_3 = GaussianDistributionKernel{ mu, sigma_3 };
    const auto kernel_4 = GaussianDistributionKernel{ mu, sigma_4 };
    const auto kernel_5 = GaussianDistributionKernel{ mu, sigma_5 };

    for (const auto value : { 0.05, 1.0, 10.0, 20.0 }) {
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

    const auto sigma = 1.25;

    const auto mu_1 = -20.2;
    const auto mu_2 = -0.05;
    const auto mu_3 = 0.0;
    const auto mu_4 = 0.95;
    const auto mu_5 = 12.01;

    const auto kernel_1 = GaussianDistributionKernel{ mu_1, sigma };
    const auto kernel_2 = GaussianDistributionKernel{ mu_2, sigma };
    const auto kernel_3 = GaussianDistributionKernel{ mu_3, sigma };
    const auto kernel_4 = GaussianDistributionKernel{ mu_4, sigma };
    const auto kernel_5 = GaussianDistributionKernel{ mu_5, sigma };

    for (const auto value : { 0.05, 1.0, 10.0, 20.0 }) {
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

    const auto sigma = 1.25;
    const auto mu = 2.01;

    const auto kernel = GaussianDistributionKernel{ mu, sigma };

    auto source_positions = std::vector<Vec3d>{};
    auto target_positions = std::vector<Vec3d>{};

    source_positions.emplace_back(0.0, 0.0, 0.0);
    source_positions.emplace_back(5.63347, 6.77419, 6.68022);
    source_positions.emplace_back(0.434595, 1.3763, 9.85119);
    source_positions.emplace_back(9.71033, 6.27692, 5.29921);
    source_positions.emplace_back(8.00937, 4.2314, 4.9292);

    target_positions.emplace_back(0.0, 0.0, 0.0);
    target_positions.emplace_back(5.63347, 6.77419, 6.68022);
    target_positions.emplace_back(0.434595, 1.3763, 9.85119);
    target_positions.emplace_back(9.71033, 6.27692, 5.29921);
    target_positions.emplace_back(8.00937, 4.2314, 4.9292);

    for (const auto& source_position : source_positions) {
        for (const auto& target_position : target_positions) {
            const auto calculated_difference = (target_position - source_position).calculate_2_norm();

            const auto probability_1 = kernel.get_probability(calculated_difference);
            const auto probability_2 = kernel.get_probability(source_position, target_position);

            ASSERT_NEAR(probability_1, probability_2, eps);
        }
    }
}
