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

#include "algorithm/Kernel/Gamma.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include <gtest/gtest.h>

#include <vector>

#include "test_kernel.h"

TEST_F(GammaKernelTest, testDefaultConstruction) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = GammaDistributionKernel{};

    ASSERT_EQ(kernel.get_k(), GammaDistributionKernel::default_k);
    ASSERT_EQ(kernel.get_theta(), GammaDistributionKernel::default_theta);
}

TEST_F(GammaKernelTest, testConstruction) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_NO_THROW(std::ignore = GammaDistributionKernel(0.1, 1.0));
    ASSERT_NO_THROW(std::ignore = GammaDistributionKernel(1.0, 0.1));
    ASSERT_NO_THROW(std::ignore = GammaDistributionKernel(1.1, 1.0));

    ASSERT_NO_THROW(std::ignore = GammaDistributionKernel(0.1, 750.0));
    ASSERT_NO_THROW(std::ignore = GammaDistributionKernel(1.0, 750.0));
    ASSERT_NO_THROW(std::ignore = GammaDistributionKernel(1.1, 750.0));
}

TEST_F(GammaKernelTest, testConstructionExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(0.1, 0.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(0.0, 0.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(1.1, 0.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(-5.0, 0.0), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(0.1, -2.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(0.0, -2.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(1.1, -2.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(-5.0, -2.0), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(0.0, 0.1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(0.0, 0.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(0.0, 1.1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(0.0, 5.0), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(-2.0, 0.1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(-2.0, -0.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(-2.0, 1.1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = GammaDistributionKernel(-2.0, -5.0), RelearnException);
}

TEST_F(GammaKernelTest, testGetter) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto kernel = GammaDistributionKernel{ 8541.58, 156478.587 };
    ASSERT_EQ(kernel.get_k(), 8541.58);
    ASSERT_EQ(kernel.get_theta(), 156478.587);
}

TEST_F(GammaKernelTest, testPositions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = GammaDistributionKernel{ 7.5, 1.1 };

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

TEST_F(GammaKernelTest, testDefaultProbabilities) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = GammaDistributionKernel{};

    const auto probability_1 = kernel.get_probability(0.0);
    const auto probability_2 = kernel.get_probability(0.5);
    const auto probability_3 = kernel.get_probability(1.2);
    const auto probability_4 = kernel.get_probability(5.2);
    const auto probability_5 = kernel.get_probability(6.6);
    const auto probability_6 = kernel.get_probability(10.1);
    const auto probability_7 = kernel.get_probability(150.5);

    ASSERT_NEAR(probability_1, 1.0, eps);
    ASSERT_GT(probability_1, probability_2);
    ASSERT_GT(probability_2, probability_3);
    ASSERT_GT(probability_3, probability_4);
    ASSERT_GT(probability_4, probability_5);
    ASSERT_GT(probability_5, probability_6);
    ASSERT_GT(probability_6, probability_7);
    ASSERT_GT(probability_7, 0.0);
}

TEST_F(GammaKernelTest, testProbabilities) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = GammaDistributionKernel{ 2.0, 2.0 };

    const auto probability_1 = kernel.get_probability(0.0);
    const auto probability_2 = kernel.get_probability(0.5);
    const auto probability_3 = kernel.get_probability(1.2);
    const auto probability_4 = kernel.get_probability(5.2);
    const auto probability_5 = kernel.get_probability(6.6);
    const auto probability_6 = kernel.get_probability(10.1);
    const auto probability_7 = kernel.get_probability(150.5);

    ASSERT_NEAR(probability_1, 0.0, eps);
    ASSERT_NEAR(probability_2, 0.09735, eps);
    ASSERT_NEAR(probability_3, 0.16464, eps);
    ASSERT_NEAR(probability_4, 0.09655, eps);
    ASSERT_NEAR(probability_5, 0.06085, eps);
    ASSERT_NEAR(probability_6, 0.01618, eps);
    ASSERT_NEAR(probability_7, 0.0, eps);
}
