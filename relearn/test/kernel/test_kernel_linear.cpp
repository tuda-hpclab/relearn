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

#include "algorithm/Kernel/Linear.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include <gtest/gtest.h>

#include <vector>

#include "test_kernel.h"

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

    ASSERT_NO_THROW(std::ignore = LinearDistributionKernel(0.2));
    ASSERT_NO_THROW(std::ignore = LinearDistributionKernel(1.1));
    ASSERT_NO_THROW(std::ignore = LinearDistributionKernel(841248.45));
}

TEST_F(LinearKernelTest, testConstructionExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_THROW_NO_PRINT(std::ignore = LinearDistributionKernel(-0.001), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = LinearDistributionKernel(-1.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = LinearDistributionKernel(-1.1), RelearnException);
}

TEST_F(LinearKernelTest, testGetter) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto kernel = LinearDistributionKernel{ 841248.45 };
    ASSERT_EQ(kernel.get_cutoff(), 841248.45);
}

TEST_F(LinearKernelTest, testDefaultProbability) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto kernel = LinearDistributionKernel{};

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

    const auto kernel = LinearDistributionKernel{ 120.4 };

    const auto probability_1 = kernel.get_probability(120.4);
    const auto probability_2 = kernel.get_probability(120.5);
    const auto probability_3 = kernel.get_probability(0.0);
    const auto probability_4 = kernel.get_probability(60.2);
    const auto probability_5 = kernel.get_probability(30.1);
    const auto probability_6 = kernel.get_probability(90.3);
    const auto probability_7 = kernel.get_probability(150.5);

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

    const auto kernel = LinearDistributionKernel{ 120.4 };

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
