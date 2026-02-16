/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_bounding_box.h"

#include "Config.h"
#include "RelearnTest.hpp"

#include "util/BoundingBox.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <tuple>
#include <vector>

TEST_F(BoundingBoxTest, testEmptyConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto bb = BoundingBox<double>{};

    const auto uninitialized = Vec3d{ Constants::uninitialized };

    ASSERT_EQ(bb.get_minimum(), uninitialized);
    ASSERT_EQ(bb.get_maximum(), uninitialized);
}

TEST_F(BoundingBoxTest, testConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto v_1 = SimulationFactory::get_random_position(mt);
    const auto v_2 = SimulationFactory::get_random_position(mt);

    auto v_min = v_1;
    v_min.calculate_componentwise_minimum(v_2);

    auto v_max = v_1;
    v_max.calculate_componentwise_maximum(v_2);

    const auto bb = BoundingBox<double>{ v_min, v_max };

    ASSERT_EQ(bb.get_minimum(), v_min);
    ASSERT_EQ(bb.get_maximum(), v_max);
}

TEST_F(BoundingBoxTest, testStructuredBinding) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto v_1 = SimulationFactory::get_random_position(mt);
    const auto v_2 = SimulationFactory::get_random_position(mt);

    auto v_min = v_1;
    v_min.calculate_componentwise_minimum(v_2);

    auto v_max = v_1;
    v_max.calculate_componentwise_maximum(v_2);

    auto bb = BoundingBox<double>{ v_min, v_max };

    const auto& [v_min_1, v_max_1] = bb;
    ASSERT_EQ(v_min, v_min_1);
    ASSERT_EQ(v_max, v_max_1);

    auto& [v_min_2, v_max_2] = bb;
    ASSERT_EQ(v_min, v_min_2);
    ASSERT_EQ(v_max, v_max_2);

    auto [v_min_3, v_max_3] = bb;
    ASSERT_EQ(v_min, v_min_3);
    ASSERT_EQ(v_max, v_max_3);

    v_min_2 = SimulationFactory::get_random_position(mt);
    v_max_2 = SimulationFactory::get_random_position(mt);

    ASSERT_EQ(bb.get_minimum(), v_min_2);
    ASSERT_EQ(bb.get_maximum(), v_max_2);
}

TEST_F(BoundingBoxTest, testConstructorExceptionXlarge) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x1 = SimulationFactory::get_random_position_element(mt);
    const auto y1 = SimulationFactory::get_random_position_element(mt);
    const auto z1 = SimulationFactory::get_random_position_element(mt);

    const auto x2 = SimulationFactory::get_random_position_element(mt);
    const auto y2 = SimulationFactory::get_random_position_element(mt);
    const auto z2 = SimulationFactory::get_random_position_element(mt);

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const auto v_min = Vec3<double>{ x_min, y_min, z_min };
    const auto v_max = Vec3<double>{ x_max, y_max, z_max };

    const auto x3 = RandomFactory::get_random_double<double>(x_min, x_max, mt) + x_max - x_min;
    const auto y3 = RandomFactory::get_random_double<double>(y_min, y_max, mt);
    const auto z3 = RandomFactory::get_random_double<double>(z_min, z_max, mt);

    const auto v_wrong = Vec3<double>{ x3, y3, z3 };

    ASSERT_THROW_NO_PRINT(std::ignore = BoundingBox(v_max, v_min), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BoundingBox(v_wrong, v_max), RelearnException);
}

TEST_F(BoundingBoxTest, testConstructorExceptionYlarge) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x1 = SimulationFactory::get_random_position_element(mt);
    const auto y1 = SimulationFactory::get_random_position_element(mt);
    const auto z1 = SimulationFactory::get_random_position_element(mt);

    const auto x2 = SimulationFactory::get_random_position_element(mt);
    const auto y2 = SimulationFactory::get_random_position_element(mt);
    const auto z2 = SimulationFactory::get_random_position_element(mt);

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const auto v_min = Vec3<double>{ x_min, y_min, z_min };
    const auto v_max = Vec3<double>{ x_max, y_max, z_max };

    const auto x3 = RandomFactory::get_random_double<double>(x_min, x_max, mt);
    const auto y3 = RandomFactory::get_random_double<double>(y_min, y_max, mt) + y_max - y_min;
    const auto z3 = RandomFactory::get_random_double<double>(z_min, z_max, mt);

    const auto v_wrong = Vec3<double>{ x3, y3, z3 };

    ASSERT_THROW_NO_PRINT(std::ignore = BoundingBox(v_max, v_min), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BoundingBox(v_wrong, v_max), RelearnException);
}

TEST_F(BoundingBoxTest, testConstructorExceptionZlarge) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x1 = SimulationFactory::get_random_position_element(mt);
    const auto y1 = SimulationFactory::get_random_position_element(mt);
    const auto z1 = SimulationFactory::get_random_position_element(mt);

    const auto x2 = SimulationFactory::get_random_position_element(mt);
    const auto y2 = SimulationFactory::get_random_position_element(mt);
    const auto z2 = SimulationFactory::get_random_position_element(mt);

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const auto v_min = Vec3<double>{ x_min, y_min, z_min };
    const auto v_max = Vec3<double>{ x_max, y_max, z_max };

    const auto x3 = RandomFactory::get_random_double<double>(x_min, x_max, mt);
    const auto y3 = RandomFactory::get_random_double<double>(y_min, y_max, mt);
    const auto z3 = RandomFactory::get_random_double<double>(z_min, z_max, mt) + z_max - z_min;

    const auto v_wrong = Vec3<double>{ x3, y3, z3 };

    ASSERT_THROW_NO_PRINT(std::ignore = BoundingBox(v_max, v_min), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BoundingBox(v_wrong, v_max), RelearnException);
}

TEST_F(BoundingBoxTest, testConstructorExceptionXsmall) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x1 = SimulationFactory::get_random_position_element(mt);
    const auto y1 = SimulationFactory::get_random_position_element(mt);
    const auto z1 = SimulationFactory::get_random_position_element(mt);

    const auto x2 = SimulationFactory::get_random_position_element(mt);
    const auto y2 = SimulationFactory::get_random_position_element(mt);
    const auto z2 = SimulationFactory::get_random_position_element(mt);

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const auto v_min = Vec3<double>{ x_min, y_min, z_min };
    const auto v_max = Vec3<double>{ x_max, y_max, z_max };

    const auto x3 = RandomFactory::get_random_double<double>(x_min, x_max, mt) - x_max + x_min;
    const auto y3 = RandomFactory::get_random_double<double>(y_min, y_max, mt);
    const auto z3 = RandomFactory::get_random_double<double>(z_min, z_max, mt);

    const auto v_wrong = Vec3<double>{ x3, y3, z3 };

    ASSERT_THROW_NO_PRINT(std::ignore = BoundingBox(v_max, v_min), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BoundingBox(v_min, v_wrong), RelearnException);
}

TEST_F(BoundingBoxTest, testConstructorExceptionYsmall) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x1 = SimulationFactory::get_random_position_element(mt);
    const auto y1 = SimulationFactory::get_random_position_element(mt);
    const auto z1 = SimulationFactory::get_random_position_element(mt);

    const auto x2 = SimulationFactory::get_random_position_element(mt);
    const auto y2 = SimulationFactory::get_random_position_element(mt);
    const auto z2 = SimulationFactory::get_random_position_element(mt);

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const auto v_min = Vec3<double>{ x_min, y_min, z_min };
    const auto v_max = Vec3<double>{ x_max, y_max, z_max };

    const auto x3 = RandomFactory::get_random_double<double>(x_min, x_max, mt);
    const auto y3 = RandomFactory::get_random_double<double>(y_min, y_max, mt) - y_max + y_min;
    const auto z3 = RandomFactory::get_random_double<double>(z_min, z_max, mt);

    const auto v_wrong = Vec3<double>{ x3, y3, z3 };

    ASSERT_THROW_NO_PRINT(std::ignore = BoundingBox(v_max, v_min), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BoundingBox(v_min, v_wrong), RelearnException);
}

TEST_F(BoundingBoxTest, testConstructorExceptionZsmall) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x1 = SimulationFactory::get_random_position_element(mt);
    const auto y1 = SimulationFactory::get_random_position_element(mt);
    const auto z1 = SimulationFactory::get_random_position_element(mt);

    const auto x2 = SimulationFactory::get_random_position_element(mt);
    const auto y2 = SimulationFactory::get_random_position_element(mt);
    const auto z2 = SimulationFactory::get_random_position_element(mt);

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const auto v_min = Vec3<double>{ x_min, y_min, z_min };
    const auto v_max = Vec3<double>{ x_max, y_max, z_max };

    const auto x3 = RandomFactory::get_random_double<double>(x_min, x_max, mt);
    const auto y3 = RandomFactory::get_random_double<double>(y_min, y_max, mt);
    const auto z3 = RandomFactory::get_random_double<double>(z_min, z_max, mt) - z_max + z_min;

    const auto v_wrong = Vec3<double>{ x3, y3, z3 };

    ASSERT_THROW_NO_PRINT(std::ignore = BoundingBox(v_max, v_min), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = BoundingBox(v_min, v_wrong), RelearnException);
}

TEST_F(BoundingBoxTest, testCheckInBoxTrue) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto v_1 = SimulationFactory::get_random_position(mt);
    const auto v_2 = SimulationFactory::get_random_position(mt);

    auto v_min = v_1;
    v_min.calculate_componentwise_minimum(v_2);

    auto v_max = v_1;
    v_max.calculate_componentwise_maximum(v_2);

    const auto bb = BoundingBox{ v_min, v_max };

    ASSERT_TRUE(bb.check_in_box(v_min));
    ASSERT_TRUE(bb.check_in_box(v_max));

    const auto pos = SimulationFactory::get_random_position_in_box(bb, mt);

    ASSERT_TRUE(bb.check_in_box(pos));
}

TEST_F(BoundingBoxTest, testCheckInBoxFalse) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x1 = SimulationFactory::get_random_position_element(mt);
    const auto y1 = SimulationFactory::get_random_position_element(mt);
    const auto z1 = SimulationFactory::get_random_position_element(mt);

    const auto x2 = SimulationFactory::get_random_position_element(mt);
    const auto y2 = SimulationFactory::get_random_position_element(mt);
    const auto z2 = SimulationFactory::get_random_position_element(mt);

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const auto x_ok = RandomFactory::get_random_double<double>(x_min, x_max, mt);
    const auto y_ok = RandomFactory::get_random_double<double>(y_min, y_max, mt);
    const auto z_ok = RandomFactory::get_random_double<double>(z_min, z_max, mt);

    const auto x_small = RandomFactory::get_random_double<double>(x_min, x_max, mt) - x_min + x_max;
    const auto y_small = RandomFactory::get_random_double<double>(y_min, y_max, mt) - y_min + y_max;
    const auto z_small = RandomFactory::get_random_double<double>(z_min, z_max, mt) - z_min + z_max;

    const auto x_large = RandomFactory::get_random_double<double>(x_min, x_max, mt) - x_max + x_min;
    const auto y_large = RandomFactory::get_random_double<double>(y_min, y_max, mt) - y_max + y_min;
    const auto z_large = RandomFactory::get_random_double<double>(z_min, z_max, mt) - z_max + z_min;

    const auto v_min = Vec3<double>{ x_min, y_min, z_min };
    const auto v_max = Vec3<double>{ x_max, y_max, z_max };

    const auto bb = BoundingBox{ v_min, v_max };

    ASSERT_FALSE(bb.check_in_box(Vec3d(x_ok, y_ok, z_small)));
    ASSERT_FALSE(bb.check_in_box(Vec3d(x_ok, y_ok, z_large)));
    ASSERT_FALSE(bb.check_in_box(Vec3d(x_ok, y_small, z_ok)));
    ASSERT_FALSE(bb.check_in_box(Vec3d(x_ok, y_large, z_ok)));
    ASSERT_FALSE(bb.check_in_box(Vec3d(x_small, y_ok, z_ok)));
    ASSERT_FALSE(bb.check_in_box(Vec3d(x_large, y_ok, z_ok)));
}

TEST_F(BoundingBoxTest, testGetMaximumDifference) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x1 = SimulationFactory::get_random_position_element(mt);
    const auto y1 = SimulationFactory::get_random_position_element(mt);
    const auto z1 = SimulationFactory::get_random_position_element(mt);

    const auto x2 = SimulationFactory::get_random_position_element(mt);
    const auto y2 = SimulationFactory::get_random_position_element(mt);
    const auto z2 = SimulationFactory::get_random_position_element(mt);

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const auto v_min = Vec3<double>{ x_min, y_min, z_min };
    const auto v_max = Vec3<double>{ x_max, y_max, z_max };

    const auto bb = BoundingBox{ v_min, v_max };

    const auto maximum_difference = bb.get_maximum_difference();

    const auto expected = (v_max - v_min).get_maximum();

    ASSERT_NEAR(maximum_difference, expected, eps);
}

TEST_F(BoundingBoxTest, testGetMiddlePoint) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x1 = SimulationFactory::get_random_position_element(mt);
    const auto y1 = SimulationFactory::get_random_position_element(mt);
    const auto z1 = SimulationFactory::get_random_position_element(mt);

    const auto x2 = SimulationFactory::get_random_position_element(mt);
    const auto y2 = SimulationFactory::get_random_position_element(mt);
    const auto z2 = SimulationFactory::get_random_position_element(mt);

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const auto v_min = Vec3<double>{ x_min, y_min, z_min };
    const auto v_max = Vec3<double>{ x_max, y_max, z_max };

    const auto bb = BoundingBox{ v_min, v_max };

    const auto mid_point = bb.get_midpoint();
    const auto expected_mid_point = v_min.get_midpoint(v_max);

    const auto diff = mid_point - expected_mid_point;

    ASSERT_NEAR(0.0, diff.calculate_1_norm(), eps);
}

TEST_F(BoundingBoxTest, testEqualsEpsSame) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x1 = SimulationFactory::get_random_position_element(mt);
    const auto y1 = SimulationFactory::get_random_position_element(mt);
    const auto z1 = SimulationFactory::get_random_position_element(mt);

    const auto x2 = SimulationFactory::get_random_position_element(mt);
    const auto y2 = SimulationFactory::get_random_position_element(mt);
    const auto z2 = SimulationFactory::get_random_position_element(mt);

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const auto v_min = Vec3<double>{ x_min, y_min, z_min };
    const auto v_max = Vec3<double>{ x_max, y_max, z_max };

    const auto bb = BoundingBox{ v_min, v_max };
    const auto bb_clone = BoundingBox{ v_min, v_max };

    ASSERT_TRUE(bb.equals_eps(bb));
    ASSERT_TRUE(bb.equals_eps(bb_clone));
    ASSERT_TRUE(bb_clone.equals_eps(bb));
    ASSERT_TRUE(bb_clone.equals_eps(bb_clone));
}

TEST_F(BoundingBoxTest, testEqualsEpsTrue) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x1 = SimulationFactory::get_random_position_element(mt);
    const auto y1 = SimulationFactory::get_random_position_element(mt);
    const auto z1 = SimulationFactory::get_random_position_element(mt);

    const auto x2 = SimulationFactory::get_random_position_element(mt);
    const auto y2 = SimulationFactory::get_random_position_element(mt);
    const auto z2 = SimulationFactory::get_random_position_element(mt);

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const auto x_small = x_min - (Constants::eps / 3.0);
    const auto y_small = y_min - (Constants::eps / 3.0);
    const auto z_small = z_min - (Constants::eps / 3.0);

    const auto x_large = x_max + (Constants::eps / 3.0);
    const auto y_large = y_max + (Constants::eps / 3.0);
    const auto z_large = z_max + (Constants::eps / 3.0);

    auto vec = std::vector<BoundingBox<double>>{};
    vec.emplace_back(Vec3d{ x_min, y_min, z_min }, Vec3d{ x_max, y_max, z_max });
    vec.emplace_back(Vec3d{ x_small, y_min, z_min }, Vec3d{ x_max, y_max, z_max });
    vec.emplace_back(Vec3d{ x_min, y_small, z_min }, Vec3d{ x_max, y_max, z_max });
    vec.emplace_back(Vec3d{ x_min, y_min, z_small }, Vec3d{ x_max, y_max, z_max });
    vec.emplace_back(Vec3d{ x_min, y_min, z_min }, Vec3d{ x_large, y_max, z_max });
    vec.emplace_back(Vec3d{ x_min, y_min, z_min }, Vec3d{ x_max, y_large, z_max });
    vec.emplace_back(Vec3d{ x_min, y_min, z_min }, Vec3d{ x_max, y_max, z_large });

    for (auto i = 0U; i < vec.size(); i++) {
        for (auto j = 0U; j < vec.size(); j++) {
            const auto& bb_1 = vec[i];
            const auto& bb_2 = vec[j];

            ASSERT_TRUE(bb_1.equals_eps(bb_2));
            ASSERT_TRUE(bb_2.equals_eps(bb_1));
        }
    }
}

TEST_F(BoundingBoxTest, testEqualsEpsFalse) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x1 = SimulationFactory::get_random_position_element(mt);
    const auto y1 = SimulationFactory::get_random_position_element(mt);
    const auto z1 = SimulationFactory::get_random_position_element(mt);

    const auto x2 = SimulationFactory::get_random_position_element(mt);
    const auto y2 = SimulationFactory::get_random_position_element(mt);
    const auto z2 = SimulationFactory::get_random_position_element(mt);

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const auto x_small = x_min - (Constants::eps * 3.0);
    const auto y_small = y_min - (Constants::eps * 3.0);
    const auto z_small = z_min - (Constants::eps * 3.0);

    const auto x_large = x_max + (Constants::eps * 3.0);
    const auto y_large = y_max + (Constants::eps * 3.0);
    const auto z_large = z_max + (Constants::eps * 3.0);

    auto vec = std::vector<BoundingBox<double>>{};
    vec.emplace_back(Vec3d{ x_min, y_min, z_min }, Vec3d{ x_max, y_max, z_max });
    vec.emplace_back(Vec3d{ x_small, y_min, z_min }, Vec3d{ x_max, y_max, z_max });
    vec.emplace_back(Vec3d{ x_min, y_small, z_min }, Vec3d{ x_max, y_max, z_max });
    vec.emplace_back(Vec3d{ x_min, y_min, z_small }, Vec3d{ x_max, y_max, z_max });
    vec.emplace_back(Vec3d{ x_min, y_min, z_min }, Vec3d{ x_large, y_max, z_max });
    vec.emplace_back(Vec3d{ x_min, y_min, z_min }, Vec3d{ x_max, y_large, z_max });
    vec.emplace_back(Vec3d{ x_min, y_min, z_min }, Vec3d{ x_max, y_max, z_large });

    for (auto i = 0U; i < vec.size(); i++) {
        for (auto j = 0U; j < vec.size() && i != j; j++) {
            const auto& bb_1 = vec[i];
            const auto& bb_2 = vec[j];

            ASSERT_FALSE(bb_1.equals_eps(bb_2));
            ASSERT_FALSE(bb_2.equals_eps(bb_1));
        }
    }
}

TEST_F(BoundingBoxTest, testEquality) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x1 = SimulationFactory::get_random_position_element(mt);
    const auto y1 = SimulationFactory::get_random_position_element(mt);
    const auto z1 = SimulationFactory::get_random_position_element(mt);

    const auto x2 = SimulationFactory::get_random_position_element(mt);
    const auto y2 = SimulationFactory::get_random_position_element(mt);
    const auto z2 = SimulationFactory::get_random_position_element(mt);

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const auto x_small = x_min - (Constants::eps * 3.0);
    const auto y_small = y_min - (Constants::eps * 3.0);
    const auto z_small = z_min - (Constants::eps * 3.0);

    const auto x_large = x_max + (Constants::eps * 3.0);
    const auto y_large = y_max + (Constants::eps * 3.0);
    const auto z_large = z_max + (Constants::eps * 3.0);

    auto vec = std::vector<BoundingBox<double>>{};
    vec.emplace_back(Vec3d{ x_min, y_min, z_min }, Vec3d{ x_max, y_max, z_max });
    vec.emplace_back(Vec3d{ x_small, y_min, z_min }, Vec3d{ x_max, y_max, z_max });
    vec.emplace_back(Vec3d{ x_min, y_small, z_min }, Vec3d{ x_max, y_max, z_max });
    vec.emplace_back(Vec3d{ x_min, y_min, z_small }, Vec3d{ x_max, y_max, z_max });
    vec.emplace_back(Vec3d{ x_min, y_min, z_min }, Vec3d{ x_large, y_max, z_max });
    vec.emplace_back(Vec3d{ x_min, y_min, z_min }, Vec3d{ x_max, y_large, z_max });
    vec.emplace_back(Vec3d{ x_min, y_min, z_min }, Vec3d{ x_max, y_max, z_large });

    for (auto i = 0U; i < vec.size(); i++) {
        for (auto j = 0U; j < vec.size(); j++) {
            if (i == j) {
                ASSERT_EQ(vec[i], vec[j]);
            } else {
                ASSERT_NE(vec[i], vec[j]);
            }
        }
    }
}

TEST_F(BoundingBoxTest, testOrder) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto bb_1 = BoundingBox{ Vec3d{ 0.0, 0.0, 0.0 }, Vec3d{ 1.0, 1.0, 1.0 } };
    const auto bb_2 = BoundingBox{ Vec3d{ 0.0, 0.0, 0.0 }, Vec3d{ 2.0, 2.0, 2.0 } };
    const auto bb_3 = BoundingBox{ Vec3d{ 0.0, 0.0, 0.0 }, Vec3d{ 0.5, 0.5, 0.5 } };

    const auto bb_4 = BoundingBox{ Vec3d{ 0.0, 0.0, 1.0 }, Vec3d{ 1.0, 1.0, 1.0 } };
    const auto bb_5 = BoundingBox{ Vec3d{ 0.0, 1.0, 0.0 }, Vec3d{ 2.0, 2.0, 2.0 } };
    const auto bb_6 = BoundingBox{ Vec3d{ 1.0, 0.0, 0.0 }, Vec3d{ 4.0, 0.0, 1.0 } };

    const auto bb_7 = BoundingBox{ Vec3d{ -3.0, -2.0, -2.0 }, Vec3d{ 1.0, 1.0, 1.0 } };
    const auto bb_8 = BoundingBox{ Vec3d{ -2.0, -3.0, -2.0 }, Vec3d{ 2.0, 2.0, 2.0 } };
    const auto bb_9 = BoundingBox{ Vec3d{ -2.0, -2.0, -3.0 }, Vec3d{ -1.0, -1.0, -1.0 } };

    ASSERT_FALSE(bb_1 < bb_1);
    ASSERT_TRUE(bb_1 < bb_2);
    ASSERT_FALSE(bb_1 < bb_3);
    ASSERT_TRUE(bb_1 < bb_4);
    ASSERT_TRUE(bb_1 < bb_5);
    ASSERT_TRUE(bb_1 < bb_6);
    ASSERT_FALSE(bb_1 < bb_7);
    ASSERT_FALSE(bb_1 < bb_8);
    ASSERT_FALSE(bb_1 < bb_9);

    ASSERT_TRUE(bb_1 < bb_2);
    ASSERT_FALSE(bb_2 < bb_2);
    ASSERT_FALSE(bb_2 < bb_3);
    ASSERT_TRUE(bb_2 < bb_4);
    ASSERT_TRUE(bb_2 < bb_5);
    ASSERT_TRUE(bb_2 < bb_6);
    ASSERT_FALSE(bb_2 < bb_7);
    ASSERT_FALSE(bb_2 < bb_8);
    ASSERT_FALSE(bb_2 < bb_9);

    ASSERT_TRUE(bb_3 < bb_1);
    ASSERT_TRUE(bb_3 < bb_2);
    ASSERT_FALSE(bb_3 < bb_3);
    ASSERT_TRUE(bb_3 < bb_4);
    ASSERT_TRUE(bb_3 < bb_5);
    ASSERT_TRUE(bb_3 < bb_6);
    ASSERT_FALSE(bb_3 < bb_7);
    ASSERT_FALSE(bb_3 < bb_8);
    ASSERT_FALSE(bb_3 < bb_9);

    ASSERT_FALSE(bb_4 < bb_1);
    ASSERT_FALSE(bb_4 < bb_2);
    ASSERT_FALSE(bb_4 < bb_3);
    ASSERT_FALSE(bb_4 < bb_4);
    ASSERT_TRUE(bb_4 < bb_5);
    ASSERT_TRUE(bb_4 < bb_6);
    ASSERT_FALSE(bb_4 < bb_7);
    ASSERT_FALSE(bb_4 < bb_8);
    ASSERT_FALSE(bb_4 < bb_9);

    ASSERT_FALSE(bb_5 < bb_1);
    ASSERT_FALSE(bb_5 < bb_2);
    ASSERT_FALSE(bb_5 < bb_3);
    ASSERT_FALSE(bb_5 < bb_4);
    ASSERT_FALSE(bb_5 < bb_5);
    ASSERT_TRUE(bb_5 < bb_6);
    ASSERT_FALSE(bb_5 < bb_7);
    ASSERT_FALSE(bb_5 < bb_8);
    ASSERT_FALSE(bb_5 < bb_9);

    ASSERT_FALSE(bb_6 < bb_1);
    ASSERT_FALSE(bb_6 < bb_2);
    ASSERT_FALSE(bb_6 < bb_3);
    ASSERT_FALSE(bb_6 < bb_4);
    ASSERT_FALSE(bb_6 < bb_5);
    ASSERT_FALSE(bb_6 < bb_6);
    ASSERT_FALSE(bb_6 < bb_7);
    ASSERT_FALSE(bb_6 < bb_8);
    ASSERT_FALSE(bb_6 < bb_9);

    ASSERT_TRUE(bb_7 < bb_1);
    ASSERT_TRUE(bb_7 < bb_2);
    ASSERT_TRUE(bb_7 < bb_3);
    ASSERT_TRUE(bb_7 < bb_4);
    ASSERT_TRUE(bb_7 < bb_5);
    ASSERT_TRUE(bb_7 < bb_6);
    ASSERT_FALSE(bb_7 < bb_7);
    ASSERT_TRUE(bb_7 < bb_8);
    ASSERT_TRUE(bb_7 < bb_9);

    ASSERT_TRUE(bb_8 < bb_1);
    ASSERT_TRUE(bb_8 < bb_2);
    ASSERT_TRUE(bb_8 < bb_3);
    ASSERT_TRUE(bb_8 < bb_4);
    ASSERT_TRUE(bb_8 < bb_5);
    ASSERT_TRUE(bb_8 < bb_6);
    ASSERT_FALSE(bb_8 < bb_7);
    ASSERT_FALSE(bb_8 < bb_8);
    ASSERT_TRUE(bb_8 < bb_9);

    ASSERT_TRUE(bb_9 < bb_1);
    ASSERT_TRUE(bb_9 < bb_2);
    ASSERT_TRUE(bb_9 < bb_3);
    ASSERT_TRUE(bb_9 < bb_4);
    ASSERT_TRUE(bb_9 < bb_5);
    ASSERT_TRUE(bb_9 < bb_6);
    ASSERT_FALSE(bb_9 < bb_7);
    ASSERT_FALSE(bb_9 < bb_8);
    ASSERT_FALSE(bb_9 < bb_9);
}

TEST_F(BoundingBoxTest, testPrint) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x_min = 1.5;
    const auto y_min = -0.2364;
    const auto z_min = 1026.0215;

    const auto x_max = 12.1;
    const auto y_max = -0.115;
    const auto z_max = 4523;

    const auto v_min = Vec3<double>{ x_min, y_min, z_min };
    const auto v_max = Vec3<double>{ x_max, y_max, z_max };

    const auto bb = BoundingBox{ v_min, v_max };

    auto out_stream = std::stringstream{};
    out_stream << std::setprecision(std::numeric_limits<double>::digits10);
    out_stream << bb;

    const auto* const expected_output = "[(1.5, -0.2364, 1026.0215), (12.1, -0.115, 4523)]";

    ASSERT_EQ(out_stream.str(), expected_output);
}

TEST_F(BoundingBoxTest, testPrintFmt) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto x_min = 1.5;
    const auto y_min = -0.2364;
    const auto z_min = 1026.0215;

    const auto x_max = 12.1;
    const auto y_max = -0.115;
    const auto z_max = 4523;

    const auto v_min = Vec3<double>{ x_min, y_min, z_min };
    const auto v_max = Vec3<double>{ x_max, y_max, z_max };

    const auto bb = BoundingBox{ v_min, v_max };

    const auto output = fmt::format("{}", bb);
    const auto* const expected_output = "((1.5, -0.2364, 1026.0215), (12.1, -0.115, 4523))";

    ASSERT_EQ(output, expected_output);
}
