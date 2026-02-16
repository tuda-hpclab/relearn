/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_space_filling_curve.h"

#include "Config.h"

#include "structure/Morton.h"
#include "util/RelearnException.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

TEST_F(SpaceFillingCurveTest, testMortonConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    for (auto refinement_level = std::uint8_t{ 0 }; refinement_level < Constants::max_lvl_subdomains; refinement_level++) {
        const auto morton = Morton{ refinement_level };
        const auto res = morton.get_current_refinement_level();

        ASSERT_EQ(refinement_level, res);
    }
}

TEST_F(SpaceFillingCurveTest, testMortonTranslationBijection) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    for (auto refinement_level = std::uint8_t{ 0 }; refinement_level < std::uint8_t{ 6 }; refinement_level++) {
        const auto morton = Morton{ refinement_level };

        const auto num_boxes_per_dimension = std::size_t{ 1 } << refinement_level;
        const auto total_num_boxes = num_boxes_per_dimension * num_boxes_per_dimension * num_boxes_per_dimension;

        for (auto x = std::size_t{ 0 }; x < num_boxes_per_dimension; x++) {
            for (auto y = std::size_t{ 0 }; y < num_boxes_per_dimension; y++) {
                for (auto z = std::size_t{ 0 }; z < num_boxes_per_dimension; z++) {
                    const auto index3d = Vec3s{ x, y, z };

                    const auto index1d = morton.map_3d_to_1d(index3d);
                    const auto res = morton.map_1d_to_3d(index1d);

                    ASSERT_EQ(index3d, res);
                    ASSERT_LT(index1d, total_num_boxes);
                }
            }
        }
    }
}

TEST_F(SpaceFillingCurveTest, testMortonTranslationStochastic) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto refinement_level = SimulationFactory::get_large_refinement_level(mt);
    const auto morton = Morton{ refinement_level };

    const auto num_boxes_per_dimension = std::size_t{ 1 } << refinement_level;
    const auto total_num_boxes = num_boxes_per_dimension * num_boxes_per_dimension * num_boxes_per_dimension;

    for (auto rep = 0; rep < 1000; rep++) {
        const auto x = RandomFactory::get_random_integer<std::size_t>(0, num_boxes_per_dimension - 1, mt);
        const auto y = RandomFactory::get_random_integer<std::size_t>(0, num_boxes_per_dimension - 1, mt);
        const auto z = RandomFactory::get_random_integer<std::size_t>(0, num_boxes_per_dimension - 1, mt);

        const auto index3d = Morton::coordinates_3d{ x, y, z };

        const auto index1d = morton.map_3d_to_1d(index3d);
        const auto res = morton.map_1d_to_3d(index1d);

        ASSERT_EQ(index3d, res);
        ASSERT_LT(index1d, total_num_boxes);
    }
}
