/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"
#include "test_barnes_hut.h"

#include "algorithm/BarnesHutInternal/BarnesHutInverted.h"
#include "algorithm/BarnesHutInternal/BarnesHutInvertedCell.h"
#include "algorithm/Internal/octree/Octree.h"
#include "structure/Morton.h"

#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <iostream>
#include <memory>

TEST_F(BarnesHutInvertedTest, testBarnesHutInvertedGetterSetter) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto level = RelearnTypes::level_type{ 0 };
    const auto morton = std::make_shared<Morton>(level);

    ASSERT_NO_THROW(BarnesHutInverted algorithm(RelearnTypes::bounding_box_type{ min, max }, morton););

    auto algorithm = BarnesHutInverted(RelearnTypes::bounding_box_type{ min, max }, morton);
    ASSERT_EQ(algorithm.get_acceptance_criterion(), Constants::bh_default_theta);

    const auto random_acceptance_criterion = RandomFactory::get_random_double(RelearnTypes::acceptance_criterion_type{ 0 }, Constants::bh_max_theta, mt);
    auto algorithm_2 = BarnesHutInverted(RelearnTypes::bounding_box_type{ min, max }, morton, random_acceptance_criterion);
    ASSERT_EQ(algorithm_2.get_acceptance_criterion(), random_acceptance_criterion);
}
