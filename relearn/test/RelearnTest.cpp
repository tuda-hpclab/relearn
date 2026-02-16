/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnTest.hpp"

#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/BarnesHutInternal/BarnesHutInvertedCell.h"
#include "algorithm/FMMInternal/FastMultipoleMethodCell.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "io/LogFiles.h"
#include "neuron_monitor/test_neuron_monitor.h"
#include "util/MemoryHolder.h"

#include "mpi-wrapper/MPIWrapper.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <vector>

std::size_t RelearnTest::iterations = 10;
double RelearnTest::eps = 0.001;

bool RelearnTest::use_predetermined_seed = false;
std::mt19937::result_type RelearnTest::predetermined_seed = 2328571864;

void RelearnTest::SetUp() {
    if (use_predetermined_seed) {
        std::cerr << "Using predetermined seed: " << predetermined_seed << '\n';
        mt.seed(predetermined_seed);
    } else {
        const auto now = std::chrono::high_resolution_clock::now();
        const auto time_since_epoch = now.time_since_epoch();
        const auto time_since_epoch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch).count();

        const auto seed = static_cast<unsigned int>(time_since_epoch_ns);

        std::cerr << "Test seed: " << seed << '\n';
        mt.seed(seed);
    }
}

void RelearnTest::TearDown() {

}

RelearnMemoryTest::RelearnMemoryTest() {
}

int main(int argc, char** argv) {
    mpiPP::MPIWrapper::init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);

    LogFiles::disable = true;

    const auto tests_return_code = RUN_ALL_TESTS();

    LogFiles::disable = false;

    mpiPP::MPIWrapper::finalize();

    return tests_return_code;
}
