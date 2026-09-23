/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnTest.hpp"

#include "algorithm/Internal/octree/OctreeNode.h"
#include "cuda/CudaBaseBridgeFunctions.h"
#include "io/LogFiles.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/MPIWrapper.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>

#ifdef RELEARN_CUDA_ENABLED
#include "cuda/input/SynapticEquallyWeightedActivityInput.h"
#include "cuda/random/RandomNumberHost.h"
#endif

std::size_t RelearnTest::iterations = 10;
double RelearnTest::eps = 0.001;

bool RelearnTest::use_predetermined_seed = false;
std::mt19937::result_type RelearnTest::predetermined_seed = 3621434206;

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

#ifdef RELEARN_CUDA_ENABLED
    cudaResetLastError_bridge();
    cudaDeviceSynchronize_bridge();
#endif
}

void RelearnTest::TearDown() {
#ifdef RELEARN_CUDA_ENABLED
    // Tests that construct SynapticEquallyWeightedActivityInput directly (rather than through
    // CombinedActivityInput, which already calls this) would otherwise leave the lazily-created
    // static cuco::static_set alive until static destruction at process exit -- by then the CUDA
    // driver may already be shutting down, turning its device-memory free into a thrown
    // cuco::cuda_error that aborts the process (destructors are implicitly noexcept).
    release_synaptic_activity_set();

    // RandomNumbers::register_random_numbers() hands out slots from a fixed-size pool
    // (max_number_random_keys) for the entire process lifetime and never reclaims them on its
    // own -- across enough tests (e.g. under --gtest_repeat) that pool is exhausted partway
    // through. Reset it after every test so each one starts with a clean slate.
    RandomNumbers::reset();
#endif
}

RelearnMemoryTest::RelearnMemoryTest() = default;

int main(int argc, char** argv) {
    mpiPP::MPIWrapper::init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);

    LogFiles::disable = true;

    const auto tests_return_code = RUN_ALL_TESTS();

    LogFiles::disable = false;

    mpiPP::MPIWrapper::finalize();

    return tests_return_code;
}
