/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "main.h"

#include "factory/octree/octree_factory.h"

#include <benchmark/benchmark.h>

#ifdef RELEARN_CUDA_ENABLED
#include "cuda/input/SynapticEquallyWeightedActivityInput.h"
#include "cuda/random/RandomNumberHost.h"
#endif

#include <mpi-wrapper/MPIWrapper.h>

int main(int argc, char** argv) {
    mpiPP::MPIWrapper::init(argc, argv);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

#ifdef RELEARN_CUDA_ENABLED
    // Any benchmark that exercised SpikeMode::Set lazily allocated a static cuco::static_set that
    // would otherwise survive until static destruction, by which point the CUDA driver may already
    // be shutting down -- turning its device-memory free into a thrown cuco::cuda_error that
    // aborts the process (destructors are implicitly noexcept). Free it here instead, matching the
    // same fix applied to RelearnTest::TearDown().
    release_synaptic_activity_set();

    // Frees all cuRAND state registered via RandomNumbers::register_random_numbers() during any
    // benchmark (e.g. the synapse-deletion benchmarks) -- same shutdown-ordering rationale as
    // release_synaptic_activity_set() above.
    RandomNumbers::reset();
#endif
}
