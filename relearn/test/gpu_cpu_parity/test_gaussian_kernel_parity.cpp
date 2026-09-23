/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#ifdef RELEARN_CUDA_ENABLED

#include "test_gaussian_kernel_parity.h"

#include "RelearnTest.hpp"

#include "algorithm/Kernel/Gaussian.h"
#include "util/Vec3.h"

#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <cmath>

class GaussianKernelParityTest : public RelearnTest { };

namespace {

// GPU code works in double precision throughout; the CPU reference (GaussianDistributionKernel)
// works in RelearnTypes::acceptance_criterion_type (float) throughout. A tolerance purely accounting for that
// precision gap should still easily catch a wrong-formula bug, which produces differences many
// orders of magnitude larger than float rounding error for any non-trivial distance.
constexpr double relative_tolerance = 1e-3;

void assert_matches_cpu_reference(double source_x, double source_y, double source_z,
                                  double target_x, double target_y, double target_z,
                                  double number_free_elements, double sigma) {
    const auto squared_sigma_inv = 1.0 / (sigma * sigma);

    const auto gpu_result = device_calculate_attractiveness_to_connect(
        source_x, source_y, source_z, target_x, target_y, target_z, number_free_elements, squared_sigma_inv);

    const auto source_position = RelearnTypes::position_type{ static_cast<RelearnTypes::space_type>(source_x), static_cast<RelearnTypes::space_type>(source_y), static_cast<RelearnTypes::space_type>(source_z) };
    const auto target_position = RelearnTypes::position_type{ static_cast<RelearnTypes::space_type>(target_x), static_cast<RelearnTypes::space_type>(target_y), static_cast<RelearnTypes::space_type>(target_z) };

    const auto kernel = GaussianDistributionKernel(0.0, static_cast<RelearnTypes::acceptance_criterion_type>(sigma));
    const auto cpu_result = kernel.get_probability(source_position, target_position, static_cast<RelearnTypes::counter_type>(number_free_elements));

    ASSERT_NEAR(gpu_result, cpu_result, std::abs(cpu_result) * relative_tolerance + 1e-9)
        << "GPU calculate_attractiveness_to_connect diverged from the CPU GaussianDistributionKernel reference "
        << "(gpu=" << gpu_result << ", cpu=" << cpu_result << ", distance="
        << std::sqrt((target_x - source_x) * (target_x - source_x) + (target_y - source_y) * (target_y - source_y) + (target_z - source_z) * (target_z - source_z))
        << ", sigma=" << sigma << ")";
}

} // namespace

// At zero distance (source == target) the GPU function takes a deliberate autapse-prevention
// shortcut baked into calculate_attractiveness_to_connect itself (see its "prevent autapse"
// guard) and returns 0, unlike the bare CPU GaussianDistributionKernel::get_probability formula,
// which would evaluate to exp(0) = 1 at zero distance -- on the CPU side, autapse exclusion
// instead happens structurally, by never presenting a neuron as its own candidate target during
// the octree traversal. This is intentional divergence at this layer, not a parity bug.
TEST_F(GaussianKernelParityTest, testAutapseAtZeroDistanceReturnsZeroOnGpu) {
    const auto gpu_result = device_calculate_attractiveness_to_connect(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 / (750.0 * 750.0));
    ASSERT_EQ(gpu_result, 0.0);
}

TEST_F(GaussianKernelParityTest, testMatchesCpuReferenceForAxisAlignedDistances) {
    // Sweep a range of distances spanning well below, around, and well above sigma, where a
    // dist^2-vs-dist^4 formula mismatch would diverge most visibly.
    for (const double distance : { 1.0, 10.0, 100.0, 500.0, 750.0, 1000.0, 2000.0, 5000.0 }) {
        assert_matches_cpu_reference(0.0, 0.0, 0.0, distance, 0.0, 0.0, 3.0, 750.0);
    }
}

TEST_F(GaussianKernelParityTest, testMatchesCpuReferenceForRandomPositionsAndSigmas) {
    for (auto trial = 0U; trial < 200U; ++trial) {
        const auto source = SimulationFactory::get_random_position(mt);
        const auto target = SimulationFactory::get_random_position(mt);
        const auto number_free_elements = RandomFactory::get_random_integer<std::uint32_t>(0, 20, mt);
        const auto sigma = RandomFactory::get_random_double<double>(1.0, 2000.0, mt);

        assert_matches_cpu_reference(
            static_cast<double>(source.get_x()), static_cast<double>(source.get_y()), static_cast<double>(source.get_z()),
            static_cast<double>(target.get_x()), static_cast<double>(target.get_y()), static_cast<double>(target.get_z()),
            static_cast<double>(number_free_elements), sigma);
    }
}

TEST_F(GaussianKernelParityTest, testZeroFreeElementsIsZeroOnBothSides) {
    assert_matches_cpu_reference(0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 750.0);
}

#endif
