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

#include "RelearnTest.hpp"

#include "cuda/CudaBaseBridgeFunctions.h"
#include "cuda/CudaConfig.h"
#include "cuda/calcium/Calcium.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"

#include <cmath>
#include <vector>

class CalciumGPUTest : public RelearnTest { };

namespace {

// Reference Euler integration replicating update_current_calcium_kernel's documented recurrence
// exactly (see source/cuda/calcium/Calcium.cu), used to check the GPU kernel does what its own
// formula says rather than just "doesn't crash".
CudaConfig::calcium_type expected_after_h_steps(CudaConfig::calcium_type c0, unsigned int h,
                                                CudaConfig::calcium_type tau_C, CudaConfig::calcium_type beta, bool fired) {
    const auto scale = CudaConfig::calcium_type{ 1 } / static_cast<CudaConfig::calcium_type>(h);
    const auto tau_C_inverse = -CudaConfig::calcium_type{ 1 } / tau_C;
    auto c = c0;
    for (auto i = 0U; i < h; ++i) {
        if (fired) {
            c += scale * (c * tau_C_inverse + beta);
        } else {
            c += scale * (c * tau_C_inverse);
        }
    }
    return c;
}

struct DeviceCalciumBuffers {
    UpdateStatus* disable_flags{};
    FiredStatus* fired{};
    CudaConfig::calcium_type* calcium{};
    CudaConfig::number_neurons_type num_neurons{};

    DeviceCalciumBuffers(const std::vector<UpdateStatus>& h_disable_flags, const std::vector<FiredStatus>& h_fired,
                         const std::vector<CudaConfig::calcium_type>& h_calcium)
        : num_neurons(static_cast<CudaConfig::number_neurons_type>(h_calcium.size())) {
        cudaMalloc_bridge(reinterpret_cast<void**>(&disable_flags), sizeof(UpdateStatus) * num_neurons);
        cudaMalloc_bridge(reinterpret_cast<void**>(&fired), sizeof(FiredStatus) * num_neurons);
        cudaMalloc_bridge(reinterpret_cast<void**>(&calcium), sizeof(CudaConfig::calcium_type) * num_neurons);

        cudaMemcpy_to_device_bridge(disable_flags, h_disable_flags.data(), sizeof(UpdateStatus) * num_neurons);
        cudaMemcpy_to_device_bridge(fired, h_fired.data(), sizeof(FiredStatus) * num_neurons);
        cudaMemcpy_to_device_bridge(calcium, h_calcium.data(), sizeof(CudaConfig::calcium_type) * num_neurons);
    }

    ~DeviceCalciumBuffers() {
        cudaFree_bridge(disable_flags);
        cudaFree_bridge(fired);
        cudaFree_bridge(calcium);
    }

    DeviceCalciumBuffers(const DeviceCalciumBuffers&) = delete;
    DeviceCalciumBuffers& operator=(const DeviceCalciumBuffers&) = delete;

    [[nodiscard]] std::vector<CudaConfig::calcium_type> download_calcium() const {
        auto result = std::vector<CudaConfig::calcium_type>(num_neurons);
        cudaMemcpy_to_host_bridge(result.data(), calcium, sizeof(CudaConfig::calcium_type) * num_neurons);
        return result;
    }
};

constexpr auto calcium_tolerance(CudaConfig::calcium_type expected) {
    return std::abs(expected) * 1e-4 + 1e-6;
}

} // namespace

TEST_F(CalciumGPUTest, testUpdateCurrentCalciumMatchesEulerFormulaWhenInactive) {
    constexpr unsigned int h = 10;
    constexpr CudaConfig::calcium_type tau_C = 20.0;
    constexpr CudaConfig::calcium_type beta = 0.5;

    const auto h_calcium = std::vector<CudaConfig::calcium_type>{ 1.0, 2.0, 0.5, 0.0 };
    const auto h_disable_flags = std::vector<UpdateStatus>(h_calcium.size(), UpdateStatus::Enabled);
    const auto h_fired = std::vector<FiredStatus>(h_calcium.size(), FiredStatus::Inactive);

    auto buffers = DeviceCalciumBuffers(h_disable_flags, h_fired, h_calcium);
    update_current_calcium_entry(NeuronsExtraInfoGPUHandleConst{ buffers.num_neurons, 0, 1, buffers.disable_flags }, buffers.fired, CalciumHandle{ .calcium = buffers.calcium }, h, tau_C, beta);
    const auto result = buffers.download_calcium();

    for (auto i = 0UL; i < h_calcium.size(); ++i) {
        const auto expected = expected_after_h_steps(h_calcium[i], h, tau_C, beta, false);
        ASSERT_NEAR(result[i], expected, calcium_tolerance(expected));
    }
}

TEST_F(CalciumGPUTest, testUpdateCurrentCalciumMatchesEulerFormulaWhenFired) {
    constexpr unsigned int h = 10;
    constexpr CudaConfig::calcium_type tau_C = 20.0;
    constexpr CudaConfig::calcium_type beta = 0.5;

    const auto h_calcium = std::vector<CudaConfig::calcium_type>{ 1.0, 2.0, 0.5, 0.0 };
    const auto h_disable_flags = std::vector<UpdateStatus>(h_calcium.size(), UpdateStatus::Enabled);
    const auto h_fired = std::vector<FiredStatus>(h_calcium.size(), FiredStatus::Fired);

    auto buffers = DeviceCalciumBuffers(h_disable_flags, h_fired, h_calcium);
    update_current_calcium_entry(NeuronsExtraInfoGPUHandleConst{ buffers.num_neurons, 0, 1, buffers.disable_flags }, buffers.fired, CalciumHandle{ .calcium = buffers.calcium }, h, tau_C, beta);
    const auto result = buffers.download_calcium();

    for (auto i = 0UL; i < h_calcium.size(); ++i) {
        const auto expected = expected_after_h_steps(h_calcium[i], h, tau_C, beta, true);
        ASSERT_NEAR(result[i], expected, calcium_tolerance(expected));
    }
}

TEST_F(CalciumGPUTest, testUpdateCurrentCalciumSkipsDisabledNeurons) {
    constexpr unsigned int h = 10;
    constexpr CudaConfig::calcium_type tau_C = 20.0;
    constexpr CudaConfig::calcium_type beta = 0.5;

    const auto h_calcium = std::vector<CudaConfig::calcium_type>{ 1.0, 2.0 };
    const auto h_disable_flags = std::vector<UpdateStatus>{ UpdateStatus::Disabled, UpdateStatus::Enabled };
    const auto h_fired = std::vector<FiredStatus>{ FiredStatus::Fired, FiredStatus::Fired };

    auto buffers = DeviceCalciumBuffers(h_disable_flags, h_fired, h_calcium);
    update_current_calcium_entry(NeuronsExtraInfoGPUHandleConst{ buffers.num_neurons, 0, 1, buffers.disable_flags }, buffers.fired, CalciumHandle{ .calcium = buffers.calcium }, h, tau_C, beta);
    const auto result = buffers.download_calcium();

    // Disabled neuron 0 must be left untouched.
    ASSERT_EQ(result[0], h_calcium[0]);
    // Enabled neuron 1 must still update normally.
    const auto expected = expected_after_h_steps(h_calcium[1], h, tau_C, beta, true);
    ASSERT_NEAR(result[1], expected, calcium_tolerance(expected));
}

TEST_F(CalciumGPUTest, testAbsoluteDecaySubtractsFixedAmountForEnabledNeuronsOnly) {
    const auto h_target_calcium = std::vector<CudaConfig::calcium_type>{ 5.0, 5.0 };
    const auto h_disable_flags = std::vector<UpdateStatus>{ UpdateStatus::Disabled, UpdateStatus::Enabled };
    const auto h_fired = std::vector<FiredStatus>(2, FiredStatus::Inactive);
    constexpr CudaConfig::calcium_type decay_amount = 0.75;

    auto buffers = DeviceCalciumBuffers(h_disable_flags, h_fired, h_target_calcium);
    update_target_calcium_absolute_decay_entry(NeuronsExtraInfoGPUHandleConst{ buffers.num_neurons, 0, 1, buffers.disable_flags }, CalciumHandle{ .target_calcium = buffers.calcium }, decay_amount);
    const auto result = buffers.download_calcium();

    ASSERT_EQ(result[0], h_target_calcium[0]);
    ASSERT_NEAR(result[1], h_target_calcium[1] - decay_amount, calcium_tolerance(h_target_calcium[1] - decay_amount));
}

TEST_F(CalciumGPUTest, testRelativeDecayMultipliesByFactorForEnabledNeuronsOnly) {
    const auto h_target_calcium = std::vector<CudaConfig::calcium_type>{ 8.0, 8.0 };
    const auto h_disable_flags = std::vector<UpdateStatus>{ UpdateStatus::Disabled, UpdateStatus::Enabled };
    const auto h_fired = std::vector<FiredStatus>(2, FiredStatus::Inactive);
    constexpr CudaConfig::calcium_type decay_factor = 0.9;

    auto buffers = DeviceCalciumBuffers(h_disable_flags, h_fired, h_target_calcium);
    update_target_calcium_relative_decay_entry(NeuronsExtraInfoGPUHandleConst{ buffers.num_neurons, 0, 1, buffers.disable_flags }, CalciumHandle{ .target_calcium = buffers.calcium }, decay_factor);
    const auto result = buffers.download_calcium();

    ASSERT_EQ(result[0], h_target_calcium[0]);
    ASSERT_NEAR(result[1], h_target_calcium[1] * decay_factor, calcium_tolerance(h_target_calcium[1] * decay_factor));
}

#endif
