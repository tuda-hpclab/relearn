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
#include "cuda/neuron_model/NeuronModels.h"
#include "cuda/random/RandomNumberHost.h"
#include "cuda/random/RandomNumberKeys.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/models/aeif/Parameters.h"
#include "neurons/models/fitzhughnagumo/Parameters.h"
#include "neurons/models/izhikevich/Parameters.h"
#include "neurons/models/poisson/Parameters.h"

#include <cmath>
#include <vector>

class NeuronModelsGPUTest : public RelearnTest { };

namespace {

using membrane_potential_type = RelearnTypes::activity_type;

constexpr auto value_tolerance(membrane_potential_type expected) {
    return std::abs(expected) * 1e-4 + 1e-6;
}

FiredRecorderHandle no_recorder(FiredStatus* d_fired) {
    return FiredRecorderHandle{ d_fired, nullptr, 0 };
}

// Minimal RAII wrapper for a device array of T, round-tripping host data through cudaMalloc_bridge
// / cudaMemcpy_*_bridge (the plain-C++-callable wrappers already used by production .cpp files
// like RandomNumbersHost.cpp, so no CUDA-specific .cu launcher file is needed for these tests --
// the neuron model entry points themselves are ordinary host functions).
template <typename T>
struct DeviceArrayFixture {
    T* ptr{};
    std::size_t count{};

    explicit DeviceArrayFixture(const std::vector<T>& host_data)
        : count(host_data.size()) {
        cudaMalloc_bridge(reinterpret_cast<void**>(&ptr), sizeof(T) * count);
        cudaMemcpy_to_device_bridge(ptr, host_data.data(), sizeof(T) * count);
    }

    ~DeviceArrayFixture() {
        cudaFree_bridge(ptr);
    }

    DeviceArrayFixture(const DeviceArrayFixture&) = delete;
    DeviceArrayFixture& operator=(const DeviceArrayFixture&) = delete;

    [[nodiscard]] std::vector<T> download() const {
        auto result = std::vector<T>(count);
        cudaMemcpy_to_host_bridge(result.data(), ptr, sizeof(T) * count);
        return result;
    }
};

} // namespace

// ────────────────────────────────────────────────────────────────────────────
// Izhikevich -- CPU reference formula (IzhikevichModel.cpp's non-CUDA update_activity) is
// mathematically identical to the GPU kernel, so this is a true parity check.
// ────────────────────────────────────────────────────────────────────────────

namespace {

membrane_potential_type izhikevich_reference_x(membrane_potential_type x0, membrane_potential_type u0, unsigned h,
                                               const models::izhikevich::Parameters<membrane_potential_type>& p, membrane_potential_type input, bool* fired_out) {
    const auto scale = membrane_potential_type{ 1 } / static_cast<membrane_potential_type>(h);
    auto x = x0;
    auto u = u0;
    *fired_out = false;
    for (auto step = 0U; step < h; ++step) {
        const auto x_increase = (p.get_k1() * x * x) + (p.get_k2() * x) + p.get_k3() - u + input;
        const auto u_increase = p.get_a() * (p.get_b() * x - u);
        x += x_increase * scale;
        u += u_increase * scale;
        if (x >= p.get_V_spike()) {
            x = p.get_c();
            u += p.get_d();
            *fired_out = true;
            break;
        }
    }
    return x;
}

} // namespace

TEST_F(NeuronModelsGPUTest, testIzhikevichMatchesReferenceFormulaWithoutSpiking) {
    const auto parameters = models::izhikevich::Parameters<membrane_potential_type>(0.02, 0.2, -65.0, 8.0, 30.0, 0.04, 5.0, 140.0);
    constexpr unsigned h = 5;

    const auto h_x = std::vector<membrane_potential_type>{ -65.0, -60.0 };
    const auto h_u = std::vector<membrane_potential_type>{ -13.0, -12.0 };
    const auto h_input = std::vector<CudaConfig::input_type>{ 0.0, 1.0 };
    const auto h_disable = std::vector<UpdateStatus>(h_x.size(), UpdateStatus::Enabled);
    auto h_fired = std::vector<FiredStatus>(h_x.size(), FiredStatus::Inactive);

    auto d_x = DeviceArrayFixture<membrane_potential_type>(h_x);
    auto d_u = DeviceArrayFixture<membrane_potential_type>(h_u);
    auto d_input = DeviceArrayFixture<CudaConfig::input_type>(h_input);
    auto d_disable = DeviceArrayFixture<UpdateStatus>(h_disable);
    auto d_fired = DeviceArrayFixture<FiredStatus>(h_fired);

    const auto extra_info = NeuronExtraInfoHandle{ static_cast<CudaConfig::number_neurons_type>(h_x.size()), d_disable.ptr };
    update_activity_izhikevich_entry(extra_info, h, parameters, d_input.ptr,
                                     IzhikevichDeviceState{ d_x.ptr, d_u.ptr }, no_recorder(d_fired.ptr));

    const auto result_x = d_x.download();
    const auto result_fired = d_fired.download();

    for (auto i = 0UL; i < h_x.size(); ++i) {
        auto expected_fired = false;
        const auto expected_x = izhikevich_reference_x(h_x[i], h_u[i], h, parameters, h_input[i], &expected_fired);
        ASSERT_NEAR(result_x[i], expected_x, value_tolerance(expected_x));
        ASSERT_EQ(result_fired[i] == FiredStatus::Fired, expected_fired);
    }
}

TEST_F(NeuronModelsGPUTest, testIzhikevichSpikesAndResetsWhenDriveIsStrong) {
    const auto parameters = models::izhikevich::Parameters<membrane_potential_type>(0.02, 0.2, -65.0, 8.0, 30.0, 0.04, 5.0, 140.0);
    constexpr unsigned h = 20;

    const auto h_x = std::vector<membrane_potential_type>{ -65.0 };
    const auto h_u = std::vector<membrane_potential_type>{ -13.0 };
    // A large, sustained input current should reliably push the neuron past V_spike.
    const auto h_input = std::vector<CudaConfig::input_type>{ 50.0 };
    const auto h_disable = std::vector<UpdateStatus>{ UpdateStatus::Enabled };
    auto h_fired = std::vector<FiredStatus>{ FiredStatus::Inactive };

    auto d_x = DeviceArrayFixture<membrane_potential_type>(h_x);
    auto d_u = DeviceArrayFixture<membrane_potential_type>(h_u);
    auto d_input = DeviceArrayFixture<CudaConfig::input_type>(h_input);
    auto d_disable = DeviceArrayFixture<UpdateStatus>(h_disable);
    auto d_fired = DeviceArrayFixture<FiredStatus>(h_fired);

    const auto extra_info = NeuronExtraInfoHandle{ 1, d_disable.ptr };
    update_activity_izhikevich_entry(extra_info, h, parameters, d_input.ptr,
                                     IzhikevichDeviceState{ d_x.ptr, d_u.ptr }, no_recorder(d_fired.ptr));

    const auto result_x = d_x.download();
    const auto result_fired = d_fired.download();

    ASSERT_EQ(result_fired[0], FiredStatus::Fired);
    ASSERT_EQ(result_x[0], parameters.get_c());
}

TEST_F(NeuronModelsGPUTest, testIzhikevichSkipsDisabledNeurons) {
    const auto parameters = models::izhikevich::Parameters<membrane_potential_type>(0.02, 0.2, -65.0, 8.0, 30.0, 0.04, 5.0, 140.0);
    constexpr unsigned h = 10;

    const auto h_x = std::vector<membrane_potential_type>{ -65.0 };
    const auto h_u = std::vector<membrane_potential_type>{ -13.0 };
    const auto h_input = std::vector<CudaConfig::input_type>{ 50.0 };
    const auto h_disable = std::vector<UpdateStatus>{ UpdateStatus::Disabled };
    auto h_fired = std::vector<FiredStatus>{ FiredStatus::Inactive };

    auto d_x = DeviceArrayFixture<membrane_potential_type>(h_x);
    auto d_u = DeviceArrayFixture<membrane_potential_type>(h_u);
    auto d_input = DeviceArrayFixture<CudaConfig::input_type>(h_input);
    auto d_disable = DeviceArrayFixture<UpdateStatus>(h_disable);
    auto d_fired = DeviceArrayFixture<FiredStatus>(h_fired);

    const auto extra_info = NeuronExtraInfoHandle{ 1, d_disable.ptr };
    update_activity_izhikevich_entry(extra_info, h, parameters, d_input.ptr,
                                     IzhikevichDeviceState{ d_x.ptr, d_u.ptr }, no_recorder(d_fired.ptr));

    ASSERT_EQ(d_x.download()[0], h_x[0]);
}

// ────────────────────────────────────────────────────────────────────────────
// AEIF -- CPU reference formula (AEIFModel.cpp's non-CUDA update_activity) is mathematically
// identical to the GPU kernel.
// ────────────────────────────────────────────────────────────────────────────

namespace {

membrane_potential_type aeif_reference_x(membrane_potential_type x0, membrane_potential_type w0, unsigned h,
                                         const models::aeif::Parameters<membrane_potential_type>& p, membrane_potential_type input, bool* fired_out) {
    const auto scale = membrane_potential_type{ 1 } / static_cast<membrane_potential_type>(h);
    const auto d_T_inverse = membrane_potential_type{ 1 } / p.get_d_T();
    const auto tau_w_inverse = membrane_potential_type{ 1 } / p.get_tau_w();
    const auto C_inverse = membrane_potential_type{ 1 } / p.get_C();
    auto x = x0;
    auto w = w0;
    *fired_out = false;
    for (auto step = 0U; step < h; ++step) {
        const auto linear_part = -p.get_g_L() * (x - p.get_E_L());
        const auto exp_part = p.get_g_L() * p.get_d_T() * std::exp((x - p.get_V_T()) * d_T_inverse);
        const auto x_increase = (linear_part + exp_part - w + input) * C_inverse;
        const auto w_increase = (p.get_a() * (x - p.get_E_L()) - w) * tau_w_inverse;
        x += x_increase * scale;
        w += w_increase * scale;
        if (x >= p.get_V_spike()) {
            x = p.get_E_L();
            w += p.get_b();
            *fired_out = true;
            break;
        }
    }
    return x;
}

} // namespace

TEST_F(NeuronModelsGPUTest, testAeifMatchesReferenceFormula) {
    const auto parameters = models::aeif::Parameters<membrane_potential_type>(281.0, 30.0, -70.6, -50.4, 2.0, 144.0, 4.0, 80.5, -40.0);
    constexpr unsigned h = 10;

    const auto h_x = std::vector<membrane_potential_type>{ -70.6, -60.0 };
    const auto h_w = std::vector<membrane_potential_type>{ 0.0, 5.0 };
    const auto h_input = std::vector<CudaConfig::input_type>{ 0.0, 200.0 };
    const auto h_disable = std::vector<UpdateStatus>(h_x.size(), UpdateStatus::Enabled);
    auto h_fired = std::vector<FiredStatus>(h_x.size(), FiredStatus::Inactive);

    auto d_x = DeviceArrayFixture<membrane_potential_type>(h_x);
    auto d_w = DeviceArrayFixture<membrane_potential_type>(h_w);
    auto d_input = DeviceArrayFixture<CudaConfig::input_type>(h_input);
    auto d_disable = DeviceArrayFixture<UpdateStatus>(h_disable);
    auto d_fired = DeviceArrayFixture<FiredStatus>(h_fired);

    const auto extra_info = NeuronExtraInfoHandle{ static_cast<CudaConfig::number_neurons_type>(h_x.size()), d_disable.ptr };
    update_activity_aeif_entry(extra_info, h, parameters, d_input.ptr,
                               AeifDeviceState{ d_x.ptr, d_w.ptr }, no_recorder(d_fired.ptr));

    const auto result_x = d_x.download();
    const auto result_fired = d_fired.download();

    for (auto i = 0UL; i < h_x.size(); ++i) {
        auto expected_fired = false;
        const auto expected_x = aeif_reference_x(h_x[i], h_w[i], h, parameters, h_input[i], &expected_fired);
        ASSERT_NEAR(result_x[i], expected_x, value_tolerance(expected_x));
        ASSERT_EQ(result_fired[i] == FiredStatus::Fired, expected_fired);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FitzHugh-Nagumo -- the GPU kernel's cubic-nullcline coefficient (1/3, the textbook FHN
// constant) previously diverged from the CPU reference's 1/2 (see FitzHughNagumoModel.cpp's
// non-CUDA update_activity); the GPU kernel has been fixed to match the CPU reference exactly,
// so this is now a true parity check that guards against the two drifting apart again.
// ────────────────────────────────────────────────────────────────────────────

namespace {

membrane_potential_type fhn_reference_x(membrane_potential_type x0, membrane_potential_type w0, unsigned h,
                                        const models::fitzhughnagumo::Parameters<membrane_potential_type>& p, membrane_potential_type input, bool* fired_out) {
    const auto scale = membrane_potential_type{ 1 } / static_cast<membrane_potential_type>(h);
    auto x = x0;
    auto w = w0;
    for (auto step = 0U; step < h; ++step) {
        const auto x_increase = x - (x * x * x * (membrane_potential_type{ 1 } / membrane_potential_type{ 2 })) - w + input;
        const auto w_increase = p.get_phi() * (x + p.get_a() - p.get_b() * w);
        x += x_increase * scale;
        w += w_increase * scale;
    }
    *fired_out = w > x - x * x * x * (membrane_potential_type{ 1 } / membrane_potential_type{ 2 }) && x > membrane_potential_type{ 1 };
    return x;
}

} // namespace

TEST_F(NeuronModelsGPUTest, testFitzHughNagumoMatchesReferenceFormula) {
    const auto parameters = models::fitzhughnagumo::Parameters<membrane_potential_type>(0.7, 0.8, 0.08, -1.2, -0.6);
    constexpr unsigned h = 10;

    const auto h_x = std::vector<membrane_potential_type>{ -1.2, 0.5 };
    const auto h_w = std::vector<membrane_potential_type>{ -0.6, 0.2 };
    const auto h_input = std::vector<CudaConfig::input_type>{ 0.0, 1.5 };
    const auto h_disable = std::vector<UpdateStatus>(h_x.size(), UpdateStatus::Enabled);
    auto h_fired = std::vector<FiredStatus>(h_x.size(), FiredStatus::Inactive);

    auto d_x = DeviceArrayFixture<membrane_potential_type>(h_x);
    auto d_w = DeviceArrayFixture<membrane_potential_type>(h_w);
    auto d_input = DeviceArrayFixture<CudaConfig::input_type>(h_input);
    auto d_disable = DeviceArrayFixture<UpdateStatus>(h_disable);
    auto d_fired = DeviceArrayFixture<FiredStatus>(h_fired);

    const auto extra_info = NeuronExtraInfoHandle{ static_cast<CudaConfig::number_neurons_type>(h_x.size()), d_disable.ptr };
    update_activity_fitzhughnagumo_entry(extra_info, h, parameters, d_input.ptr,
                                         FitzHughNagumoDeviceState{ d_x.ptr, d_w.ptr }, no_recorder(d_fired.ptr));

    const auto result_x = d_x.download();
    const auto result_fired = d_fired.download();

    for (auto i = 0UL; i < h_x.size(); ++i) {
        auto expected_fired = false;
        const auto expected_x = fhn_reference_x(h_x[i], h_w[i], h, parameters, h_input[i], &expected_fired);
        ASSERT_NEAR(result_x[i], expected_x, value_tolerance(expected_x));
        ASSERT_EQ(result_fired[i] == FiredStatus::Fired, expected_fired);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Poisson -- involves a cuRAND draw, so exact-value parity isn't meaningful; instead check the
// structural contract: refractory countdown, and a deterministic fire/no-fire regime that holds
// regardless of the drawn random value.
// ────────────────────────────────────────────────────────────────────────────

TEST_F(NeuronModelsGPUTest, testPoissonRefractoryNeuronNeverFiresAndCountsDown) {
    const auto parameters = models::poisson::Parameters<membrane_potential_type>(0.0, 1000.0, 5U);
    constexpr unsigned h = 1;
    const auto random_key = RandomNumbers::register_random_numbers(RandomNumberKey::POISSON_INPUT, RandomNumberType::UNIFORM, 1, 4242);

    const auto h_x = std::vector<membrane_potential_type>{ 1e6 };
    const auto h_input = std::vector<CudaConfig::input_type>{ 0.0 };
    const auto h_disable = std::vector<UpdateStatus>{ UpdateStatus::Enabled };
    auto h_fired = std::vector<FiredStatus>{ FiredStatus::Inactive };
    const auto h_refractory = std::vector<unsigned int>{ 3U };

    auto d_x = DeviceArrayFixture<membrane_potential_type>(h_x);
    auto d_input = DeviceArrayFixture<CudaConfig::input_type>(h_input);
    auto d_disable = DeviceArrayFixture<UpdateStatus>(h_disable);
    auto d_fired = DeviceArrayFixture<FiredStatus>(h_fired);
    auto d_refractory = DeviceArrayFixture<unsigned int>(h_refractory);

    const auto extra_info = NeuronExtraInfoHandle{ 1, d_disable.ptr };
    update_activity_poisson_entry(extra_info, h, parameters, d_input.ptr,
                                  PoissonDeviceState{ d_x.ptr, d_refractory.ptr }, no_recorder(d_fired.ptr), random_key);

    ASSERT_EQ(d_fired.download()[0], FiredStatus::Inactive);
    ASSERT_EQ(d_refractory.download()[0], 2U);
}

TEST_F(NeuronModelsGPUTest, testPoissonFiresDeterministicallyWhenPotentialFarExceedsMaxThreshold) {
    const auto parameters = models::poisson::Parameters<membrane_potential_type>(0.0, 1000.0, 5U);
    constexpr unsigned h = 1;
    const auto random_key = RandomNumbers::register_random_numbers(RandomNumberKey::POISSON_INPUT, RandomNumberType::UNIFORM, 1, 4243);

    // curand_uniform_double never exceeds 1.0, so a potential this large guarantees a spike
    // regardless of the drawn threshold.
    const auto h_x = std::vector<membrane_potential_type>{ 1e6 };
    const auto h_input = std::vector<CudaConfig::input_type>{ 0.0 };
    const auto h_disable = std::vector<UpdateStatus>{ UpdateStatus::Enabled };
    auto h_fired = std::vector<FiredStatus>{ FiredStatus::Inactive };
    const auto h_refractory = std::vector<unsigned int>{ 0U };

    auto d_x = DeviceArrayFixture<membrane_potential_type>(h_x);
    auto d_input = DeviceArrayFixture<CudaConfig::input_type>(h_input);
    auto d_disable = DeviceArrayFixture<UpdateStatus>(h_disable);
    auto d_fired = DeviceArrayFixture<FiredStatus>(h_fired);
    auto d_refractory = DeviceArrayFixture<unsigned int>(h_refractory);

    const auto extra_info = NeuronExtraInfoHandle{ 1, d_disable.ptr };
    update_activity_poisson_entry(extra_info, h, parameters, d_input.ptr,
                                  PoissonDeviceState{ d_x.ptr, d_refractory.ptr }, no_recorder(d_fired.ptr), random_key);

    ASSERT_EQ(d_fired.download()[0], FiredStatus::Fired);
    ASSERT_EQ(d_refractory.download()[0], parameters.get_refractory_period());
}

#endif
