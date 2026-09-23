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
#include "cuda/input/ActivityInput.h"
#include "cuda/random/RandomNumberHost.h"
#include "cuda/random/RandomNumberKeys.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "neurons/enums/UpdateStatus.h"

#include <cmath>
#include <numeric>
#include <vector>

class ActivityInputGPUTest : public RelearnTest { };

namespace {

// Same device-array RAII helper pattern used in test_neuron_models_gpu.cpp / test_calcium_gpu.cpp;
// these entry points are plain host-callable functions, so no .cu launcher file is needed here.
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

TEST_F(ActivityInputGPUTest, testConstantActivityInputSetsBaseInputForEnabledNeuronsOnly) {
    constexpr auto base_input = CudaConfig::input_type{ 2.5 };
    const auto h_disable = std::vector<UpdateStatus>{ UpdateStatus::Enabled, UpdateStatus::Disabled, UpdateStatus::Enabled, UpdateStatus::Enabled };
    auto h_input = std::vector<CudaConfig::input_type>(h_disable.size(), -1.0);

    auto d_disable = DeviceArrayFixture<UpdateStatus>(h_disable);
    auto d_input = DeviceArrayFixture<CudaConfig::input_type>(h_input);

    const auto stream = StreamWrapper::default_stream();
    // Only process the middle range [1, 3) -- neuron 0 and neuron 3 must be left untouched.
    update_constant_activity_input_range_entry(1, 3, d_disable.ptr, d_input.ptr, base_input, stream);
    cudaDeviceSynchronize_bridge();

    const auto result = d_input.download();
    ASSERT_EQ(result[0], -1.0) << "neuron outside the processed range must be untouched";
    ASSERT_EQ(result[1], 0.0) << "disabled neuron inside the range must get 0, not base_input";
    ASSERT_EQ(result[2], base_input);
    ASSERT_EQ(result[3], -1.0) << "neuron outside the processed range must be untouched";
}

TEST_F(ActivityInputGPUTest, testCombinedActivityInputSumsSubInputsElementwise) {
    constexpr auto num_neurons = 3UL;
    constexpr auto num_sub_inputs = 3UL;

    // Three sub-input arrays; combined value for neuron i should be the sum across sub-inputs.
    const auto h_sub0 = std::vector<CudaConfig::input_type>{ 1.0, 2.0, 3.0 };
    const auto h_sub1 = std::vector<CudaConfig::input_type>{ 10.0, 20.0, 30.0 };
    const auto h_sub2 = std::vector<CudaConfig::input_type>{ 100.0, 200.0, 300.0 };

    auto d_sub0 = DeviceArrayFixture<CudaConfig::input_type>(h_sub0);
    auto d_sub1 = DeviceArrayFixture<CudaConfig::input_type>(h_sub1);
    auto d_sub2 = DeviceArrayFixture<CudaConfig::input_type>(h_sub2);

    const auto h_sub_ptrs = std::vector<CudaConfig::input_type*>{ d_sub0.ptr, d_sub1.ptr, d_sub2.ptr };
    auto d_sub_ptrs = DeviceArrayFixture<CudaConfig::input_type*>(h_sub_ptrs);

    auto h_input = std::vector<CudaConfig::input_type>(num_neurons, -1.0);
    auto d_input = DeviceArrayFixture<CudaConfig::input_type>(h_input);

    const auto stream = StreamWrapper::default_stream();
    update_combined_activity_input_range_entry(0, num_neurons, num_sub_inputs, d_input.ptr, d_sub_ptrs.ptr, stream);
    cudaDeviceSynchronize_bridge();

    const auto result = d_input.download();
    for (auto i = 0UL; i < num_neurons; ++i) {
        const auto expected = h_sub0[i] + h_sub1[i] + h_sub2[i];
        ASSERT_DOUBLE_EQ(result[i], expected);
    }
}

TEST_F(ActivityInputGPUTest, testNormalActivityInputZerosDisabledNeuronsAndRespectsRange) {
    const auto h_disable = std::vector<UpdateStatus>{ UpdateStatus::Enabled, UpdateStatus::Disabled, UpdateStatus::Enabled };
    auto h_input = std::vector<CudaConfig::input_type>(h_disable.size(), -1.0);
    const auto random_key = RandomNumbers::register_random_numbers(RandomNumberKey::NORMAL_INPUT, RandomNumberType::NORMAL, h_disable.size(), 555);

    auto d_disable = DeviceArrayFixture<UpdateStatus>(h_disable);
    auto d_input = DeviceArrayFixture<CudaConfig::input_type>(h_input);

    const auto stream = StreamWrapper::default_stream();
    update_normal_activity_input_range_entry(0, static_cast<CudaConfig::number_neurons_type>(h_disable.size()),
                                             d_disable.ptr, d_input.ptr, 0.0, 1.0, random_key, stream);
    cudaDeviceSynchronize_bridge();

    const auto result = d_input.download();
    ASSERT_EQ(result[1], 0.0) << "disabled neuron must receive exactly 0 input, not a random draw";
}

TEST_F(ActivityInputGPUTest, testNormalActivityInputMatchesRequestedMeanAndStddevStatistically) {
    constexpr auto num_neurons = 20000UL;
    constexpr auto mean = CudaConfig::input_type{ 5.0 };
    constexpr auto stddev = CudaConfig::input_type{ 2.0 };

    const auto h_disable = std::vector<UpdateStatus>(num_neurons, UpdateStatus::Enabled);
    auto h_input = std::vector<CudaConfig::input_type>(num_neurons, 0.0);
    const auto random_key = RandomNumbers::register_random_numbers(RandomNumberKey::NORMAL_INPUT, RandomNumberType::NORMAL, num_neurons, 556);

    auto d_disable = DeviceArrayFixture<UpdateStatus>(h_disable);
    auto d_input = DeviceArrayFixture<CudaConfig::input_type>(h_input);

    const auto stream = StreamWrapper::default_stream();
    update_normal_activity_input_range_entry(0, static_cast<CudaConfig::number_neurons_type>(num_neurons),
                                             d_disable.ptr, d_input.ptr, mean, stddev, random_key, stream);
    cudaDeviceSynchronize_bridge();

    const auto result = d_input.download();
    const auto sum = std::accumulate(result.begin(), result.end(), 0.0);
    const auto sample_mean = sum / static_cast<double>(result.size());

    auto sq_diff_sum = 0.0;
    for (const auto v : result) {
        const auto diff = static_cast<double>(v) - sample_mean;
        sq_diff_sum += diff * diff;
    }
    const auto sample_stddev = std::sqrt(sq_diff_sum / static_cast<double>(result.size()));

    // Loose statistical bounds -- catches a badly wrong scale/offset (e.g. stddev not applied,
    // or mean/stddev swapped) without being sensitive to sampling noise at this sample size.
    ASSERT_NEAR(sample_mean, mean, 0.15) << "sample mean " << sample_mean << " far from requested mean " << mean;
    ASSERT_NEAR(sample_stddev, stddev, 0.15) << "sample stddev " << sample_stddev << " far from requested stddev " << stddev;
}

#endif
