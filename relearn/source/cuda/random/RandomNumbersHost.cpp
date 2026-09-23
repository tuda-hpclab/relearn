/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RandomNumberHost.h"
#include "RandomNumberKeys.h"

#include "cuda/CudaBaseBridgeFunctions.h"
#include "cuda/CudaConfig.h"
#include "cuda/memory/DeviceArray.h"
#include "cuda/random/RandomBridge.h"
#include "util/RelearnException.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace RandomNumbers {
namespace {
bool already_initialized{ false };
std::optional<DeviceArray<RandomNumbersConfig>> d_configs_device{};
std::size_t mem_usage{};
std::vector<std::uint32_t> already_initialized_key{};

std::uint32_t number_registered_generators{ 0 };

// Every register_random_numbers() call allocates its own cuRAND state array; kept alive here
// (as opaque byte buffers -- the concrete curand state struct is only known inside
// RandomNumber.cu) purely so reset() can free them all again by destroying the vector.
std::vector<DeviceArray<std::byte>> allocated_curand_states{};
} // namespace

std::uint32_t register_random_numbers(RandomNumberKey key, RandomNumberType type,
                                      std::uint64_t number_neurons, std::uint64_t seed) {

    cudaDeviceSynchronize_bridge();

    if (!already_initialized) {
        d_configs_device.emplace(max_number_random_keys);
        init_random_configs(d_configs_device->device_ptr());
        already_initialized = true;
    }

    if (already_initialized_key.size() <= key) {
        already_initialized_key.resize(static_cast<std::size_t>(key) + 1, 0U);
    }
    auto& key_count = already_initialized_key[key];
    key_count++;
    number_registered_generators++;
    if (number_registered_generators > max_number_random_keys) {
        RelearnException::fail("RandomNumbers::RandomNumbers: number_registered_generators exceeded");
    }

    auto& d_curand_state = allocated_curand_states.emplace_back(getCurandStateSize() * number_neurons);
    mem_usage += getCurandStateSize() * number_neurons;
    curand_setup_entry(static_cast<CudaConfig::number_neurons_type>(number_neurons), d_curand_state.void_device_ptr(), seed);

    RandomNumbersConfig _config{};
    _config.number_neurons = number_neurons;
    _config.d_use_pre_drawn_cpu = false;
    _config.d_pre_drawn_state = nullptr;
    _config.d_pre_drawn = nullptr;
    _config.d_curand_states = d_curand_state.void_device_ptr();
    _config.random_type = type;

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access) - d_configs_device is always populated here: either just emplaced above, or already_initialized was true (implying a prior emplace that only reset() clears)
    cudaMemcpy_to_device_bridge(d_configs_device->device_ptr() + number_registered_generators - 1U, &_config, sizeof(RandomNumbersConfig));

    cudaDeviceSynchronize_bridge();

    return number_registered_generators - 1;
}

std::size_t get_memory_usage() {
    return mem_usage;
}

void reset() {
    d_configs_device.reset();
    allocated_curand_states.clear();

    already_initialized = false;
    mem_usage = 0;
    already_initialized_key.clear();
    number_registered_generators = 0;
}
}; // namespace RandomNumbers