/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */
#ifdef RELEARN_CUDA_ENABLED

#include "main.h"

#include "cuda/CudaTypes.h"
#include "cuda/algorithm/NaiveInternalCUDA/NaiveCUDA_CU.h"
#include "cuda/memory/DeviceArray.h"
#include "types/BasicTypes.h"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace {

constexpr auto sigma = 750.0;
constexpr auto squared_sigma_inv = 1.0 / (sigma * sigma);

// Every neuron submits exactly one axon task and every other neuron has a vacant dendrite --
// the worst case for the naive (all-pairs) kernel since no candidate can be skipped.
void BM_Naive_CUDA_Find_Target_Neurons(benchmark::State& state) {
    const auto number_neurons = static_cast<std::uint64_t>(state.range(0));

    auto positions = std::vector<SimpleVec3d>(number_neurons);
    for (auto i = std::uint64_t{ 0 }; i < number_neurons; ++i) {
        positions[i] = SimpleVec3d{ static_cast<double>(i % 100), static_cast<double>((i / 100) % 100), static_cast<double>(i / 10000) };
    }

    auto vacant_dendrites = std::vector<RelearnTypes::counter_type>(number_neurons, 1U);

    // One task per source neuron, mapping task i -> source neuron i.
    auto mapping = std::vector<RelearnTypes::counter_type>(number_neurons);
    for (auto i = std::uint64_t{ 0 }; i < number_neurons; ++i) {
        mapping[i] = static_cast<RelearnTypes::counter_type>(i);
    }

    const auto target_size = number_neurons;
    auto random_nums = std::vector<double>(target_size, 0.5);
    auto target_array = std::vector<std::uint64_t>(target_size, std::numeric_limits<std::uint64_t>::max());

    auto d_pos = DeviceArray<SimpleVec3d>(std::span<const SimpleVec3d>(positions));

    for (auto _ : state) {
        NaiveCUDA_CU::find_target_neurons(d_pos, number_neurons, NaiveCUDA_CU::NaiveTargetSelectionTask{ mapping, vacant_dendrites, random_nums, target_size }, target_array, squared_sigma_inv);

        state.PauseTiming();
        benchmark::DoNotOptimize(target_array);
        state.ResumeTiming();
    }
}

} // namespace

BENCHMARK(BM_Naive_CUDA_Find_Target_Neurons)->Unit(benchmark::kMillisecond)->Arg(small_number_neurons)->Iterations(medium_number_iterations);

#endif
