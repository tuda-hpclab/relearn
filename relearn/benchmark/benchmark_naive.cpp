/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "main.h"

#include "algorithm/Internal/octree/NodeCache.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/NaiveInternal/NaiveBase.h"
#include "algorithm/NaiveInternal/NaiveCell.h"
#include "benchmark/benchmark.h"
#include "neurons/enums/SynapticElementType.h"

#include <cstddef>

namespace {
void BM_Naive_Find_Target_Neuron(benchmark::State& state) {
    constexpr auto inner_iterations = 1000;
    constexpr auto pos = Vec3d{ 10, 10, 10 };
    const auto number_nodes = state.range(0);

    auto node_cache = NodeCache<NaiveCell>{};
    node_cache.set_is_already_downloaded();

    auto root = get_octree<NaiveCell>(static_cast<size_t>(number_nodes));

    auto kernel = GaussianDistributionKernel{};

    for (auto _ : state) {
        for (auto i = 0; i < inner_iterations; i++) {
            state.ResumeTiming();
            const auto& target_opt = NaiveBase<NaiveCell>::find_target_neuron(kernel, node_cache, NeuronID{ 0 }, pos, SignalType::Excitatory, &root);

            state.PauseTiming();
            if (!target_opt.has_value()) {
                continue;
            }
            const auto& target = target_opt.value();
            const auto& [target_rank, target_id] = target;

            benchmark::DoNotOptimize(target_rank.get_rank());
            benchmark::DoNotOptimize(target_id.get_neuron_id());
        }
    }
}

void BM_Naive_Find_Target_Neurons(benchmark::State& state) {
    const auto number_nodes = state.range(0);
    const auto pos = Vec3d{ 10, 10, 10 };

    auto node_cache = NodeCache<NaiveCell>{};
    node_cache.set_is_already_downloaded();

    auto root = get_octree<NaiveCell>(static_cast<size_t>(number_nodes));

    auto kernel = GaussianDistributionKernel{};

    for (auto _ : state) {
        const auto& target_map = NaiveBase<NaiveCell>::find_target_neurons(kernel, node_cache, NeuronID{ 0 }, pos, 1000, &root, SignalType::Excitatory);

        state.PauseTiming();

        const auto& target = target_map[target_map.size() - 1];
        const auto& [target_rank, target_id] = target;

        benchmark::DoNotOptimize(target_rank.get_rank());
        benchmark::DoNotOptimize(target_id.get_target().get_neuron_id());

        state.ResumeTiming();
    }
}
} // namespace

BENCHMARK(BM_Naive_Find_Target_Neuron)->Unit(benchmark::kMillisecond)->Arg(small_number_neurons)->Iterations(small_number_iterations);
BENCHMARK(BM_Naive_Find_Target_Neurons)->Unit(benchmark::kMillisecond)->Arg(small_number_neurons)->Iterations(small_number_iterations);