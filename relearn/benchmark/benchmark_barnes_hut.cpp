/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "main.h"

#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/Internal/octree/NodeCache.h"
#include "algorithm/Kernel/Gaussian.h"
#include "neurons/enums/SynapticElementType.h"
#include "types/BasicTypes.h"

#include <benchmark/benchmark.h>

#include <cpp-utility/Cast.hpp>

#include <cstddef>
#include <cstdint>

namespace {
void BM_Test_Acceptance_Criterion(benchmark::State& state) {
    constexpr auto default_theta = utility::as<RelearnTypes::acceptance_criterion_type>(0.3);
    constexpr auto inner_iterations = 1000000;

    const auto pos = RelearnTypes::position_type{ 2.0, 2.0, 2.0 };
    const auto root = get_octree<BarnesHutCell>(5);

    auto sum = 0;

    for (auto _ : state) {
        for (auto i = 0; i < inner_iterations; i++) {
            const auto accept = BarnesHutBase<BarnesHutCell>::test_acceptance_criterion(pos, &root, ElementType::Dendrite, SignalType::Excitatory, default_theta);
            sum += static_cast<char>(accept);
        }
    }

    benchmark::DoNotOptimize(sum);
}

void BM_Get_Nodes_To_Consider(benchmark::State& state) {
    constexpr auto default_theta = utility::as<RelearnTypes::acceptance_criterion_type>(0.3);
    constexpr auto inner_iterations = 1000;

    const auto number_nodes = state.range(0);

    const auto pos = RelearnTypes::position_type{ 1.0, 1.0, 1.0 };
    auto root = get_octree<BarnesHutCell>(static_cast<std::size_t>(number_nodes));

    auto sum = static_cast<std::size_t>(0);

    auto node_cache = NodeCache<BarnesHutCell>{};
    node_cache.set_is_already_downloaded();

    for (auto _ : state) {
        for (auto i = 0; i < inner_iterations; i++) {
            const auto& possible_targets = BarnesHutBase<BarnesHutCell>::get_nodes_to_consider(node_cache, pos, &root, ElementType::Dendrite, SignalType::Excitatory, default_theta, false);
            sum += possible_targets.size();
        }
    }

    benchmark::DoNotOptimize(sum);
}

void BM_Find_Target_Neuron(benchmark::State& state) {
    constexpr auto default_theta = utility::as<RelearnTypes::acceptance_criterion_type>(0.3);
    constexpr auto inner_iterations = 1000;

    const auto number_nodes = state.range(0);

    const auto pos = RelearnTypes::position_type{ 10, 10, 10 };
    auto root = get_octree<BarnesHutCell>(static_cast<std::size_t>(number_nodes));

    const auto rni = RankNeuronId{ mpiPP::MPIRank{ 1 }, NeuronID{ 0 } };

    auto node_cache = NodeCache<BarnesHutCell>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (auto _ : state) {
        for (auto i = 0; i < inner_iterations; i++) {
            const auto& target_opt = BarnesHutBase<BarnesHutCell>::find_target_neuron(kernel, node_cache, rni, pos, &root, ElementType::Dendrite, SignalType::Excitatory, default_theta);

            state.PauseTiming();

            const auto& target = target_opt.value();
            const auto& [target_rank, target_id] = target;

            benchmark::DoNotOptimize(target_rank.get_rank());
            benchmark::DoNotOptimize(target_id.get_neuron_id());

            state.ResumeTiming();
        }
    }
}

void BM_Find_Target_Neurons(benchmark::State& state) {
    constexpr auto default_theta = utility::as<RelearnTypes::acceptance_criterion_type>(0.3);

    const auto number_nodes = static_cast<std::uint32_t>(state.range(0));

    const auto pos = RelearnTypes::position_type{ 10, 10, 10 };
    auto root = get_octree<BarnesHutCell>(static_cast<std::size_t>(number_nodes));

    const auto rni = RankNeuronId{ mpiPP::MPIRank{ 1 }, NeuronID{ 0 } };

    auto node_cache = NodeCache<BarnesHutCell>{};
    node_cache.set_is_already_downloaded();

    auto kernel = GaussianDistributionKernel{};

    for (auto _ : state) {
        const auto& target_opt = BarnesHutBase<BarnesHutCell>::find_target_neurons(kernel, node_cache, rni, pos, 1000, &root, ElementType::Dendrite, SignalType::Excitatory, default_theta);

        state.PauseTiming();

        const auto& target = target_opt[target_opt.size() - 1];
        const auto& [target_rank, target_id] = target;

        benchmark::DoNotOptimize(target_rank.get_rank());
        benchmark::DoNotOptimize(target_id.get_target().get_neuron_id());

        state.ResumeTiming();
    }
}
} // namespace

BENCHMARK(BM_Test_Acceptance_Criterion)->Unit(benchmark::kMillisecond)->Iterations(small_number_iterations);
BENCHMARK(BM_Get_Nodes_To_Consider)->Unit(benchmark::kMillisecond)->Arg(medium_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Find_Target_Neuron)->Unit(benchmark::kMillisecond)->Arg(medium_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Find_Target_Neurons)->Unit(benchmark::kMillisecond)->Arg(small_number_neurons)->Iterations(large_number_iterations);