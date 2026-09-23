/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "main.h"

#include "neurons/synaptic_elements/Axons.h"
#include "neurons/synaptic_elements/MultiPositionAxons.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "util/Vec3.h"

#include "adapter/extra_info/ExtraInfoAdapter.h"

#include "factory/extra_info/extra_info_factory.h"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <vector>

namespace {
std::vector<RelearnTypes::position_type> get_positions(const std::size_t number_positions) {
    auto result = std::vector<RelearnTypes::position_type>{};
    result.reserve(number_positions);

    for (auto i = std::size_t{ 0 }; i < number_positions; ++i) {
        result.emplace_back(1.0, 2.0, 3.0);
    }

    return result;
}

std::vector<std::vector<RelearnTypes::position_type>> get_axon_positions(const RelearnTypes::number_neurons_type number_neurons, const std::size_t number_axon_positions) {
    auto result = std::vector<std::vector<RelearnTypes::position_type>>{};
    result.reserve(number_neurons);

    for (auto i = std::size_t{ 0 }; i < number_neurons; ++i) {
        result.emplace_back(get_positions(number_axon_positions));
    }

    return result;
}

std::vector<std::vector<RelearnTypes::position_type>> get_axon_positions(const RelearnTypes::number_neurons_type number_neurons) {
    auto result = std::vector<std::vector<RelearnTypes::position_type>>{};
    result.reserve(number_neurons);

    for (auto i = std::size_t{ 0 }; i < number_neurons; ++i) {
        result.emplace_back(get_positions((i % 10) + 1));
    }

    return result;
}

void BM_Axon_GetBoutonPosition(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto postions = get_positions(number_neurons);
    extra_info->set_positions(postions);

    auto axons = Axons{};
    axons.set_extra_infos(extra_info);
    axons.init(number_neurons);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    auto holder = std::vector<RelearnTypes::position_type>{};
    holder.resize(number_neurons);

    for (auto _ : state) {
        for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons; ++i) {
            holder[i] = axons.get_bouton_position(i);
        }

        state.PauseTiming();

        auto sum = ranges::accumulate(holder, RelearnTypes::position_type{});

        benchmark::DoNotOptimize(sum);
        state.ResumeTiming();
    }
}

void BM_Axon_GetBoutonPosition_MultiPosition_1(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto postions = get_axon_positions(number_neurons, 1);

    auto axons = MultiPositionAxons{};
    axons.set_bouton_positions(postions);
    axons.set_extra_infos(extra_info);
    axons.init(number_neurons);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    auto holder = std::vector<RelearnTypes::position_type>{};
    holder.resize(number_neurons);

    for (auto _ : state) {
        for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons; ++i) {
            holder[i] = axons.get_bouton_position(i);
        }

        state.PauseTiming();

        auto sum = ranges::accumulate(holder, RelearnTypes::position_type{});

        benchmark::DoNotOptimize(sum);
        state.ResumeTiming();
    }
}

void BM_Axon_GetBoutonPosition_MultiPosition_5(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto postions = get_axon_positions(number_neurons, 5);

    auto axons = MultiPositionAxons{};
    axons.set_bouton_positions(postions);
    axons.set_extra_infos(extra_info);
    axons.init(number_neurons);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    auto holder = std::vector<RelearnTypes::position_type>{};
    holder.resize(number_neurons);

    for (auto _ : state) {
        for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons; ++i) {
            holder[i] = axons.get_bouton_position(i);
        }

        state.PauseTiming();

        auto sum = ranges::accumulate(holder, RelearnTypes::position_type{});

        benchmark::DoNotOptimize(sum);
        state.ResumeTiming();
    }
}

void BM_Axon_GetBoutonPosition_MultiPosition_15(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto postions = get_axon_positions(number_neurons, 15);

    auto axons = MultiPositionAxons{};
    axons.set_bouton_positions(postions);
    axons.set_extra_infos(extra_info);
    axons.init(number_neurons);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    auto holder = std::vector<RelearnTypes::position_type>{};
    holder.resize(number_neurons);

    for (auto _ : state) {
        for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons; ++i) {
            holder[i] = axons.get_bouton_position(i);
        }

        state.PauseTiming();

        auto sum = ranges::accumulate(holder, RelearnTypes::position_type{});

        benchmark::DoNotOptimize(sum);
        state.ResumeTiming();
    }
}

void BM_Axon_GetBoutonPosition_MultiPosition_X(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto postions = get_axon_positions(number_neurons);

    auto axons = MultiPositionAxons{};
    axons.set_bouton_positions(postions);
    axons.set_extra_infos(extra_info);
    axons.init(number_neurons);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    auto holder = std::vector<RelearnTypes::position_type>{};
    holder.resize(number_neurons);

    for (auto _ : state) {
        for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons; ++i) {
            holder[i] = axons.get_bouton_position(i);
        }

        state.PauseTiming();

        auto sum = ranges::accumulate(holder, RelearnTypes::position_type{});

        benchmark::DoNotOptimize(sum);
        state.ResumeTiming();
    }
}

} // namespace

BENCHMARK(BM_Axon_GetBoutonPosition)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons })->Iterations(small_number_iterations);
BENCHMARK(BM_Axon_GetBoutonPosition_MultiPosition_1)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons })->Iterations(small_number_iterations);
BENCHMARK(BM_Axon_GetBoutonPosition_MultiPosition_5)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons })->Iterations(small_number_iterations);
BENCHMARK(BM_Axon_GetBoutonPosition_MultiPosition_15)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons })->Iterations(small_number_iterations);
BENCHMARK(BM_Axon_GetBoutonPosition_MultiPosition_X)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons })->Iterations(small_number_iterations);
