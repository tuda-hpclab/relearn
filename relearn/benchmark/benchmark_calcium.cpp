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

#include "neurons/enums/FiredStatus.h"
#include "types/BasicTypes.h"

#include "adapter/extra_info/ExtraInfoAdapter.h"

#include "factory/calcium/calcium_factory.h"
#include "factory/extra_info/extra_info_factory.h"

#include <benchmark/benchmark.h>

#include <range/v3/numeric/accumulate.hpp>

#include <omp.h>

#include <vector>

#ifdef RELEARN_CUDA_ENABLED

namespace {
void BM_CalciumCalculator_No_Decay_No_Fired(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));
    const auto decay = state.range(1);
    const auto beta = state.range(2);

    omp_set_num_threads(1);

    auto fired_status = LazySyncedArray<FiredStatus>{};
    fired_status.resize(number_neurons, FiredStatus::Inactive);

    auto calcium_calculator = CalciumFactory::construct_calcium_calculator_no_decay();
    calcium_calculator->init(number_neurons);
    calcium_calculator->set_tau_C(static_cast<RelearnTypes::calcium_type>(decay));
    calcium_calculator->set_beta(RelearnTypes::calcium_type{ 1 } / static_cast<RelearnTypes::calcium_type>(beta));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    calcium_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    for (auto _ : state) {
        calcium_calculator->update_calcium(100, fired_status.get_device_ptr_const());

        state.PauseTiming();

        const auto values = calcium_calculator->get_calcium();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        state.ResumeTiming();
    }
}

void BM_CalciumCalculator_No_Decay_All_Fired(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));
    const auto decay = state.range(1);
    const auto beta = state.range(2);

    omp_set_num_threads(1);

    auto fired_status = LazySyncedArray<FiredStatus>{};
    fired_status.resize(number_neurons, FiredStatus::Fired);

    auto calcium_calculator = CalciumFactory::construct_calcium_calculator_no_decay();
    calcium_calculator->init(number_neurons);
    calcium_calculator->set_tau_C(static_cast<RelearnTypes::calcium_type>(decay));
    calcium_calculator->set_beta(RelearnTypes::calcium_type{ 1 } / static_cast<RelearnTypes::calcium_type>(beta));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    calcium_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    for (auto _ : state) {
        calcium_calculator->update_calcium(100, fired_status.get_device_ptr_const());
        state.PauseTiming();

        const auto values = calcium_calculator->get_calcium();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        state.ResumeTiming();
    }
}

void BM_CalciumCalculator_Relative_Decay_No_Fired(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    omp_set_num_threads(1);

    auto fired_status = LazySyncedArray<FiredStatus>{};
    fired_status.resize(number_neurons, FiredStatus::Inactive);

    auto calcium_calculator = CalciumFactory::construct_calcium_calculator_relative_decay();
    calcium_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    calcium_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    for (auto _ : state) {
        calcium_calculator->update_calcium(100, fired_status.get_device_ptr_const());
        state.PauseTiming();

        const auto values = calcium_calculator->get_calcium();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        state.ResumeTiming();
    }
}

void BM_CalciumCalculator_Relative_Decay_All_Fired(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    omp_set_num_threads(1);

    auto fired_status = LazySyncedArray<FiredStatus>{};
    fired_status.resize(number_neurons, FiredStatus::Fired);

    auto calcium_calculator = CalciumFactory::construct_calcium_calculator_relative_decay();
    calcium_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    calcium_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    for (auto _ : state) {
        calcium_calculator->update_calcium(100, fired_status.get_device_ptr_const());
        state.PauseTiming();

        const auto values = calcium_calculator->get_calcium();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        state.ResumeTiming();
    }
}

void BM_CalciumCalculator_Absolute_Decay_No_Fired(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    omp_set_num_threads(1);

    auto fired_status = LazySyncedArray<FiredStatus>{};
    fired_status.resize(number_neurons, FiredStatus::Inactive);

    auto calcium_calculator = CalciumFactory::construct_calcium_calculator_absolute_decay();
    calcium_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    calcium_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    for (auto _ : state) {
        calcium_calculator->update_calcium(100, fired_status.get_device_ptr_const());
        state.PauseTiming();

        const auto values = calcium_calculator->get_calcium();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        state.ResumeTiming();
    }
}

void BM_CalciumCalculator_Absolute_Decay_All_Fired(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    omp_set_num_threads(1);

    auto fired_status = LazySyncedArray<FiredStatus>{};
    fired_status.resize(number_neurons, FiredStatus::Fired);

    auto calcium_calculator = CalciumFactory::construct_calcium_calculator_absolute_decay();
    calcium_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    calcium_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    for (auto _ : state) {
        calcium_calculator->update_calcium(100, fired_status.get_device_ptr_const());
        state.PauseTiming();

        const auto values = calcium_calculator->get_calcium();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        state.ResumeTiming();
    }
}
} // namespace

BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 1000, 100 })->Iterations(large_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 5000, 100 })->Iterations(large_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 10000, 100 })->Iterations(large_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 15000, 100 })->Iterations(large_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 20000, 100 })->Iterations(large_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 50000, 100 })->Iterations(large_number_iterations);

BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 1000, 1000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 5000, 1000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 10000, 1000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 15000, 1000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 20000, 1000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 50000, 1000 })->Iterations(small_number_iterations);

BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 1000, 10000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 5000, 10000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 10000, 10000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 15000, 10000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 20000, 10000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 50000, 10000 })->Iterations(small_number_iterations);

BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 1000, 100 })->Iterations(large_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 5000, 100 })->Iterations(large_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 10000, 100 })->Iterations(large_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 15000, 100 })->Iterations(large_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 20000, 100 })->Iterations(large_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 50000, 100 })->Iterations(large_number_iterations);

BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 1000, 1000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 5000, 1000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 10000, 1000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 15000, 1000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 20000, 1000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 50000, 1000 })->Iterations(small_number_iterations);

BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 1000, 10000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 5000, 10000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 10000, 10000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 15000, 10000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 20000, 10000 })->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, 50000, 10000 })->Iterations(small_number_iterations);

BENCHMARK(BM_CalciumCalculator_Relative_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_Relative_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(small_number_iterations);

BENCHMARK(BM_CalciumCalculator_Absolute_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(small_number_iterations);
BENCHMARK(BM_CalciumCalculator_Absolute_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(small_number_iterations);

#endif
