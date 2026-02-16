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

#include "neurons/helper/CachedChoiceFunction.h"

#include "factory/activity_input/activity_input_factory.h"
#include "factory/extra_info/extra_info_factory.h"

#include <benchmark/benchmark.h>
#include <omp.h>
#include <range/v3/numeric/accumulate.hpp>

#include <cmath>
#include <vector>

namespace {
void BM_Constant_Activity_Input(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    omp_set_num_threads(1);

    auto activity_input = ActivityInputFactory::construct_constant_activity(1.1);
    activity_input->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    activity_input->set_extra_infos(extra_info);

    for (auto _ : state) {
        activity_input->update_input(101);
        state.PauseTiming();

        const auto values = activity_input->get_input();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        state.ResumeTiming();
    }
}

void BM_Normal_Activity_Input(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    omp_set_num_threads(1);

    auto activity_input = ActivityInputFactory::construct_normal_activity(4.2, 0.42);
    activity_input->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    activity_input->set_extra_infos(extra_info);

    for (auto _ : state) {
        activity_input->update_input(101);
        state.PauseTiming();

        const auto values = activity_input->get_input();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        state.ResumeTiming();
    }
}

void BM_Fast_Normal_Activity_Input(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    omp_set_num_threads(1);

    auto activity_input = ActivityInputFactory::construct_fast_normal_activity(4.2, 0.42, 10);
    activity_input->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    activity_input->set_extra_infos(extra_info);

    for (auto _ : state) {
        activity_input->update_input(101);
        state.PauseTiming();

        const auto values = activity_input->get_input();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        state.ResumeTiming();
    }
}

void BM_Combined_Activity_Input(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    omp_set_num_threads(1);

    auto other_inputs = std::vector{ ActivityInputFactory::construct_fast_normal_activity(4.2, 0.42, 10), ActivityInputFactory::construct_normal_activity(4.2, 0.42), ActivityInputFactory::construct_constant_activity(1.1) };

    auto activity_input = ActivityInputFactory::construct_combined_activity(other_inputs);
    activity_input->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    activity_input->set_extra_infos(extra_info);

    for (auto _ : state) {
        activity_input->update_input(101);
        state.PauseTiming();

        const auto values = activity_input->get_input();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        state.ResumeTiming();
    }
}

void BM_Flexible_Activity_Input_Easy(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    omp_set_num_threads(1);

    auto inputs = std::vector<std::shared_ptr<ActivityInput>>{ ActivityInputFactory::construct_constant_activity(2.0), ActivityInputFactory::construct_constant_activity(42.0) };

    auto even_neuron_ids = std::vector<NeuronID>{};
    auto odd_neuron_ids = std::vector<NeuronID>{};

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        if (neuron_id.get_neuron_id() % 2 == 0) {
            even_neuron_ids.push_back(neuron_id);
        } else {
            odd_neuron_ids.push_back(neuron_id);
        }
    }

    constexpr auto max_step = std::numeric_limits<RelearnTypes::step_type>::max();
    auto changes = std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, std::size_t, std::vector<NeuronID>>>{
        { 0, max_step, 0, even_neuron_ids }, { 0, max_step, 1, odd_neuron_ids }
    };

    auto choice_function = std::make_unique<CachedChoiceFunction>(std::move(changes), number_neurons);

    auto activity_input = ActivityInputFactory::construct_flexible_activity(inputs, std::move(choice_function));
    activity_input->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    activity_input->set_extra_infos(extra_info);

    auto step = 0U;
    for (auto _ : state) {
        activity_input->update_input(step);
        state.PauseTiming();

        const auto values = activity_input->get_input();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        step++;
        state.ResumeTiming();
    }
}

void BM_Flexible_Activity_Input_Complex(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));
    const auto number_inputs = static_cast<std::size_t>(state.range(1));
    const auto update_interval = static_cast<RelearnTypes::step_type>(state.range(2));

    omp_set_num_threads(1);

    auto inputs = std::vector<std::shared_ptr<ActivityInput>>{};
    for (auto i = 0U; i < number_inputs; i++) {
        inputs.push_back(ActivityInputFactory::construct_constant_activity(i * 10.0));
    }

    auto neuron_ids = std::vector<std::vector<NeuronID>>{};
    neuron_ids.resize(number_inputs);
    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        auto index = neuron_id.get_neuron_id() % number_inputs;
        neuron_ids[index].emplace_back(neuron_id);
    }

    auto changes = std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, std::size_t, std::vector<NeuronID>>>{};

    for (auto step = 0U; step < state.max_iterations; step += update_interval) {
        for (auto index = 0U; index < number_inputs; index++) {
            changes.emplace_back(step, step + update_interval, index, neuron_ids[index]);
        }
    }

    auto choice_function = std::make_unique<CachedChoiceFunction>(std::move(changes), number_neurons);

    auto activity_input = ActivityInputFactory::construct_flexible_activity(inputs, std::move(choice_function));
    activity_input->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    activity_input->set_extra_infos(extra_info);

    auto step = 0U;
    for (auto _ : state) {
        activity_input->update_input(step);
        state.PauseTiming();

        const auto values = activity_input->get_input();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        step++;
        state.ResumeTiming();
    }
}

void BM_Scale_Activity_Input_Multiply(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    omp_set_num_threads(1);

    auto input = ActivityInputFactory::construct_constant_activity(1.1);

    auto function = [](const double d) -> double { return 2.3 * d; };

    auto activity_input = ActivityInputFactory::construct_scale_activity(function, input);
    activity_input->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    activity_input->set_extra_infos(extra_info);

    for (auto _ : state) {
        activity_input->update_input(101);
        state.PauseTiming();

        const auto values = activity_input->get_input();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        state.ResumeTiming();
    }
}

void BM_Scale_Activity_Input_Logarithm(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    omp_set_num_threads(1);

    auto input = ActivityInputFactory::construct_constant_activity(1.1);

    auto function = [](const double d) -> double { return 2.3 * std::log10((0.03 * d) + 1.0); };

    auto activity_input = ActivityInputFactory::construct_scale_activity(function, input);
    activity_input->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    activity_input->set_extra_infos(extra_info);

    for (auto _ : state) {
        activity_input->update_input(101);
        state.PauseTiming();

        const auto values = activity_input->get_input();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        state.ResumeTiming();
    }
}

void BM_Scale_Activity_Input_Tanh(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    omp_set_num_threads(1);

    auto input = ActivityInputFactory::construct_constant_activity(1.1);

    auto function = [](const double d) -> double { return 2.3 * std::tanh(0.03 * d); };

    auto activity_input = ActivityInputFactory::construct_scale_activity(function, input);
    activity_input->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    activity_input->set_extra_infos(extra_info);

    for (auto _ : state) {
        activity_input->update_input(101);
        state.PauseTiming();

        const auto values = activity_input->get_input();

        benchmark::DoNotOptimize(ranges::accumulate(values, 0.0));
        state.ResumeTiming();
    }
}
} // namespace

constexpr static auto number_inputs = 50U;
constexpr static auto update_increments = 10U;
BENCHMARK(BM_Combined_Activity_Input)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Constant_Activity_Input)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Flexible_Activity_Input_Easy)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Flexible_Activity_Input_Complex)->Unit(benchmark::kMillisecond)->Args({ large_number_neurons, number_inputs, update_increments })->Iterations(large_number_iterations);
BENCHMARK(BM_Normal_Activity_Input)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Fast_Normal_Activity_Input)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Scale_Activity_Input_Multiply)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Scale_Activity_Input_Logarithm)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Scale_Activity_Input_Tanh)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(large_number_iterations);
