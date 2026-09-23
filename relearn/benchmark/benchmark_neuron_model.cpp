/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "main.h"

#include "neurons/models/NeuronModel.h"
#include "util/OMPHelper.h"

#include "adapter/extra_info/ExtraInfoAdapter.h"

#include "factory/extra_info/extra_info_factory.h"
#include "factory/network_graph/network_graph_factory.h"
#include "factory/neuron_model/neuron_model_factory.h"

#include <benchmark/benchmark.h>

#include <range/v3/numeric/accumulate.hpp>

#include <omp.h>

#include <memory>
#include <utility>

namespace {
void BM_NeuronModel_Update(benchmark::State& state, std::unique_ptr<NeuronModel> model) {
    omp_set_num_threads(static_cast<int>(state.range(1)));

#pragma omp parallel
    {
    }
    model->update_electrical_activity(101);

    for (auto _ : state) {
        model->update_activity();

        state.PauseTiming();

        const auto x = model->get_x();

        benchmark::DoNotOptimize(ranges::accumulate(x, 0.0));
        state.ResumeTiming();
    }
}

void BM_NeuronModel_Update_Benchmark(benchmark::State& state, std::unique_ptr<NeuronModel> model) {
    omp_set_num_threads(static_cast<int>(state.range(1)));

#pragma omp parallel
    {
    }

    model->update_electrical_activity_benchmark(101);

    for (auto _ : state) {
        model->update_activity_benchmark();

        state.PauseTiming();

        const auto x = model->get_x();

        benchmark::DoNotOptimize(ranges::accumulate(x, 0.0));
        state.ResumeTiming();
    }
}

void BM_NeuronModel_Update_AEIF(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    auto model = NeuronModelFactory::construct_aeif_model();
    model->init(number_neurons);
    model->set_extra_infos(extra_info);
    model->set_network_graph(network_graph);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    BM_NeuronModel_Update(state, std::move(model));
}

void BM_NeuronModel_Update_AEIF_Benchmark(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    auto model = NeuronModelFactory::construct_aeif_model();
    model->init(number_neurons);
    model->set_extra_infos(extra_info);
    model->set_network_graph(network_graph);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    BM_NeuronModel_Update_Benchmark(state, std::move(model));
}

void BM_NeuronModel_Update_FitzHughNagumo(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    auto model = NeuronModelFactory::construct_fitzhughnaguma_model();
    model->init(number_neurons);
    model->set_extra_infos(extra_info);
    model->set_network_graph(network_graph);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    BM_NeuronModel_Update(state, std::move(model));
}

void BM_NeuronModel_Update_FitzHughNagumo_Benchmark(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    auto model = NeuronModelFactory::construct_fitzhughnaguma_model();
    model->init(number_neurons);
    model->set_extra_infos(extra_info);
    model->set_network_graph(network_graph);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    BM_NeuronModel_Update_Benchmark(state, std::move(model));
}

void BM_NeuronModel_Update_Izhikevich(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    auto model = NeuronModelFactory::construct_izhikevich_model();
    model->init(number_neurons);
    model->set_extra_infos(extra_info);
    model->set_network_graph(network_graph);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    BM_NeuronModel_Update(state, std::move(model));
}

void BM_NeuronModel_Update_Izhikevich_Benchmark(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    auto model = NeuronModelFactory::construct_izhikevich_model();
    model->init(number_neurons);
    model->set_extra_infos(extra_info);
    model->set_network_graph(network_graph);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    BM_NeuronModel_Update_Benchmark(state, std::move(model));
}

void BM_NeuronModel_Update_Poisson(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    auto model = NeuronModelFactory::construct_poisson_model();
    model->init(number_neurons);
    model->set_extra_infos(extra_info);
    model->set_network_graph(network_graph);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    BM_NeuronModel_Update(state, std::move(model));
}

void BM_NeuronModel_Update_Poisson_Benchmark(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    auto model = NeuronModelFactory::construct_poisson_model();
    model->init(number_neurons);
    model->set_extra_infos(extra_info);
    model->set_network_graph(network_graph);

    NeuronsExtraInfoAdapter::enable_all(extra_info);

    BM_NeuronModel_Update_Benchmark(state, std::move(model));
}

} // namespace

BENCHMARK(BM_NeuronModel_Update_AEIF)->Unit(benchmark::kMillisecond)->ArgsProduct({ { large_number_neurons }, { 1, 2, 4, 8 } })->Iterations(50);
BENCHMARK(BM_NeuronModel_Update_AEIF_Benchmark)->Unit(benchmark::kMillisecond)->ArgsProduct({ { large_number_neurons }, { 1, 2, 4, 8 } })->Iterations(50);
BENCHMARK(BM_NeuronModel_Update_FitzHughNagumo)->Unit(benchmark::kMillisecond)->ArgsProduct({ { large_number_neurons }, { 1, 2, 4, 8 } })->Iterations(50);
BENCHMARK(BM_NeuronModel_Update_FitzHughNagumo_Benchmark)->Unit(benchmark::kMillisecond)->ArgsProduct({ { large_number_neurons }, { 1, 2, 4, 8 } })->Iterations(50);
BENCHMARK(BM_NeuronModel_Update_Izhikevich)->Unit(benchmark::kMillisecond)->ArgsProduct({ { large_number_neurons }, { 1, 2, 4, 8 } })->Iterations(50);
BENCHMARK(BM_NeuronModel_Update_Izhikevich_Benchmark)->Unit(benchmark::kMillisecond)->ArgsProduct({ { large_number_neurons }, { 1, 2, 4, 8 } })->Iterations(50);
BENCHMARK(BM_NeuronModel_Update_Poisson)->Unit(benchmark::kMillisecond)->ArgsProduct({ { large_number_neurons }, { 1, 2, 4, 8 } })->Iterations(50);
BENCHMARK(BM_NeuronModel_Update_Poisson_Benchmark)->Unit(benchmark::kMillisecond)->ArgsProduct({ { large_number_neurons }, { 1, 2, 4, 8 } })->Iterations(50);
