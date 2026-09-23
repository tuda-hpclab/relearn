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

#include "neurons/firing/FiredStatusCommunicationMap.h"
#include "neurons/models/NeuronModel.h"

#include "adapter/extra_info/ExtraInfoAdapter.h"

#include "factory/extra_info/extra_info_factory.h"
#include "factory/network_graph/network_graph_factory.h"

#include <benchmark/benchmark.h>

#include <memory>

namespace {

// void BM_FM(benchmark::State& state) {
//     const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));
//     const auto number_ranks = state.range(1);
//     const auto number_distant_edges_per_neuron = static_cast<std::size_t>(state.range(2));
//     const auto fire_rate = static_cast<double>(state.range(3)) / 1000.0;
//     const auto type = state.range(4);
//
//
//     auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
//     extra_info->init(number_neurons);
//
//     NeuronsExtraInfoAdapter::enable_all(extra_info);
//
//     const auto network_graphs = NetworkGraphFactory::construct_multi_rank_network_graph(number_neurons,
//                                                                                         static_cast<std::size_t>(number_ranks), 0, number_distant_edges_per_neuron, mt);
//     const auto network_graph = network_graphs[0];
//     network_graph->sync_with_gpu();
//
//     const auto fsr = std::make_shared<FiredStatusRecorder>();
//     fsr->init(number_neurons);
//
//     auto fsc = std::make_unique<FiredStatusCommunicationMap>(number_ranks);
//
//     fsc->set_extra_infos(extra_info);
//     fsc->set_network_graph(network_graph);
//     fsc->set_fired_status_recorder(fsr);
//     fsc->init(number_neurons);
//
//     for (auto _ : state) {
//         if(type == 0) {
//             fsc->commit_local_fired_status_cpu();
//         } else if(type==1) {
//             fsc->commit_local_fired_status_thrust();
//         } else if(type==2) {
//             fsc->commit_local_fired_status_cub();
//         }
//
//         state.PauseTiming();
//         for(const auto neuron_id : NeuronID::range(number_neurons)) {
//             const auto fired = RandomFactory::get_random_bool(fire_rate,mt);
//             fsr->set_fired(neuron_id, fired ? FiredStatus::Fired : FiredStatus::Inactive);
//         }
//         state.ResumeTiming();
//     }
//
// }

} // namespace

#ifndef RELEARN_CUDA_ENABLED
// BENCHMARK(BM_FM)->Unit(benchmark::kMillisecond)->ArgsProduct({{100, 10'000}, {2}, {0,10}, {70, 500}, {0}})->Iterations(10);
#else
// BENCHMARK(BM_FM)->Unit(benchmark::kMillisecond)->ArgsProduct({{100, 10'000}, {2,32}, {0,10}, {70, 500}, {0, 1, 2}})->Iterations(10);
// BENCHMARK(BM_FM)->Unit(benchmark::kMillisecond)->ArgsProduct({{100, 10'000, 100'000}, {2}, {0,10, 100, 10'000}, {70, 500}, {0, 1, 2}})->Iterations(10);
#endif
