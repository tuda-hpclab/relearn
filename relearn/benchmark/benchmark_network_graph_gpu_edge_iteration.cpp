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

#include "benchmark_network_graph_gpu_edge_iteration.h"

#include "main.h"

#include "cuda/CudaBaseBridgeFunctions.h"
#include "cuda/CudaConfig.h"
#include "cuda/input/Handle.h"
#include "cuda/memory/DeviceArray.h"
#include "cuda/network_graph/NetworkGraphGPU.h"
#include "cuda/network_graph/NetworkHandle.h"
#include "util/NeuronID.h"

#include <benchmark/benchmark.h>

#include <mpi-wrapper/core/MPIRank.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

using num_neurons_t = CudaConfig::number_neurons_type;

// Config 2 -- production NetworkGPUType::MEMORY_POOL shape: unweighted MemoryPool everywhere (edge count is
// expanded by |weight| instead of storing weights directly). Smart2 (a separate, simpler
// incoming-only layout) was removed from GPUEdgesBase; OnlyOutgoing/MemoryPool are the only incoming layouts now.
std::unique_ptr<NetworkGraphGPUBase> build_graph_config2(const num_neurons_t number_neurons,
                                                         const std::size_t max_edges_per_neuron) {
    auto graph = std::make_unique<NetworkGraphGPUBase>(1, 0);
    graph->init(NetworkGraphGPUParams{
        true,
        LayoutType::MemoryPool, Features{ false, false },
        LayoutType::MemoryPool, Features{ false, false },
        LayoutType::MemoryPool, Features{ false, true },
        LayoutType::MemoryPool, Features{
                                    false,
                                    true,
                                },
        max_edges_per_neuron, number_neurons });
    return graph;
}

// Config 3 -- MemoryPool local edges with the deletion flag enabled (exercises tombstoned-entry
// compaction on rebuild()).
std::unique_ptr<NetworkGraphGPUBase> build_graph_config3(const num_neurons_t number_neurons,
                                                         const std::size_t max_edges_per_neuron) {
    auto graph = std::make_unique<NetworkGraphGPUBase>(1, 0);
    graph->init(NetworkGraphGPUParams{
        true,
        LayoutType::MemoryPool, Features{ true, false },
        LayoutType::MemoryPool, Features{
                                    true,
                                    false,
                                },
        LayoutType::MemoryPool, Features{
                                    true,
                                    true,
                                },
        LayoutType::MemoryPool, Features{
                                    true,
                                    true,
                                },
        max_edges_per_neuron, number_neurons });
    return graph;
}

// Config 4 -- production NetworkGPUType::MEMORY_POOL_WEIGHTED shape: weighted MemoryPool everywhere, no
// deletion tracking. LayoutType::CSR (a separate raw-array storage backend) was removed; weighted
// and unweighted MemoryPool cover what CSR used to test.
std::unique_ptr<NetworkGraphGPUBase> build_graph_config4(const num_neurons_t number_neurons,
                                                         const std::size_t max_edges_per_neuron) {
    auto graph = std::make_unique<NetworkGraphGPUBase>(1, 0);
    graph->init(NetworkGraphGPUParams{
        true,
        LayoutType::MemoryPool, Features{
                                    true,
                                    false,
                                },
        LayoutType::MemoryPool, Features{
                                    true,
                                    false,
                                },
        LayoutType::MemoryPool, Features{
                                    true,
                                    true,
                                },
        LayoutType::MemoryPool, Features{
                                    true,
                                    true,
                                },
        max_edges_per_neuron, number_neurons });
    return graph;
}

void populate_graph(NetworkGraphGPUBase* graph, const num_neurons_t number_neurons, const std::size_t edges_per_neuron) {
    auto local_in = std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>(number_neurons);
    auto local_out = std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>(number_neurons);
    const auto distant_in = std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>(number_neurons);
    const auto distant_out = std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>(number_neurons);

    for (auto n = 0U; n < number_neurons; n++) {
        for (auto s = 0U; s < edges_per_neuron; s++) {
            const auto target = (n + s + 1) % number_neurons;
            const auto weight = (s % 2 == 0) ? 1 : -1;
            local_in[n].emplace_back(NeuronID{ target }, weight);
            local_out[n].emplace_back(NeuronID{ target }, weight);
        }
    }
    graph->update_edges(local_in, local_out, distant_in, distant_out);
    graph->rebuild();
}

void CustomArgsNetworkGraphGPU(benchmark::Benchmark* b) {
    if constexpr (excessive_testing) {
        for (const auto neurons : { 1000, 5000, 10000 }) {
            for (const auto edges : { 10, 50, 100 }) {
                b->Args({ neurons, edges });
            }
        }
    } else {
        b->Args({ 5000, 1000 });
    }
}

void run_benchmark(benchmark::State& state,
                   std::unique_ptr<NetworkGraphGPUBase>& graph,
                   [[maybe_unused]] const num_neurons_t number_neurons) {
    const auto handle = graph->get_handle();
    const DeviceArray<uint64_t> d_edge_count{ 1, static_cast<uint64_t>(0) };

    for (auto _ : state) {
        launch_kernel_iterate_edges(handle, 0, d_edge_count.device_ptr());
        cudaDeviceSynchronize_bridge();

        state.PauseTiming();
        benchmark::DoNotOptimize(d_edge_count.device_ptr());
        state.ResumeTiming();
    }
}

void BM_NetworkGraphGPU_EdgeIteration_Config2(benchmark::State& state) {
    const auto number_neurons = static_cast<num_neurons_t>(state.range(0));
    const auto edges_per_neuron = static_cast<std::size_t>(state.range(1));
    // NetworkGraphGPUBase::init() only passes local_init_size (~non_overflow_size_factor * this cap,
    // since number_ranks==1 forces local_edges_ratio to 1.0 here) to the local edge graphs' init(),
    // and that becomes GPUEdgesBase::expected_synapses_per_neuron -- the strict size-check bound
    // update_edges() bulk-loads against. So the cap must clear edges_per_neuron / non_overflow_size_factor,
    // not just edges_per_neuron, or a full (non-excessive-testing) run throws "Too many edges".
    auto graph = build_graph_config2(number_neurons, std::max(CudaConfig::expected_synapses_per_neuron,
                                                              static_cast<std::size_t>(static_cast<double>(edges_per_neuron + 16) / CudaConfig::non_overflow_size_factor) + 1));
    populate_graph(graph.get(), number_neurons, edges_per_neuron);
    run_benchmark(state, graph, number_neurons);
}

void BM_NetworkGraphGPU_EdgeIteration_Config3(benchmark::State& state) {
    const auto number_neurons = static_cast<num_neurons_t>(state.range(0));
    const auto edges_per_neuron = static_cast<std::size_t>(state.range(1));
    // NetworkGraphGPUBase::init() only passes local_init_size (~non_overflow_size_factor * this cap,
    // since number_ranks==1 forces local_edges_ratio to 1.0 here) to the local edge graphs' init(),
    // and that becomes GPUEdgesBase::expected_synapses_per_neuron -- the strict size-check bound
    // update_edges() bulk-loads against. So the cap must clear edges_per_neuron / non_overflow_size_factor,
    // not just edges_per_neuron, or a full (non-excessive-testing) run throws "Too many edges".
    auto graph = build_graph_config3(number_neurons, std::max(CudaConfig::expected_synapses_per_neuron,
                                                              static_cast<std::size_t>(static_cast<double>(edges_per_neuron + 16) / CudaConfig::non_overflow_size_factor) + 1));
    populate_graph(graph.get(), number_neurons, edges_per_neuron);
    run_benchmark(state, graph, number_neurons);
}

void BM_NetworkGraphGPU_EdgeIteration_Config4(benchmark::State& state) {
    const auto number_neurons = static_cast<num_neurons_t>(state.range(0));
    const auto edges_per_neuron = static_cast<std::size_t>(state.range(1));
    // NetworkGraphGPUBase::init() only passes local_init_size (~non_overflow_size_factor * this cap,
    // since number_ranks==1 forces local_edges_ratio to 1.0 here) to the local edge graphs' init(),
    // and that becomes GPUEdgesBase::expected_synapses_per_neuron -- the strict size-check bound
    // update_edges() bulk-loads against. So the cap must clear edges_per_neuron / non_overflow_size_factor,
    // not just edges_per_neuron, or a full (non-excessive-testing) run throws "Too many edges".
    auto graph = build_graph_config4(number_neurons, std::max(CudaConfig::expected_synapses_per_neuron,
                                                              static_cast<std::size_t>(static_cast<double>(edges_per_neuron + 16) / CudaConfig::non_overflow_size_factor) + 1));
    populate_graph(graph.get(), number_neurons, edges_per_neuron);
    run_benchmark(state, graph, number_neurons);
}

} // namespace

BENCHMARK(BM_NetworkGraphGPU_EdgeIteration_Config2)
    ->Unit(benchmark::kMicrosecond)
    ->Apply(CustomArgsNetworkGraphGPU)
    ->Iterations(small_number_iterations);

BENCHMARK(BM_NetworkGraphGPU_EdgeIteration_Config3)
    ->Unit(benchmark::kMicrosecond)
    ->Apply(CustomArgsNetworkGraphGPU)
    ->Iterations(small_number_iterations);

BENCHMARK(BM_NetworkGraphGPU_EdgeIteration_Config4)
    ->Unit(benchmark::kMicrosecond)
    ->Apply(CustomArgsNetworkGraphGPU)
    ->Iterations(small_number_iterations);

#endif // RELEARN_CUDA_ENABLED
