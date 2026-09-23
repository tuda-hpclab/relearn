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

#include "main.h"

#include "cuda/CudaBaseBridgeFunctions.h"
#include "cuda/CudaConfig.h"
#include "cuda/input/Handle.h"
#include "cuda/input/SynapticEquallyWeightedActivityInput.h"
#include "cuda/network_graph/NetworkGPUType.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "neurons/NetworkGraph.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/helper/RankNeuronId.h"
#include "types/SynapseTypes.h"
#include "util/NeuronID.h"

#include <benchmark/benchmark.h>

#include <mpi-wrapper/core/MPIRank.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

// launch() dispatches on 4 (SpikeMode, FireInformation) pairings that are both valid *and*
// semantically reachable through production usage (see test_synaptic_equally_weighted_launch.cpp
// for why the other two combinations are never constructed by real code):
//   - (LocalVector spike, LocalVector fire, Spikes iteration)  -- production's real local-input
//     path (update_local_input()).
//   - (Set spike,         LocalVector fire, Spikes iteration)  -- alternate local encoding; never
//     used by production but valid and otherwise unbenchmarked.
//   - (Set spike,         NeuronIDs fire,   Neurons iteration) -- production's real distant-input
//     path (update_distant_input()) when Config::do_binary_search is false.
//   - (BinarySearch spike, NeuronIDs fire,  Neurons iteration) -- same, when do_binary_search is
//     true (the default).
// All four are benchmarked below so a regression in any one dispatch path is caught.

namespace {

// Fraction of neurons/synapse-sources treated as fired for the benchmarks below.
constexpr auto fired_fraction = 0.07; // 7%

// Returns how many of `total` items should be marked as fired for a given fraction.
std::size_t fired_count(std::size_t total, double fraction) {
    return static_cast<std::size_t>(std::llround(static_cast<double>(total) * fraction));
}

// True for exactly `count` of the indices in [0, total), evenly spread across the range
// (a Bresenham-style rate limiter), so e.g. count=7,total=100 fires roughly every 14th index.
bool select_fired(std::size_t index, std::size_t total, std::size_t count) {
    return (index * count) / total != ((index + 1) * count) / total;
}

template <typename T>
struct DeviceArrayFixture {
    T* ptr{};
    std::size_t count{};

    explicit DeviceArrayFixture(const std::vector<T>& host_data)
        : count(host_data.size()) {
        cudaMalloc_bridge(reinterpret_cast<void**>(&ptr), sizeof(T) * count);
        cudaMemcpy_to_device_bridge(ptr, host_data.data(), sizeof(T) * count);
    }

    ~DeviceArrayFixture() {
        cudaFree_bridge(ptr);
    }

    DeviceArrayFixture(const DeviceArrayFixture&) = delete;
    DeviceArrayFixture& operator=(const DeviceArrayFixture&) = delete;
};

// Builds a single-rank network where every neuron has `synapses_per_neuron` incoming local
// synapses from distinct (non-self) sources, mirroring a realistic fan-in during a simulation
// step, and returns it ready for launch() (edges synced and the bloom filter rebuilt).
std::shared_ptr<NetworkGraph> build_local_outgoing_graph(CudaConfig::number_neurons_type number_neurons, std::size_t synapses_per_neuron) {
    auto ng = std::make_shared<NetworkGraph>(mpiPP::MPIRank{ 0 }, 1);
    ng->init(number_neurons, NetworkGPUType::MEMORY_POOL);

    for (auto target = CudaConfig::number_neurons_type{ 0 }; target < number_neurons; ++target) {
        for (auto s = std::size_t{ 0 }; s < synapses_per_neuron; ++s) {
            const auto source = (target + static_cast<CudaConfig::number_neurons_type>(s) + 1) % number_neurons;
            ng->add_synapse(PlasticLocalSynapse{ NeuronID{ target }, NeuronID{ source }, 1 });
        }
    }

    ng->sync_with_gpu();
    ng->rebuild();

    return ng;
}

struct DistantFixture {
    std::shared_ptr<NetworkGraph> graph;
    std::vector<int> h_incoming_displ;
    std::vector<CudaConfig::number_neurons_type> h_incoming_ids;
};

// Builds a network where every local neuron has `synapses_per_neuron` incoming distant synapses
// spread round-robin across `number_foreign_ranks` other ranks, and `fraction` of the remote
// neurons that contribute an edge are marked as fired -- the common case for the cross-rank input
// path. Each foreign rank's own counter doubles as the neuron IDs assigned to its edges, and the
// fired subset is an evenly spread selection of that range, so every rank's segment of fired IDs
// comes out contiguous and already sorted, exactly what FireStatusCommunicatorUncompressedHandle's
// per-rank (Set or BinarySearch) lookup requires.
DistantFixture build_distant_incoming_graph(CudaConfig::number_neurons_type number_neurons, std::size_t synapses_per_neuron, int number_foreign_ranks, double fraction) {
    constexpr auto my_rank = CudaConfig::mpi_rank_type{ 0 };
    const auto number_ranks = number_foreign_ranks + 1;

    auto ng = std::make_shared<NetworkGraph>(mpiPP::MPIRank{ my_rank }, number_ranks);
    ng->init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto number_ranks_u = static_cast<std::size_t>(number_ranks);
    auto next_remote_id = std::vector<CudaConfig::number_neurons_type>(number_ranks_u, 0);

    for (auto target = CudaConfig::number_neurons_type{ 0 }; target < number_neurons; ++target) {
        for (auto s = std::size_t{ 0 }; s < synapses_per_neuron; ++s) {
            const auto foreign_rank = 1 + static_cast<int>(s % static_cast<std::size_t>(number_foreign_ranks));
            const auto foreign_rank_u = static_cast<std::size_t>(foreign_rank);
            const auto remote_id = next_remote_id[foreign_rank_u]++;
            ng->add_synapse(PlasticDistantInSynapse{ NeuronID{ target }, RankNeuronId{ mpiPP::MPIRank{ foreign_rank }, NeuronID{ remote_id } }, 1 });
        }
    }

    ng->sync_with_gpu();
    ng->rebuild();

    // number_ranks + 1 entries; this rank's own segment (index 0) is always empty since a rank
    // never looks up its own fired status through the distant path.
    auto h_incoming_displ = std::vector<int>(number_ranks_u + 1, 0);
    auto number_fired_per_rank = std::vector<std::size_t>(number_ranks_u, 0);
    for (auto rank = std::size_t{ 1 }; rank < number_ranks_u; ++rank) {
        number_fired_per_rank[rank] = fired_count(static_cast<std::size_t>(next_remote_id[rank]), fraction);
        h_incoming_displ[rank + 1] = h_incoming_displ[rank] + static_cast<int>(number_fired_per_rank[rank]);
    }

    auto h_incoming_ids = std::vector<CudaConfig::number_neurons_type>(static_cast<std::size_t>(h_incoming_displ.back()));
    for (auto rank = std::size_t{ 1 }; rank < number_ranks_u; ++rank) {
        const auto total = static_cast<std::size_t>(next_remote_id[rank]);
        auto out = static_cast<std::size_t>(h_incoming_displ[rank]);
        for (auto i = CudaConfig::number_neurons_type{ 0 }; i < next_remote_id[rank]; ++i) {
            if (select_fired(static_cast<std::size_t>(i), total, number_fired_per_rank[rank])) {
                h_incoming_ids[out] = i;
                ++out;
            }
        }
    }

    return DistantFixture{ ng, std::move(h_incoming_displ), std::move(h_incoming_ids) };
}

void BM_SynapticEquallyWeighted_Launch(benchmark::State& state, SpikeMode spike, FireInformation fire, IterationMode iteration) {
    const auto number_neurons = static_cast<CudaConfig::number_neurons_type>(state.range(0));
    const auto synapses_per_neuron = static_cast<std::size_t>(state.range(1));
    constexpr auto number_foreign_ranks = 3;
    const auto local = fire == FireInformation::LocalVector;

    std::shared_ptr<NetworkGraph> ng;
    std::unique_ptr<FireStatusCommunicatorHandle> fire_handle;
    std::unique_ptr<DeviceArrayFixture<FiredStatus>> d_fired;
    std::unique_ptr<DeviceArrayFixture<int>> d_incoming_displ;
    std::unique_ptr<DeviceArrayFixture<CudaConfig::number_neurons_type>> d_incoming_ids;

    if (local) {
        ng = build_local_outgoing_graph(number_neurons, synapses_per_neuron + 2);

        // Evenly spread `fired_fraction` of the population as fired across each launch.
        const auto number_fired = fired_count(static_cast<std::size_t>(number_neurons), fired_fraction);
        auto h_fired = std::vector<FiredStatus>(number_neurons);
        for (auto i = CudaConfig::number_neurons_type{ 0 }; i < number_neurons; ++i) {
            h_fired[i] = select_fired(static_cast<std::size_t>(i), static_cast<std::size_t>(number_neurons), number_fired) ? FiredStatus::Fired : FiredStatus::Inactive;
        }
        d_fired = std::make_unique<DeviceArrayFixture<FiredStatus>>(h_fired);
        fire_handle = std::make_unique<FireStatusLocalVectorHandle>(d_fired->ptr);
    } else {
        CudaConfig::local_edges_ratio = 0;
        auto distant = build_distant_incoming_graph(number_neurons, synapses_per_neuron, number_foreign_ranks, fired_fraction);
        ng = distant.graph;

        d_incoming_displ = std::make_unique<DeviceArrayFixture<int>>(distant.h_incoming_displ);
        d_incoming_ids = std::make_unique<DeviceArrayFixture<CudaConfig::number_neurons_type>>(distant.h_incoming_ids);
        fire_handle = std::make_unique<FireStatusCommunicatorUncompressedHandle>(d_incoming_displ->ptr, d_incoming_ids->ptr);
    }

    auto h_input = std::vector<CudaConfig::input_type>(number_neurons, 0.0);
    auto d_input = DeviceArrayFixture<CudaConfig::input_type>(h_input);

    const auto config = LaunchConfig{ WeightMode::Weighted, spike, NetworkMode::Default, fire, iteration };
    const auto handles = LaunchHandles{ ng->get_gpu_handle_const(), fire_handle.get() };
    const auto stream = StreamWrapper::default_stream();

    for (auto _ : state) {
        auto event = launch(CudaConfig::number_neurons_type{ 0 }, number_neurons, d_input.ptr, config, handles, stream,
                            /* my_rank */ 0, local, /* local_distant_helper */ false, /* synapse_conductance */ 1.0);

        cudaDeviceSynchronize_bridge();

        state.PauseTiming();
        benchmark::DoNotOptimize(event);
        state.ResumeTiming();
    }
}

} // namespace

BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, LocalVector_LocalVector_Spikes, SpikeMode::LocalVector, FireInformation::LocalVector, IterationMode::Spikes)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 10 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);
BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, Set_LocalVector_Spikes, SpikeMode::Set, FireInformation::LocalVector, IterationMode::Spikes)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 10 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);
BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, LocalVector_LocalVector_Neurons, SpikeMode::LocalVector, FireInformation::LocalVector, IterationMode::Neurons)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 10 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);
BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, Set_LocalVector_Neurons, SpikeMode::Set, FireInformation::LocalVector, IterationMode::Neurons)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 10 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);
BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, Set_NeuronIDs_Neurons, SpikeMode::Set, FireInformation::NeuronIDs, IterationMode::Neurons)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 10 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);
BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, BinarySearch_NeuronIDs_Neurons, SpikeMode::BinarySearch, FireInformation::NeuronIDs, IterationMode::Neurons)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 10 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);

// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, LocalVector_LocalVector_Spikes, SpikeMode::LocalVector, FireInformation::LocalVector, IterationMode::Spikes)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 10 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, Set_LocalVector_Spikes, SpikeMode::Set, FireInformation::LocalVector, IterationMode::Spikes)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 10 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, LocalVector_LocalVector_Neurons, SpikeMode::LocalVector, FireInformation::LocalVector, IterationMode::Neurons)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 10 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, Set_LocalVector_Neurons, SpikeMode::Set, FireInformation::LocalVector, IterationMode::Neurons)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 10 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, Set_NeuronIDs_Neurons, SpikeMode::Set, FireInformation::NeuronIDs, IterationMode::Neurons)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 10 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, BinarySearch_NeuronIDs_Neurons, SpikeMode::BinarySearch, FireInformation::NeuronIDs, IterationMode::Neurons)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 10 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);







BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, LocalVector_LocalVector_Spikes, SpikeMode::LocalVector, FireInformation::LocalVector, IterationMode::Spikes)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 100 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);
BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, Set_LocalVector_Spikes, SpikeMode::Set, FireInformation::LocalVector, IterationMode::Spikes)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 100 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);
BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, LocalVector_LocalVector_Neurons, SpikeMode::LocalVector, FireInformation::LocalVector, IterationMode::Neurons)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 100 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);
BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, Set_LocalVector_Neurons, SpikeMode::Set, FireInformation::LocalVector, IterationMode::Neurons)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 100 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);
BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, Set_NeuronIDs_Neurons, SpikeMode::Set, FireInformation::NeuronIDs, IterationMode::Neurons)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 100 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);
BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, BinarySearch_NeuronIDs_Neurons, SpikeMode::BinarySearch, FireInformation::NeuronIDs, IterationMode::Neurons)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 100 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);
//
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, LocalVector_LocalVector_Spikes, SpikeMode::LocalVector, FireInformation::LocalVector, IterationMode::Spikes)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 100 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, Set_LocalVector_Spikes, SpikeMode::Set, FireInformation::LocalVector, IterationMode::Spikes)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 100 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, LocalVector_LocalVector_Neurons, SpikeMode::LocalVector, FireInformation::LocalVector, IterationMode::Neurons)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 100 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, Set_LocalVector_Neurons, SpikeMode::Set, FireInformation::LocalVector, IterationMode::Neurons)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 100 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, Set_NeuronIDs_Neurons, SpikeMode::Set, FireInformation::NeuronIDs, IterationMode::Neurons)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 100 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_Launch, BinarySearch_NeuronIDs_Neurons, SpikeMode::BinarySearch, FireInformation::NeuronIDs, IterationMode::Neurons)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 100 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);

#endif
