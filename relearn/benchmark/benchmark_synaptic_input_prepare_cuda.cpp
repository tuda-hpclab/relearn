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

#include "Config.h"
#include "cuda/CudaBaseBridgeFunctions.h"
#include "cuda/CudaConfig.h"
#include "cuda/firing/FireStatusCommunicatorGPUUncompressed.h"
#include "cuda/input/Handle.h"
#include "cuda/input/SynapticEquallyWeightedActivityInput.h"
#include "cuda/memory/DeviceArray.h"
#include "cuda/network_graph/NetworkGPUType.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "factory/extra_info/extra_info_factory.h"
#include "neurons/NetworkGraph.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "neurons/helper/RankNeuronId.h"
#include "types/SynapseTypes.h"
#include "util/NeuronID.h"

#include <benchmark/benchmark.h>

#include <mpi-wrapper/core/MPIRank.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

// benchmark_synaptic_input_cuda.cpp's distant-path benchmarks hand-construct an already-sorted
// per-rank fired-ids array, so they only ever measure launch()'s lookup kernel. That hides the
// cost FireStatusCommunicatorGPUUncompressed::commit_local_fired_status() pays every step to
// collect and, for BinarySearch, sort those ids (via prepare_spikes(), see SpikePreparation.h).
//
// This file benchmarks both costs together: commit_local_fired_status() (collect + sort-if-
// necessary) and launch() (lookup) inside the same timed iteration. The two aren't wired together
// through a real (or faked) MPI exchange -- commit_local_fired_status() sorts its own outgoing
// data (discarded afterwards, exactly like a real rank sorting what it's about to send), while
// launch() looks up against a separately, already-sorted fixture built the same way
// benchmark_synaptic_input_cuda.cpp's build_distant_incoming_graph() builds it. That is
// representative because in production a rank only ever *sorts* its own outgoing ids -- data it
// *receives* over MPI arrives already sorted (or not) by whichever remote rank sent it, so a
// receiver never re-sorts incoming ids itself.

namespace {

// Fraction of neurons treated as fired for the benchmarks below -- mirrors the constant of the
// same name in benchmark_synaptic_input_cuda.cpp.
constexpr auto fired_fraction = 0.07; // 7%

std::size_t fired_count(std::size_t total, double fraction) {
    return static_cast<std::size_t>(std::llround(static_cast<double>(total) * fraction));
}

// True for exactly `count` of the indices in [0, total), evenly spread across the range
// (a Bresenham-style rate limiter), so e.g. count=7,total=100 fires roughly every 14th index.
bool select_fired(std::size_t index, std::size_t total, std::size_t count) {
    return (index * count) / total != ((index + 1) * count) / total;
}

struct DistantFixture {
    std::shared_ptr<NetworkGraph> graph;
    std::vector<int> h_incoming_displ;
    std::vector<CudaConfig::number_neurons_type> h_incoming_ids;
};

// Builds a network where every local neuron has `synapses_per_neuron` incoming distant synapses
// *and* `synapses_per_neuron` outgoing distant synapses, both spread round-robin across
// `number_foreign_ranks` other ranks with the same split -- so commit_local_fired_status()'s
// collection kernels (which count/copy per *edge*, not deduplicated per (neuron, rank); see
// count_spikes_per_rank_kernel/fill_spikes_kernel in SpikePreparation.cu) see exactly the same
// per-rank edge counts that the lookup side below is sized from. `fraction` of the remote
// neurons that contribute an incoming edge are marked as fired for the lookup fixture, built
// exactly like benchmark_synaptic_input_cuda.cpp's build_distant_incoming_graph() so it's
// already sorted without needing a real (or faked) exchange; the outgoing edges instead feed
// commit_local_fired_status()'s own (separately fired, separately discarded) collection/sort.
DistantFixture build_distant_prepare_fixture(CudaConfig::number_neurons_type number_neurons, std::size_t synapses_per_neuron, int number_foreign_ranks) {
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

        for (auto s = std::size_t{ 0 }; s < synapses_per_neuron; ++s) {
            const auto foreign_rank = 1 + static_cast<int>(s % static_cast<std::size_t>(number_foreign_ranks));
            ng->add_synapse(PlasticDistantOutSynapse{ RankNeuronId{ mpiPP::MPIRank{ foreign_rank }, NeuronID{ 0 } }, NeuronID{ target }, 1 });
        }
    }

    ng->sync_with_gpu();
    ng->rebuild();

    auto h_incoming_displ = std::vector<int>(number_ranks_u + 1, 0);
    auto number_fired_per_rank = std::vector<std::size_t>(number_ranks_u, 0);
    for (auto rank = std::size_t{ 1 }; rank < number_ranks_u; ++rank) {
        number_fired_per_rank[rank] = fired_count(static_cast<std::size_t>(next_remote_id[rank]), fired_fraction);
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

void BM_SynapticEquallyWeighted_LaunchWithPrepare(benchmark::State& state, SpikeMode spike) {
    const auto number_neurons = static_cast<CudaConfig::number_neurons_type>(state.range(0));
    const auto synapses_per_neuron = static_cast<std::size_t>(state.range(1));
    constexpr auto number_foreign_ranks = 3;
    constexpr auto number_ranks = number_foreign_ranks + 1;
    constexpr auto my_rank = 0;

    // commit_local_fired_status() reads this global flag (not a launch()-style parameter) to
    // decide whether prepare_spikes() sorts each rank's segment -- keep it in sync with `spike` so
    // the collected ids are sorted exactly when BinarySearch needs them to be, and restore the
    // previous value so this benchmark doesn't leak its choice into whatever runs after it.
    const auto previous_do_binary_search = Config::do_binary_search;
    Config::do_binary_search = (spike == SpikeMode::BinarySearch);
    CudaConfig::local_edges_ratio = 0;
    const struct BinarySearchGuard {
        bool previous;
        ~BinarySearchGuard() { Config::do_binary_search = previous; }
    } binary_search_guard{ previous_do_binary_search };

    auto fixture = build_distant_prepare_fixture(number_neurons, synapses_per_neuron, number_foreign_ranks);
    auto& ng = fixture.graph;

    auto d_incoming_displ = DeviceArray<int>(std::span<const int>{ fixture.h_incoming_displ });
    auto d_incoming_ids = DeviceArray<CudaConfig::number_neurons_type>(std::span<const CudaConfig::number_neurons_type>{ fixture.h_incoming_ids });
    const auto fire_handle = FireStatusCommunicatorUncompressedHandle{ d_incoming_displ.device_ptr(), d_incoming_ids.device_ptr() };

    auto fcm = std::make_shared<FireStatusCommunicatorGPUUncompressed>(mpiPP::MPIRank{ my_rank }, number_ranks);
    // finalize() joins fcm's background MPI thread; std::thread's destructor calls std::terminate()
    // if still joinable, so this must run on every exit path.
    const struct FinalizeGuard {
        std::shared_ptr<FireStatusCommunicatorGPUUncompressed> fcm;
        ~FinalizeGuard() { fcm->finalize(); }
    } finalize_guard{ fcm };

    auto fsr = std::make_shared<FiredStatusRecorder>();
    fsr->init(number_neurons);

    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons);

    fcm->init(number_neurons);
    fcm->set_network_graph(ng);
    fcm->set_fired_status_recorder(fsr);
    fcm->set_extra_infos(extra_infos);

    // Evenly spread `fired_fraction` of the local population as fired, so
    // commit_local_fired_status()'s own (discarded) outgoing collection processes a realistically
    // sized set -- same scheme as benchmark_synaptic_input_cuda.cpp.
    const auto number_fired = fired_count(static_cast<std::size_t>(number_neurons), fired_fraction);
    for (auto i = CudaConfig::number_neurons_type{ 0 }; i < number_neurons; ++i) {
        if (select_fired(static_cast<std::size_t>(i), static_cast<std::size_t>(number_neurons), number_fired)) {
            fsr->set_fired(NeuronID{ i }, FiredStatus::Fired);
        }
    }

    // fill_entry<T>() (used by DeviceArray's size+value constructor) isn't explicitly instantiated
    // for CudaConfig::input_type (double), so upload an explicit zeroed host buffer instead.
    const auto h_input = std::vector<CudaConfig::input_type>(number_neurons, 0.0);
    auto d_input = DeviceArray<CudaConfig::input_type>(std::span<const CudaConfig::input_type>{ h_input });

    const auto config = LaunchConfig{ WeightMode::Weighted, spike, NetworkMode::Default, FireInformation::NeuronIDs, IterationMode::Neurons };
    const auto handles = LaunchHandles{ ng->get_gpu_handle_const(), &fire_handle };
    const auto stream = StreamWrapper::default_stream();

    auto step = std::uint32_t{ 0 };

    for (auto _ : state) {
        fcm->commit_local_fired_status(++step);
        cudaDeviceSynchronize_bridge();

        auto event = launch(CudaConfig::number_neurons_type{ 0 }, number_neurons, d_input.device_ptr(), config, handles, stream,
                            my_rank, /* local */ false, /* local_distant_helper */ false, /* synapse_conductance */ 1.0);

        cudaDeviceSynchronize_bridge();

        state.PauseTiming();
        benchmark::DoNotOptimize(event);
        state.ResumeTiming();
    }
}

} // namespace

BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_LaunchWithPrepare, Set_NeuronIDs_Neurons, SpikeMode::Set)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 10 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);
BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_LaunchWithPrepare, BinarySearch_NeuronIDs_Neurons, SpikeMode::BinarySearch)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 10 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);


BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_LaunchWithPrepare, Set_NeuronIDs_Neurons, SpikeMode::Set)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 100 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);
BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_LaunchWithPrepare, BinarySearch_NeuronIDs_Neurons, SpikeMode::BinarySearch)
    ->Unit(benchmark::kMillisecond)
    ->Args({ small_number_neurons, 100 })
    ->ArgNames({ "neurons", "synapses" })
    ->Iterations(10);

// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_LaunchWithPrepare, Set_NeuronIDs_Neurons, SpikeMode::Set)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 10 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_LaunchWithPrepare, BinarySearch_NeuronIDs_Neurons, SpikeMode::BinarySearch)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 10 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);
//
//
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_LaunchWithPrepare, Set_NeuronIDs_Neurons, SpikeMode::Set)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 100 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);
// BENCHMARK_CAPTURE(BM_SynapticEquallyWeighted_LaunchWithPrepare, BinarySearch_NeuronIDs_Neurons, SpikeMode::BinarySearch)
//     ->Unit(benchmark::kMillisecond)
//     ->Args({ 512000, 100 })
//     ->ArgNames({ "neurons", "synapses" })
//     ->Iterations(10);



#endif
