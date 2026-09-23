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

#include "cuda/CudaConfig.h"
#include "cuda/memory/DeviceArray.h"
#include "cuda/network_graph/NetworkGPUType.h"
#include "cuda/random/RandomNumberHost.h"
#include "cuda/random/RandomNumberKeys.h"
#include "cuda/spikes/ExchangeAlgorithm.h"
#include "cuda/synaptic_elements/SynapticElementsHandle.h"
#include "neurons/NetworkGraph.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseCreationResponse.h"
#include "neurons/synaptic_elements/Dendrites.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"

#include "factory/extra_info/extra_info_factory.h"

#include <benchmark/benchmark.h>

#include <mpi-wrapper/core/MPIRank.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace {

// Every source neuron sends exactly one synapse-creation request to a distinct (non-self)
// target, and every target has several vacant dendrites so (almost) every request succeeds --
// the common per-step case, as opposed to the single-target/heavy-contention scenario already
// covered by the fairness regression test in test_exchange_algorithm.cpp.
void BM_ExchangeAlgorithm_Process_Requests(benchmark::State& state) {
    const auto number_neurons = static_cast<CudaConfig::number_neurons_type>(state.range(0));
    constexpr auto my_rank = CudaConfig::mpi_rank_type{ 0 };
    constexpr auto vacant_dendrites_per_neuron = 5.0;

    auto network_graph = std::make_shared<NetworkGraph>(mpiPP::MPIRank{ my_rank }, /*number_ranks=*/1,
                                                        /*expected_synapses_per_neuron=*/static_cast<std::size_t>(medium_number_iterations) + 20);
    network_graph->init(number_neurons, NetworkGPUType::MEMORY_POOL);

    auto h_source_ids = std::vector<CudaConfig::number_neurons_type>(number_neurons);
    auto h_target_ids = std::vector<CudaConfig::number_neurons_type>(number_neurons);
    auto h_source_ranks = std::vector<CudaConfig::mpi_rank_type>(number_neurons, my_rank);
    for (auto i = CudaConfig::number_neurons_type{ 0 }; i < number_neurons; ++i) {
        h_source_ids[i] = i;
        h_target_ids[i] = (i + 1) % number_neurons;
    }

    auto d_source_ids = DeviceArray<CudaConfig::number_neurons_type>(std::span<const CudaConfig::number_neurons_type>(h_source_ids));
    auto d_target_ids = DeviceArray<CudaConfig::number_neurons_type>(std::span<const CudaConfig::number_neurons_type>(h_target_ids));
    auto d_source_ranks = DeviceArray<CudaConfig::mpi_rank_type>(std::span<const CudaConfig::mpi_rank_type>(h_source_ranks));

    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons);
    const auto info_handle = extra_infos->get_gpu_handle();

    // Inhibitory dendrites are never dereferenced for an Excitatory request, but the handle
    // object itself must still be valid.
    auto inhibitory_dendrites = Dendrites{};
    inhibitory_dendrites.init(number_neurons);

    auto step = std::uint64_t{ 0 };

    for (auto _ : state) {
        // A fresh Dendrites instance each round -- each target's vacant-dendrite count is
        // consumed by successful requests, so reusing one across iterations would exhaust it
        // after a handful of rounds and stop exercising the actual creation path.
        state.PauseTiming();
        auto excitatory_dendrites = Dendrites{};
        excitatory_dendrites.set_grown_elements_calculator(
            [](const RelearnTypes::number_neurons_type) -> double { return vacant_dendrites_per_neuron; }, SignalType::Excitatory);
        excitatory_dendrites.init(number_neurons);

        auto h_responses = std::vector<SynapseCreationResponse>(number_neurons, SynapseCreationResponse::Failed);
        auto d_responses = DeviceArray<SynapseCreationResponse>(std::span<const SynapseCreationResponse>(h_responses));
        state.ResumeTiming();

        auto created = process_requests_entry_aware(
            SynapseCreationRequestHandle{ d_source_ids.device_ptr(), d_source_ranks.device_ptr(), d_target_ids.device_ptr(), SignalType::Excitatory, number_neurons },
            d_responses.device_ptr(),
            excitatory_dendrites.get_cuda_handle(SignalType::Excitatory), inhibitory_dendrites.get_cuda_handle(SignalType::Inhibitory),
            info_handle, network_graph->get_gpu_handle(), /*seed=*/42, step);

        state.PauseTiming();
        benchmark::DoNotOptimize(created);
        ++step;
        state.ResumeTiming();
    }
}

} // namespace

BENCHMARK(BM_ExchangeAlgorithm_Process_Requests)->Unit(benchmark::kMillisecond)->Arg(small_number_neurons)->Iterations(medium_number_iterations);

#endif
