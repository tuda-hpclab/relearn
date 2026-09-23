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

#include "adapter/network_graph/NetworkGraphAdapter.h"
#include "adapter/synapses/SynapsesAdapter.h"

#include "factory/network_graph/network_graph_factory.h"
#include "factory/synapses/synapses_factory.h"

#include <benchmark/benchmark.h>

#include <mpi-wrapper/core/MPIRank.h>

namespace {
void BM_NetworkGraph_InsertLocal(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));
    const auto number_synapses_per_neuron = static_cast<RelearnTypes::number_neurons_type>(state.range(1));

    for (auto _ : state) {
        state.PauseTiming();

        const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);
        const auto& synapses = SynapsesFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);

        state.ResumeTiming();

        NetworkGraphAdapter::add_synapses(*network_graph, synapses);
    }
}

void BM_NetworkGraph_RemoveLocal(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));
    const auto number_synapses_per_neuron = static_cast<RelearnTypes::number_neurons_type>(state.range(1));

    for (auto _ : state) {
        state.PauseTiming();

        const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);
        const auto& synapses = SynapsesFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
        NetworkGraphAdapter::add_synapses(*network_graph, synapses);

        const auto& inverted_synapses = SynapsesAdapter::invert_synapses(synapses);

        state.ResumeTiming();

        NetworkGraphAdapter::add_synapses(*network_graph, inverted_synapses);
    }
}

void BM_NetworkGraph_InsertDistantIn(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));
    const auto number_synapses_per_neuron = static_cast<RelearnTypes::number_neurons_type>(state.range(1));

    for (auto _ : state) {
        state.PauseTiming();

        const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons, mpiPP::MPIRank::root_rank(), 32);
        const auto& synapses = SynapsesFactory::generate_distant_in_synapses(number_neurons, number_synapses_per_neuron);

        state.ResumeTiming();

        NetworkGraphAdapter::add_synapses(*network_graph, synapses);
    }
}

void BM_NetworkGraph_RemoveDistantIn(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));
    const auto number_synapses_per_neuron = static_cast<RelearnTypes::number_neurons_type>(state.range(1));

    for (auto _ : state) {
        state.PauseTiming();

        const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons, mpiPP::MPIRank::root_rank(), 32);
        const auto& synapses = SynapsesFactory::generate_distant_in_synapses(number_neurons, number_synapses_per_neuron);
        NetworkGraphAdapter::add_synapses(*network_graph, synapses);

        const auto& inverted_synapses = SynapsesAdapter::invert_synapses(synapses);

        state.ResumeTiming();

        NetworkGraphAdapter::add_synapses(*network_graph, inverted_synapses);
    }
}

void BM_NetworkGraph_InsertDistantOut(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));
    const auto number_synapses_per_neuron = static_cast<RelearnTypes::number_neurons_type>(state.range(1));

    for (auto _ : state) {
        state.PauseTiming();

        const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons, mpiPP::MPIRank::root_rank(), 32);
        const auto& synapses = SynapsesFactory::generate_distant_out_synapses(number_neurons, number_synapses_per_neuron);

        state.ResumeTiming();

        NetworkGraphAdapter::add_synapses(*network_graph, synapses);
    }
}

void BM_NetworkGraph_RemoveDistantOut(benchmark::State& state) {
    const auto number_neurons = static_cast<RelearnTypes::number_neurons_type>(state.range(0));
    const auto number_synapses_per_neuron = static_cast<RelearnTypes::number_neurons_type>(state.range(1));

    for (auto _ : state) {
        state.PauseTiming();

        const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons, mpiPP::MPIRank::root_rank(), 32);

        const auto& synapses = SynapsesFactory::generate_distant_out_synapses(number_neurons, number_synapses_per_neuron);
        NetworkGraphAdapter::add_synapses(*network_graph, synapses);

        const auto& inverted_synapses = SynapsesAdapter::invert_synapses(synapses);

        state.ResumeTiming();

        NetworkGraphAdapter::add_synapses(*network_graph, inverted_synapses);
    }
}

void CustomArgsNetwork(benchmark::Benchmark* b) {
    if constexpr (excessive_testing) {
        const auto neuron_sizes = { 1000,
                                    2000,
                                    5000,
                                    10000 };

        const auto synapse_sizes = {
            10, 20, 50, 100
        };

        for (const auto neuron_size : neuron_sizes) {
            for (const auto synapse_size : synapse_sizes) {
                b->Args({ neuron_size, synapse_size });
            }
        }
    } else {
        b->Args({ 5000, 20 });
    }
}
} // namespace

BENCHMARK(BM_NetworkGraph_InsertLocal)->Unit(benchmark::kMillisecond)->Apply(CustomArgsNetwork)->Iterations(small_number_iterations);
BENCHMARK(BM_NetworkGraph_RemoveLocal)->Unit(benchmark::kMillisecond)->Apply(CustomArgsNetwork)->Iterations(small_number_iterations);

BENCHMARK(BM_NetworkGraph_InsertDistantIn)->Unit(benchmark::kMillisecond)->Apply(CustomArgsNetwork)->Iterations(small_number_iterations);
BENCHMARK(BM_NetworkGraph_RemoveDistantIn)->Unit(benchmark::kMillisecond)->Apply(CustomArgsNetwork)->Iterations(small_number_iterations);

BENCHMARK(BM_NetworkGraph_InsertDistantOut)->Unit(benchmark::kMillisecond)->Apply(CustomArgsNetwork)->Iterations(small_number_iterations);
BENCHMARK(BM_NetworkGraph_RemoveDistantOut)->Unit(benchmark::kMillisecond)->Apply(CustomArgsNetwork)->Iterations(small_number_iterations);
