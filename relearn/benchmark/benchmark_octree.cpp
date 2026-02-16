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

#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/Internal/octree/OctreeNodeHelper.h"
#include "neurons/enums/SynapticElementType.h"

#include <benchmark/benchmark.h>

namespace {
void BM_Update_Octree(benchmark::State& state) {
    const auto number_nodes = static_cast<RelearnTypes::number_neurons_type>(state.range(0));

    auto root = get_octree<BarnesHutCell>(number_nodes);

    for (auto _ : state) {
        OctreeNodeUpdater<BarnesHutCell>::update_tree(&root);

        state.PauseTiming();

        const auto norm1 = root.get_cell().get_dendrites_position_for(SignalType::Excitatory).value().calculate_1_norm();
        const auto norm2 = root.get_cell().get_dendrites_position_for(SignalType::Inhibitory).value().calculate_1_norm();

        const auto num1 = root.get_cell().get_number_dendrites_for(SignalType::Excitatory);
        const auto num2 = root.get_cell().get_number_dendrites_for(SignalType::Inhibitory);

        benchmark::DoNotOptimize(norm1 + norm2 + num1 + num2);

        state.ResumeTiming();
    }
}
} // namespace

BENCHMARK(BM_Update_Octree)->Unit(benchmark::kMillisecond)->Arg(small_number_neurons)->Iterations(medium_number_iterations);
BENCHMARK(BM_Update_Octree)->Unit(benchmark::kMillisecond)->Arg(medium_number_neurons)->Iterations(medium_number_iterations);
BENCHMARK(BM_Update_Octree)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(medium_number_iterations);
