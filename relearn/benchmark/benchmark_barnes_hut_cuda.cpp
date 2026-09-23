/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#ifdef RELEARN_CUDA_ENABLED

#include "Config.h"
#include "main.h"

#include "algorithm/BarnesHutInternalCUDA/BarnesHutCUDACell.h"
#include "algorithm/BarnesHutInternalCUDA/LinearizedTree.h"
#include "algorithm/Internal/octree/Octree.h"
#include "algorithm/Internal/octree/OctreeNodeHelper.h"
#include "cuda/CudaConfig.h"
#include "cuda/CudaTypes.h"
#include "cuda/algorithm/BarnesHutInternalCUDA/BarnesHutCUDA_CU.h"
#include "cuda/memory/DeviceArray.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "neurons/enums/SynapticElementType.h"
#include "structure/Morton.h"
#include "types/BasicTypes.h"
#include "util/BoundingBox.h"
#include "util/NeuronID.h"
#include "util/Vec3.h"

#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

constexpr auto sigma = 750.0;
constexpr auto squared_sigma_inv = 1.0 / (sigma * sigma);

RelearnTypes::bounding_box_type get_unit_box() {
    return RelearnTypes::bounding_box_type{ RelearnTypes::position_type{ 0.0, 0.0, 0.0 }, RelearnTypes::position_type{ 1.0, 1.0, 1.0 } };
}

// Builds a fully local (single-MPI-rank) octree with one leaf per neuron, all leaves holding
// `vacant_dendrites_per_neuron` vacant excitatory and inhibitory dendrites, mirroring the setup
// used in test_barnes_hut_cuda.cpp.
std::shared_ptr<Octree<BarnesHutCUDACell>> build_octree(std::size_t number_neurons, RelearnTypes::counter_type vacant_dendrites_per_neuron) {
    const auto box = get_unit_box();
    const auto& [minimum, maximum] = box;

    auto space_filling_curve = std::make_shared<Morton>(0);
    auto octree = std::make_shared<Octree<BarnesHutCUDACell>>(box, space_filling_curve, false);

    for (auto i = std::size_t{ 0 }; i < number_neurons; i++) {
        const auto position = SimulationFactory::get_random_position_in_box(minimum, maximum, _mt);
        octree->insert(position, NeuronID(i));
    }

    octree->initializes_leaf_nodes(number_neurons);

    const auto& leaf_nodes = octree->get_leaf_nodes();
    for (auto i = std::size_t{ 0 }; i < number_neurons; i++) {
        leaf_nodes[i]->set_cell_number_excitatory_dendrites(vacant_dendrites_per_neuron);
        leaf_nodes[i]->set_cell_number_inhibitory_dendrites(vacant_dendrites_per_neuron);
    }

    OctreeNodeUpdater<BarnesHutCUDACell>::update_tree(octree->get_root());

    return octree;
}

struct DeviceOctreeFixture {
    std::shared_ptr<Octree<BarnesHutCUDACell>> octree;
    LinearizedTree linearized_tree;

    LinearizedTreeDeviceStorage tree_storage{};
    NeuronPopulationDeviceStorage population_storage{};

    std::shared_ptr<StreamWrapper> stream = std::make_shared<StreamWrapper>();

    [[nodiscard]] LinearizedTreeDeviceHandle tree() const {
        return tree_storage.handle(static_cast<CudaConfig::bh_index_type>(linearized_tree.size()));
    }

    [[nodiscard]] NeuronPopulationDeviceHandle population() const {
        return population_storage.handle();
    }
};

DeviceOctreeFixture make_device_fixture(std::size_t number_neurons, RelearnTypes::counter_type vacant_dendrites_per_neuron) {
    auto octree = build_octree(number_neurons, vacant_dendrites_per_neuron);

    const auto axons = std::vector<RelearnTypes::counter_type>(number_neurons, 1U);
    auto linearized_tree = LinearizedTree(octree, std::vector(number_neurons, SignalType::Excitatory), axons);
    const auto details = linearized_tree.get_neuron_details(SignalType::Excitatory);

    auto [tree_storage, mem_usage_1] = BarnesHutCUDA_CU::init_neurons(
        linearized_tree.get_child_indices(), linearized_tree.get_parent_indices(), linearized_tree.get_subdomain_lengths(),
        linearized_tree.get_neuron_ids(), linearized_tree.get_rma_offset_to_neuron_id(), linearized_tree.get_node_types());

    auto [population_storage, mem_usage_2] = BarnesHutCUDA_CU::init_neuron_details(
        details.position, details.vacant_dendritic_elements, details.vacant_axonal_elements, details.ranks);

    auto fixture = DeviceOctreeFixture{ octree, linearized_tree };
    fixture.tree_storage = std::move(tree_storage);
    fixture.population_storage = std::move(population_storage);

    return fixture;
}

// Recomputes the bottom-up dendrite-count / position aggregation for the whole octree.
void BM_BH_CUDA_Update_Octree(benchmark::State& state) {
    const auto number_neurons = static_cast<std::size_t>(state.range(0));
    auto fixture = make_device_fixture(number_neurons, 1U);

    for (auto _ : state) {
        BarnesHutCUDA_CU::calculate_updated_octree_host(fixture.population(), fixture.tree(),
                                                        fixture.linearized_tree.get_level_indices(), fixture.stream);
    }
}

// Every leaf neuron submits one axon task, competing (via the Gaussian kernel) for dendrites
// spread uniformly across the whole population -- the common case during a simulation step.
void BM_BH_CUDA_Find_Target_Neurons(benchmark::State& state) {
    const auto number_neurons = static_cast<std::size_t>(state.range(0));
    auto fixture = make_device_fixture(number_neurons, 1U);

    BarnesHutCUDA_CU::calculate_updated_octree_host(fixture.population(), fixture.tree(),
                                                    fixture.linearized_tree.get_level_indices(), fixture.stream);

    auto mapping = std::vector<CudaConfig::bh_index_type>{};
    const auto& node_types = fixture.linearized_tree.get_node_types();
    for (auto i = 0U; i < fixture.linearized_tree.size(); ++i) {
        if (node_types[i] == NodeType::Leaf) {
            mapping.push_back(i);
        }
    }
    const DeviceArray<CudaConfig::bh_index_type> d_mapping{ std::span<const CudaConfig::bh_index_type>(mapping) };

    auto exc_stream = std::make_shared<StreamWrapper>();
    auto inh_stream = std::make_shared<StreamWrapper>();

    auto seed = std::uint64_t{ 0 };

    for (auto _ : state) {
        const auto tree = fixture.tree();
        const auto population = fixture.population();
        auto [result_exc, result_inh] = BarnesHutCUDA_CU::find_target_neurons(
            seed, 0ULL,
            population, d_mapping.device_ptr(), mapping.size(),
            population, nullptr, std::size_t{ 0 },
            static_cast<CudaConfig::number_neurons_type>(number_neurons), tree,
            RelearnTypes::as<CudaConfig::gaussian_type>(0.3), population.ranks, /* my_rank */ 0,
            RelearnTypes::as<CudaConfig::gaussian_type>(squared_sigma_inv), /* number_ranks */ 1,
            static_cast<CudaConfig::number_neurons_type>(number_neurons), exc_stream, inh_stream);

        state.PauseTiming();

        auto& [counts_exc, source_ids_exc, source_positions_exc, target_ids_exc] = result_exc;
        benchmark::DoNotOptimize(counts_exc);
        ++seed;

        state.ResumeTiming();
    }
}

} // namespace

BENCHMARK(BM_BH_CUDA_Update_Octree)->Unit(benchmark::kMillisecond)->Arg(small_number_neurons)->Iterations(medium_number_iterations);
BENCHMARK(BM_BH_CUDA_Find_Target_Neurons)->Unit(benchmark::kMillisecond)->Arg(small_number_neurons)->Iterations(medium_number_iterations);

#endif
