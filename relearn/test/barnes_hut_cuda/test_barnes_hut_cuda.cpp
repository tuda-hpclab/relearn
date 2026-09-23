/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_barnes_hut_cuda.h"
#ifdef RELEARN_CUDA_ENABLED

#include "Config.h"

#include "algorithm/Algorithm.h"
#include "algorithm/BarnesHutInternalCUDA/BarnesHutCUDA.h"
#include "algorithm/BarnesHutInternalCUDA/BarnesHutCUDACell.h"
#include "algorithm/BarnesHutInternalCUDA/LinearizedTree.h"
#include "algorithm/Internal/octree/Octree.h"
#include "algorithm/Internal/octree/OctreeNodeHelper.h"
#include "cuda/CudaConfig.h"
#include "cuda/CudaTypes.h"
#include "cuda/algorithm/BarnesHutInternalCUDA/BarnesHutCUDA_CU.h"
#include "cuda/memory/DeviceArray.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "structure/Morton.h"
#include "structure/SpaceFillingCurve.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "util/BoundingBox.h"
#include "util/NeuronID.h"
#include "util/Random.h"
#include "util/Vec3.h"

#include "factory/kernel/kernel_factory.h"
#include "factory/network_graph/network_graph_factory.h"
#include "factory/neuron_types/neuron_types_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"
#include "factory/synaptic_elements/synaptic_elements_factory.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

namespace {

/**
 * Converts a Vec3-like position (as used throughout the host code) into the plain SimpleVec3d
 * struct that the BarnesHutCUDA_CU kernels operate on.
 */
SimpleVec3d to_simple_vec3d(const RelearnTypes::position_type& position) {
    return SimpleVec3d{ static_cast<double>(position.get_x()), static_cast<double>(position.get_y()), static_cast<double>(position.get_z()) };
}

/**
 * Builds a fully local (single-MPI-rank) Octree<BarnesHutCUDACell> with one leaf per entry in
 * `positions`, with the given per-neuron excitatory/inhibitory dendrite counts, and returns the
 * ready-to-use octree (leaf cells populated, virtual nodes aggregated bottom-up).
 */
std::shared_ptr<Octree<BarnesHutCUDACell>> build_octree(
    const RelearnTypes::bounding_box_type& box,
    const std::vector<RelearnTypes::position_type>& positions,
    const std::vector<RelearnTypes::counter_type>& excitatory_dendrites,
    const std::vector<RelearnTypes::counter_type>& inhibitory_dendrites) {
    const auto number_neurons = positions.size();

    auto space_filling_curve = std::make_shared<Morton>(0);
    auto octree = std::make_shared<Octree<BarnesHutCUDACell>>(box, space_filling_curve, false);

    for (auto i = std::size_t{ 0 }; i < number_neurons; i++) {
        octree->insert(positions[i], NeuronID(i));
    }

    octree->initializes_leaf_nodes(number_neurons);

    const auto& leaf_nodes = octree->get_leaf_nodes();
    for (auto i = std::size_t{ 0 }; i < number_neurons; i++) {
        leaf_nodes[i]->set_cell_number_excitatory_dendrites(excitatory_dendrites[i]);
        leaf_nodes[i]->set_cell_number_inhibitory_dendrites(inhibitory_dendrites[i]);
    }

    OctreeNodeUpdater<BarnesHutCUDACell>::update_tree(octree->get_root());

    return octree;
}

/**
 * A small, tightly bounded box (positions in [0, 1]^3). Used for the target-selection tests
 * where the Gaussian attractiveness kernel would otherwise underflow for the huge distances
 * that SimulationFactory's default boxes produce.
 */
RelearnTypes::bounding_box_type get_unit_box() {
    return RelearnTypes::bounding_box_type{ RelearnTypes::position_type{ 0.0, 0.0, 0.0 }, RelearnTypes::position_type{ 1.0, 1.0, 1.0 } };
}

} // namespace

// ────────────────────────────────────────────────────────────────────────────
// test_acceptance_criterion_host
// ────────────────────────────────────────────────────────────────────────────

TEST_F(BarnesHutCUDATest, testACLeafDendritesAlwaysAccepted) {
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto source_position = to_simple_vec3d(SimulationFactory::get_random_position_in_box(minimum, maximum, mt));

    for (auto it = 0U; it < iterations; ++it) {
        const auto acceptance_criterion = RandomFactory::get_random_double<CudaConfig::gaussian_type>(static_cast<CudaConfig::gaussian_type>(eps), Constants::bh_max_theta, mt);
        const auto target_position = to_simple_vec3d(SimulationFactory::get_random_position_in_box(minimum, maximum, mt));

        const auto diff_vector = maximum - minimum;
        const auto target_subdomain_length = diff_vector.get_maximum();

        // Leaf targets are always accepted as long as they have vacant dendrites and are not an autapse
        const auto results = BarnesHutCUDA_CU::test_acceptance_criterion_host(1, source_position, target_position, 2, target_subdomain_length, acceptance_criterion, true);

        ASSERT_TRUE(results[0]);
    }
}

TEST_F(BarnesHutCUDATest, testACLeafDendritesSamePositionRejected) {
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto source_position = to_simple_vec3d(SimulationFactory::get_random_position_in_box(minimum, maximum, mt));

    for (auto it = 0U; it < iterations; ++it) {
        const auto acceptance_criterion = RandomFactory::get_random_double<CudaConfig::gaussian_type>(static_cast<CudaConfig::gaussian_type>(eps), Constants::bh_max_theta, mt);

        const auto diff_vector = maximum - minimum;
        const auto target_subdomain_length = diff_vector.get_maximum();

        // Source and target are the same position -> autapse -> always rejected, even for leaves
        const auto results = BarnesHutCUDA_CU::test_acceptance_criterion_host(1, source_position, source_position, 2, target_subdomain_length, acceptance_criterion, true);

        ASSERT_FALSE(results[0]);
    }
}

TEST_F(BarnesHutCUDATest, testACNoDendritesRejected) {
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    for (auto it = 0U; it < iterations; ++it) {
        const auto acceptance_criterion = RandomFactory::get_random_double<CudaConfig::gaussian_type>(static_cast<CudaConfig::gaussian_type>(eps), Constants::bh_max_theta, mt);
        const auto source_position = to_simple_vec3d(SimulationFactory::get_random_position_in_box(minimum, maximum, mt));
        const auto target_position = to_simple_vec3d(SimulationFactory::get_random_position_in_box(minimum, maximum, mt));

        const auto diff_vector = maximum - minimum;
        const auto target_subdomain_length = diff_vector.get_maximum();

        // No vacant dendrites -> rejected regardless of leaf/parent or distance
        const auto results_leaf = BarnesHutCUDA_CU::test_acceptance_criterion_host(1, source_position, target_position, 0, target_subdomain_length, acceptance_criterion, true);
        ASSERT_FALSE(results_leaf[0]);

        const auto results_parent = BarnesHutCUDA_CU::test_acceptance_criterion_host(1, source_position, target_position, 0, target_subdomain_length, acceptance_criterion, false);
        ASSERT_FALSE(results_parent[0]);
    }
}

TEST_F(BarnesHutCUDATest, testACParentDendritesMatchesFormula) {
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto source_vector = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);
    const auto source_position = to_simple_vec3d(source_vector);

    for (auto it = 0U; it < iterations; ++it) {
        const auto acceptance_criterion = RandomFactory::get_random_double<CudaConfig::gaussian_type>(static_cast<CudaConfig::gaussian_type>(eps), Constants::bh_max_theta, mt);
        const auto target_vector = SimulationFactory::get_random_position_in_box(minimum, maximum, mt);
        const auto target_position = to_simple_vec3d(target_vector);

        const auto diff_vector = maximum - minimum;
        const auto target_subdomain_length = diff_vector.get_maximum();

        const auto results = BarnesHutCUDA_CU::test_acceptance_criterion_host(1, source_position, target_position, 2, target_subdomain_length, acceptance_criterion, false);

        const auto distance = (target_vector - source_vector).calculate_2_norm();
        const auto expected = distance > 0.0 && (static_cast<double>(target_subdomain_length) < static_cast<double>(acceptance_criterion) * distance);
        ASSERT_EQ(results[0], expected);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// calculate_updated_octree_host / get_updated_octree
// ────────────────────────────────────────────────────────────────────────────

TEST_F(BarnesHutCUDATest, testUpdateOctreeThrowsOnNullptr) {
    auto linearized_tree = LinearizedTree(5);
    auto stream = std::make_shared<StreamWrapper>();

    auto [tree_storage, mem_usage_1] = BarnesHutCUDA_CU::init_neurons(
        linearized_tree.get_child_indices(), linearized_tree.get_parent_indices(), linearized_tree.get_subdomain_lengths(),
        linearized_tree.get_neuron_ids(), linearized_tree.get_rma_offset_to_neuron_id(), linearized_tree.get_node_types());
    const auto tree = tree_storage.handle(static_cast<CudaConfig::bh_index_type>(linearized_tree.size()));
    [[maybe_unused]] const auto* const d_child_begin_index = tree.child_index;
    [[maybe_unused]] const auto* const d_parent_index = tree.parent_index;
    [[maybe_unused]] const auto* const d_subdomain_length = tree.subdomain_length;
    [[maybe_unused]] const auto* const d_neuron_ids = tree.neuron_ids;
    [[maybe_unused]] const auto* const d_node_types = tree.node_types;

    const auto details = linearized_tree.get_neuron_details(SignalType::Excitatory);
    auto [population_storage, mem_usage_2] = BarnesHutCUDA_CU::init_neuron_details(
        details.position, details.vacant_dendritic_elements, details.vacant_axonal_elements, details.ranks);
    const auto population = population_storage.handle();
    [[maybe_unused]] auto* const d_positions = population.positions;
    [[maybe_unused]] auto* const d_vacant_dendrites = population.vacant_dendrites;
    [[maybe_unused]] auto* const d_vacant_axons = population.vacant_axons;
    [[maybe_unused]] auto* const d_ranks = population.ranks;

    const auto level_indices = linearized_tree.get_level_indices();

    // positions nullptr
    ASSERT_THROW(BarnesHutCUDA_CU::calculate_updated_octree_host(NeuronPopulationDeviceHandle{ nullptr, d_vacant_dendrites, d_vacant_axons, d_ranks }, tree, level_indices, stream), std::runtime_error);
    // child begin index nullptr
    ASSERT_THROW(BarnesHutCUDA_CU::calculate_updated_octree_host(population, LinearizedTreeDeviceHandle{ nullptr, d_parent_index, d_subdomain_length, d_node_types, d_neuron_ids, tree.tree_size }, level_indices, stream), std::runtime_error);
    // vacant dendrites nullptr
    ASSERT_THROW(BarnesHutCUDA_CU::calculate_updated_octree_host(NeuronPopulationDeviceHandle{ d_positions, nullptr, d_vacant_axons, d_ranks }, tree, level_indices, stream), std::runtime_error);
}

TEST_F(BarnesHutCUDATest, testUpdateOctreeNoDendritesAnywhereLeavesTreeUnchanged) {
    constexpr auto number_neurons = 40U;
    const auto box = get_unit_box();
    const auto& [minimum, maximum] = box;

    auto positions = std::vector<RelearnTypes::position_type>{};
    positions.reserve(number_neurons);
    for (auto i = 0U; i < number_neurons; i++) {
        positions.push_back(SimulationFactory::get_random_position_in_box(minimum, maximum, mt));
    }

    const auto zero_dendrites = std::vector<RelearnTypes::counter_type>(number_neurons, 0U);
    auto octree = build_octree(box, positions, zero_dendrites, zero_dendrites);

    const auto searched_signal_type = NeuronTypesFactory::get_random_signal_type(mt);
    const auto axons = std::vector<RelearnTypes::counter_type>(number_neurons, 1U);

    auto linearized_tree = LinearizedTree(octree, std::vector(number_neurons, searched_signal_type), axons);
    const auto details = linearized_tree.get_neuron_details(searched_signal_type);

    auto [tree_storage, mem_usage_1] = BarnesHutCUDA_CU::init_neurons(
        linearized_tree.get_child_indices(), linearized_tree.get_parent_indices(), linearized_tree.get_subdomain_lengths(),
        linearized_tree.get_neuron_ids(), linearized_tree.get_rma_offset_to_neuron_id(), linearized_tree.get_node_types());
    const auto tree = tree_storage.handle(static_cast<CudaConfig::bh_index_type>(linearized_tree.size()));
    [[maybe_unused]] const auto* const d_child_begin_index = tree.child_index;
    [[maybe_unused]] const auto* const d_parent_index = tree.parent_index;
    [[maybe_unused]] const auto* const d_subdomain_length = tree.subdomain_length;
    [[maybe_unused]] const auto* const d_neuron_ids = tree.neuron_ids;
    [[maybe_unused]] const auto* const d_node_types = tree.node_types;

    auto [population_storage, mem_usage_2] = BarnesHutCUDA_CU::init_neuron_details(
        details.position, details.vacant_dendritic_elements, details.vacant_axonal_elements, details.ranks);
    const auto population = population_storage.handle();
    [[maybe_unused]] auto* const d_positions = population.positions;
    [[maybe_unused]] auto* const d_vacant_dendrites = population.vacant_dendrites;
    [[maybe_unused]] auto* const d_vacant_axons = population.vacant_axons;
    [[maybe_unused]] auto* const d_ranks = population.ranks;

    auto stream = std::make_shared<StreamWrapper>();
    [[maybe_unused]] const auto tree_size = static_cast<CudaConfig::number_neurons_type>(linearized_tree.size());

    BarnesHutCUDA_CU::calculate_updated_octree_host(population, tree, linearized_tree.get_level_indices(), stream);

    auto updated_positions = std::vector<SimpleVec3d>{};
    auto updated_vacant_dendrites = std::vector<CudaConfig::synaptic_count_type>{};
    BarnesHutCUDA_CU::get_updated_octree(population, linearized_tree.size(), updated_positions, updated_vacant_dendrites);

    // if there are no dendrites anywhere, the roll-up may not change any value
    for (auto i = 0U; i < linearized_tree.size(); ++i) {
        ASSERT_EQ(updated_vacant_dendrites[i], details.vacant_dendritic_elements[i]);

        ASSERT_DOUBLE_EQ(updated_positions[i].x, details.position[i].x);
        ASSERT_DOUBLE_EQ(updated_positions[i].y, details.position[i].y);
        ASSERT_DOUBLE_EQ(updated_positions[i].z, details.position[i].z);
    }
}

TEST_F(BarnesHutCUDATest, testUpdateOctreeFullTreeAggregatesBottomUp) {
    constexpr auto number_neurons = 100U;
    const auto box = get_unit_box();
    const auto& [minimum, maximum] = box;

    auto positions = std::vector<RelearnTypes::position_type>{};
    positions.reserve(number_neurons);
    for (auto i = 0U; i < number_neurons; i++) {
        positions.push_back(SimulationFactory::get_random_position_in_box(minimum, maximum, mt));
    }

    // every leaf has exactly 1 vacant excitatory dendrite and 0 inhibitory ones
    const auto exc_dendrites = std::vector<RelearnTypes::counter_type>(number_neurons, 1U);
    const auto inh_dendrites = std::vector<RelearnTypes::counter_type>(number_neurons, 0U);
    auto octree = build_octree(box, positions, exc_dendrites, inh_dendrites);

    const auto axons = std::vector<RelearnTypes::counter_type>(number_neurons, 0U);
    auto linearized_tree = LinearizedTree(octree, std::vector(number_neurons, SignalType::Excitatory), axons);
    const auto details = linearized_tree.get_neuron_details(SignalType::Excitatory);

    auto [tree_storage, mem_usage_1] = BarnesHutCUDA_CU::init_neurons(
        linearized_tree.get_child_indices(), linearized_tree.get_parent_indices(), linearized_tree.get_subdomain_lengths(),
        linearized_tree.get_neuron_ids(), linearized_tree.get_rma_offset_to_neuron_id(), linearized_tree.get_node_types());
    const auto tree = tree_storage.handle(static_cast<CudaConfig::bh_index_type>(linearized_tree.size()));
    [[maybe_unused]] const auto* const d_child_begin_index = tree.child_index;
    [[maybe_unused]] const auto* const d_parent_index = tree.parent_index;
    [[maybe_unused]] const auto* const d_subdomain_length = tree.subdomain_length;
    [[maybe_unused]] const auto* const d_neuron_ids = tree.neuron_ids;
    [[maybe_unused]] const auto* const d_node_types = tree.node_types;

    auto [population_storage, mem_usage_2] = BarnesHutCUDA_CU::init_neuron_details(
        details.position, details.vacant_dendritic_elements, details.vacant_axonal_elements, details.ranks);
    const auto population = population_storage.handle();
    [[maybe_unused]] auto* const d_positions = population.positions;
    [[maybe_unused]] auto* const d_vacant_dendrites = population.vacant_dendrites;
    [[maybe_unused]] auto* const d_vacant_axons = population.vacant_axons;
    [[maybe_unused]] auto* const d_ranks = population.ranks;

    auto stream = std::make_shared<StreamWrapper>();
    [[maybe_unused]] const auto tree_size = static_cast<CudaConfig::number_neurons_type>(linearized_tree.size());

    BarnesHutCUDA_CU::calculate_updated_octree_host(population, tree, linearized_tree.get_level_indices(), stream);

    auto updated_positions = std::vector<SimpleVec3d>{};
    auto updated_vacant_dendrites = std::vector<CudaConfig::synaptic_count_type>{};
    BarnesHutCUDA_CU::get_updated_octree(population, linearized_tree.size(), updated_positions, updated_vacant_dendrites);

    const auto& node_types = linearized_tree.get_node_types();
    const auto& child_indices = linearized_tree.get_child_indices();

    auto expected_vacant_dendrites = std::vector<CudaConfig::synaptic_count_type>(details.vacant_dendritic_elements.begin(), details.vacant_dendritic_elements.end());

    for (auto i = static_cast<std::int64_t>(linearized_tree.size()) - 1; i >= 0; --i) {
        const auto idx = static_cast<std::size_t>(i);
        const auto type = node_types[idx];

        if (type == NodeType::Leaf) {
            ASSERT_EQ(updated_vacant_dendrites[idx], 1U);
        } else if (type == NodeType::Placeholder) {
            ASSERT_EQ(updated_vacant_dendrites[idx], 0U);
        } else if (type == NodeType::VirtualNode) {
            const auto child_begin = child_indices[idx];
            auto dendrites_sum = CudaConfig::synaptic_count_type{ 0 };
            for (auto j = 0U; j < 8U; ++j) {
                dendrites_sum += expected_vacant_dendrites[child_begin + j];
            }
            expected_vacant_dendrites[idx] = dendrites_sum;
            ASSERT_EQ(updated_vacant_dendrites[idx], dendrites_sum);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// find_target_neurons
// ────────────────────────────────────────────────────────────────────────────

TEST_F(BarnesHutCUDATest, testFindTargetNeuronsThrowsOnInvalidInput) {
    auto linearized_tree = LinearizedTree(5);
    auto exc_stream = std::make_shared<StreamWrapper>();
    auto inh_stream = std::make_shared<StreamWrapper>();

    auto [tree_storage, mem_usage_1] = BarnesHutCUDA_CU::init_neurons(
        linearized_tree.get_child_indices(), linearized_tree.get_parent_indices(), linearized_tree.get_subdomain_lengths(),
        linearized_tree.get_neuron_ids(), linearized_tree.get_rma_offset_to_neuron_id(), linearized_tree.get_node_types());
    const auto tree = tree_storage.handle(static_cast<CudaConfig::bh_index_type>(linearized_tree.size()));
    [[maybe_unused]] const auto* const d_child_begin_index = tree.child_index;
    [[maybe_unused]] const auto* const d_parent_index = tree.parent_index;
    [[maybe_unused]] const auto* const d_subdomain_length = tree.subdomain_length;
    [[maybe_unused]] const auto* const d_neuron_ids = tree.neuron_ids;
    [[maybe_unused]] const auto* const d_node_types = tree.node_types;

    const auto details = linearized_tree.get_neuron_details(SignalType::Excitatory);
    auto [population_storage, mem_usage_2] = BarnesHutCUDA_CU::init_neuron_details(
        details.position, details.vacant_dendritic_elements, details.vacant_axonal_elements, details.ranks);
    const auto population = population_storage.handle();
    [[maybe_unused]] auto* const d_positions = population.positions;
    [[maybe_unused]] auto* const d_vacant_dendrites = population.vacant_dendrites;
    [[maybe_unused]] auto* const d_vacant_axons = population.vacant_axons;
    [[maybe_unused]] auto* const d_ranks = population.ranks;

    const auto tree_size = static_cast<CudaConfig::bh_index_type>(linearized_tree.size());

    const auto call = [&](SimpleVec3d* positions, const CudaConfig::bh_index_type* child_index, const CudaConfig::bh_index_type* parent_index,
                          const CudaConfig::gaussian_type* subdomain_length, CudaConfig::synaptic_count_type* vacant_dendrites,
                          const CudaConfig::gaussian_type acceptance_criterion) {
        const auto population_arg = NeuronPopulationDeviceHandle{ positions, vacant_dendrites, nullptr, nullptr };
        const auto tree_arg = LinearizedTreeDeviceHandle{ child_index, parent_index, subdomain_length, d_node_types, d_neuron_ids, tree_size };
        return BarnesHutCUDA_CU::find_target_neurons(
            0ULL, 0ULL,
            population_arg, nullptr, std::size_t{ 0 },
            population_arg, nullptr, std::size_t{ 0 },
            tree_size, tree_arg,
            acceptance_criterion, d_ranks, /* my_rank */ 0,
            /* squared_sigma_inv */ 1.0, /* number_ranks */ 1, tree_size, exc_stream, inh_stream);
    };

    // positions array null
    ASSERT_THROW(call(nullptr, d_child_begin_index, d_parent_index, d_subdomain_length, d_vacant_dendrites, Constants::bh_default_theta), std::runtime_error);
    // child idx array null
    ASSERT_THROW(call(d_positions, nullptr, d_parent_index, d_subdomain_length, d_vacant_dendrites, Constants::bh_default_theta), std::runtime_error);
    // parent idx array null
    ASSERT_THROW(call(d_positions, d_child_begin_index, nullptr, d_subdomain_length, d_vacant_dendrites, Constants::bh_default_theta), std::runtime_error);
    // subdomain length array null
    ASSERT_THROW(call(d_positions, d_child_begin_index, d_parent_index, nullptr, d_vacant_dendrites, Constants::bh_default_theta), std::runtime_error);
    // vacant dendrites array null
    ASSERT_THROW(call(d_positions, d_child_begin_index, d_parent_index, d_subdomain_length, nullptr, Constants::bh_default_theta), std::runtime_error);
    // acceptance criterion negative
    ASSERT_THROW(call(d_positions, d_child_begin_index, d_parent_index, d_subdomain_length, d_vacant_dendrites, RelearnTypes::as<CudaConfig::gaussian_type>(-0.1)), std::runtime_error);
    // acceptance criterion too large
    ASSERT_THROW(call(d_positions, d_child_begin_index, d_parent_index, d_subdomain_length, d_vacant_dendrites, RelearnTypes::as<CudaConfig::gaussian_type>(static_cast<double>(Constants::bh_max_theta) + 0.1)), std::runtime_error);
}

TEST_F(BarnesHutCUDATest, testFindTargetNeuronsNoDendritesAnywhereFindsNoTargets) {
    constexpr auto number_neurons = 24U;
    const auto box = get_unit_box();
    const auto& [minimum, maximum] = box;

    auto positions = std::vector<RelearnTypes::position_type>{};
    positions.reserve(number_neurons);
    for (auto i = 0U; i < number_neurons; i++) {
        positions.push_back(SimulationFactory::get_random_position_in_box(minimum, maximum, mt));
    }

    const auto zero_dendrites = std::vector<RelearnTypes::counter_type>(number_neurons, 0U);
    auto octree = build_octree(box, positions, zero_dendrites, zero_dendrites);

    const auto axons = std::vector<RelearnTypes::counter_type>(number_neurons, 1U);
    auto linearized_tree = LinearizedTree(octree, std::vector(number_neurons, SignalType::Excitatory), axons);
    const auto details = linearized_tree.get_neuron_details(SignalType::Excitatory);

    auto [tree_storage, mem_usage_1] = BarnesHutCUDA_CU::init_neurons(
        linearized_tree.get_child_indices(), linearized_tree.get_parent_indices(), linearized_tree.get_subdomain_lengths(),
        linearized_tree.get_neuron_ids(), linearized_tree.get_rma_offset_to_neuron_id(), linearized_tree.get_node_types());
    const auto tree = tree_storage.handle(static_cast<CudaConfig::bh_index_type>(linearized_tree.size()));
    [[maybe_unused]] const auto* const d_child_begin_index = tree.child_index;
    [[maybe_unused]] const auto* const d_parent_index = tree.parent_index;
    [[maybe_unused]] const auto* const d_subdomain_length = tree.subdomain_length;
    [[maybe_unused]] const auto* const d_neuron_ids = tree.neuron_ids;
    [[maybe_unused]] const auto* const d_node_types = tree.node_types;

    auto [population_storage, mem_usage_2] = BarnesHutCUDA_CU::init_neuron_details(
        details.position, details.vacant_dendritic_elements, details.vacant_axonal_elements, details.ranks);
    const auto population = population_storage.handle();
    [[maybe_unused]] auto* const d_positions = population.positions;
    [[maybe_unused]] auto* const d_vacant_dendrites = population.vacant_dendrites;
    [[maybe_unused]] auto* const d_vacant_axons = population.vacant_axons;
    [[maybe_unused]] auto* const d_ranks = population.ranks;

    auto exc_stream = std::make_shared<StreamWrapper>();
    auto inh_stream = std::make_shared<StreamWrapper>();
    [[maybe_unused]] const auto tree_size = static_cast<CudaConfig::number_neurons_type>(linearized_tree.size());

    BarnesHutCUDA_CU::calculate_updated_octree_host(population, tree, linearized_tree.get_level_indices(), exc_stream);

    // every leaf neuron sends exactly one axon
    auto mapping = std::vector<CudaConfig::bh_index_type>{};
    const auto& node_types = linearized_tree.get_node_types();
    for (auto i = 0U; i < linearized_tree.size(); ++i) {
        if (node_types[i] == NodeType::Leaf) {
            mapping.push_back(i);
        }
    }
    const DeviceArray<CudaConfig::bh_index_type> d_mapping{ std::span<const CudaConfig::bh_index_type>(mapping) };

    auto [result_exc, result_inh] = BarnesHutCUDA_CU::find_target_neurons(
        42ULL, 0ULL,
        population, d_mapping.device_ptr(), mapping.size(),
        population, nullptr, std::size_t{ 0 },
        static_cast<CudaConfig::number_neurons_type>(number_neurons), tree,
        static_cast<CudaConfig::gaussian_type>(eps), d_ranks, /* my_rank */ 0,
        /* squared_sigma_inv */ 0.5, /* number_ranks */ 1, static_cast<CudaConfig::number_neurons_type>(number_neurons), exc_stream, inh_stream);

    const auto& [counts_exc, source_ids_exc, source_positions_exc, target_ids_exc] = result_exc;
    const auto& [counts_inh, source_ids_inh, source_positions_inh, target_ids_inh] = result_inh;

    const auto total_found_exc = std::accumulate(counts_exc.begin(), counts_exc.end(), 0);
    const auto total_found_inh = std::accumulate(counts_inh.begin(), counts_inh.end(), 0);

    ASSERT_EQ(total_found_exc, 0);
    ASSERT_EQ(total_found_inh, 0);
}

TEST_F(BarnesHutCUDATest, testFindTargetNeuronsAvoidsAutapse) {
    constexpr auto number_neurons = 24U;
    const auto box = get_unit_box();
    const auto& [minimum, maximum] = box;

    auto positions = std::vector<RelearnTypes::position_type>{};
    positions.reserve(number_neurons);
    for (auto i = 0U; i < number_neurons; i++) {
        positions.push_back(SimulationFactory::get_random_position_in_box(minimum, maximum, mt));
    }

    // every leaf has vacant dendrites of both kinds -> every neuron is a viable target
    const auto dendrites = std::vector<RelearnTypes::counter_type>(number_neurons, 1U);
    auto octree = build_octree(box, positions, dendrites, dendrites);

    const auto axons = std::vector<RelearnTypes::counter_type>(number_neurons, 1U);
    auto linearized_tree = LinearizedTree(octree, std::vector(number_neurons, SignalType::Excitatory), axons);
    const auto details = linearized_tree.get_neuron_details(SignalType::Excitatory);

    auto [tree_storage, mem_usage_1] = BarnesHutCUDA_CU::init_neurons(
        linearized_tree.get_child_indices(), linearized_tree.get_parent_indices(), linearized_tree.get_subdomain_lengths(),
        linearized_tree.get_neuron_ids(), linearized_tree.get_rma_offset_to_neuron_id(), linearized_tree.get_node_types());
    const auto tree = tree_storage.handle(static_cast<CudaConfig::bh_index_type>(linearized_tree.size()));
    [[maybe_unused]] const auto* const d_child_begin_index = tree.child_index;
    [[maybe_unused]] const auto* const d_parent_index = tree.parent_index;
    [[maybe_unused]] const auto* const d_subdomain_length = tree.subdomain_length;
    [[maybe_unused]] const auto* const d_neuron_ids = tree.neuron_ids;
    [[maybe_unused]] const auto* const d_node_types = tree.node_types;

    auto [population_storage, mem_usage_2] = BarnesHutCUDA_CU::init_neuron_details(
        details.position, details.vacant_dendritic_elements, details.vacant_axonal_elements, details.ranks);
    const auto population = population_storage.handle();
    [[maybe_unused]] auto* const d_positions = population.positions;
    [[maybe_unused]] auto* const d_vacant_dendrites = population.vacant_dendrites;
    [[maybe_unused]] auto* const d_vacant_axons = population.vacant_axons;
    [[maybe_unused]] auto* const d_ranks = population.ranks;

    auto exc_stream = std::make_shared<StreamWrapper>();
    auto inh_stream = std::make_shared<StreamWrapper>();
    [[maybe_unused]] const auto tree_size = static_cast<CudaConfig::number_neurons_type>(linearized_tree.size());

    BarnesHutCUDA_CU::calculate_updated_octree_host(population, tree, linearized_tree.get_level_indices(), exc_stream);

    auto mapping = std::vector<CudaConfig::bh_index_type>{};
    const auto& node_types = linearized_tree.get_node_types();
    for (auto i = 0U; i < linearized_tree.size(); ++i) {
        if (node_types[i] == NodeType::Leaf) {
            mapping.push_back(i);
        }
    }
    const DeviceArray<CudaConfig::bh_index_type> d_mapping{ std::span<const CudaConfig::bh_index_type>(mapping) };

    auto [result_exc, result_inh] = BarnesHutCUDA_CU::find_target_neurons(
        123ULL, 0ULL,
        population, d_mapping.device_ptr(), mapping.size(),
        population, nullptr, std::size_t{ 0 },
        static_cast<CudaConfig::number_neurons_type>(number_neurons), tree,
        static_cast<CudaConfig::gaussian_type>(eps), d_ranks, /* my_rank */ 0,
        /* squared_sigma_inv */ 0.5, /* number_ranks */ 1, static_cast<CudaConfig::number_neurons_type>(number_neurons), exc_stream, inh_stream);

    const auto& [counts_exc, source_ids_exc, source_positions_exc, target_ids_exc] = result_exc;
    const auto& [counts_inh, source_ids_inh, source_positions_inh, target_ids_inh] = result_inh;

    ASSERT_EQ(counts_exc.size(), 1U);
    ASSERT_EQ(counts_inh.size(), 1U);
    // no inhibitory axon tasks were submitted
    ASSERT_EQ(counts_inh[0], 0);

    const auto total_found = static_cast<std::size_t>(counts_exc[0]);
    // sanity check that the pipeline actually found something with these (generous) parameters
    ASSERT_GT(total_found, 0U);

    const auto host_source_ids = source_ids_exc.get_device_data();
    const auto host_target_ids = target_ids_exc.get_device_data();

    for (auto i = 0U; i < total_found; ++i) {
        ASSERT_NE(host_source_ids[i], host_target_ids[i]);
    }
}

TEST_F(BarnesHutCUDATest, testFindTargetNeuronsRoutesFoundTargetsToTheirOwningRank) {
    constexpr auto number_neurons = 16U;
    const auto box = get_unit_box();
    const auto& [minimum, maximum] = box;

    auto positions = std::vector<RelearnTypes::position_type>{};
    positions.reserve(number_neurons);
    for (auto i = 0U; i < number_neurons; i++) {
        positions.push_back(SimulationFactory::get_random_position_in_box(minimum, maximum, mt));
    }

    // Every neuron has a single vacant dendrite, except for one "foreign" neuron which
    // dominates the attractiveness so that it is (almost certainly) picked at least once.
    auto dendrites = std::vector<RelearnTypes::counter_type>(number_neurons, 1U);
    constexpr auto foreign_neuron_id = CudaConfig::number_neurons_type{ 0 };
    dendrites[foreign_neuron_id] = 1000U;

    auto octree = build_octree(box, positions, dendrites, dendrites);

    const auto axons = std::vector<RelearnTypes::counter_type>(number_neurons, 1U);
    auto linearized_tree = LinearizedTree(octree, std::vector(number_neurons, SignalType::Excitatory), axons);
    const auto details = linearized_tree.get_neuron_details(SignalType::Excitatory);

    auto [tree_storage, mem_usage_1] = BarnesHutCUDA_CU::init_neurons(
        linearized_tree.get_child_indices(), linearized_tree.get_parent_indices(), linearized_tree.get_subdomain_lengths(),
        linearized_tree.get_neuron_ids(), linearized_tree.get_rma_offset_to_neuron_id(), linearized_tree.get_node_types());
    const auto tree = tree_storage.handle(static_cast<CudaConfig::bh_index_type>(linearized_tree.size()));
    [[maybe_unused]] const auto* const d_child_begin_index = tree.child_index;
    [[maybe_unused]] const auto* const d_parent_index = tree.parent_index;
    [[maybe_unused]] const auto* const d_subdomain_length = tree.subdomain_length;
    [[maybe_unused]] const auto* const d_neuron_ids = tree.neuron_ids;
    [[maybe_unused]] const auto* const d_node_types = tree.node_types;

    auto [population_storage, mem_usage_2] = BarnesHutCUDA_CU::init_neuron_details(
        details.position, details.vacant_dendritic_elements, details.vacant_axonal_elements, details.ranks);
    const auto population = population_storage.handle();
    [[maybe_unused]] auto* const d_positions = population.positions;
    [[maybe_unused]] auto* const d_vacant_dendrites = population.vacant_dendrites;
    [[maybe_unused]] auto* const d_vacant_axons = population.vacant_axons;

    // Locate the linearized-tree index of the leaf that represents `foreign_neuron_id`, and
    // build a rank array where only that entry belongs to a different (foreign) MPI rank.
    const auto& node_types = linearized_tree.get_node_types();
    const auto& neuron_ids = linearized_tree.get_neuron_ids();

    auto foreign_index = std::optional<std::size_t>{};
    for (auto i = 0U; i < linearized_tree.size(); ++i) {
        if (node_types[i] == NodeType::Leaf && neuron_ids[i] == foreign_neuron_id) {
            foreign_index = i;
            break;
        }
    }
    ASSERT_TRUE(foreign_index.has_value());

    constexpr CudaConfig::mpi_rank_type my_rank = 0;
    constexpr CudaConfig::mpi_rank_type foreign_rank = 1;
    constexpr CudaConfig::mpi_rank_type number_ranks = 2;

    auto host_ranks = std::vector<CudaConfig::mpi_rank_type>(details.ranks.begin(), details.ranks.end());
    host_ranks[*foreign_index] = foreign_rank;
    const DeviceArray<CudaConfig::mpi_rank_type> d_ranks{ std::span<const CudaConfig::mpi_rank_type>(host_ranks) };

    auto exc_stream = std::make_shared<StreamWrapper>();
    auto inh_stream = std::make_shared<StreamWrapper>();
    [[maybe_unused]] const auto tree_size = static_cast<CudaConfig::number_neurons_type>(linearized_tree.size());

    BarnesHutCUDA_CU::calculate_updated_octree_host(population, tree, linearized_tree.get_level_indices(), exc_stream);

    auto mapping = std::vector<CudaConfig::bh_index_type>{};
    for (auto i = 0U; i < linearized_tree.size(); ++i) {
        if (node_types[i] == NodeType::Leaf) {
            mapping.push_back(i);
        }
    }
    const DeviceArray<CudaConfig::bh_index_type> d_mapping{ std::span<const CudaConfig::bh_index_type>(mapping) };

    auto [result_exc, result_inh] = BarnesHutCUDA_CU::find_target_neurons(
        777ULL, 0ULL,
        population, d_mapping.device_ptr(), mapping.size(),
        population, nullptr, std::size_t{ 0 },
        static_cast<CudaConfig::number_neurons_type>(number_neurons), tree,
        static_cast<CudaConfig::gaussian_type>(eps), d_ranks.device_ptr(), my_rank,
        /* squared_sigma_inv */ 0.5, number_ranks, static_cast<CudaConfig::number_neurons_type>(number_neurons), exc_stream, inh_stream);

    const auto& [counts_exc, source_ids_exc, source_positions_exc, target_ids_exc] = result_exc;

    ASSERT_EQ(counts_exc.size(), static_cast<std::size_t>(number_ranks));

    // At least one source neuron should have picked the (overwhelmingly attractive) foreign neuron
    ASSERT_GT(counts_exc[foreign_rank], 0);

    const auto host_target_ids = target_ids_exc.get_device_data();
    const auto offset = static_cast<std::size_t>(counts_exc[my_rank]);
    const auto foreign_count = static_cast<std::size_t>(counts_exc[foreign_rank]);

    // Every entry routed into the foreign-rank bucket must refer to the foreign neuron
    for (auto i = offset; i < offset + foreign_count; ++i) {
        ASSERT_EQ(host_target_ids[i], foreign_neuron_id);
    }
}

// Regression test for a nondeterminism bug in find_target_neurons's
// compaction step: the per-rank output arrays used to be filled via an atomicAdd whose slot
// order varies with GPU thread scheduling, so two calls with the identical seed/step could
// produce the identical *set* of (source, target) picks in a different *order*. The fix sorts
// each rank's segment into a canonical (source, target) order after the kernel, so the exact
// same seed/step must now reproduce byte-identical output order too. Uses the same "one
// overwhelmingly attractive neuron" setup as testFindTargetNeuronsRoutesFoundTargetsToTheirOwningRank
// so many source neurons genuinely contend for the same compaction slots, which is exactly the
// scenario the old atomicAdd ordering was sensitive to.
TEST_F(BarnesHutCUDATest, testFindTargetNeuronsIsReproducibleGivenSameSeedAndStep) {
    constexpr auto number_neurons = 16U;
    const auto box = get_unit_box();
    const auto& [minimum, maximum] = box;

    auto positions = std::vector<RelearnTypes::position_type>{};
    positions.reserve(number_neurons);
    for (auto i = 0U; i < number_neurons; i++) {
        positions.push_back(SimulationFactory::get_random_position_in_box(minimum, maximum, mt));
    }

    // Every neuron has a single vacant dendrite, except for one neuron which dominates the
    // attractiveness so that (almost) every source neuron's random walk converges on it --
    // creating genuine contention for the same compaction slots.
    auto dendrites = std::vector<RelearnTypes::counter_type>(number_neurons, 1U);
    constexpr auto popular_neuron_id = CudaConfig::number_neurons_type{ 0 };
    dendrites[popular_neuron_id] = 1000U;

    auto octree = build_octree(box, positions, dendrites, dendrites);

    const auto axons = std::vector<RelearnTypes::counter_type>(number_neurons, 1U);
    auto linearized_tree = LinearizedTree(octree, std::vector(number_neurons, SignalType::Excitatory), axons);
    const auto details = linearized_tree.get_neuron_details(SignalType::Excitatory);

    auto [tree_storage, mem_usage_1] = BarnesHutCUDA_CU::init_neurons(
        linearized_tree.get_child_indices(), linearized_tree.get_parent_indices(), linearized_tree.get_subdomain_lengths(),
        linearized_tree.get_neuron_ids(), linearized_tree.get_rma_offset_to_neuron_id(), linearized_tree.get_node_types());
    const auto tree = tree_storage.handle(static_cast<CudaConfig::bh_index_type>(linearized_tree.size()));
    [[maybe_unused]] const auto* const d_child_begin_index = tree.child_index;
    [[maybe_unused]] const auto* const d_parent_index = tree.parent_index;
    [[maybe_unused]] const auto* const d_subdomain_length = tree.subdomain_length;
    [[maybe_unused]] const auto* const d_neuron_ids = tree.neuron_ids;
    [[maybe_unused]] const auto* const d_node_types = tree.node_types;

    auto [population_storage, mem_usage_2] = BarnesHutCUDA_CU::init_neuron_details(
        details.position, details.vacant_dendritic_elements, details.vacant_axonal_elements, details.ranks);
    const auto population = population_storage.handle();
    [[maybe_unused]] auto* const d_positions = population.positions;
    [[maybe_unused]] auto* const d_vacant_dendrites = population.vacant_dendrites;
    [[maybe_unused]] auto* const d_vacant_axons = population.vacant_axons;
    [[maybe_unused]] auto* const d_ranks = population.ranks;

    constexpr CudaConfig::mpi_rank_type my_rank = 0;
    constexpr CudaConfig::mpi_rank_type number_ranks = 1;
    [[maybe_unused]] const auto tree_size = static_cast<CudaConfig::number_neurons_type>(linearized_tree.size());

    auto exc_stream = std::make_shared<StreamWrapper>();
    auto inh_stream = std::make_shared<StreamWrapper>();
    BarnesHutCUDA_CU::calculate_updated_octree_host(population, tree, linearized_tree.get_level_indices(), exc_stream);

    auto mapping = std::vector<CudaConfig::bh_index_type>{};
    for (auto i = 0U; i < linearized_tree.size(); ++i) {
        if (linearized_tree.get_node_types()[i] == NodeType::Leaf) {
            mapping.push_back(i);
        }
    }
    const DeviceArray<CudaConfig::bh_index_type> d_mapping{ std::span<const CudaConfig::bh_index_type>(mapping) };

    constexpr auto seed = 777ULL;
    constexpr auto step = 0ULL;

    // find_target_neurons only reads the octree/dendrite state built
    // above -- calling it twice against the same, unmodified state with the same seed/step is
    // exactly the "repeat the same round" scenario the fix needs to be reproducible for.
    auto [result_exc_1, result_inh_1] = BarnesHutCUDA_CU::find_target_neurons(
        seed, step,
        population, d_mapping.device_ptr(), mapping.size(),
        population, nullptr, std::size_t{ 0 },
        static_cast<CudaConfig::number_neurons_type>(number_neurons), tree,
        static_cast<CudaConfig::gaussian_type>(eps), d_ranks, my_rank,
        /* squared_sigma_inv */ 0.5, number_ranks, static_cast<CudaConfig::number_neurons_type>(number_neurons), exc_stream, inh_stream);

    auto [result_exc_2, result_inh_2] = BarnesHutCUDA_CU::find_target_neurons(
        seed, step,
        population, d_mapping.device_ptr(), mapping.size(),
        population, nullptr, std::size_t{ 0 },
        static_cast<CudaConfig::number_neurons_type>(number_neurons), tree,
        static_cast<CudaConfig::gaussian_type>(eps), d_ranks, my_rank,
        /* squared_sigma_inv */ 0.5, number_ranks, static_cast<CudaConfig::number_neurons_type>(number_neurons), exc_stream, inh_stream);

    const auto& [counts_exc_1, source_ids_exc_1, source_positions_exc_1, target_ids_exc_1] = result_exc_1;
    const auto& [counts_exc_2, source_ids_exc_2, source_positions_exc_2, target_ids_exc_2] = result_exc_2;

    ASSERT_GT(counts_exc_1[my_rank], 1) << "need genuine contention (more than one request) for this test to be meaningful";
    ASSERT_EQ(counts_exc_1, counts_exc_2);

    const auto host_source_ids_1 = source_ids_exc_1.get_device_data();
    const auto host_source_ids_2 = source_ids_exc_2.get_device_data();
    const auto host_target_ids_1 = target_ids_exc_1.get_device_data();
    const auto host_target_ids_2 = target_ids_exc_2.get_device_data();

    ASSERT_EQ(host_source_ids_1, host_source_ids_2) << "same seed/step must reproduce the exact same request order, not just the same set of requests";
    ASSERT_EQ(host_target_ids_1, host_target_ids_2);

    // The "call twice and diff" check above can pass by accident: on an otherwise idle GPU, two
    // back-to-back launches of the same kernel with identical inputs can end up with identical
    // atomicAdd scheduling anyway, even without the sort, so it doesn't reliably discriminate
    // fixed from unfixed. Directly check the actual property the fix establishes instead: within
    // this (single-rank) segment, entries must come out in canonical ascending (source_id,
    // target_id) order -- which the pre-fix atomicAdd-ordered output had no reason to satisfy.
    for (auto i = std::size_t{ 1 }; i < host_source_ids_1.size(); ++i) {
        const auto prev_key = (static_cast<std::uint64_t>(host_source_ids_1[i - 1]) << 32) | static_cast<std::uint64_t>(host_target_ids_1[i - 1]);
        const auto curr_key = (static_cast<std::uint64_t>(host_source_ids_1[i]) << 32) | static_cast<std::uint64_t>(host_target_ids_1[i]);
        ASSERT_LE(prev_key, curr_key) << "entry " << i << " is out of canonical (source_id, target_id) order";
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BarnesHutCUDA (the class itself, not just the free BarnesHutCUDA_CU functions)
// ────────────────────────────────────────────────────────────────────────────

// End-to-end regression test for this session's conversion of BarnesHutCUDA's device buffers
// from raw cudaMalloc'd pointers to DeviceArray-owned storage (tree_storage/population_ex_storage/
// population_inh_storage/vacant_exc_axons_mapping/vacant_inh_axons_mapping in BarnesHutCUDA.h),
// and of the ExchangeAlgorithm.h entry points it calls (process_calculation_requests_entry_aware/
// process_requests_entry_aware/process_responses_entry_aware) from long flat pointer-list
// parameters to the new SynapseCreationRequestHandle/SynapseCreationResponseHandle/
// BHCalculationRequestHandle/RemoteNodeRankHandle structs. Exercises the full
// update_connectivity_cuda_aware() pipeline (init_octree -> find_target_neurons_cuda_aware ->
// process_requests_aware -> process_responses_aware) end-to-end rather than just the individual
// free functions, and checks it actually creates the expected synapse instead of throwing or
// silently doing nothing.
TEST_F(BarnesHutCUDATest, testUpdateConnectivityConnectsTwoNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI rank.\n";
        }
        return;
    }

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 2 };
    const auto box = get_unit_box();

    const auto neuron_positions = std::vector<RelearnTypes::position_type>{
        RelearnTypes::position_type{ 0.1F, 0.1F, 0.1F },
        RelearnTypes::position_type{ 0.9F, 0.9F, 0.9F },
    };

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_neurons);
    extra_infos->set_positions(neuron_positions);

    const auto signal_types = std::vector<SignalType>(number_neurons, SignalType::Excitatory);
    // Every neuron offers exactly one vacant axon and one vacant excitatory dendrite -- with only
    // two neurons and autapses forbidden, each one's axon is forced to connect to the other's
    // dendrite, giving a fully deterministic expected outcome.
    auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements_with_fixed_number_axons_dendrites(extra_infos, signal_types, 1.0, 1.0);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    auto barnes_hut_cuda = BarnesHutCUDA(box, std::make_shared<Morton>(0));
    barnes_hut_cuda.set_probability_kernel(KernelFactory::get_standard_gaussian());
    barnes_hut_cuda.set_synaptic_elements(synaptic_elements);
    barnes_hut_cuda.set_network_graph(network_graph);
    barnes_hut_cuda.set_neuron_extra_infos(extra_infos);

    barnes_hut_cuda.init(number_neurons);

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    // prepare_update_connectivity() is re-declared protected in BarnesHutCUDA (unlike NaiveCUDA,
    // where it's public), but Algorithm::prepare_update_connectivity() itself is public and
    // virtual -- access control is checked against the static type, so calling through an
    // Algorithm& still resolves (via virtual dispatch) to BarnesHutCUDA's override.
    auto& algorithm = static_cast<Algorithm&>(barnes_hut_cuda);
    ASSERT_NO_THROW(algorithm.prepare_update_connectivity(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites));
    ASSERT_NO_THROW(std::ignore = barnes_hut_cuda.update_connectivity(number_neurons));

    const auto& number_connected_axons = synaptic_elements->get_connected_elements(SynapticElementType::Axon);
    const auto total_connected_axons = std::accumulate(number_connected_axons.begin(), number_connected_axons.end(), RelearnTypes::counter_type{ 0 });

    ASSERT_EQ(total_connected_axons, 2) << "Each of the two neurons has exactly one vacant axon and must connect it to the other neuron's dendrite";
}

TEST_F(BarnesHutCUDATest, testSetAcceptanceCriterionAcceptsPositiveValue) {
    auto barnes_hut_cuda = BarnesHutCUDA(get_unit_box(), std::make_shared<Morton>(0));
    ASSERT_NO_THROW(barnes_hut_cuda.set_acceptance_criterion(RelearnTypes::as<CudaConfig::gaussian_type>(0.5)));
}

TEST_F(BarnesHutCUDATest, testSetAcceptanceCriterionRejectsNonPositiveValue) {
    auto barnes_hut_cuda = BarnesHutCUDA(get_unit_box(), std::make_shared<Morton>(0));
    ASSERT_THROW(barnes_hut_cuda.set_acceptance_criterion(RelearnTypes::as<CudaConfig::gaussian_type>(0.0)), RelearnException);
    ASSERT_THROW(barnes_hut_cuda.set_acceptance_criterion(RelearnTypes::as<CudaConfig::gaussian_type>(-1.0)), RelearnException);
}

// Regression test that free() is safe to call after init_octree() has actually populated the
// device-owned storage (tree_storage/population_ex_storage/population_inh_storage/
// vacant_exc_axons_mapping/vacant_inh_axons_mapping), and that calling it a second time (now that
// everything is already reset to empty DeviceArrays) does not double-free or crash.
TEST_F(BarnesHutCUDATest, testFreeIsSafeAfterInitOctreeAndIdempotent) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI rank.\n";
        }
        return;
    }

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 2 };
    const auto box = get_unit_box();

    const auto neuron_positions = std::vector<RelearnTypes::position_type>{
        RelearnTypes::position_type{ 0.1F, 0.1F, 0.1F },
        RelearnTypes::position_type{ 0.9F, 0.9F, 0.9F },
    };

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_neurons);
    extra_infos->set_positions(neuron_positions);

    const auto signal_types = std::vector<SignalType>(number_neurons, SignalType::Excitatory);
    auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements_with_fixed_number_axons_dendrites(extra_infos, signal_types, 1.0, 1.0);
    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    auto barnes_hut_cuda = BarnesHutCUDA(box, std::make_shared<Morton>(0));
    barnes_hut_cuda.set_probability_kernel(KernelFactory::get_standard_gaussian());
    barnes_hut_cuda.set_synaptic_elements(synaptic_elements);
    barnes_hut_cuda.set_network_graph(network_graph);
    barnes_hut_cuda.set_neuron_extra_infos(extra_infos);
    barnes_hut_cuda.init(number_neurons);

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    auto& algorithm = static_cast<Algorithm&>(barnes_hut_cuda);
    ASSERT_NO_THROW(algorithm.prepare_update_connectivity(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites));

    ASSERT_NO_THROW(barnes_hut_cuda.free());
    ASSERT_NO_THROW(barnes_hut_cuda.free());
}

#endif
