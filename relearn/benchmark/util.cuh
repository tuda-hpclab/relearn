#pragma once

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

#include "main.h"

#include "algorithm/BarnesHutInternalCUDA/LinearizedTree.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "cuda/CudaConversion.h"

#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"
#include "factory/synaptic_elements/synaptic_elements_factory.h"

#include <vector>

inline LinearizedTree get_barnes_hut_cuda_linearized_tree(const std::size_t number_nodes, const std::uint32_t vacant_elements) {
    const auto root = get_octree<BarnesHutCUDACell>(number_nodes);

    auto axons_count = 0U;
    auto vacant_axonal_elements = std::vector<std::uint32_t>(number_nodes);
    auto vacant_dendritic_elements = std::vector<std::uint32_t>(number_nodes);

    for (auto i = 0U; i < number_nodes; ++i) {
        // begin vacant axons
        auto random_vacant_axons = 0U;
        if (axons_count <= vacant_elements) {
            random_vacant_axons = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 2, _mt);

            axons_count += random_vacant_axons;

            if (axons_count > vacant_elements) {
                random_vacant_axons -= axons_count - vacant_elements;
            }
        }
        vacant_axonal_elements[i] = random_vacant_axons;
        // end vacant axons
        vacant_dendritic_elements[i] = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 5, _mt);
    }

    std::ranges::shuffle(vacant_axonal_elements, _mt);
    return LinearizedTree(&root, SignalType::Excitatory, vacant_axonal_elements, vacant_dendritic_elements);
}

inline LinearizedTree get_barnes_hut_cuda_linearized_tree(const std::size_t number_nodes, const NeuronID source_neuron_id, const std::uint32_t source_vacant_elements) {
    const auto root = get_octree<BarnesHutCUDACell>(number_nodes);
    constexpr auto signal_type = SignalType::Excitatory;
    // auto tree = LinearizedTree(root, signal_type, number_nodes, std::vector<std::uint32_t>(number_nodes, 0), std::vector<std::uint32_t>(number_nodes, 0));
    auto tree = LinearizedTree(&root, std::vector(number_nodes, signal_type), std::vector(number_nodes, 0U));
    tree.set_neuron_details(signal_type, source_neuron_id, source_vacant_elements);
    return tree;
}

inline void naive_cuda_fill_vectors(std::vector<CudaTypes::cuda_real3>& positions, std::vector<uint32_t>& vacant_axons,
                                    std::vector<uint32_t>& vacant_dendrites, std::vector<uint64_t>& start_index, uint64_t& target_size,
                                    const NeuronID vacant_axons_target, const std::uint32_t max_axons) {

    bool position_axons_on_target = vacant_axons_target.is_initialized();

    auto axons_count = 0U;
    vacant_axons.resize(positions.size());
    vacant_dendrites.resize(positions.size());
    start_index.resize(positions.size());

    // fill neurons with random information
    for (auto i = 0U; i < positions.size(); ++i) {
        const auto random_neuron_position = SimulationFactory::get_random_position_in_box({ 0, 0, 0 }, { 1, 1, 1 }, _mt);
        const auto random_vacant_dendrites = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 5, _mt);

        auto random_vacant_axons = 0U;
        if (axons_count <= max_axons && !position_axons_on_target) {
            random_vacant_axons = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 2, _mt);

            axons_count += random_vacant_axons;

            if (axons_count > max_axons) {
                random_vacant_axons -= axons_count - max_axons;
            }
        }

        positions[i] = CudaTypes::to_cuda_real3(random_neuron_position);
        vacant_axons[i] = random_vacant_axons;
        vacant_dendrites[i] = random_vacant_dendrites;
        start_index[i] = target_size;
        target_size += random_vacant_axons > 0 ? random_vacant_axons : 1;
    }

    std::ranges::shuffle(vacant_axons, _mt);

    if (position_axons_on_target) {
        const auto axon_target_id = vacant_axons_target.get_neuron_id();
        vacant_axons[axon_target_id] = max_axons;
    }
}

inline void naive_cuda_fill_vectors(std::vector<CudaTypes::cuda_real3>& positions, std::vector<uint32_t>& vacant_axons, std::vector<uint32_t>& vacant_dendrites, std::vector<uint64_t>& start_index, uint64_t& target_size, const std::uint32_t max_axons) {
    naive_cuda_fill_vectors(positions, vacant_axons, vacant_dendrites, start_index, target_size, NeuronID::uninitialized_id(), max_axons);
}

inline void naive_cuda_fill_vectors(std::vector<CudaTypes::cuda_real3>& positions, std::vector<uint32_t>& vacant_axons,
                                    std::vector<uint32_t>& vacant_dendrites, std::vector<uint64_t>& start_index, uint64_t& target_size,
                                    std::vector<uint64_t>& neuronId_vacant_axons, const NeuronID vacant_axons_target, const std::uint32_t max_axons) {

    auto position_axons_on_target = vacant_axons_target.is_initialized();

    auto axons_count = 0U;
    vacant_axons.resize(positions.size());
    vacant_dendrites.resize(positions.size());
    start_index.resize(positions.size());

    // fill neurons with random information
    for (auto i = 0U; i < positions.size(); ++i) {
        const auto random_neuron_position = SimulationFactory::get_random_position_in_box({ 0, 0, 0 }, { 1, 1, 1 }, _mt);
        const auto random_vacant_dendrites = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 5, _mt);

        auto random_vacant_axons = 0U;
        if (axons_count <= max_axons && !position_axons_on_target) {
            random_vacant_axons = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 2, _mt);

            axons_count += random_vacant_axons;

            if (axons_count > max_axons) {
                random_vacant_axons -= axons_count - max_axons;
            }
        }

        positions[i] = CudaTypes::to_cuda_real3(random_neuron_position);
        vacant_axons[i] = random_vacant_axons;
        vacant_dendrites[i] = random_vacant_dendrites;
        start_index[i] = target_size;
        target_size += random_vacant_axons > 0 ? random_vacant_axons : 1;
    }

    auto rng = std::default_random_engine{};
    std::shuffle(vacant_axons.begin(), vacant_axons.end(), rng);

    if (position_axons_on_target) {
        const auto axon_target_id = vacant_axons_target.get_neuron_id();
        vacant_axons[axon_target_id] = max_axons;
    }

    // optimized neuron placemeent -> no sparse vector
    target_size = max_axons;
    neuronId_vacant_axons.resize(0);
    for (auto i = 0U; i < vacant_axons.size(); ++i) {
        for (auto j = 0U; j < vacant_axons[i]; ++j) {
            neuronId_vacant_axons.emplace_back(i);
        }
    }
}

inline void naive_cuda_fill_vectors(std::vector<float3>& positions, std::vector<uint32_t>& vacant_axons,
                                    std::vector<uint32_t>& vacant_dendrites, std::vector<uint32_t>& start_index, uint32_t& target_size,
                                    std::vector<uint32_t>& neuronId_vacant_axons, const NeuronID vacant_axons_target, const uint32_t max_axons) {

    bool position_axons_on_target = vacant_axons_target.is_initialized();

    uint32_t axons_count = 0L;
    vacant_axons.resize(positions.size());
    vacant_dendrites.resize(positions.size());
    start_index.resize(positions.size());

    // fill neurons with random information
    for (auto i = 0U; i < positions.size(); ++i) {
        const auto random_neuron_position = SimulationFactory::get_random_position_in_box({ 0, 0, 0 }, { 1, 1, 1 }, _mt);
        const auto random_vacant_dendrites = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 5, _mt);

        auto random_vacant_axons = 0U;
        if (axons_count <= max_axons && !position_axons_on_target) {
            random_vacant_axons = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 2, _mt);

            axons_count += random_vacant_axons;

            if (axons_count > max_axons) {
                random_vacant_axons -= axons_count - max_axons;
            }
        }

        // The experimental kernels are single precision, independent of the precision the rest of the device code uses
        positions[i] = float3{ RelearnTypes::as<float>(random_neuron_position.get_x()), RelearnTypes::as<float>(random_neuron_position.get_y()), RelearnTypes::as<float>(random_neuron_position.get_z()) };

        vacant_axons[i] = random_vacant_axons;
        vacant_dendrites[i] = random_vacant_dendrites;
        start_index[i] = target_size;
        target_size += random_vacant_axons > 0 ? random_vacant_axons : 1;
    }
    auto rng = std::default_random_engine{};

    std::shuffle(vacant_axons.begin(), vacant_axons.end(), rng);

    if (position_axons_on_target) {
        const auto axon_target_id = vacant_axons_target.get_neuron_id();
        vacant_axons[axon_target_id] = max_axons;
    }

    // optimized neuron placemeent -> no sparse vector
    target_size = max_axons;
    neuronId_vacant_axons.resize(0);
    for (auto i = 0U; i < vacant_axons.size(); ++i) {
        for (auto j = 0U; j < vacant_axons[i]; ++j) {
            neuronId_vacant_axons.emplace_back(i);
        }
    }
}

inline void naive_cuda_fill_vectors(std::vector<CudaTypes::cuda_real3>& positions, std::vector<uint32_t>& vacant_axons,
                                    std::vector<uint32_t>& vacant_dendrites, std::vector<uint64_t>& start_index, uint64_t& target_size,
                                    std::vector<uint64_t>& neuron_id_vacant_axons) {
    naive_cuda_fill_vectors(positions, vacant_axons, vacant_dendrites, start_index, target_size, neuron_id_vacant_axons, NeuronID::uninitialized_id(), 1000);
}

#endif