#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "algorithm/Internal/octree/OctreeNode.h"
#include "algorithm/Internal/octree/OctreeNodeHelper.h"
#include "types/BasicTypes.h"
#include "util/MemoryHolder.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/Vec3.h"

#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <random>
#include <stack>

class OctreeFactory {
public:
    template <typename AdditionalCellAttributes>
    static OctreeNode<AdditionalCellAttributes> get_standard_tree(const RelearnTypes::number_neurons_type number_neurons, std::shared_ptr<MemoryHolder<AdditionalCellAttributes>> memory_holder, const RelearnTypes::position_type& min_pos, const RelearnTypes::position_type& max_pos, std::mt19937& mt) {
        auto get_synaptic_count = [&mt]() { return RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(1, 2, mt); };

        auto root = OctreeNode<AdditionalCellAttributes>{};
        root.set_level(0);
        root.set_rank(mpiPP::MPIInfo::get_my_rank());

        root.set_cell_neuron_id(NeuronID(0));
        root.set_cell_size(min_pos, max_pos);
        root.set_cell_neuron_position(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));

        for (const auto id : NeuronIDRange::range(1, number_neurons)) {
            root.insert(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt), id, memory_holder);
        }

        auto stack = std::stack<OctreeNode<AdditionalCellAttributes>*>{};
        stack.push(&root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_dendrite) {
                    current->set_cell_number_excitatory_dendrites(get_synaptic_count());
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_dendrite) {
                    current->set_cell_number_inhibitory_dendrites(get_synaptic_count());
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_axon) {
                    current->set_cell_number_excitatory_axons(get_synaptic_count());
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_axon) {
                    current->set_cell_number_inhibitory_axons(get_synaptic_count());
                }

                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&root);

        return root;
    }

    template <typename AdditionalCellAttributes>
    static OctreeNode<AdditionalCellAttributes> get_tree_no_axons(const RelearnTypes::number_neurons_type number_neurons, std::shared_ptr<MemoryHolder<AdditionalCellAttributes>> memory_holder, const RelearnTypes::position_type& min_pos, const RelearnTypes::position_type& max_pos, std::mt19937& mt) {
        auto get_synaptic_count = [&mt]() { return RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 1, mt); };

        OctreeNode<AdditionalCellAttributes> root{};
        root.set_level(0);
        root.set_rank(mpiPP::MPIInfo::get_my_rank());

        root.set_cell_neuron_id(NeuronID(0));
        root.set_cell_size(min_pos, max_pos);
        root.set_cell_neuron_position(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));

        for (const auto id : NeuronIDRange::range(1, number_neurons)) {
            root.insert(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt), id, memory_holder);
        }

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(&root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_dendrite) {
                    current->set_cell_number_excitatory_dendrites(get_synaptic_count());
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_dendrite) {
                    current->set_cell_number_inhibitory_dendrites(get_synaptic_count());
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_axon) {
                    current->set_cell_number_excitatory_axons(0);
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_axon) {
                    current->set_cell_number_inhibitory_axons(0);
                }

                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&root);

        return root;
    }

    template <typename AdditionalCellAttributes>
    static OctreeNode<AdditionalCellAttributes> get_tree_no_dendrites(const RelearnTypes::number_neurons_type number_neurons, std::shared_ptr<MemoryHolder<AdditionalCellAttributes>> memory_holder, const RelearnTypes::position_type& min_pos, const RelearnTypes::position_type& max_pos, std::mt19937& mt) {
        auto get_synaptic_count = [&mt]() { return RandomFactory::get_random_integer<typename OctreeNode<AdditionalCellAttributes>::counter_type>(0, 1, mt); };

        OctreeNode<AdditionalCellAttributes> root{};
        root.set_level(0);
        root.set_rank(mpiPP::MPIInfo::get_my_rank());

        root.set_cell_neuron_id(NeuronID(0));
        root.set_cell_size(min_pos, max_pos);
        root.set_cell_neuron_position(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));

        for (const auto id : NeuronIDRange::range(1, number_neurons)) {
            root.insert(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt), id, memory_holder);
        }

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(&root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_dendrite) {
                    current->set_cell_number_excitatory_dendrites(0);
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_dendrite) {
                    current->set_cell_number_inhibitory_dendrites(0);
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_axon) {
                    current->set_cell_number_excitatory_axons(get_synaptic_count());
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_axon) {
                    current->set_cell_number_inhibitory_axons(get_synaptic_count());
                }

                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&root);

        return root;
    }

    template <typename AdditionalCellAttributes>
    static OctreeNode<AdditionalCellAttributes> get_tree_no_synaptic_elements(const RelearnTypes::number_neurons_type number_neurons, std::shared_ptr<MemoryHolder<AdditionalCellAttributes>> memory_holder, const RelearnTypes::position_type& min_pos, const RelearnTypes::position_type& max_pos, std::mt19937& mt) {
        OctreeNode<AdditionalCellAttributes> root{};
        root.set_level(0);
        root.set_rank(mpiPP::MPIInfo::get_my_rank());

        root.set_cell_neuron_id(NeuronID(0));
        root.set_cell_size(min_pos, max_pos);
        root.set_cell_neuron_position(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));

        for (const auto id : NeuronIDRange::range(1, number_neurons)) {
            root.insert(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt), id, memory_holder);
        }

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(&root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_dendrite) {
                    current->set_cell_number_excitatory_dendrites(0);
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_dendrite) {
                    current->set_cell_number_inhibitory_dendrites(0);
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_excitatory_axon) {
                    current->set_cell_number_excitatory_axons(0);
                }

                if constexpr (OctreeNode<AdditionalCellAttributes>::has_inhibitory_axon) {
                    current->set_cell_number_inhibitory_axons(0);
                }

                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(&root);

        return root;
    }
};
