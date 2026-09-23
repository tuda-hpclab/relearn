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

#include "algorithm/Internal/octree/Octree.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "neurons/NeuronsExtraInfo.h"
#include "structure/SpaceFillingCurve.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <cstdint>
#include <memory>
#include <utility>

template <typename AdditionalCellAttributes>
class OctreeAlgorithm {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using counter_type = RelearnTypes::counter_type;

    /**
     * @brief Constructs a new octree algorithm
     * @param bounding_box The bounding box of the whole simulation
     * @param _space_filling_curve The space-filling curve to use, not nullptr
     * @exception Throws a RelearnException if _space_filling_curve is nullptr
     */
    OctreeAlgorithm(const RelearnTypes::bounding_box_type& bounding_box, std::shared_ptr<SpaceFillingCurve> _space_filling_curve, const bool rma_required)
        : octree(std::make_unique<Octree<AdditionalCellAttributes>>(bounding_box, std::move(_space_filling_curve), rma_required)) { }

    /**
     * @brief Sets the extra infos for the neurons. They hold the positions and update flags for the neurons.
     * @param infos The extra infos, not empty
     * @exception throws a RelearnException if infos is empty
     */
    void set_neuron_extra_infos(std::shared_ptr<NeuronsExtraInfo> infos) {
        RelearnException::check(infos != nullptr, "OctreeAlgorithm::set_neuron_extra_infos: infos is empty");
        octree_internal_extra_infos = std::move(infos);
    }

    /**
     * @brief Initializes the algorithm to include number_neurons many local neurons.
     * @param number_neurons The number of local neurons to store in this class
     */
    void init(const number_neurons_type number_neurons) {
        for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
            const auto pos = octree_internal_extra_infos->get_position(neuron_id);
            octree->insert(pos, neuron_id);
        }

        octree->initializes_leaf_nodes(number_neurons);

        local_number_neurons = number_neurons;
    }

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    void create_neurons(const number_neurons_type creation_count) {
        const auto current_size = local_number_neurons;
        const auto new_size = current_size + creation_count;

        for (const auto neuron_id : NeuronIDRange::range(current_size, new_size)) {
            const auto pos = octree_internal_extra_infos->get_position(neuron_id);
            octree->insert(pos, neuron_id);
        }

        octree->initializes_leaf_nodes(new_size);

        local_number_neurons = new_size;
    }

    /**
     * @brief Performs all required steps to disable all neurons that are specified.
     *      Disables incrementally, i.e., previously disabled neurons are not enabled.
     * @param neuron_ids The local neuron ids that should be disabled
     * @exception Throws a RelearnException if a specified id is too large
     */
    void disable_neurons([[maybe_unused]] const std::span<const NeuronID> neuron_ids) { }

    /**
     * @brief Updates the octree according to the necessities of the algorithm. Updates only those neurons for which the extra infos specify so.
     *      Performs communication via MPI
     * @exception Can throw a RelearnException
     */
    void update_tree(const std::span<const SignalType> signal_types,
                     const std::span<const counter_type> vacant_axons,
                     const std::span<const counter_type> vacant_excitatory_dendrites,
                     const std::span<const counter_type> vacant_inhibitory_dendrites) {

        // Update my leaf nodes
        Timers::start(TimerRegion::UPDATE_LEAF_NODES);
        update_leaf_nodes(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites);
        Timers::stop_and_add(TimerRegion::UPDATE_LEAF_NODES);

        // Update the octree
        octree->synchronize_tree();
    }

    /**
     * @brief Returns the stored octree
     * @return The octree
     */
    [[nodiscard]] const std::shared_ptr<Octree<AdditionalCellAttributes>>& get_octree() const noexcept {
        return octree;
    }

    /**
     * @brief Returns the root of the stored octree
     * @return The root
     */
    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_octree_root() const noexcept {
        return octree->get_root();
    }

    /**
     * @brief Returns the level of branch nodes of the stored octree
     * @return The level of branch nodes
     */
    [[nodiscard]] RelearnTypes::level_type get_level_of_branch_nodes() const noexcept {
        return octree->get_level_of_branch_nodes();
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
        const auto my_footprint = sizeof(*this);
        footprint->emplace("OctreeAlgorithm", my_footprint);

        octree->record_memory_footprint(footprint);
    }

private:
    /**
     * @brief Updates all leaf nodes in the octree by the algorithm if the extra infos specify so.
     * @exception Throws a RelearnException if the number of flags is different than the number of leaf nodes, or if there is an internal error
     */
    void update_leaf_nodes(const std::span<const SignalType> signal_types,
                           const std::span<const counter_type> vacant_axons,
                           const std::span<const counter_type> vacant_excitatory_dendrites,
                           const std::span<const counter_type> vacant_inhibitory_dendrites) {

        const auto& leaf_nodes = octree->get_leaf_nodes();
        const auto num_leaf_nodes = leaf_nodes.size();

        const auto num_disable_flags = octree_internal_extra_infos->get_size();

        const auto num_signal_types = signal_types.size();
        const auto num_vacant_axons = vacant_axons.size();
        const auto num_vacant_excitatory_dendrites = vacant_excitatory_dendrites.size();
        const auto num_vacant_inhibitory_dendrites = vacant_inhibitory_dendrites.size();

        const auto all_same_size = num_leaf_nodes == num_disable_flags
                                   && num_leaf_nodes == num_signal_types
                                   && num_leaf_nodes == num_vacant_axons
                                   && num_leaf_nodes == num_vacant_excitatory_dendrites
                                   && num_leaf_nodes == num_vacant_inhibitory_dendrites;

        RelearnException::check(all_same_size, "OctreeAlgorithm::update_leaf_nodes: The vectors were of different sizes");

        for (const auto& neuron_id : NeuronIDRange::range(num_leaf_nodes)) {
            const auto local_neuron_id = neuron_id.get_neuron_id();

            auto* node = leaf_nodes[local_neuron_id];
            RelearnException::check(node != nullptr, "OctreeAlgorithm::update_leaf_nodes: node was nullptr: {}", neuron_id);

            const auto& cell = node->get_cell();
            const auto other_neuron_id = cell.get_neuron_id();
            RelearnException::check(neuron_id == other_neuron_id, "OctreeAlgorithm::update_leaf_nodes: The nodes are not in order {} != {}", neuron_id, other_neuron_id);

            if (!octree_internal_extra_infos->does_update_plasticity(neuron_id)) {
                if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                    node->set_cell_number_excitatory_dendrites(0);
                }

                if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                    node->set_cell_number_inhibitory_dendrites(0);
                }

                if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                    node->set_cell_number_excitatory_axons(0);
                }

                if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                    node->set_cell_number_inhibitory_axons(0);
                }
                continue;
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                const auto number_vacant_excitatory_dendrites = vacant_excitatory_dendrites[local_neuron_id];
                node->set_cell_number_excitatory_dendrites(number_vacant_excitatory_dendrites);
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                const auto number_vacant_inhibitory_dendrites = vacant_inhibitory_dendrites[local_neuron_id];
                node->set_cell_number_inhibitory_dendrites(number_vacant_inhibitory_dendrites);
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                const auto signal_type = signal_types[local_neuron_id];

                if (signal_type == SignalType::Excitatory) {
                    const auto number_vacant_axons = vacant_axons[local_neuron_id];
                    node->set_cell_number_excitatory_axons(number_vacant_axons);
                } else {
                    node->set_cell_number_excitatory_axons(0);
                }
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                const auto signal_type = signal_types[local_neuron_id];

                if (signal_type == SignalType::Inhibitory) {
                    const auto number_vacant_axons = vacant_axons[local_neuron_id];
                    node->set_cell_number_inhibitory_axons(number_vacant_axons);
                } else {
                    node->set_cell_number_inhibitory_axons(0);
                }
            }
        }
    }

    std::shared_ptr<Octree<AdditionalCellAttributes>> octree{};
    std::shared_ptr<NeuronsExtraInfo> octree_internal_extra_infos;

    number_neurons_type local_number_neurons{};
};
