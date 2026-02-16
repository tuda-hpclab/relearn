#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"
#include "Types.h"

#include "algorithm/Internal/octree/NodeCache.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "algorithm/Kernel/Kernel.h"
#include "algorithm/Kernel/KernelBase.h"
#include "algorithm/NaiveInternal/NaiveCell.h"
#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <array>
#include <optional>
#include <stack>
#include <tuple>
#include <vector>
#include <utility>

/**
 * This class provides all computational elements of the Barnes-Hut algorithm.
 * It purely calculates things, but does not change any visible state. However, it might download nodes via MPI
 * @tparam AdditionalCellAttributes The cell attributes that are necessary for the instance
 */
template <typename AdditionalCellAttributes>
class NaiveBase {
public:
    using position_type = RelearnTypes::position_type;
    using counter_type = RelearnTypes::counter_type;

    virtual ~NaiveBase() = default;

    /**
     * @brief Returns an optional RankNeuronId that the algorithm determined for the given source neuron. No actual request is made.
     *      Might perform MPI communication via NodeCache::get_children()
     * @param kernel The probability kernel
     * @param cache The cache that contains the children of the nodes
     * @param src_neuron_id The neuron's id that wants to connect. Is used to disallow autapses (connections to itself)
     * @param axon_position The neuron's position that wants to connect. Is used in probability computations
     * @param dendrite_type_needed The signal type that is searched.
     * @param root Where the source neuron should start to search for targets. It is not const because the children might be changed if the node is remote. Not nullptr
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorithm found a matching neuron, it's id and MPI rank are returned.
     */
    [[nodiscard]] static std::optional<RankNeuronId> find_target_neuron(const KernelBase& kernel, const NodeCache<AdditionalCellAttributes>& cache, const NeuronID& src_neuron_id, const position_type& axon_position, SignalType dendrite_type_needed, OctreeNode<AdditionalCellAttributes>* const root) {
        auto* root_of_subtree = root;
        while (true) {
            const auto& possible_targets = get_nodes_for_interval(cache, axon_position, root_of_subtree, dendrite_type_needed);

            auto* node_selected = Kernel<AdditionalCellAttributes>::pick_target(kernel, { mpiPP::MPIInfo::get_my_rank(), src_neuron_id },
                                                                                axon_position, possible_targets, ElementType::Dendrite, dendrite_type_needed);
            if (node_selected == nullptr) {
                return {};
            }

            // A chosen child is a valid target
            if (const auto done = node_selected->is_leaf(); done) {
                return RankNeuronId{ node_selected->get_mpi_rank(), node_selected->get_cell_neuron_id() };
            }

            // We need to choose again, starting from the chosen virtual neuron
            root_of_subtree = node_selected;
        } // while
    }

    /**
     * @brief Returns an optional RankNeuronId that the algorithm determined for the given source neuron. No actual request is made.
     *      Might perform MPI communication via NodeCache::get_children()
     * @param kernel The probability kernel
     * @param cache The cache that contains the children of the nodes
     * @param src_neuron_id The neuron's id that wants to connect. Is used to disallow autapses (connections to itself)
     * @param axon_position The neuron's position that wants to connect. Is used in probability computations
     * @param number_vacant_elements The source neuron's number of vacant elements
     * @param root Where the source neuron should start to search for targets. It is not const because the children might be changed if the node is remote. Not nullptr
     * @param dendrite_type_needed The signal type that is searched.
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorithm found a matching neuron, it's id and MPI rank are returned.
     */
    [[nodiscard]] static std::vector<std::pair<mpiPP::MPIRank, SynapseCreationRequest>> find_target_neurons(const KernelBase& kernel, const NodeCache<AdditionalCellAttributes>& cache, const NeuronID& src_neuron_id, const position_type& axon_position,
                                                                                                            const counter_type number_vacant_elements, OctreeNode<AdditionalCellAttributes>* const root, const SignalType dendrite_type_needed) {

        auto requests = std::vector<std::pair<mpiPP::MPIRank, SynapseCreationRequest>>{};
        requests.reserve(number_vacant_elements);

        // For all vacant axons of neuron "neuron_id"
        for (auto j = 0U; j < number_vacant_elements; j++) {
            /**
             * Find target neuron for connecting and
             * connect if target neuron has still dendrite available.
             *
             * The target neuron might not have any dendrites left
             * as other axons might already have connected to them.
             * Right now, those collisions are handled in a first-come-first-served fashion.
             */
            const auto& rank_neuron_id = find_target_neuron(kernel, cache, src_neuron_id, axon_position, dendrite_type_needed, root);
            if (!rank_neuron_id.has_value()) {
                // If finding failed, it won't succeed in later iterations
                break;
            }

            const auto& [target_rank, target_id] = rank_neuron_id.value();
            const auto creation_request = SynapseCreationRequest(target_id, src_neuron_id, dendrite_type_needed);
            /**
             * Append request for synapse creation to rank "target_rank"
             * Note that "target_rank" could also be my own rank.
             */
            requests.emplace_back(target_rank, creation_request);
        }

        return requests;
    }

    [[nodiscard]] static std::tuple<bool, bool> acceptance_criterion_test(
        const position_type& /*axon_position*/,
        const OctreeNode<NaiveCell>* node_with_dendrite,
        SignalType dendrite_type_needed) {

        RelearnException::check(node_with_dendrite != nullptr, "Naive::acceptance_criterion_test: node_with_dendrite was nullptr");

        const auto& cell = node_with_dendrite->get_cell();
        const auto has_vacant_dendrites = cell.get_number_dendrites_for(dendrite_type_needed) != 0;
        const auto is_parent = node_with_dendrite->is_parent();

        // Accept leaf only
        return std::make_tuple(!is_parent, has_vacant_dendrites);
    }

    [[nodiscard]] static std::vector<OctreeNode<NaiveCell>*> get_nodes_for_interval(
        const NodeCache<NaiveCell>& cache,
        const position_type& axon_position,
        OctreeNode<NaiveCell>* root,
        SignalType dendrite_type_needed) {
        if (root == nullptr) {
            return {};
        }

        if (root->get_cell().get_number_dendrites_for(dendrite_type_needed) == 0) {
            return {};
        }

        if (root->is_leaf()) {
            /**
             * The root node is a leaf and thus contains the target neuron.
             *
             * NOTE: Root is not intended to be a leaf but we handle this as well.
             * Without pushing root onto the stack, it would not make it into the "vector" of nodes.
             */

            const auto [accept, _] = acceptance_criterion_test(axon_position, root, dendrite_type_needed);
            if (accept) {
                return { root };
            }

            return {};
        }

        auto stack = std::stack<OctreeNode<NaiveCell>*>{};

        const auto add_children_to_stack = [&stack, &cache](OctreeNode<NaiveCell>* node) {
            auto children = std::array<OctreeNode<NaiveCell>*, Constants::number_oct>{ nullptr };

            // Node is owned by this rank
            if (node->is_actual_id()) {
                // Node is owned by this rank, so the pointers are good
                children = node->get_children();
            } else {
                // Node owned by different rank, so we have to download the data to local nodes
                children = cache.get_children(node);
            }

            for (auto* iter : children) {
                if (iter != nullptr) {
                    stack.push(iter);
                }
            }
        };

        // The algorithm expects that root is not considered directly, rather its children
        add_children_to_stack(root);

        auto nodes_to_consider = std::vector<OctreeNode<NaiveCell>*>{};
        nodes_to_consider.reserve(Constants::number_oct);

        while (!stack.empty()) {
            // Get top-of-stack node and remove it from stack
            auto* stack_elem = stack.top();
            stack.pop();

            /**
             * Should node be used for probability interval?
             *
             * Only take those that have dendrites available
             */
            const auto [accept, has_vacant_dendrites] = acceptance_criterion_test(axon_position, stack_elem, dendrite_type_needed);

            if (accept) {
                // Insert node into vector
                nodes_to_consider.emplace_back(stack_elem);
                continue;
            }

            if (!has_vacant_dendrites) {
                continue;
            }

            add_children_to_stack(stack_elem);
        } // while

        return nodes_to_consider;
    }
};