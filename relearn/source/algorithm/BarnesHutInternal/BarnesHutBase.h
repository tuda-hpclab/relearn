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
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/DistantNeuronRequests.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "util/RelearnException.h"

#include "cpp-utility/data-structure/Stack.hpp"

#include "mpi-wrapper/MPIRank.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

/**
 * This class provides all computational elements of the Barnes-Hut algorithm.
 * It purely calculates things, but does not change any visible state. However, it might download nodes via MPI
 * @tparam AdditionalCellAttributes The cell attributes that are necessary for the instance
 */
template <typename AdditionalCellAttributes>
class BarnesHutBase {
public:
    using position_type = RelearnTypes::position_type;
    using counter_type = RelearnTypes::counter_type;

    /**
     * This enum indicates for an OctreeNode what the acceptance status is
     * It can be:
     * - Discard (no matching elements there)
     * - Expand (would be too much approximation, need to expand)
     * - Accept (can use the node for the algorithm)
     */
    enum class AcceptanceStatus : char {
        Discard = 0,
        Expand = 1,
        Accept = 2,
    };

    /**
     * @brief Tests the Barnes-Hut criterion on the source position and the target wrt. to required element type and signal type
     * @param source_position The source position of the calculation
     * @param target_node The target node within the Octree that should be considered, not nullptr
     * @param element_type The type of elements that are searched for
     * @param signal_type The signal type of the elements that are searched for
     * @param acceptance_criterion The acceptance criterion, must be > 0.0 and <= Constants::bh_max_theta
     * @exception Throws a RelearnException if a parameter did not meet the requirements or there was an algorithmic error
     * @return The acceptance status for the node, i.e., if it must be discarded, can be accepted, or must be expanded.
     */
    [[nodiscard]] static AcceptanceStatus test_acceptance_criterion(const position_type& source_position, const OctreeNode<AdditionalCellAttributes>* target_node,
                                                                    const ElementType element_type, const SignalType signal_type, const double acceptance_criterion) {
        RelearnException::check(acceptance_criterion > 0.0,
                                "BarnesHutBase::test_acceptance_criterion: The acceptance criterion was not positive: ({})", acceptance_criterion);
        RelearnException::check(acceptance_criterion <= Constants::bh_max_theta,
                                "BarnesHutBase::test_acceptance_criterion: The acceptance criterion must not be larger than {}: ({})", Constants::bh_max_theta, acceptance_criterion);
        RelearnException::check(target_node != nullptr,
                                "BarnesHutBase::test_acceptance_criterion: target_node was nullptr");

        const auto& cell = target_node->get_cell();

        // Never accept a node with zero vacant elements
        if (const auto number_vacant_elements = cell.get_number_elements_for(element_type, signal_type); number_vacant_elements == 0) {
            return AcceptanceStatus::Discard;
        }

        // Check distance between source and target
        const auto& target_position = cell.get_position_for(element_type, signal_type);

        // NOTE: This assertion fails when considering inner nodes that don't have the required elements.
        RelearnException::check(target_position.has_value(), "BarnesHutBase::test_acceptance_criterion: target_position was bad");

        // Calc Euclidean distance between source and target neuron
        const auto& distance_vector = target_position.value() - source_position;
        const auto distance = distance_vector.calculate_2_norm();

        // No autapse
        if (distance == 0.0) {
            return AcceptanceStatus::Discard;
        }

        // Always accept a leaf node
        if (const auto is_leaf = target_node->is_leaf(); is_leaf) {
            return AcceptanceStatus::Accept;
        }

        const auto length = cell.get_maximal_dimension_difference();

        // Original Barnes-Hut acceptance criterion
        // const auto ret_val = (length / distance) < acceptance_criterion;
        const auto ret_val = length < (acceptance_criterion * distance);
        return ret_val ? AcceptanceStatus::Accept : AcceptanceStatus::Expand;
    }

    /**
     * @brief Searches all neurons that must be considered as targets starting at root.
     *      Might perform MPI communication via NodeCache::get_children()
     * @param cache The cache that contains the children of the nodes
     * @param source_position The position of the source
     * @param root The start where the source searches for targets, not nullptr
     * @param element_type The element type that the source searches
     * @param signal_type The signal type that the source searches
     * @param acceptance_criterion The acceptance criterion, must be > 0.0 and <= Constants::bh_max_theta
     * @param accept_early_far_node If true, then nodes that belong to another MPI rank and (don't) satisfy the acceptance criterion are still returned
     * @exception Throws a RelearnException if the requirements on the acceptance criterion are not fulfilled or root is nullptr
     * @return The vector of all nodes from which the source can choose. Might be empty
     */
    [[nodiscard]] static std::vector<OctreeNode<AdditionalCellAttributes>*> get_nodes_to_consider(const NodeCache<AdditionalCellAttributes>& cache, const position_type& source_position, OctreeNode<AdditionalCellAttributes>* const root,
                                                                                                  const ElementType element_type, const SignalType signal_type, const double acceptance_criterion, const bool accept_early_far_node = false) {
        RelearnException::check(acceptance_criterion > 0.0,
                                "BarnesHutBase::get_nodes_to_consider: The acceptance criterion was not positive: ({})", acceptance_criterion);
        RelearnException::check(acceptance_criterion <= Constants::bh_max_theta,
                                "BarnesHutBase::get_nodes_to_consider: The acceptance criterion must not be larger than {}: ({})", Constants::bh_max_theta, acceptance_criterion);
        RelearnException::check(root != nullptr,
                                "BarnesHutBase::get_nodes_to_consider: root was nullptr");

        if (root->get_cell().get_number_elements_for(element_type, signal_type) == 0) {
            return {};
        }

        if (root->is_leaf()) {
            /**
             * The root node is a leaf and thus contains the target neuron.
             *
             * NOTE: Root is not intended to be a leaf but we handle this as well.
             * Without pushing root onto the stack, it would not make it into the "vector" of nodes.
             */

            const auto status = test_acceptance_criterion(source_position, root, element_type, signal_type, acceptance_criterion);
            if (status != AcceptanceStatus::Discard) {
                return { root };
            }

            return {};
        }

        auto stack = utility::Stack<OctreeNode<AdditionalCellAttributes>*>(Constants::number_prealloc_space);

        const auto add_children = [&stack, &cache](OctreeNode<AdditionalCellAttributes>* node) {
            const auto& children = cache.get_children(node);

            for (auto* it : children) {
                if (it != nullptr) {
                    stack.emplace_back(it);
                }
            }
        };

        // The algorithm expects that root is not considered directly, rather its children
        add_children(root);

        auto nodes_to_consider = std::vector<OctreeNode<AdditionalCellAttributes>*>{};
        nodes_to_consider.reserve(Constants::number_prealloc_space);

        while (!stack.empty()) {
            // Get top-of-stack node and remove it
            auto* node = stack.pop_back();

            /**
             * Should node be used for probability interval?
             * Only take those that have the required elements
             */
            const auto status = test_acceptance_criterion(source_position, node, element_type, signal_type, acceptance_criterion);

            if (status == AcceptanceStatus::Discard) {
                continue;
            }

            if (status == AcceptanceStatus::Accept) {
                // Insert node into vector
                nodes_to_consider.emplace_back(node);
                continue;
            }

            if (accept_early_far_node && !node->is_actual_id()) {
                nodes_to_consider.emplace_back(node);
                continue;
            }

            // Need to expand
            add_children(node);
        } // while

        return nodes_to_consider;
    }

    /**
     * @brief Returns an optional RankNeuronId that the algorithm determined for the given source neuron. No actual request is made.
     *      Might perform MPI communication via NodeCache::get_children()
     * @param kernel The probability kernel
     * @param cache The cache that contains the children of the nodes
     * @param source_neuron_id The source neuron's id
     * @param source_position The source neuron's position
     * @param root The starting position where to look, not nullptr
     * @param element_type The element type the source is looking for
     * @param signal_type The signal type the source is looking for
     * @param acceptance_criterion The acceptance criterion, must be > 0.0 and <= Constants::bh_max_theta
     * @exception Throws a RelearnException if the requirements on the acceptance criterion are not fulfilled or root is nullptr
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorithm found a matching neuron, its RankNeuronId is returned
     */
    [[nodiscard]] static std::optional<RankNeuronId> find_target_neuron(const KernelBase& kernel, const NodeCache<AdditionalCellAttributes>& cache, const RankNeuronId& source_neuron_id, const position_type& source_position, OctreeNode<AdditionalCellAttributes>* const root,
                                                                        const ElementType element_type, const SignalType signal_type, const double acceptance_criterion) {
        RelearnException::check(acceptance_criterion > 0.0,
                                "BarnesHutBase::find_target_neuron: The acceptance criterion was not positive: ({})", acceptance_criterion);
        RelearnException::check(acceptance_criterion <= Constants::bh_max_theta,
                                "BarnesHutBase::find_target_neuron: The acceptance criterion must not be larger than {}: ({})", Constants::bh_max_theta, acceptance_criterion);
        RelearnException::check(root != nullptr, "BarnesHutBase::find_target_neuron: root was nullptr");

        if (root->contains(source_neuron_id)) {
            return {};
        }

        if (root->is_leaf()) {
            return RankNeuronId{ root->get_mpi_rank(), root->get_cell_neuron_id() };
        }

        for (auto root_of_subtree = root; true;) {
            const auto& possible_targets = get_nodes_to_consider(cache, source_position, root_of_subtree, element_type, signal_type, acceptance_criterion);

            auto* node_selected = Kernel<AdditionalCellAttributes>::pick_target(kernel, source_neuron_id, source_position, possible_targets, element_type, signal_type);
            if (node_selected == nullptr) {
                return {};
            }

            // A chosen child is a valid target
            if (const auto done = node_selected->is_leaf(); done) {
                return RankNeuronId{ node_selected->get_mpi_rank(), node_selected->get_cell_neuron_id() };
            }

            // We need to choose again, starting from the chosen virtual neuron
            root_of_subtree = node_selected;
        }
    }

    /**
     * @brief Finds target neurons for a specified source neuron. No actual request is made.
     *      Might perform MPI communication via NodeCache::get_children()
     * @param kernel The probability kernel
     * @param cache The cache that contains the children of the nodes
     * @param source_neuron_id The source neuron's id
     * @param source_position The source neuron's position
     * @param number_vacant_elements The source neuron's number of vacant elements
     * @param root Where the source neuron should start to search for targets. It is not const because the children might be changed if the node is remote. Not nullptr
     * @param element_type The element type the source neuron searches
     * @param signal_type The signal type the source neuron searches
     * @param acceptance_criterion The acceptance criterion, must be > 0.0 and <= Constants::bh_max_theta
     * @exception Throws a RelearnException if the requirements on the acceptance criterion are not fulfilled or root is nullptr
     * @return A vector of pairs with (a) the target mpi rank and (b) the request for that rank
     */
    [[nodiscard]] static std::vector<std::pair<mpiPP::MPIRank, SynapseCreationRequest>> find_target_neurons(const KernelBase& kernel, const NodeCache<AdditionalCellAttributes>& cache, const RankNeuronId& source_neuron_id, const position_type& source_position,
                                                                                                            const counter_type number_vacant_elements, OctreeNode<AdditionalCellAttributes>* const root, const ElementType element_type, const SignalType signal_type, const double acceptance_criterion) {
        RelearnException::check(acceptance_criterion > 0.0,
                                "BarnesHutBase::find_target_neurons: The acceptance criterion was not positive: ({})", acceptance_criterion);
        RelearnException::check(acceptance_criterion <= Constants::bh_max_theta,
                                "BarnesHutBase::find_target_neurons: The acceptance criterion must not be larger than {}: ({})", Constants::bh_max_theta, acceptance_criterion);
        RelearnException::check(root != nullptr, "BarnesHutBase::find_target_neurons: root was nullptr");

        auto requests = std::vector<std::pair<mpiPP::MPIRank, SynapseCreationRequest>>{};
        requests.reserve(number_vacant_elements);

        for (auto j = 0U; j < number_vacant_elements; j++) {
            // Find one target at the time
            const auto& rank_neuron_id = find_target_neuron(kernel, cache, source_neuron_id, source_position, root, element_type, signal_type, acceptance_criterion);
            if (!rank_neuron_id.has_value()) {
                // If finding failed, it won't succeed in later iterations
                break;
            }

            const auto& [target_rank, target_id] = rank_neuron_id.value();
            const auto creation_request = SynapseCreationRequest(target_id, source_neuron_id.get_neuron_id(), signal_type);

            requests.emplace_back(target_rank, creation_request);
        }

        return requests;
    }

    /**
     * @brief Converts a chosen target node to a DistantNeuronRequest (with associated MPIRank).
     * @param source_neuron_id The source neuron's id
     * @param source_position The source neuron's position
     * @param target_node The chosen target node, not nullptr
     * @param signal_type The searched signal type
     * @param level_of_branch_nodes The level of branch nodes
     * @exception Throws a RelearnExcpetion if target_node is nullptr
     * @return If the target node cannot be used as-is (its level is smaller than the level of branch nodes), returns empty.
     *      Otherwise, constructs the correct DistantNeuronRequest and returns it
     */
    [[nodiscard]] static std::optional<std::pair<mpiPP::MPIRank, DistantNeuronRequest>> convert_target_node(const RankNeuronId& source_neuron_id, const position_type& source_position,
                                                                                                            const OctreeNode<AdditionalCellAttributes>* const target_node, const SignalType signal_type, const std::uint16_t level_of_branch_nodes) {
        RelearnException::check(target_node != nullptr, "BarnesHutBase::convert_target_node: target_node was nullptr");

        const auto& cell = target_node->get_cell();
        const auto target_rank = target_node->get_mpi_rank();

        if (const auto is_leaf = target_node->is_leaf(); is_leaf) {
            const auto neuron_request = DistantNeuronRequest(
                source_neuron_id.get_neuron_id(),
                source_position,
                cell.get_neuron_id().get_neuron_id(),
                DistantNeuronRequest::TargetNeuronType::Leaf,
                signal_type);

            return std::make_pair(target_rank, neuron_request);
        }

        if (target_node->get_level() < level_of_branch_nodes) {
            return {};
        }

        const auto neuron_request = DistantNeuronRequest(
            source_neuron_id.get_neuron_id(),
            source_position,
            cell.get_neuron_id().get_rma_offset(),
            DistantNeuronRequest::TargetNeuronType::VirtualNode,
            signal_type);

        return std::make_pair(target_rank, neuron_request);
    }

    /**
     * @brief Returns an optional pair of target rank and request. This is used for sending the request to the other rank and calculating further. No actual request is sent yet.
     *      Might perform MPI communication via NodeCache::get_children()
     * @param kernel The probability kernel
     * @param cache The cache that contains the children of the nodes
     * @param source_neuron_id The source neuron's id
     * @param source_position The source neuron's position
     * @param root The starting position where to look, not nullptr
     * @param element_type The element type the source is looking for
     * @param signal_type The signal type the source is looking for
     * @param level_of_branch_nodes The level of branch nodes in the Octree
     * @param acceptance_criterion The acceptance criterion, must be > 0.0 and <= Constants::bh_max_theta
     * @exception Throws a RelearnException if the requirements on the acceptance criterion are not fulfilled or root is nullptr
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorithm found a matching neuron, it is returned as the rank that shall further the calculations and the distant request
     */
    [[nodiscard]] static std::optional<std::pair<mpiPP::MPIRank, DistantNeuronRequest>> find_target_neuron_location_aware(const KernelBase& kernel, const NodeCache<AdditionalCellAttributes>& cache, const RankNeuronId& source_neuron_id, const position_type& source_position,
                                                                                                                          OctreeNode<AdditionalCellAttributes>* const root, const ElementType element_type, const SignalType signal_type, const std::uint16_t level_of_branch_nodes, const double acceptance_criterion) {
        RelearnException::check(acceptance_criterion > 0.0,
                                "BarnesHutBase::find_target_neuron_location_aware: The acceptance criterion was not positive: ({})", acceptance_criterion);
        RelearnException::check(acceptance_criterion <= Constants::bh_max_theta,
                                "BarnesHutBase::find_target_neuron_location_aware: The acceptance criterion must not be larger than {}: ({})", Constants::bh_max_theta, acceptance_criterion);
        RelearnException::check(root != nullptr, "BarnesHutBase::find_target_neuron_location_aware: root was nullptr");

        for (auto root_of_subtree = root; true;) {
            const auto& possible_targets = get_nodes_to_consider(cache, source_position, root_of_subtree, element_type, signal_type, acceptance_criterion, true);

            auto* node_selected = Kernel<AdditionalCellAttributes>::pick_target(kernel, source_neuron_id, source_position, possible_targets, element_type, signal_type);
            if (node_selected == nullptr) {
                return {};
            }

            const auto& request = convert_target_node(source_neuron_id, source_position, node_selected, signal_type, level_of_branch_nodes);
            if (request.has_value()) {
                return request;
            }

            // If the level of the selected node is too low, continue the search
            // We need to choose again, starting from the chosen virtual neuron
            root_of_subtree = node_selected;
        }
    }

    /**
     * @brief Finds target neurons for a specified source neuron. No actual request is made.
     *      Might perform MPI communication via NodeCache::get_children()
     * @param kernel The probability kernel
     * @param cache The cache that contains the children of the nodes
     * @param source_neuron_id The source neuron's id
     * @param source_position The source neuron's position
     * @param number_vacant_elements The number of vacant elements of the source neuron
     * @param root Where the source neuron should start to search for targets, not nullptr
     * @param element_type The element type the source neuron searches
     * @param signal_type The signal type the source neuron searches
     * @param level_of_branch_nodes The level of branch nodes in the Octree
     * @param acceptance_criterion The acceptance criterion, must be > 0.0 and <= Constants::bh_max_theta
     * @exception Throws a RelearnException if the requirements on the acceptance criterion are not fulfilled or root is nullptr
     * @return A vector of pairs with (a) the target mpi rank and (b) the request for that rank
     */
    [[nodiscard]] static std::vector<std::pair<mpiPP::MPIRank, DistantNeuronRequest>> find_target_neurons_location_aware(const KernelBase& kernel, const NodeCache<AdditionalCellAttributes>& cache, const RankNeuronId& source_neuron_id, const position_type& source_position,
                                                                                                                         const counter_type number_vacant_elements, OctreeNode<AdditionalCellAttributes>* const root, const ElementType element_type,
                                                                                                                         const SignalType signal_type, const std::uint16_t level_of_branch_nodes, const double acceptance_criterion) {

        RelearnException::check(acceptance_criterion > 0.0,
                                "BarnesHutBase::find_target_neurons_location_aware: The acceptance criterion was not positive: ({})", acceptance_criterion);
        RelearnException::check(acceptance_criterion <= Constants::bh_max_theta,
                                "BarnesHutBase::find_target_neurons_location_aware: The acceptance criterion must not be larger than {}: ({})", Constants::bh_max_theta, acceptance_criterion);
        RelearnException::check(root != nullptr, "BarnesHutBase::find_target_neurons_location_aware: root was nullptr");

        auto requests = std::vector<std::pair<mpiPP::MPIRank, DistantNeuronRequest>>{};
        requests.reserve(number_vacant_elements);

        for (auto j = 0U; j < number_vacant_elements; j++) {
            // Find one target at the time
            const auto& neuron_request = find_target_neuron_location_aware(kernel, cache, source_neuron_id, source_position, root, element_type, signal_type, level_of_branch_nodes, acceptance_criterion);
            if (!neuron_request.has_value()) {
                // If finding failed, it won't succeed in later iterations
                break;
            }

            requests.emplace_back(neuron_request.value());
        }

        return requests;
    }
};
