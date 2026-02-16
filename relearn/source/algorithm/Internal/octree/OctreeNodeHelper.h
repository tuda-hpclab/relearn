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

#include "algorithm/Internal/octree/Cell.h"
#include "algorithm/Internal/octree/NodeCache.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "util/RelearnException.h"

#include "cpp-utility/data-structure/Stack.hpp"

#include <optional>
#include <utility>
#include <vector>

template <typename AdditionalCellAttributes>
class OctreeNodeUpdater {
private:
    using position_type = typename Cell<AdditionalCellAttributes>::position_type;
    using counter_type = typename Cell<AdditionalCellAttributes>::counter_type;
    using box_size_type = typename Cell<AdditionalCellAttributes>::box_size_type;

    constexpr static bool has_excitatory_dendrite = AdditionalCellAttributes::has_excitatory_dendrite;
    constexpr static bool has_inhibitory_dendrite = AdditionalCellAttributes::has_inhibitory_dendrite;
    constexpr static bool has_excitatory_axon = AdditionalCellAttributes::has_excitatory_axon;
    constexpr static bool has_inhibitory_axon = AdditionalCellAttributes::has_inhibitory_axon;

public:
    /**
     * @brief Update the node based on its children. Saves the sum of vacant elements and the weighted average of the positions in the node
     * @param node The node to update
     * @exception Throws a RelearnException if node is nullptr or one of the children had vacant elements but not a position
     */
    static void update_node(OctreeNode<AdditionalCellAttributes>* node) {
        RelearnException::check(node != nullptr, "OctreeNodeUpdater::update_node: node was nullptr");

        if constexpr (has_excitatory_dendrite) {
            auto my_position = position_type{ 0.0, 0.0, 0.0 };
            auto my_free_elements = counter_type(0);

            for (const auto& child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                const auto& child_cell = child->get_cell();

                // Sum up number of free elements
                const auto child_free_elements = child_cell.get_number_excitatory_dendrites();
                my_free_elements += child_free_elements;

                const auto& opt_child_position = child_cell.get_excitatory_dendrites_position();

                // We can use position if it's valid or if corresponding number of elements is 0
                RelearnException::check(opt_child_position.has_value() || (0 == child_free_elements), "OctreeNodeUpdater::update_node: The child had excitatory dendrites, but no position. ID: {}", child_cell.get_neuron_id());

                if (opt_child_position.has_value()) {
                    const auto& child_position = opt_child_position.value();

                    const auto& [child_cell_xyz_min, child_cell_xyz_max] = child_cell.get_size();
                    const auto is_in_box = child_position.check_in_box(child_cell_xyz_min, child_cell_xyz_max);

                    RelearnException::check(is_in_box, "OctreeNodeUpdater::update_node: The excitatory dendrites of the child are not in its cell");

                    const auto& scaled_position = child_position * static_cast<double>(child_free_elements);
                    my_position += scaled_position;
                }
            }

            node->set_cell_number_excitatory_dendrites(my_free_elements);

            /**
             * For calculating the new weighted position, make sure that we don't
             * divide by 0. This happens if the my number of dendrites is 0.
             */
            if (0 == my_free_elements) {
                node->set_cell_excitatory_dendrites_position({});
            } else {
                const auto& [child_cell_xyz_min, child_cell_xyz_max] = node->get_cell().get_size();

                // This clipping exists because of potential rounding errors
                auto scaled_position = my_position / my_free_elements;
                scaled_position.clip_to_box(child_cell_xyz_min, child_cell_xyz_max);

                node->set_cell_excitatory_dendrites_position(std::optional<position_type>{ scaled_position });
            }
        }

        if constexpr (has_inhibitory_dendrite) {
            auto my_position = position_type{ 0.0, 0.0, 0.0 };
            auto my_free_elements = counter_type(0);

            for (const auto& child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                const auto& child_cell = child->get_cell();

                // Sum up number of free elements
                const auto child_free_elements = child_cell.get_number_inhibitory_dendrites();
                my_free_elements += child_free_elements;

                const auto& opt_child_position = child_cell.get_inhibitory_dendrites_position();

                // We can use position if it's valid or if corresponding number of elements is 0
                RelearnException::check(opt_child_position.has_value() || (0 == child_free_elements), "OctreeNodeUpdater::update_node: The child had inhibitory dendrites, but no position. ID: {}", child_cell.get_neuron_id());

                if (opt_child_position.has_value()) {
                    const auto& child_position = opt_child_position.value();

                    const auto& [child_cell_xyz_min, child_cell_xyz_max] = child_cell.get_size();
                    const auto is_in_box = child_position.check_in_box(child_cell_xyz_min, child_cell_xyz_max);

                    RelearnException::check(is_in_box, "OctreeNodeUpdater::update_node: The inhibitory dendrites of the child are not in its cell");

                    const auto& scaled_position = child_position * static_cast<double>(child_free_elements);
                    my_position += scaled_position;
                }
            }

            node->set_cell_number_inhibitory_dendrites(my_free_elements);

            /**
             * For calculating the new weighted position, make sure that we don't
             * divide by 0. This happens if the my number of dendrites is 0.
             */
            if (0 == my_free_elements) {
                node->set_cell_inhibitory_dendrites_position({});
            } else {
                const auto& [child_cell_xyz_min, child_cell_xyz_max] = node->get_cell().get_size();

                // This clipping exists because of potential rounding errors
                auto scaled_position = my_position / my_free_elements;
                scaled_position.clip_to_box(child_cell_xyz_min, child_cell_xyz_max);

                node->set_cell_inhibitory_dendrites_position(std::optional<position_type>{ scaled_position });
            }
        }

        if constexpr (has_excitatory_axon) {
            auto my_position = position_type{ 0.0, 0.0, 0.0 };
            auto my_free_elements = counter_type(0);

            for (const auto& child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                const auto& child_cell = child->get_cell();

                // Sum up number of free elements
                const auto child_free_elements = child_cell.get_number_excitatory_axons();
                my_free_elements += child_free_elements;

                const auto& opt_child_position = child_cell.get_excitatory_axons_position();

                // We can use position if it's valid or if corresponding number of elements is 0
                RelearnException::check(opt_child_position.has_value() || (0 == child_free_elements), "OctreeNodeUpdater::update_node: The child had excitatory axons, but no position. ID: {}", child_cell.get_neuron_id());

                if (opt_child_position.has_value()) {
                    const auto& child_position = opt_child_position.value();

                    const auto& [child_cell_xyz_min, child_cell_xyz_max] = child_cell.get_size();
                    const auto is_in_box = child_position.check_in_box(child_cell_xyz_min, child_cell_xyz_max);

                    RelearnException::check(is_in_box, "OctreeNodeUpdater::update_node: The excitatory axons of the child are not in its cell");

                    const auto& scaled_position = child_position * static_cast<double>(child_free_elements);
                    my_position += scaled_position;
                }
            }

            node->set_cell_number_excitatory_axons(my_free_elements);

            /**
             * For calculating the new weighted position, make sure that we don't
             * divide by 0. This happens if the my number of axons is 0.
             */
            if (0 == my_free_elements) {
                node->set_cell_excitatory_axons_position({});
            } else {
                const auto& [child_cell_xyz_min, child_cell_xyz_max] = node->get_cell().get_size();

                // This clipping exists because of potential rounding errors
                auto scaled_position = my_position / my_free_elements;
                scaled_position.clip_to_box(child_cell_xyz_min, child_cell_xyz_max);

                node->set_cell_excitatory_axons_position(std::optional<position_type>{ scaled_position });
            }
        }

        if constexpr (has_inhibitory_axon) {
            auto my_position = position_type{ 0.0, 0.0, 0.0 };
            auto my_free_elements = counter_type(0);

            for (const auto& child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                const auto& child_cell = child->get_cell();

                // Sum up number of free elements
                const auto child_free_elements = child_cell.get_number_inhibitory_axons();
                my_free_elements += child_free_elements;

                const auto& opt_child_position = child_cell.get_inhibitory_axons_position();

                // We can use position if it's valid or if corresponding number of elements is 0
                RelearnException::check(opt_child_position.has_value() || (0 == child_free_elements), "OctreeNodeUpdater::update_node: The child had inhibitory axons, but no position. ID: {}", child_cell.get_neuron_id());

                if (opt_child_position.has_value()) {
                    const auto& child_position = opt_child_position.value();

                    const auto& [child_cell_xyz_min, child_cell_xyz_max] = child_cell.get_size();
                    const auto is_in_box = child_position.check_in_box(child_cell_xyz_min, child_cell_xyz_max);

                    RelearnException::check(is_in_box, "OctreeNodeUpdater::update_node: The inhibitory axons of the child are not in its cell");

                    const auto& scaled_position = child_position * static_cast<double>(child_free_elements);
                    my_position += scaled_position;
                }
            }

            node->set_cell_number_inhibitory_axons(my_free_elements);

            /**
             * For calculating the new weighted position, make sure that we don't
             * divide by 0. This happens if the my number of axons is 0.
             */
            if (0 == my_free_elements) {
                node->set_cell_inhibitory_axons_position({});
            } else {
                const auto& [child_cell_xyz_min, child_cell_xyz_max] = node->get_cell().get_size();

                // This clipping exists because of potential rounding errors
                auto scaled_position = my_position / my_free_elements;
                scaled_position.clip_to_box(child_cell_xyz_min, child_cell_xyz_max);

                node->set_cell_inhibitory_axons_position(std::optional<position_type>{ scaled_position });
            }
        }
    }

    /**
     * @brief Updates the tree until the desired level.
     *      Uses OctreeNode::get_level() to determine the depth. The nodes at that depth are still updated, but not their children.
     * @param tree The root of the tree from where to update
     * @param max_depth The depth where the updates shall stop
     * @exception Throws a RelearnException if tree is nullptr or if max_depth is smaller than the depth of local_tree_root
     */
    static void update_tree(OctreeNode<AdditionalCellAttributes>* tree, const std::uint16_t max_depth = std::numeric_limits<std::uint16_t>::max()) {
        struct StackElement {
        private:
            OctreeNode<AdditionalCellAttributes>* ptr{ nullptr };

            // True if node has been on stack already
            // twice and can be visited now
            bool already_visited{ false };

        public:
            /**
             * @brief Constructs a new object that holds the given node, which is marked as not already visited
             * @param octree_node The node that should be visited, not nullptr
             * @exception Throws a RelearnException if octree_node is nullptr
             */
            explicit StackElement(OctreeNode<AdditionalCellAttributes>* octree_node)
                : ptr(octree_node) {
                RelearnException::check(octree_node != nullptr, "StackElement::StackElement: octree_node was nullptr");
            }

            /**
             * @brief Returns the node
             * @return The node
             */
            [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_octree_node() const noexcept {
                return ptr;
            }

            /**
             * @brief Sets the flag that indicated if this node was already visited
             * @exception Throws a RelearnException if this node was already visited before
             */
            void set_visited() {
                RelearnException::check(!already_visited, "StackElement::set_visited: element is already visited");
                already_visited = true;
            }

            /**
             * @brief Returns the flag indicating if this node was already visited
             * @return True iff the node was already visited
             */
            [[nodiscard]] bool was_already_visited() const noexcept {
                return already_visited;
            }
        };

        RelearnException::check(tree != nullptr, "OctreeNodeUpdater::update_tree: tree is nullptr.");
        RelearnException::check(tree->get_level() <= max_depth, "OctreeNodeUpdater::update_tree: The root had a larger depth than max_depth.");

        auto stack = utility::Stack<StackElement>{};
        stack.emplace_back(tree);

        while (!stack.empty()) {
            auto& current_element = stack.top();
            auto* current_octree_node = current_element.get_octree_node();

            if (current_element.was_already_visited()) {
                // Make sure that the element was visited before, i.e., its children are processed
                if (current_octree_node->is_parent()) {
                    // Don't update leaf nodes, they were updated before
                    OctreeNodeUpdater<AdditionalCellAttributes>::update_node(current_octree_node);
                }

                stack.pop();
                continue;
            }

            // Mark node to be visited next time now, because it's a reference and will change once we push the other elements
            current_element.set_visited();

            const auto current_depth = current_octree_node->get_level();
            if (current_depth >= max_depth) {
                // We're at the border of where we want to update, so don't push children
                if (current_octree_node->is_parent()) {
                    // Don't update leaf nodes, they were updated before
                    OctreeNodeUpdater<AdditionalCellAttributes>::update_node(current_octree_node);
                }

                stack.pop();
                continue;
            }

            for (auto* child : current_octree_node->get_children()) {
                if (child == nullptr) {
                    continue;
                }
                stack.emplace_back(child);
            }
        } /* while */
    }
};

template <typename AdditionalCellAttributes>
class OctreeNodeExtractor {
public:
    using position_type = typename Cell<AdditionalCellAttributes>::position_type;
    using counter_type = typename Cell<AdditionalCellAttributes>::counter_type;
    using box_size_type = typename Cell<AdditionalCellAttributes>::box_size_type;

private:
    constexpr static bool has_excitatory_dendrite = AdditionalCellAttributes::has_excitatory_dendrite;
    constexpr static bool has_inhibitory_dendrite = AdditionalCellAttributes::has_inhibitory_dendrite;
    constexpr static bool has_excitatory_axon = AdditionalCellAttributes::has_excitatory_axon;
    constexpr static bool has_inhibitory_axon = AdditionalCellAttributes::has_inhibitory_axon;

public:
    /**
     * @brief Returns a vector of the positions and the number of free elements of the specified type.
     *      Filters out all all neurons that don't have a vacant element of the specified type.
     * @param node The induced subtree from where to extract the positions
     * @param element_type The requested ElementType
     * @param signal_type The requested SignalType
     * @exception Throws a RelearnException if node == nullptr or if the specified combination of ElementType and SignalType is not present in Cell
     * @return A vector of the actual positions and the number of vacant elements
     */
    [[nodiscard]] static std::vector<std::pair<position_type, counter_type>> get_all_positions_for(OctreeNode<AdditionalCellAttributes>* node, const NodeCache<AdditionalCellAttributes>& cache,
                                                                                                   const ElementType element_type, const SignalType signal_type) {
        RelearnException::check(node != nullptr, "OctreeNodeExtractor::get_all_positions_for: node is nullptr");

        if (element_type == ElementType::Axon && signal_type == SignalType::Excitatory) {
            RelearnException::check(has_excitatory_axon, "OctreeNodeExtractor::get_all_positions_for: Requested excitatory axon, but there are none based on the AdditionalCellAttributes");
        } else if (element_type == ElementType::Axon && signal_type == SignalType::Inhibitory) {
            RelearnException::check(has_inhibitory_axon, "OctreeNodeExtractor::get_all_positions_for: Requested inhibitory axon, but there are none based on the AdditionalCellAttributes");
        } else if (element_type == ElementType::Dendrite && signal_type == SignalType::Excitatory) {
            RelearnException::check(has_excitatory_dendrite, "OctreeNodeExtractor::get_all_positions_for: Requested excitatory dendrite, but there are none based on the AdditionalCellAttributes");
        } else {
            RelearnException::check(has_inhibitory_dendrite, "OctreeNodeExtractor::get_all_positions_for: Requested inhibitory dendrite, but there are none based on the AdditionalCellAttributes");
        }

        auto result = std::vector<std::pair<position_type, counter_type>>{};
        result.reserve(30);

        auto stack = utility::Stack<OctreeNode<AdditionalCellAttributes>*>{ 30 };
        stack.emplace_back(node);

        while (!stack.empty()) {
            auto* current_node = stack.pop_back();
            if (current_node == nullptr) {
                continue;
            }

            if (current_node->is_leaf()) {
                // Get number and position, depending on which types were chosen.
                const auto& cell = current_node->get_cell();
                const auto& opt_position = cell.get_position_for(element_type, signal_type);
                RelearnException::check(opt_position.has_value(), "OctreeNodeExtractor::get_all_positions_for: opt_position has no value.");

                const auto number_elements = cell.get_number_elements_for(element_type, signal_type);
                if (number_elements != 0) {
                    result.emplace_back(opt_position.value(), number_elements);
                }
                continue;
            }

            const auto& children = cache.get_children(current_node);
            for (auto* child : children) {
                if (child == nullptr) {
                    continue;
                }

                if (const auto number_elements = child->get_cell().get_number_elements_for(element_type, signal_type); number_elements == 0) {
                    continue;
                }

                // push children to stack that have relevant elements
                stack.emplace_back(child);
            }
        }

        return result;
    }
};
