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
#include "algorithm/Internal/octree/Octree.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"
#include "util/Vec3.h"

#include "mpi-wrapper/MPIRank.h"

#include <range/v3/algorithm/for_each.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <stack>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

class OctreeAdapter {
public:
    template <typename AdditionalCellAttributes>
    static std::vector<std::pair<Vec3d, size_t>> extract_virtual_neurons(const OctreeNode<AdditionalCellAttributes>* root) {
        std::vector<std::pair<Vec3d, size_t>> return_value{};

        std::stack<std::pair<const OctreeNode<AdditionalCellAttributes>*, size_t>> octree_nodes{};
        octree_nodes.emplace(root, 0);

        while (!octree_nodes.empty()) {
            // Don't change this to a reference
            const auto [current_node, level] = octree_nodes.top();
            octree_nodes.pop();

            if (current_node->get_cell().get_neuron_id().is_virtual()) {
                return_value.emplace_back(current_node->get_cell().get_neuron_position().value(), level);
            }

            if (current_node->is_parent()) {
                const auto& childs = current_node->get_children();
                for (auto i = 0ULL; i < 8ULL; i++) {
                    const auto child = childs[i];
                    if (child != nullptr) {
                        octree_nodes.emplace(child, level + 1);
                    }
                }
            }
        }

        return return_value;
    }

    template <typename AdditionalCellAttributes>
    static std::vector<const OctreeNode<AdditionalCellAttributes>*> extract_leaf_nodes(const OctreeNode<AdditionalCellAttributes>* root) {
        std::vector<const OctreeNode<AdditionalCellAttributes>*> return_value{};

        std::stack<const OctreeNode<AdditionalCellAttributes>*> octree_nodes{};
        octree_nodes.push(root);

        while (!octree_nodes.empty()) {
            const OctreeNode<AdditionalCellAttributes>* current_node = octree_nodes.top();
            octree_nodes.pop();

            if (current_node->is_leaf()) {
                return_value.emplace_back(current_node);
                continue;
            }

            const auto& childs = current_node->get_children();
            for (auto* child : childs) {
                if (child != nullptr) {
                    octree_nodes.push(child);
                }
            }
        }

        return return_value;
    }

    template <typename AdditionalCellAttributes>
    static std::vector<OctreeNode<AdditionalCellAttributes>*> extract_leaf_nodes(OctreeNode<AdditionalCellAttributes>* root) {
        std::vector<OctreeNode<AdditionalCellAttributes>*> return_value{};

        std::stack<OctreeNode<AdditionalCellAttributes>*> octree_nodes{};
        octree_nodes.push(root);

        while (!octree_nodes.empty()) {
            OctreeNode<AdditionalCellAttributes>* current_node = octree_nodes.top();
            octree_nodes.pop();

            if (current_node->is_leaf()) {
                return_value.emplace_back(current_node);
                continue;
            }

            const auto& childs = current_node->get_children();
            for (auto* child : childs) {
                if (child != nullptr) {
                    octree_nodes.push(child);
                }
            }
        }

        return return_value;
    }

    template <typename AdditionalCellAttributes>
    static std::vector<OctreeNode<AdditionalCellAttributes>*> extract_inner_nodes(OctreeNode<AdditionalCellAttributes>* root) {
        std::vector<OctreeNode<AdditionalCellAttributes>*> return_value{};

        std::stack<OctreeNode<AdditionalCellAttributes>*> octree_nodes{};
        octree_nodes.push(root);

        while (!octree_nodes.empty()) {
            OctreeNode<AdditionalCellAttributes>* current_node = octree_nodes.top();
            octree_nodes.pop();

            if (current_node->is_leaf()) {
                continue;
            }

            return_value.emplace_back(current_node);

            const auto& childs = current_node->get_children();
            for (auto* child : childs) {
                if (child != nullptr) {
                    octree_nodes.push(child);
                }
            }
        }

        return return_value;
    }

    template <typename AdditionalCellAttributes>
    static std::vector<std::pair<Vec3d, NeuronID>> extract_neurons(const OctreeNode<AdditionalCellAttributes>* root) {
        std::vector<std::pair<Vec3d, NeuronID>> return_value{};

        std::stack<const OctreeNode<AdditionalCellAttributes>*> octree_nodes{};
        octree_nodes.push(root);

        while (!octree_nodes.empty()) {
            const OctreeNode<AdditionalCellAttributes>* current_node = octree_nodes.top();
            octree_nodes.pop();

            if (current_node->is_parent()) {
                const auto& childs = current_node->get_children();
                for (auto* child : childs) {
                    if (child != nullptr) {
                        octree_nodes.push(child);
                    }
                }
            } else {
                const Cell<AdditionalCellAttributes>& cell = current_node->get_cell();
                const auto neuron_id = cell.get_neuron_id();
                const auto& opt_position = cell.get_neuron_position();

                EXPECT_TRUE(opt_position.has_value());

                const auto& position = opt_position.value();

                if (neuron_id.is_initialized() && !neuron_id.is_virtual()) {
                    return_value.emplace_back(position, neuron_id);
                }
            }
        }

        return return_value;
    }

    template <typename AdditionalCellAttributes>
    static std::vector<std::pair<Vec3d, NeuronID>> extract_neurons_tree(const Octree<AdditionalCellAttributes>& octree) {
        const auto root = octree.get_root();
        if (root == nullptr) {
            return {};
        }

        return extract_neurons<AdditionalCellAttributes>(root);
    }

    template <typename AdditionalCellAttributes>
    static std::vector<OctreeNode<AdditionalCellAttributes>*> extract_branch_nodes(OctreeNode<AdditionalCellAttributes>* root, const std::uint8_t branch_node_level) {
        std::vector<OctreeNode<AdditionalCellAttributes>*> branch_nodes{};

        std::stack<OctreeNode<AdditionalCellAttributes>*> octree_nodes{};
        octree_nodes.push(root);

        while (!octree_nodes.empty()) {
            OctreeNode<AdditionalCellAttributes>* current_node = octree_nodes.top();
            octree_nodes.pop();

            if (current_node->get_level() == branch_node_level) {
                branch_nodes.emplace_back(current_node);
                continue;
            }

            for (auto* child : current_node->get_children()) {
                if (child != nullptr) {
                    octree_nodes.push(child);
                }
            }
        }

        return branch_nodes;
    }

    template <typename AdditionalCellAttributes>
    static void mark_node_as_distributed(OctreeNode<AdditionalCellAttributes>* root, const std::uint8_t level_of_branch_nodes) {
        std::vector<OctreeNode<AdditionalCellAttributes>*> branch_nodes{};
        branch_nodes.reserve(static_cast<size_t>(std::pow(8.0, level_of_branch_nodes) * 2));

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->get_level() == level_of_branch_nodes) {
                branch_nodes.emplace_back(current);
                continue;
            }

            if (current->get_level() < level_of_branch_nodes) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
                continue;
            }
        }

        auto current_rank = 0;
        for (auto* node : branch_nodes) {
            node->set_rank(mpiPP::MPIRank(current_rank));
            current_rank++;
        }

        auto mark_children = [](OctreeNode<AdditionalCellAttributes>* node) {
            auto rank = node->get_mpi_rank();

            auto children_stack = std::stack<OctreeNode<AdditionalCellAttributes>*>{};
            children_stack.push(node);

            while (!children_stack.empty()) {
                auto* current = children_stack.top();
                children_stack.pop();

                current->set_rank(rank);

                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        children_stack.push(child);
                    }
                }
            }
        };

        ranges::for_each(branch_nodes, mark_children);
    }

    template <typename AdditionalCellAttributes>
    static void invalidate_elements(OctreeNode<AdditionalCellAttributes>* root, const ElementType element_type, const SignalType signal_type) {
        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(root);

        while (!stack.empty()) {
            OctreeNode<AdditionalCellAttributes>* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                if (element_type == ElementType::Axon) {
                    if (signal_type == SignalType::Excitatory) {
                        current->set_cell_number_excitatory_axons(0);
                        current->set_cell_excitatory_axons_position({});
                    } else {
                        current->set_cell_number_inhibitory_axons(0);
                        current->set_cell_inhibitory_axons_position({});
                    }
                } else {
                    if (signal_type == SignalType::Excitatory) {
                        current->set_cell_number_excitatory_dendrites(0);
                        current->set_cell_excitatory_dendrites_position({});
                    } else {
                        current->set_cell_number_inhibitory_dendrites(0);
                        current->set_cell_inhibitory_dendrites_position({});
                    }
                }
                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(root);
    }

    template <typename AdditionalCellAttributes>
    static void increase_number_elements(OctreeNode<AdditionalCellAttributes>* root, const ElementType element_type, const SignalType signal_type) {
        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(root);

        while (!stack.empty()) {
            OctreeNode<AdditionalCellAttributes>* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                if (element_type == ElementType::Axon) {
                    if (signal_type == SignalType::Excitatory) {
                        current->set_cell_number_excitatory_axons((current->get_cell().get_number_excitatory_axons() * 2) + 1);
                    } else {
                        current->set_cell_number_inhibitory_axons((current->get_cell().get_number_inhibitory_axons() * 2) + 1);
                    }
                } else {
                    if (signal_type == SignalType::Excitatory) {
                        current->set_cell_number_excitatory_dendrites((current->get_cell().get_number_excitatory_dendrites() * 2) + 1);
                    } else {
                        current->set_cell_number_inhibitory_dendrites((current->get_cell().get_number_inhibitory_dendrites() * 2) + 1);
                    }
                }
                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(root);
    }

    template <typename AdditionalCellAttributes>
    static OctreeNode<AdditionalCellAttributes>* find_node(const RankNeuronId& rank_neuron_id, OctreeNode<AdditionalCellAttributes>* root) {
        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf() && current->get_mpi_rank() == rank_neuron_id.get_rank() && current->get_cell_neuron_id() == rank_neuron_id.get_neuron_id()) {
                return current;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        return nullptr;
    }

    template <typename AdditionalCellAttributes>
    static std::unordered_map<RankNeuronId, OctreeNode<AdditionalCellAttributes>*> find_nodes(OctreeNode<AdditionalCellAttributes>* root) {
        std::unordered_map<RankNeuronId, OctreeNode<AdditionalCellAttributes>*> mapping{};

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                const RankNeuronId rni{ current->get_mpi_rank(), current->get_cell_neuron_id() };
                mapping.emplace(rni, current);
                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }
        }

        return mapping;
    }

    template <typename AdditionalCellAttributes>
    static std::unordered_map<std::uint64_t, OctreeNode<AdditionalCellAttributes>*> find_child_offsets(OctreeNode<AdditionalCellAttributes>* root) {
        std::unordered_map<std::uint64_t, OctreeNode<AdditionalCellAttributes>*> mapping{};

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.push(root);

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            if (current->is_leaf()) {
                continue;
            }

            for (auto* child : current->get_children()) {
                if (child != nullptr) {
                    stack.push(child);
                }
            }

            const auto virtual_id = current->get_cell().get_neuron_id().get_rma_offset();
            mapping.emplace(virtual_id, current);
        }

        return mapping;
    }
};
