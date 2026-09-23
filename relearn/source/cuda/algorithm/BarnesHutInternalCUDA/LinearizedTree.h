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

#include "BarnesHutCUDACell.h"
#include "Config.h"

#include "algorithm/Internal/octree/Octree.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "cuda/CudaConfig.h"
#include "cuda/CudaConversion.h"
#include "cuda/CudaTypes.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "util/RelearnException.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class LinearizedTree {
    using AdditionalCellAttributes = BarnesHutCUDACell;
    using index_type = CudaConfig::bh_index_type;
    using level_type = std::uint16_t;

public:
    using counter_type = RelearnTypes::counter_type;

    /**
     * @brief The struct holds the positional data, vacant axonal and dendritic elements for all neurons in a network, values are zero by default
     */
    struct NeuronDetails {
        std::span<const SimpleVec3d> position{};
        std::span<const CudaConfig::synaptic_count_type> vacant_axonal_elements{};
        std::span<const CudaConfig::synaptic_count_type> vacant_dendritic_elements{};
        std::span<const CudaConfig::mpi_rank_type> ranks{};
    };

    /**
     * @brief Construct a new LinearizedTree from an octree with given signal types, vacant axonal elements and vacant dendritic elements
     * @param octree Octree root
     * @param node_signal_types Span of signal types
     * @param vacant_axonal_elements Span of vacant axonal elements
     * @param vacant_dendritic_elements Span of vacant dendritic elements, optional
     * @exception Throws a RelearnException if the provided spans differ in size
     */
    LinearizedTree(const std::shared_ptr<Octree<AdditionalCellAttributes>>& octree,
                   const std::span<const SignalType>& node_signal_types,
                   const std::span<const counter_type>& vacant_axonal_elements,
                   const std::optional<const std::span<const counter_type>>& vacant_dendritic_elements = std::nullopt) {
        RelearnException::check(
            vacant_axonal_elements.size() == node_signal_types.size() && (!vacant_dendritic_elements.has_value() || vacant_dendritic_elements.value().size() == node_signal_types.size()),
            "The length of the given vectors differ from length of LinearizedTree");

        const auto my_rank = mpiPP::MPIInfo::get_my_rank();

        auto queue = std::queue<OctreeNode<AdditionalCellAttributes>*>();

        const auto* root = octree->get_root();
        const auto children = root->get_children();
        const auto octree_branch_nodes = octree->get_local_branch_nodes();
        std::unordered_set<NeuronID> local_branch_ids{};
        for (auto* const node : octree_branch_nodes) {
            local_branch_ids.insert(node->get_cell_neuron_id());
        }

        // fill queue from root
        for (const auto& child : children) {
            queue.push(child);
        }
        auto next_child_index = static_cast<CudaConfig::bh_index_type>(queue.size());

        auto leaf_neuron_ids = std::vector<CudaConfig::number_neurons_type>(0);
        auto leafs_inserted = 0U;

        while (!queue.empty()) {
            // set info for LinearizedTree

            if (const auto* const child = queue.front(); child == nullptr) {
                push_placeholder();
            } else {
                const auto level = child->get_level();
                const auto child_cell = child->get_cell();
                const auto child_rank = child->get_mpi_rank().get_rank();
                const auto subdomain_length = child_cell.get_maximal_dimension_difference();
                const auto is_parent = child->is_parent();
                const auto neuron_id = child_cell.get_neuron_id();
                const auto remote_node = neuron_id.is_virtual() && child_rank != my_rank.get_rank();

                if (!is_parent) {
                    // Workaround for OctreeNode.insert behaving weirdly in tests
                    if (leafs_inserted >= vacant_axonal_elements.size()) {
                        break;
                    }

                    RelearnException::check(child_cell.get_neuron_position().has_value(),
                                            "Neuron in LinearizedTree must have a position");
                    const auto position = child_cell.get_neuron_position().value();
                    const auto signal_type = node_signal_types[neuron_id.get_neuron_id()];

                    const auto num_free_axons = vacant_axonal_elements[neuron_id.get_neuron_id()];

                    const auto num_free_exc_dendrites = child_cell.get_number_elements_for(ElementType::Dendrite,
                                                                                           SignalType::Excitatory); // : vacant_dendritic_elements.value()[neuron_id.get_neuron_id()];
                    const auto num_free_inh_dendrites = child_cell.get_number_elements_for(ElementType::Dendrite,
                                                                                           SignalType::Inhibitory);

                    sum_vacant_axons += num_free_axons;

                    push_leaf(static_cast<CudaConfig::number_neurons_type>(neuron_id.get_neuron_id()),
                              SimpleVec3d{ utility::cast<double>(position.get_x()), utility::cast<double>(position.get_y()), utility::cast<double>(position.get_z()) }, num_free_axons,
                              num_free_exc_dendrites, num_free_inh_dendrites, signal_type, level, subdomain_length,
                              child_rank);

                    leaf_neuron_ids.emplace_back(size() - 1);
                    leafs_inserted++;
                } else if (remote_node) {
                    const auto rma_index = static_cast<CudaConfig::number_neurons_type>(neuron_id.get_rma_offset());
                    push_remote_node(rma_index, level, subdomain_length, child_rank);
                    remote_nodes_rank_rma_to_index[RankNeuronId{ mpiPP::MPIRank{ child_rank },
                                                                 NeuronID::virtual_id(rma_index) }] = static_cast<CudaConfig::bh_index_type>(size()) - 1U;
                } else {
                    for (const auto& sub_child : child->get_children()) {
                        queue.push(sub_child);
                    }

                    const auto n_id = child_cell.get_neuron_id();
                    if (local_branch_ids.contains(n_id)) {
                        const auto idx = static_cast<CudaConfig::bh_index_type>(subdomain_lengths.size());
                        local_branch_nodes.emplace_back(idx);
                    }

                    push_parent(next_child_index, level, subdomain_length, child_rank,
                                child_cell.get_neuron_id().get_rma_offset());
                    next_child_index += Constants::number_oct;
                }
            }
            // pop first element of queue
            queue.pop();
        }
        leaf_neuron_id_mapping = leaf_neuron_ids;
        remove_dangling_children();
        calculate_parent_indices();
        calculate_level_indices();
        shrink();
    }

    /**
     * @brief Construct a new LinearizedTree from an octree and the synaptic elements
     * @param octree Octree root
     * @param synaptic_elements Synaptic elements
     */
    LinearizedTree(const std::shared_ptr<Octree<AdditionalCellAttributes>>& octree,
                   const std::shared_ptr<SynapticElements>& synaptic_elements)
        : LinearizedTree(octree, synaptic_elements->get_signal_types(),
                         synaptic_elements->get_vacant_elements(SynapticElementType::Axon)) { }

    /**
     * @brief Construct a new LinearizedTree from an octree and a single SignalType and vectors for the vacant axonal and dendritic elements
     * @param octree Octree root
     * @param signal_type Signal type
     * @param vacant_axonal_elements  Vector of vacant axonal elements
     * @param vacant_dendritic_elements Vector of vacant dendritic elements
     * @exception Throws a new RelearnException if the given vectors differ in size
     */
    LinearizedTree(const std::shared_ptr<Octree<AdditionalCellAttributes>>& octree, const SignalType signal_type,
                   std::span<const CudaConfig::synaptic_count_type> vacant_axonal_elements,
                   std::span<const CudaConfig::synaptic_count_type> vacant_dendritic_elements)
        : LinearizedTree(octree, std::vector(vacant_axonal_elements.size(), signal_type), vacant_axonal_elements,
                         vacant_dendritic_elements) { }

    /**
     * Constructor for an empty LinearizedTree with a given size
     * @param size Size of the LinearizedTree
     */
    explicit LinearizedTree(const CudaConfig::bh_index_type size) {
        node_types.resize(size);
        neuron_ids.resize(size);
        child_indices.resize(size);
        parent_indices.resize(size);
        levels.resize(size);
        signal_types.resize(size);
        subdomain_lengths.resize(size);

        neuron_details_ex.init_empty(size);
        neuron_details_inh.init_empty(size);
    }

    ~LinearizedTree() = default;

    LinearizedTree(const LinearizedTree&) = default;
    LinearizedTree& operator=(const LinearizedTree&) = default;
    LinearizedTree(LinearizedTree&&) = default;
    LinearizedTree& operator=(LinearizedTree&&) = default;

    /**
     * @brief  Update linearized tree with synaptic elements
     * @param synaptic_elements The synaptic elements
     */
    void update_linearized_tree_on_host(const std::shared_ptr<SynapticElements>& synaptic_elements) {
        const auto& vacant_dendex = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
        const auto& vacant_dendinh = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);
        const auto& vacant_axon = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
        const auto& _signal_types = synaptic_elements->get_signal_types();

        sum_vacant_axons = 0;
        neuron_details_ex.reset();
        neuron_details_inh.reset();

        for (auto i = 0U; i < size(); ++i) {
            const auto node_type = node_types[i];
            if (node_type == NodeType::Leaf) {
                const auto neuron_id = NeuronID{ neuron_ids[i] };

                const auto signal_type = _signal_types[neuron_id.get_neuron_id()];
                const auto num_vacant_axons = vacant_axon[neuron_id.get_neuron_id()];

                sum_vacant_axons += num_vacant_axons;

                if (signal_type == SignalType::Excitatory) { // should be the same as child->has_excitatory_dendrite()?
                    update_leaf(SignalType::Excitatory, i, num_vacant_axons, vacant_dendex[neuron_id.get_neuron_id()]);
                    update_leaf(SignalType::Inhibitory, i, 0, vacant_dendinh[neuron_id.get_neuron_id()]);
                } else {
                    update_leaf(SignalType::Excitatory, i, 0, vacant_dendex[neuron_id.get_neuron_id()]);
                    update_leaf(SignalType::Inhibitory, i, num_vacant_axons, vacant_dendinh[neuron_id.get_neuron_id()]);
                }
            } else if (node_type == NodeType::RemoteNode) {
                // Remote nodes are updated separately
                continue;
            } else {
                update_non_leaf();
            }
        }
    }

    [[nodiscard]] std::span<const CudaConfig::bh_index_type> get_local_branch_nodes() const {
        return local_branch_nodes;
    }

    /**
     * @brief Update all vacant dendritic elements of a specific SignalType
     * @param vacant_dendritic_elements Vector containing number of vacant dendritic elements
     * @param signal_type The signal type
     * @exception Throws RelearnException if given vacant_dendritic_elements vector differs in size from LinearizedTree
     */
    void update_dendritic_elements(std::span<const CudaConfig::synaptic_count_type> vacant_dendritic_elements,
                                   const SignalType& signal_type) {
        RelearnException::check(vacant_dendritic_elements.size() == size(),
                                "LinearizedTree::update_dendritic_elements: Length of parameter differs from length of LinearizedTree");
        auto& neuron_details = get_details(signal_type);
        neuron_details.set_vacant_dendritic_elements(vacant_dendritic_elements);
    }

    /**
     * @brief Update the positions of all neurons of a specific signal type
     * @param positions The positions that will be used to update
     * @param signal_type The signal type that is to be updated
     */
    void set_positions(std::span<const SimpleVec3d> positions, const SignalType& signal_type) {
        RelearnException::check(positions.size() == size(),
                                "LinearizedTree::set_positions: Length of parameter differs from length of LinearizedTree");
        auto& details = get_details(signal_type);
        details.set_positions(positions);
    }

    /**
     * @brief Set SignalType specific details for a neuron with a given neuron id
     * @param type The signal type which details will be updated
     * @param neuron_id The id of the neuron that will be updated
     * @param vacant_axonal_elements A number of vacant axonal elements, overrides the current value. If no value is given, override existing value with 0
     * @param vacant_dendritic_elements A number of vacant dendritic elements, overrides the current value. If no value is given, override existing value with 0
     */
    void set_neuron_details(const SignalType& type, const NeuronID& neuron_id,
                            const CudaConfig::synaptic_count_type vacant_axonal_elements = 0,
                            const CudaConfig::synaptic_count_type vacant_dendritic_elements = 0) {
        leaf_neuron_id_mapping.clear();
        leaf_neuron_id_mapping.shrink_to_fit();

        sum_vacant_axons += vacant_axonal_elements;

        for (auto i = 0U; i < size(); ++i) {
            if (neuron_ids[i] == neuron_id.get_neuron_id()) {
                update_leaf(type, i, vacant_axonal_elements, vacant_dendritic_elements);
                break;
            }
        }
    }

    /**
     * @brief Returns an instance of a NeuronDetails SoA for a given SignalType
     * @param type The signal type
     * @return A NeuronDetails struct for the given signal type
     */
    [[nodiscard]] NeuronDetails get_neuron_details(const SignalType type) const {
        const auto& neuron_details = get_details(type);

        return NeuronDetails{
            .position = neuron_details.get_positions(),
            .vacant_axonal_elements = neuron_details.get_vacant_axonal_elements(),
            .vacant_dendritic_elements = neuron_details.get_vacant_dendritic_elements(),
            .ranks = neuron_details.get_ranks()
        };
    }

    /**
     * @brief Return the number of vacant axonal elements there are for different signal types
     * @param type the signal type
     * @return the number of vacant axonal elements for the given signal type
     */
    CudaConfig::synaptic_count_type get_sum_vacant_axons(const SignalType& type) {
        const auto& neuron_details = get_details(type);
        return neuron_details.get_start_index_count();
    }

    [[nodiscard]] std::vector<CudaConfig::number_neurons_type> get_neuron_ids() const {
        return neuron_ids;
    }

    [[nodiscard]] std::vector<NodeType> get_node_types() const {
        return node_types;
    }

    [[nodiscard]] std::vector<index_type> get_level_indices() const {
        return level_indices;
    }

    [[nodiscard]] std::vector<index_type> get_child_indices() const {
        return child_indices;
    }

    [[nodiscard]] std::vector<index_type> get_parent_indices() const {
        return parent_indices;
    }

    [[nodiscard]] std::vector<CudaConfig::gaussian_type> get_subdomain_lengths() const {
        return subdomain_lengths;
    }

    [[nodiscard]] CudaConfig::synaptic_count_type get_sum_vacant_axons() const {
        return sum_vacant_axons;
    }

    [[nodiscard]] std::size_t size() const {
        return node_types.size();
    }

    [[nodiscard]] std::span<const CudaConfig::bh_index_type> get_rma_offset_to_neuron_id() const {
        return rma_offset_to_neuron_id;
    }

    /**
     * @brief Prints all attributes of all nodes of the LinearizedTree in a table format
     * @param ss StringStream
     */
    void print(std::stringstream& ss) const {
        ss
            << "\ni\ttype\tneuronID\tchildren_begin\tparent_index\tvae\tvde\tsignal type\tlevel\tsubdomain_length\ttarget_start_index\tposition\n";
        for (std::size_t i = 0; i < neuron_ids.size(); ++i) {
            const auto& details = get_details(signal_types[i].value_or(SignalType::Excitatory)).get_details(i);
            const auto& [x, y, z] = details.position;

            auto parent_idx_str = std::string(" ");
            if (parent_indices[i] != std::numeric_limits<index_type>::max()) {
                parent_idx_str = std::to_string(parent_indices[i]);
            }

            const auto* signal_type_str = " ";
            if (signal_types[i].has_value()) {
                switch (signal_types[i].value()) {
                case SignalType::Excitatory:
                    signal_type_str = "Ex";
                    break;
                case SignalType::Inhibitory:
                    signal_type_str = "In";
                    break;
                }
            }

            auto node_type = ' ';
            switch (node_types[i]) {
            case NodeType::Leaf:
                node_type = 'l';
                break;
            case NodeType::Placeholder:
                node_type = '0';
                break;
            case NodeType::VirtualNode:
                node_type = 'v';
                break;
            case NodeType::RemoteNode:
                node_type = 'r';
                break;
            }

            ss << i << "\t" << node_type << "\t" << neuron_ids[i] << "\t" << child_indices[i] << "\t"
               << parent_idx_str << "\t" << details.vacant_axonal_elements << "\t" << details.vacant_dendritic_elements
               << "\t"
               << signal_type_str << "\t"
               << levels[i] << "\t" << subdomain_lengths[i] << "\t"
               << "(" << x << ", " << y << ", " << z << ")\n";
        }
    }

    [[nodiscard]] const std::unordered_map<RankNeuronId, CudaConfig::bh_index_type>& get_remote_nodes_rank_rma_to_index() const {
        return remote_nodes_rank_rma_to_index;
    }

private:
    struct InternalNeuron {
        SimpleVec3d position;
        CudaConfig::synaptic_count_type vacant_axonal_elements;
        CudaConfig::synaptic_count_type vacant_dendritic_elements;
        CudaConfig::mpi_rank_type rank;
    };

    class InternalNeuronDetails {
    public:
        void init_empty(const CudaConfig::bh_index_type size) {
            positions.resize(size);
            vacant_axonal_elements.resize(size);
            vacant_dendritic_elements.resize(size);
            ranks.resize(size);
            start_index_count = size;
        }

        void add_neuron(const SimpleVec3d& position, const CudaConfig::synaptic_count_type vae,
                        const CudaConfig::synaptic_count_type vde, const CudaConfig::mpi_rank_type rank) {
            positions.emplace_back(position);
            ranks.emplace_back(rank);
            vacant_axonal_elements.emplace_back(vae);
            vacant_dendritic_elements.emplace_back(vde);
            const auto number_vacant_axonal_elements = vae > 0 ? vae : 1;
            start_index_count += static_cast<CudaConfig::bh_index_type>(number_vacant_axonal_elements);
        }

        void add_placeholder(const CudaConfig::mpi_rank_type rank) {
            positions.emplace_back(SimpleVec3d{ 0, 0, 0 });
            vacant_axonal_elements.emplace_back(0);
            vacant_dendritic_elements.emplace_back(0);
            ranks.emplace_back(rank);
            ++start_index_count;
        }

        void update_leaf(const index_type index, const CudaConfig::synaptic_count_type vae,
                         const CudaConfig::synaptic_count_type vde) {
            vacant_axonal_elements[index] = vae;
            vacant_dendritic_elements[index] = vde;
            const auto number_vacant_axonal_elements = vae > 0 ? vae : 1;
            start_index_count += static_cast<CudaConfig::bh_index_type>(number_vacant_axonal_elements);
        }

        void set_position(const index_type index, const SimpleVec3d& pos) {
            RelearnException::check(index < positions.size(),
                                    "InternalNeuronDetails::set_position: Out of bounds index {} >= {}", index,
                                    positions.size());
            positions[index] = pos;
        }

        void set_free_elements(const index_type index, const CudaConfig::synaptic_count_type vae,
                               const CudaConfig::synaptic_count_type vde) {
            RelearnException::check(index < positions.size(),
                                    "InternalNeuronDetails::set_position: Out of bounds index {} >= {}", index,
                                    positions.size());
            vacant_axonal_elements[index] = vae;
            vacant_dendritic_elements[index] = vde;
        }

        void advance_start_index() { ++start_index_count; }

        void reset() {
            start_index_count = 0;
        }

        void shrink() {
            vacant_axonal_elements.shrink_to_fit();
            vacant_dendritic_elements.shrink_to_fit();
            positions.shrink_to_fit();
            ranks.shrink_to_fit();
        }

        [[nodiscard]] InternalNeuron get_details(const std::size_t i) const {
            return { positions[i], vacant_axonal_elements[i], vacant_dendritic_elements[i],
                     ranks[i] };
        }

        void
        set_vacant_dendritic_elements(std::span<const CudaConfig::synaptic_count_type> _vacant_dendritic_elements) {
            vacant_dendritic_elements.assign(_vacant_dendritic_elements.begin(), _vacant_dendritic_elements.end());
        }

        void set_positions(std::span<const SimpleVec3d> _positions) {
            positions.assign(_positions.begin(), _positions.end());
        }

        [[nodiscard]] std::span<const CudaConfig::synaptic_count_type> get_vacant_axonal_elements() const {
            return vacant_axonal_elements;
        }

        [[nodiscard]] std::span<const CudaConfig::synaptic_count_type> get_vacant_dendritic_elements() const {
            return vacant_dendritic_elements;
        }

        [[nodiscard]] std::span<const SimpleVec3d> get_positions() const {
            return positions;
        }

        [[nodiscard]] std::span<const CudaConfig::mpi_rank_type> get_ranks() const {
            return ranks;
        }

        [[nodiscard]] index_type get_start_index_count() const {
            return start_index_count;
        }

    private:
        std::vector<CudaConfig::synaptic_count_type> vacant_axonal_elements{};
        std::vector<CudaConfig::synaptic_count_type> vacant_dendritic_elements{};
        std::vector<SimpleVec3d> positions{};
        std::vector<CudaConfig::mpi_rank_type> ranks{};
        index_type start_index_count = 0;
    };

    void calculate_level_indices() {
        auto _levels = std::vector<index_type>(1, 0);
        auto max_level = 0;
        for (auto i = 0U; i < levels.size(); ++i) {
            if (const auto level = levels[i]; level > max_level) {
                if (level != std::numeric_limits<level_type>::max()) {
                    max_level = level;
                    _levels.emplace_back(i - i % 8); // given, that every neuron has 8 children in LinearizedTree. If not replace the subtrahend with (i % neurons_count[i])
                }
            }
        }
        level_indices = _levels;
    }

    void push_leaf(const CudaConfig::number_neurons_type neuron_id, const SimpleVec3d& position,
                   const CudaConfig::synaptic_count_type vacant_axonal_elements,
                   const CudaConfig::synaptic_count_type vacant_exc_dendritic_elements,
                   const CudaConfig::synaptic_count_type vacant_inh_dendritic_elements, const SignalType& signal_type,
                   const level_type level, const CudaConfig::gaussian_type subdomain_length, const CudaConfig::mpi_rank_type rank) {
        node_types.emplace_back(NodeType::Leaf);
        neuron_ids.emplace_back(neuron_id);
        child_indices.emplace_back(0);
        parent_indices.emplace_back(std::numeric_limits<index_type>::max());
        levels.emplace_back(level);
        signal_types.emplace_back(signal_type);
        subdomain_lengths.emplace_back(subdomain_length);
        RelearnException::check(rank != std::numeric_limits<CudaConfig::mpi_rank_type>::max(),
                                "LinearizeTree::push_leaf: Pushes leaf with invalid rank");

        neuron_details_ex.add_neuron(position,
                                     signal_type == SignalType::Excitatory ? vacant_axonal_elements : 0,
                                     vacant_exc_dendritic_elements, rank);
        neuron_details_inh.add_neuron(position,
                                      signal_type == SignalType::Inhibitory ? vacant_axonal_elements : 0,
                                      vacant_inh_dendritic_elements, rank);
    }

    void push_remote_node(const CudaConfig::number_neurons_type rma_offset, const level_type level,
                          const CudaConfig::gaussian_type subdomain_length, const CudaConfig::mpi_rank_type rank) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();
        RelearnException::check(rank != my_rank, "LinearizedTree::push_remote_node: Remote node is local");
        RelearnException::check(rank != std::numeric_limits<CudaConfig::mpi_rank_type>::max(),
                                "LinearizedTree::push_remote_node: Remote node is invalid");

        node_types.emplace_back(NodeType::RemoteNode);
        neuron_ids.emplace_back(rma_offset);
        child_indices.emplace_back(0);
        parent_indices.emplace_back(std::numeric_limits<index_type>::max());
        levels.emplace_back(level);
        signal_types.emplace_back(std::nullopt);
        subdomain_lengths.emplace_back(subdomain_length);

        neuron_details_ex.add_placeholder(rank);
        neuron_details_inh.add_placeholder(rank);
    }

    void push_parent(const index_type child_index, const level_type level, const CudaConfig::gaussian_type subdomain_length,
                     const CudaConfig::mpi_rank_type rank, const std::uint64_t rma_offset) {
        node_types.emplace_back(NodeType::VirtualNode);
        neuron_ids.emplace_back(std::numeric_limits<index_type>::max());
        child_indices.emplace_back(child_index);
        parent_indices.emplace_back(std::numeric_limits<index_type>::max());
        levels.emplace_back(level);
        signal_types.emplace_back(std::nullopt);
        subdomain_lengths.emplace_back(subdomain_length);
        const auto rma_index = rma_offset;
        if (rma_offset_to_neuron_id.size() < rma_index + 1) {
            rma_offset_to_neuron_id.resize(rma_index + 1, std::numeric_limits<CudaConfig::bh_index_type>::max());
        }
        rma_offset_to_neuron_id[rma_index] = static_cast<CudaConfig::bh_index_type>(subdomain_lengths.size() - 1U);
        RelearnException::check(rank != std::numeric_limits<CudaConfig::mpi_rank_type>::max(),
                                "LinearizeTree::push_parent Pushes leaf with invalid rank");

        neuron_details_ex.add_placeholder(rank);
        neuron_details_inh.add_placeholder(rank);
    }

    void push_placeholder() {
        node_types.emplace_back(NodeType::Placeholder);
        neuron_ids.emplace_back(std::numeric_limits<index_type>::max());
        child_indices.emplace_back(0);
        parent_indices.emplace_back(std::numeric_limits<index_type>::max());
        levels.emplace_back(std::numeric_limits<level_type>::max());
        signal_types.emplace_back(std::nullopt);
        subdomain_lengths.emplace_back(0);

        neuron_details_ex.add_placeholder(std::numeric_limits<CudaConfig::mpi_rank_type>::max());
        neuron_details_inh.add_placeholder(std::numeric_limits<CudaConfig::mpi_rank_type>::max());
    }

    void update_leaf(const SignalType signal_type, const index_type i,
                     const CudaConfig::synaptic_count_type vacant_axonal_element,
                     const CudaConfig::synaptic_count_type vacant_dendritic_element) {
        auto& group = get_details(signal_type);
        group.update_leaf(i, vacant_axonal_element, vacant_dendritic_element);
        // get_details(opposite(signal_type)).advance_start_index();
    }

    void update_remote_node() {
    }

    void update_non_leaf() {
        neuron_details_ex.advance_start_index();
        neuron_details_inh.advance_start_index();
    }

    /**
     * The breadth-first search of the constructor can stop before it has pushed every node (see the workaround for
     * OctreeNode::insert). Nodes that were pushed earlier then reference children that never made it into the arrays.
     * The CUDA kernels follow those indices without knowing about it and would read past the end of the device
     * arrays, so the dangling references are cut here and the nodes are turned into placeholders.
     */
    void remove_dangling_children() {
        for (auto i = 0U; i < size(); ++i) {
            const auto child_index = child_indices[i];
            if (child_index != 0 && child_index + Constants::number_oct > size()) {
                child_indices[i] = 0;
                node_types[i] = NodeType::Placeholder;
            }
        }
    }

    void calculate_parent_indices() {
        // set parent index for every child
        for (auto i = 0U; i < size(); ++i) {
            const auto child_idx = child_indices[i];
            if (child_idx != 0U) {
                parent_indices[child_idx] = i;
            }
        }
    }

    void shrink() {
        node_types.shrink_to_fit();
        neuron_ids.shrink_to_fit();
        child_indices.shrink_to_fit();
        parent_indices.shrink_to_fit();
        levels.shrink_to_fit();
        signal_types.shrink_to_fit();
        subdomain_lengths.shrink_to_fit();

        neuron_details_ex.shrink();
        neuron_details_inh.shrink();
    }

    [[nodiscard]] InternalNeuronDetails& get_details(const SignalType type) {
        return type == SignalType::Excitatory ? neuron_details_ex : neuron_details_inh;
    }

    [[nodiscard]] const InternalNeuronDetails& get_details(const SignalType type) const {
        return type == SignalType::Excitatory ? neuron_details_ex : neuron_details_inh;
    }

    [[nodiscard]] static SignalType opposite(const SignalType type) {
        return type == SignalType::Excitatory ? SignalType::Inhibitory : SignalType::Excitatory;
    }

    std::vector<NodeType> node_types{};
    std::vector<CudaConfig::number_neurons_type> neuron_ids{};
    std::vector<index_type> child_indices{};
    std::vector<index_type> parent_indices{};
    std::vector<level_type> levels{};
    std::vector<index_type> level_indices{};
    std::vector<std::optional<SignalType>> signal_types{};
    std::vector<CudaConfig::gaussian_type> subdomain_lengths{};
    InternalNeuronDetails neuron_details_ex;
    InternalNeuronDetails neuron_details_inh;
    CudaConfig::synaptic_count_type sum_vacant_axons = 0;
    std::vector<CudaConfig::number_neurons_type> leaf_neuron_id_mapping{};
    std::vector<CudaConfig::bh_index_type> rma_offset_to_neuron_id{};
    std::unordered_map<RankNeuronId, CudaConfig::bh_index_type> remote_nodes_rank_rma_to_index{};
    std::vector<CudaConfig::bh_index_type> local_branch_nodes{};
};