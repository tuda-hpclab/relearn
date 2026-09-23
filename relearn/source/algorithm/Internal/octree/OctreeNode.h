#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"

#include "algorithm/Internal/octree/Cell.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/MemoryHolder.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>

/**
 * This class serves as the basic building blocks of the Octree.
 * Each object has up to Constants::number_oct children (can be nullptr) and a Cell which summarizes the relevant biological aspects.
 * Additionally, an object stores its its MPI rank and whether or not it is an inner node.
 */
template <typename AdditionalCellAttributes>
class OctreeNode {
public:
    using OctreeNodePtr = OctreeNode<AdditionalCellAttributes>*;
    using OctreeNodeIdx = RelearnTypes::counter_type;

    using position_type = typename Cell<AdditionalCellAttributes>::position_type;
    using counter_type = typename Cell<AdditionalCellAttributes>::counter_type;
    using level_type = RelearnTypes::level_type;

    constexpr static bool has_excitatory_dendrite = AdditionalCellAttributes::has_excitatory_dendrite;
    constexpr static bool has_inhibitory_dendrite = AdditionalCellAttributes::has_inhibitory_dendrite;
    constexpr static bool has_excitatory_axon = AdditionalCellAttributes::has_excitatory_axon;
    constexpr static bool has_inhibitory_axon = AdditionalCellAttributes::has_inhibitory_axon;

    /**
     * @brief Returns the MPI rank this node belongs to
     * @return The MPI rank
     */
    [[nodiscard]] constexpr mpiPP::MPIRank get_mpi_rank() const noexcept {
        return rank;
    }

    /**
     * @brief Returns a flag that indicates if this node is an inner node or a leaf node
     * @return True iff it is an inner node
     */
    [[nodiscard]] constexpr bool is_parent() const noexcept {
        return parent;
    }

    /**
     * @brief Returns a flag that indicates if this node is an inner node or a leaf node
     * @return True iff it is a leaf node
     */
    [[nodiscard]] constexpr bool is_leaf() const noexcept {
        return !parent;
    }

    /**
     * @brief Checks if the specified neuron is stored within *this* node
     * @param rank_neuron_id The MPI rank and neuron id in question
     * @return True iff the specified neuron is stored in this node
     */
    [[nodiscard]] constexpr bool contains(const RankNeuronId& rank_neuron_id) const {
        if (is_parent()) {
            return false;
        }

        const auto& [rni_rank, id] = rank_neuron_id;

        if (rank != rni_rank) {
            return false;
        }

        return cell.get_neuron_id() == id;
    }

    /**
     * @brief Returns a constant view on the associated child nodes. This reference is not invalidated by calls to other methods
     * @return A constant view on the associated child nodes
     */
    [[nodiscard]] constexpr const std::array<OctreeNodePtr, Constants::number_oct>& get_children() const noexcept {
        return children;
    }

    /**
     * @brief Returns the child node with the requested id (id calculation base on Cell::get_octant_for_position)
     * @exception Throws a RelearnException if idx >= Constants::number_oct
     * @return The associated child
     */
    [[nodiscard]] OctreeNodePtr get_child(const unsigned char idx) const {
        RelearnException::check(idx < Constants::number_oct, "OctreeNode::get_child const: idx was: {}", idx);
        // NOLINTNEXTLINE
        return children[idx];
    }

    /**
     * @brief Returns the child node with the requested id (id calculation base on Cell::get_octant_for_position)
     * @exception Throws a RelearnException if idx >= Constants::number_oct
     * @return The associated child
     */
    [[nodiscard]] OctreeNodePtr get_child(const unsigned char idx) {
        RelearnException::check(idx < Constants::number_oct, "OctreeNode::get_child: idx was: {}", idx);
        // NOLINTNEXTLINE
        return children[idx];
    }

    /**
     * @brief Returns a constant view on the associated cell. This reference is not invalidated by calls to other methods
     * @return A constant view on the associated cell
     */
    [[nodiscard]] constexpr const Cell<AdditionalCellAttributes>& get_cell() const noexcept {
        return cell;
    }

    /**
     * @brief Returns a flag that indicates if this object belongs to the current MPI process.
     *      Achieves so by calling MPIWrapper::get_my_rank()
     * @exception Throws a RelearnException if the MPIWrapper is not properly initialized
     * @return True iff this object belongs to the current MPI process
     */
    [[nodiscard]] bool is_actual_id() const {
        return rank == mpiPP::MPIInfo::get_my_rank();
    }

    /**
     * @brief Inserts a neuron with the specified id and the specified position into the subtree that is induced by this object.
     * @param position The position of the new neuron
     * @param neuron_id The id of the new neuron (can be Constants::uninitialized to indicate a virtual neuron), <= Constants::uninitialized
     * @param memory_holder The MemoryHolder that manages the memory for the octree
     * @exception Throws a RelearnException if one of the following happens:
     *      (a) The position is not within the cell's boundaries
     *      (b) neuron_id is unitialized
     *      (c) Allocating a new object in the shared memory window fails
     *      (d) Something went wrong within the insertion
     * @return A pointer to the newly created and inserted node
     */
    OctreeNodePtr insert(const position_type& position, const NeuronID& neuron_id, std::shared_ptr<MemoryHolder<AdditionalCellAttributes>>& memory_holder) {
        const auto& [cell_xyz_min, cell_xyz_max] = cell.get_size();
        const auto is_in_box = position.check_in_box(cell_xyz_min, cell_xyz_max);

        RelearnException::check(is_in_box, "OctreeNode::insert: position is not in box: {} in [{}, {}]", position, cell_xyz_min, cell_xyz_max);
        RelearnException::check(neuron_id.is_initialized(), "OctreeNode::insert, neuron_id is not initialized");

        const auto my_rank = mpiPP::MPIInfo::get_my_rank();

        OctreeNodePtr parent_node = nullptr;
        for (OctreeNodePtr current_node = this; nullptr != current_node;) {
            /**
             * My parent already exists.
             * Calc which child to follow, i.e., determine octant
             */
            const auto new_position_octant = current_node->get_cell().get_octant_for_position(position);

            parent_node = current_node;
            current_node = current_node->get_child(new_position_octant);
        }

        // Now we know the furthest-down node and the respective octant where to insert the position
        RelearnException::check(parent_node != nullptr, "OctreeNode::insert: parent_node is nullptr");

        if (parent_node->is_leaf()) {
            /**
             * Found my octant, but I'm the very first child of that node.
             * I.e., the node is a leaf.
             */

            if (parent_node->get_cell_neuron_id().is_virtual() && neuron_id != parent_node->get_cell_neuron_id()) {
                /**
                 * The found parent node is virtual and can just be substituted,
                 * i.e., it was constructed while constructing the upper part to the branch nodes.
                 */
                parent_node->set_cell_neuron_id(neuron_id);
                parent_node->set_cell_neuron_position({ position });
                parent_node->set_rank(my_rank);
                return parent_node;
            }

            while (true) {
                // The found parent node contains a neuron; it has to be transformed to a virtual node and the neuron shifted one down
                const auto& parent_cell = parent_node->get_cell();
                const auto& parent_neuron_id = parent_cell.get_neuron_id();

                // Determine octant for neuron
                const auto& parent_position = parent_cell.get_neuron_position();
                RelearnException::check(parent_position.has_value(), "OctreeNode::insert: While building the octree, the cell doesn't have a position");

                const auto parent_own_octant = parent_cell.get_octant_for_position(parent_position.value()); // NOLINT(bugprone-unchecked-optional-access) - guarded by RelearnException::check above
                const auto& [minimum_position, maximum_position] = parent_cell.get_size_for_octant(parent_own_octant);

                // The child copies the parent node
                auto* new_node = memory_holder->get_available(parent_node, parent_own_octant);
                new_node->set_cell_size(minimum_position, maximum_position);
                new_node->set_cell_neuron_position(parent_position);
                new_node->set_rank(parent_node->get_mpi_rank());
                new_node->set_level(parent_node->get_level() + 1);
                if (parent_neuron_id.is_actual_id()) {
                    new_node->set_cell_neuron_id(parent_neuron_id);
                } else {
                    new_node->set_cell_neuron_id(NeuronID::virtual_id());
                }

                // Set the child and mark the parent as virtual
                parent_node->set_child(new_node, parent_own_octant);

                const auto parent_node_offset = memory_holder->get_offset_from_parent(parent_node);
                parent_node->set_cell_neuron_id(NeuronID::virtual_id(parent_node_offset));

                if (const auto insert_octant = parent_cell.get_octant_for_position(position); insert_octant == parent_own_octant) {
                    // The moved parent and the position to insert still are in the same octant -> try again
                    parent_node = new_node;
                } else {
                    break;
                }
            }
        }

        // Now parent_node is virtual and the octant the new position will occupy is empty
        const auto new_position_octant = parent_node->get_cell().get_octant_for_position(position);

        auto* new_node_to_insert = memory_holder->get_available(parent_node, new_position_octant);
        RelearnException::check(new_node_to_insert != nullptr, "OctreeNode::insert: new_node_to_insert is nullptr");

        parent_node->set_child(new_node_to_insert, new_position_octant);
        const auto parent_level = parent_node->get_level();

        const auto& [minimum_position, maximum_position] = parent_node->get_cell().get_size_for_octant(new_position_octant);
        new_node_to_insert->set_cell_size(minimum_position, maximum_position);
        new_node_to_insert->set_cell_neuron_position({ position });
        new_node_to_insert->set_cell_neuron_id(neuron_id);
        new_node_to_insert->set_rank(rank);
        new_node_to_insert->set_level(parent_level + 1);

        return new_node_to_insert;
    }

    /**
     * @brief Sets the associated MPI rank
     * @param new_rank The associated MPI rank, must be initialized
     * @exception Throws a RelearnException if new_rank is not initialized
     */
    void set_rank(const mpiPP::MPIRank new_rank) {
        RelearnException::check(new_rank.is_initialized(), "OctreeNode::set_rank: new_rank not initialized", new_rank);
        rank = new_rank;
    }

    /**
     * @brief Marks this node as a parent (an inner node)
     */
    constexpr void set_parent() noexcept {
        parent = true;
    }

    /**
     * @brief Sets the node as the child with the given index and updates the parent flag accordingly
     * @param node The new child node (can be nullptr)
     * @param idx The index of the child which shall be set, < Constants::number_oct
     * @exception Throws a RelearnException if idx >= Constants::number_oct
     */
    void set_child(OctreeNodePtr node, const unsigned char idx) {
        RelearnException::check(idx < Constants::number_oct, "OctreeNode::set_child: idx is {}", idx);
        // NOLINTNEXTLINE
        children[idx] = node;

        bool has_children = false;

        for (auto* child : children) {
            if (child != nullptr) {
                has_children = true;
            }
        }

        parent = has_children;
    }

    /**
     * @brief Returns the level of this node
     * @return The level
     */
    [[nodiscard]] constexpr level_type get_level() const noexcept {
        return level;
    }

    /**
     * @brief Sets the level of this node
     * @param new_level The new level of this node
     */
    constexpr void set_level(level_type new_level) noexcept {
        level = new_level;
    }

    /**
     * @brief Resets the current object:
     *      (a) The children are newly constructed with nullptr
     *      (b) The cell is newly constructed
     *      (c) level is std::numeric_limits<level_type>::max()
     *      (d) parent is false
     *      (e) rank is MPIRank::uninitialized_rank()
     */
    constexpr void reset() noexcept {
        children = std::array<OctreeNodePtr, Constants::number_oct>{ nullptr };
        cell = Cell<AdditionalCellAttributes>{};
        level = std::numeric_limits<level_type>::max();
        parent = false;
        rank = mpiPP::MPIRank::uninitialized_rank();
    }

    /**
     * Prints this node and all of its children to the given stream with the given prefix
     * @param ss StringStream
     * @param prefix Prefix
     */
    void printSubtree(std::stringstream& ss, const std::string& prefix) const {
        if (!is_actual_id()) {
            return;
        }

        auto num_children = std::size_t{ 0 };
        for (auto* c : children) {
            if (c != nullptr) {
                num_children++;
            }
        }

        if (num_children == 0) {
            return;
        }

        ss << prefix << (num_children > 1 ? "|-- " : "");

        auto added_children = std::size_t{ 0 };
        for (auto* c : children) {
            if (c == nullptr) {
                continue;
            }
            auto str = std::string{};
            auto new_prefix = std::string{};
            if (!c->is_actual_id()) {
                const auto pos = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).value_or(position_type{ -1, -1, -1 });
                auto vacant = std::string{ "n/a" };
                if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                    vacant = std::to_string(cell.get_number_dendrites_for(SignalType::Excitatory));
                }
                str = std::string("REMOTE (") + std::to_string(c->get_mpi_rank().get_rank()) + ", " + std::to_string(c->get_cell_neuron_id().get_rma_offset()) + " position " + std::to_string(pos.get_x()) + " " + std::to_string(pos.get_y()) + " " + std::to_string(pos.get_z()) + " vacant " + vacant + ")";
            } else {
                str = c->to_string();
            }

            if (added_children < num_children - 1) {
                if (added_children > 0) {   // added fix
                    ss << prefix << "|-- "; // added fix
                } // added fix

                auto childs_child_empty = true;
                for (auto* cc : c->children) {
                    if (cc != nullptr) {
                        childs_child_empty = false;
                        break;
                    }
                }

                const auto printStrand = num_children > 1 && !childs_child_empty;
                new_prefix = prefix + (printStrand ? "|\t" : "\t");
            } else {
                ss << (num_children > 1 ? prefix : "") << "|>- ";
                new_prefix = prefix + "\t";
            }

            ss << str << "\n";
            c->printSubtree(ss, new_prefix);
            added_children++;
        }
    }

    /**
     * Human readable string with essential information about this node
     * @return Human readable string
     */
    [[nodiscard]] std::string to_string() const {
        auto ss = std::ostringstream{};
        ss << this;
        const auto pos = cell.get_position_for(ElementType::Dendrite, SignalType::Excitatory).value_or(position_type{ -1, -1, -1 });
        ss << "( rank " << get_mpi_rank().get_rank() << ", neuron id " << cell.get_neuron_id() << " position " << pos.get_x() << " " << pos.get_y() << " " << pos.get_z() << " vacant " << cell.get_number_dendrites_for(SignalType::Excitatory) << " leaf " << is_leaf() << " parent " << is_parent() << ")";
        return ss.str();
    }

    /**
     * Human readable string with essential information about this node and its children
     * @return Human readable string
     */
    [[nodiscard]] std::string to_string_with_children() const {
        auto ss = std::ostringstream{};
        ss << get_mpi_rank() << ", " << get_cell_neuron_id() << " leaf " << is_leaf() << " parent " << is_parent() << " local " << is_actual_id() << " Children [";

        for (const auto* child : get_children()) {
            ss << child << ", ";
        }
        ss << "]";
        return ss.str();
    }

    /**
     * @brief Prints the octree node to the output stream
     * @param output_stream The output stream
     * @param octree_node The octree node to print
     * @return The output stream after printing the octree node
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const OctreeNode<AdditionalCellAttributes>& octree_node) {
        output_stream << "== OctreeNode (" << &octree_node << ") ==\n";

        output_stream << "  children[8]: ";
        for (const auto* const child : octree_node.get_children()) {
            output_stream << child << " ";
        }
        output_stream << '\n';

        output_stream << "  is_parent  : " << octree_node.is_parent() << "\n\n";
        output_stream << "  rank       : " << octree_node.get_mpi_rank() << "\n\n";
        output_stream << octree_node.get_cell();
        output_stream << '\n';

        return output_stream;
    }

private:
    std::array<OctreeNodePtr, Constants::number_oct> children{ nullptr };
    Cell<AdditionalCellAttributes> cell{};

    level_type level{ std::numeric_limits<level_type>::max() };
    bool parent{ false };

    mpiPP::MPIRank rank{ mpiPP::MPIRank::uninitialized_rank() }; // MPI rank who owns this octree node

public:
    /**
     * @brief Sets the optional position for both the excitatory and inhibitory positions in the associated cell
     * @param opt_position The optional position, can be empty
     */
    void set_cell_neuron_position(const std::optional<position_type>& opt_position) {
        cell.set_neuron_position(opt_position);
    }

    /**
     * @brief Sets the number of free excitatory and inhibitory dendrites in the associated cell
     * @param number_excitatory_dendrites The number of free excitatory dendrites
     * @param number_inhibitory_dendrites The number of free inhibitory dendrites
     */
    constexpr void set_cell_number_dendrites(const counter_type number_excitatory_dendrites, const counter_type number_inhibitory_dendrites) noexcept {
        cell.set_number_excitatory_dendrites(number_excitatory_dendrites);
        cell.set_number_inhibitory_dendrites(number_inhibitory_dendrites);
    }

    /**
     * @brief Sets the number of free excitatory and inhibitory axons in the associated cell
     * @param number_excitatory_axons The number of free excitatory axons
     * @param number_inhibitory_axons The number of free inhibitory axons
     */
    constexpr void set_cell_number_axons(const counter_type number_excitatory_axons, const counter_type number_inhibitory_axons) noexcept {
        cell.set_number_excitatory_axons(number_excitatory_axons);
        cell.set_number_inhibitory_axons(number_inhibitory_axons);
    }

    /**
     * @brief Sets the number of free excitatory dendrites int he associated cell
     * @param number_dendrites The number of free excitatory dendrites
     */
    constexpr void set_cell_number_excitatory_dendrites(const counter_type number_dendrites) noexcept {
        cell.set_number_excitatory_dendrites(number_dendrites);
    }

    /**
     * @brief Sets the number of free excitatory axons int he associated cell
     * @param number_axons The number of free excitatory axons
     */
    constexpr void set_cell_number_excitatory_axons(const counter_type number_axons) noexcept {
        cell.set_number_excitatory_axons(number_axons);
    }

    /**
     * @brief Sets the number of free inhibitory dendrites int he associated cell
     * @param number_dendrites The number of free inhibitory dendrites
     */
    constexpr void set_cell_number_inhibitory_dendrites(const counter_type number_dendrites) noexcept {
        cell.set_number_inhibitory_dendrites(number_dendrites);
    }

    /**
     * @brief Sets the number of free inhibitory axons int he associated cell
     * @param number_axons The number of free inhibitory axons
     */
    constexpr void set_cell_number_inhibitory_axons(const counter_type number_axons) noexcept {
        cell.set_number_inhibitory_axons(number_axons);
    }

    /**
     * @brief Sets the optional position for the excitatory dendrites position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    constexpr void set_cell_excitatory_dendrites_position(const std::optional<position_type>& opt_position) {
        cell.set_excitatory_dendrites_position(opt_position);
    }

    /**
     * @brief Sets the optional position for the inhibitory dendrites position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    constexpr void set_cell_inhibitory_dendrites_position(const std::optional<position_type>& opt_position) {
        cell.set_inhibitory_dendrites_position(opt_position);
    }

    /**
     * @brief Sets the optional position for the excitatory axons position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    constexpr void set_cell_excitatory_axons_position(const std::optional<position_type>& opt_position) {
        cell.set_excitatory_axons_position(opt_position);
    }

    /**
     * @brief Sets the optional position for the inhibitory axons position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    constexpr void set_cell_inhibitory_axons_position(const std::optional<position_type>& opt_position) {
        cell.set_inhibitory_axons_position(opt_position);
    }

    /**
     * @brief Returns the neuron id for the associated cell
     * @return The neuron id
     */
    [[nodiscard]] constexpr NeuronID get_cell_neuron_id() const noexcept {
        return cell.get_neuron_id();
    }

    /**
     * @brief Sets the neuron id for the associated cell
     * @param neuron_id The neuron id
     * @exception Throws a RelearnException if the neuron_id is not initialized
     */
    constexpr void set_cell_neuron_id(const NeuronID& neuron_id) {
        cell.set_neuron_id(neuron_id);
    }

    /**
     * @brief Sets the min and max of the associated cell
     * @param min The minimum boundary of the cell
     * @param max The maximum boundary of the cell
     * @exception Might throw a RelearnException
     */
    constexpr void set_cell_size(const position_type& min, const position_type& max) {
        cell.set_size({ min, max });
    }

    /**
     * @brief Returns the size of the associated cell
     * @return The size of the cell
     */
    [[nodiscard]] constexpr RelearnTypes::bounding_box_type get_size() const noexcept {
        return cell.get_size();
    }

    template <std::size_t Index>
    [[nodiscard]] auto& get() & {
        if constexpr (Index == 0) {
            return children;
        }
        if constexpr (Index == 1) {
            return cell;
        }
        if constexpr (Index == 2) {
            return level;
        }
        if constexpr (Index == 3) {
            return parent;
        }
        if constexpr (Index == 4) {
            return rank;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] const auto& get() const& {
        if constexpr (Index == 0) {
            return children;
        }
        if constexpr (Index == 1) {
            return cell;
        }
        if constexpr (Index == 2) {
            return level;
        }
        if constexpr (Index == 3) {
            return parent;
        }
        if constexpr (Index == 4) {
            return rank;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] auto&& get() && {
        if constexpr (Index == 0) {
            return std::move(children);
        }
        if constexpr (Index == 1) {
            return std::move(cell);
        }
        if constexpr (Index == 2) {
            return std::move(level);
        }
        if constexpr (Index == 3) {
            return std::move(parent);
        }
        if constexpr (Index == 4) {
            return std::move(rank);
        }
    }
};

namespace std {
// NOLINTBEGIN(cert-dcl58-cpp) - specializing tuple_size/tuple_element for a program-defined type is standard-sanctioned
template <typename AdditionalCellAttributes>
struct tuple_size<::OctreeNode<AdditionalCellAttributes>> {
    static constexpr size_t value = 5;
};

template <typename AdditionalCellAttributes>
struct tuple_element<0, ::OctreeNode<AdditionalCellAttributes>> {
    using type = std::array<OctreeNode<AdditionalCellAttributes>*, Constants::number_oct>;
};

template <typename AdditionalCellAttributes>
struct tuple_element<1, ::OctreeNode<AdditionalCellAttributes>> {
    using type = Cell<AdditionalCellAttributes>;
};

template <typename AdditionalCellAttributes>
struct tuple_element<2, ::OctreeNode<AdditionalCellAttributes>> {
    using type = RelearnTypes::level_type;
};

template <typename AdditionalCellAttributes>
struct tuple_element<3, ::OctreeNode<AdditionalCellAttributes>> {
    using type = bool;
};

template <typename AdditionalCellAttributes>
struct tuple_element<4, ::OctreeNode<AdditionalCellAttributes>> {
    using type = mpiPP::MPIRank;
};
// NOLINTEND(cert-dcl58-cpp)

} // namespace std
