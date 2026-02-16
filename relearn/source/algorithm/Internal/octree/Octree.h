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
#include "algorithm/Internal/octree/OctreeNodeHelper.h"
#include "structure/SpaceFillingCurve.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/Timers.h"
#include "util/Vec3.h"

#include "cpp-utility/MemoryFootprint.hpp"
#include "cpp-utility/data-structure/Stack.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/RMAWindow.h"
#include "mpi-wrapper/collectives/MPIAllGather.h"

#include <range/v3/functional/indirect.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/indirect.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <utility>
#include <vector>

/**
 * This type represents the (spatial) Octree in which the neurons are organised.
 * It offers general information about the structure, the functionality to insert new neurons,
 * update from the bottom up, and synchronize parts with MPI.
 * It is templated by the additional cell attributes that the algorithm will need the cell to have.
 */
template <typename AdditionalCellAttributes>
class Octree {
public:
    using box_size_type = RelearnTypes::box_size_type;
    using bounding_box_type = RelearnTypes::bounding_box_type;

    /**
     * @brief Constructs a new Octree with the the given size
     * @param box_size The simulation box boundaries
     * @param _space_filling_curve The space filling curve; contains the level at which the branch nodes (that are exchanged via MPI) are, not nullptr
     * @exception Throws a RelearnException if space_filling_curve was empty
     */
    Octree(const bounding_box_type& box_size, std::shared_ptr<SpaceFillingCurve> _space_filling_curve)
        : simulation_box(box_size)
        , space_curve(std::move(_space_filling_curve))
        , rma_window{ Constants::mpi_alloc_mem } {

        RelearnException::check(space_curve != nullptr, "Octree::Octree: _space_filling_curve was empty");
        level_of_branch_nodes = space_curve->get_current_refinement_level();

        node_cache.set_rma_window(&rma_window);

        const auto num_local_trees = 1ULL << (3U * level_of_branch_nodes);
        branch_nodes.resize(num_local_trees, nullptr);

        memory_holder.init(std::span{ rma_window.get_pointer(), Constants::mpi_alloc_mem });

        construct_global_tree_part();
    }

    virtual ~Octree() = default;

    Octree(const Octree& other) = delete;
    Octree(Octree&& other) = default;

    Octree& operator=(const Octree& other) = delete;
    Octree& operator=(Octree&& other) = default;

    /**
     * @brief Returns the root of the Octree
     * @return The root of the Octree. Ownership is not transferred
     */
    [[nodiscard]] const OctreeNode<AdditionalCellAttributes>* get_root() const noexcept {
        return &root;
    }

    /**
     * @brief Returns the root of the Octree
     * @return The root of the Octree. Ownership is not transferred
     */
    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_root() noexcept {
        return &root;
    }

    /**
     * @brief Clears the node cache
     */
    void clear_cache() {
        node_cache.clear();
    }

    /**
     * @brief Returns the cache that is used for the nodes
     * @return The cache that is used for the nodes
     */
    [[nodiscard]] const NodeCache<AdditionalCellAttributes>& get_cache() const noexcept {
        return node_cache;
    }

    /**
     * @brief Returns the underling memory holder that manages the memory for the octree nodes
     * @return The underling memory holder
     */
    [[nodiscard]] const MemoryHolder<AdditionalCellAttributes>& get_memory_holder() const noexcept {
        return memory_holder;
    }

    /**
     * @brief Returns the number of branch nodes (that are exchanged via MPI)
     * @return The number of branch nodes (that are exchanged via MPI)
     */
    [[nodiscard]] std::size_t get_num_local_trees() const noexcept {
        return branch_nodes.size();
    }

    /**
     * @brief Returns the bounding box of the simulation
     * @return The simulation box' size
     */
    [[nodiscard]] const bounding_box_type& get_simulation_box() const noexcept {
        return simulation_box;
    }

    /**
     * @brief Returns the level at which the branch nodes (that are exchanged via MPI) are
     * @return The level at which the branch nodes (that are exchanged via MPI) are
     */
    [[nodiscard]] std::uint16_t get_level_of_branch_nodes() const noexcept {
        return level_of_branch_nodes;
    }

    /**
     * @brief Get all local branch nodes
     * @return All local branch nodes
     */
    [[nodiscard]] std::vector<const OctreeNode<AdditionalCellAttributes>*> get_local_branch_nodes() const {
        const auto constify = [](OctreeNode<AdditionalCellAttributes>* ptr) -> const OctreeNode<AdditionalCellAttributes>* { return ptr; };

        return branch_nodes
               | ranges::views::filter(ranges::indirect(&OctreeNode<AdditionalCellAttributes>::is_actual_id))
               | ranges::views::transform(constify)
               | ranges::to_vector;
    }

    /**
     * @brief Get all local branch nodes
     * @return All local branch nodes
     */
    [[nodiscard]] std::vector<OctreeNode<AdditionalCellAttributes>*> get_local_branch_nodes() {
        return branch_nodes
               | ranges::views::filter(ranges::indirect(&OctreeNode<AdditionalCellAttributes>::is_actual_id))
               | ranges::to_vector;
    }

    /**
     * @brief Inserts a neuron with the specified id and the specified position into the octree.
     * @param position The position of the new neuron
     * @param neuron_id The id of the new neuron, < Constants::uninitialized (only use for actual neurons, virtual neurons are inserted automatically)
     * @exception Throws a RelearnException if one of the following happens:
     *      (a) The position is not within the octree's boundaries
     *      (b) neuron_id >= Constants::uninitialized
     *      (c) Allocating a new object in the shared memory window fails
     *      (d) Something went wrong within the insertion
     */
    void insert(const box_size_type& position, const NeuronID& neuron_id) {
        RelearnException::check(neuron_id.is_initialized(), "Octree::insert: neuron_id {} was uninitialized", neuron_id);

        const auto& bounding_box = get_simulation_box();
        RelearnException::check(bounding_box.check_in_box(position), "Octree::insert: position was not in range: {} vs {}", position, bounding_box);

        auto* res = root.insert(position, neuron_id, memory_holder);
        RelearnException::check(res != nullptr, "Octree::insert: res was nullptr");
    }

    /**
     * Print a visualization of this tree to a file
     * @param file_path The file where the visualization will be stored
     */
    void print_to_file(const std::filesystem::path& file_path) const {
        auto out_stream = std::ofstream{ file_path };
        RelearnException::check(out_stream.good() && !out_stream.bad(), "Octree::print_to_file: Unable to open stream for {}", file_path.string());

        auto ss = std::stringstream{};
        print(ss);
        out_stream << ss.rdbuf();
        out_stream.flush();
        out_stream.close();
    }

    /**
     * @brief Gathers all leaf nodes and makes them available via get_leaf_nodes
     * @param num_neurons The number of neurons
     */
    void initializes_leaf_nodes(const RelearnTypes::number_neurons_type num_neurons) {
        auto leaf_nodes = std::vector<OctreeNode<AdditionalCellAttributes>*>{ num_neurons, nullptr };

        auto stack = utility::Stack<OctreeNode<AdditionalCellAttributes>*>{ num_neurons };
        stack.emplace_back(&root);

        while (!stack.empty()) {
            auto* node = stack.pop_back();

            if (node->is_leaf()) {
                const auto neuron_id = node->get_cell_neuron_id();
                RelearnException::check(neuron_id.get_neuron_id() < leaf_nodes.size(), "Octree::initializes_leaf_nodes: Neuron id was too large for leaf nodes: {}", neuron_id);
                RelearnException::check(leaf_nodes[neuron_id.get_neuron_id()] == nullptr, "Octree::initializes_leaf_nodes: Found the following neuron id multiple times: {}", neuron_id);

                leaf_nodes[neuron_id.get_neuron_id()] = node;
                continue;
            }

            for (auto* child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                if (const auto neuron_id = child->get_cell_neuron_id(); !child->is_parent() && (neuron_id.is_virtual() || !neuron_id.is_initialized())) {
                    continue;
                }

                stack.emplace_back(child);
            }
        }

        for (const auto neuron_id : NeuronID::range(num_neurons)) {
            const auto& node = leaf_nodes[neuron_id.get_neuron_id()];
            RelearnException::check(node != nullptr, "Octree::initializes_leaf_nodes: Leaf node {} is null", neuron_id);
            RelearnException::check(node->is_leaf(), "Octree::initializes_leaf_nodes: Leaf node {} is not a leaf node", neuron_id);
            RelearnException::check(node->is_actual_id(), "Octree::initializes_leaf_nodes: Leaf node {} is not local", neuron_id);
            RelearnException::check(node->get_cell().get_neuron_id() == neuron_id, "Octree::initializes_leaf_nodes: Leaf node {} has wrong neuron id {}", neuron_id, node->get_cell().get_neuron_id());
        }

        all_leaf_nodes = std::move(leaf_nodes);
    }

    /**
     * @brief Synchronizes the octree with all MPI ranks
     */
    void synchronize_tree() {
        // Update my local trees bottom-up
        update_local_trees();

        // Exchange the local trees
        synchronize_local_trees();
    }

    /**
     * @brief Returns a constant reference to all leaf nodes
     *      The reference is never invalidated
     * @return All leaf nodes
     */
    [[nodiscard]] const std::vector<OctreeNode<AdditionalCellAttributes>*>& get_leaf_nodes() const noexcept {
        return all_leaf_nodes;
    }

    /**
     * @brief Returns the branch node with the (global) index, cast to a void*
     * @param index The global index of the requested branch node
     * @exception Throws a RelearnException if index is larger than or equal to the number of branch nodes
     * @return The requested branch node
     */
    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_branch_node_pointer(const std::size_t index) {
        RelearnException::check(index < branch_nodes.size(), "Octree::get_branch_node_pointer(): index ({}) is larger than or equal to the number of branch nodes ({}).", index, branch_nodes.size());
        return branch_nodes[index];
    }

    /**
     * @brief Records the memory footprint of the current object as "Octree"
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
        const auto my_footprint = sizeof(*this)
                                  + (branch_nodes.capacity() * sizeof(OctreeNode<AdditionalCellAttributes>*))
                                  + (all_leaf_nodes.capacity() * sizeof(OctreeNode<AdditionalCellAttributes>*));
        footprint->emplace("Octree", my_footprint);

        const auto octree_node_footprint = memory_holder.get_size() * sizeof(OctreeNode<AdditionalCellAttributes>);
        footprint->emplace("OctreeNode", octree_node_footprint);
    }

private:
    /**
     * Print a visualization of this tree to a stringstream
     * @param ss stringstream
     */
    void print(std::stringstream& ss) const {
        ss << root.to_string() << '\n';
        root.printSubtree(ss, "");
        ss << '\n';
    }

    /**
     * @brief Constructs the upper portion of the tree, i.e., all nodes at depths [0, level_of_branch_nodes].
     */
    void construct_global_tree_part() {
        const auto _level_of_branch_nodes = get_level_of_branch_nodes();
        const auto num_cells_per_dimension = 1ULL << _level_of_branch_nodes; // (2^level_of_branch_nodes)
        const auto num_cells_per_dimension_cast = static_cast<double>(num_cells_per_dimension);

        auto branch_nodes_positions = std::vector<box_size_type>{};
        branch_nodes_positions.reserve(num_cells_per_dimension * num_cells_per_dimension * num_cells_per_dimension);

        const auto& [xyz_min, xyz_max] = get_simulation_box();
        const auto& [x_min, y_min, z_min] = xyz_min;
        const auto& [x_max, y_max, z_max] = xyz_max;

        const auto box_x = (x_max - x_min) / num_cells_per_dimension_cast;
        const auto box_y = (y_max - y_min) / num_cells_per_dimension_cast;
        const auto box_z = (z_max - z_min) / num_cells_per_dimension_cast;

        const auto half_x = box_x / 2.0;
        const auto half_y = box_y / 2.0;
        const auto half_z = box_z / 2.0;

        for (auto z_it = 0U; z_it < num_cells_per_dimension; z_it++) {
            for (auto y_it = 0U; y_it < num_cells_per_dimension; y_it++) {
                for (auto x_it = 0U; x_it < num_cells_per_dimension; x_it++) {
                    const auto x = (x_it * box_x) + half_x + x_min;
                    const auto y = (y_it * box_y) + half_y + y_min;
                    const auto z = (z_it * box_z) + half_z + z_min;
                    branch_nodes_positions.emplace_back(x, y, z);
                }
            }
        }

        root.set_cell_size(xyz_min, xyz_max);
        root.set_cell_neuron_id(NeuronID::virtual_id());
        root.set_cell_neuron_position(branch_nodes_positions[0]);
        root.set_rank(mpiPP::MPIInfo::get_my_rank());
        root.set_level(0);

        for (auto pos_it = 1U; pos_it < branch_nodes_positions.size(); pos_it++) {
            std::ignore = root.insert(branch_nodes_positions[pos_it], NeuronID::virtual_id(), memory_holder);
        }

        auto stack = utility::Stack<std::pair<OctreeNode<AdditionalCellAttributes>*, Vec3s>>{ Constants::number_oct * _level_of_branch_nodes };
        stack.emplace_back(&root, Vec3s{ 0, 0, 0 });

        while (!stack.empty()) {
            const auto [ptr, index3d] = stack.pop_back();

            if (!ptr->is_parent()) {
                const auto index1d = space_curve->map_3d_to_1d(index3d);
                branch_nodes[index1d] = ptr;
                continue;
            }

            for (auto id = std::size_t{ 0 }; id < Constants::number_oct; id++) {
                auto* child_node = ptr->get_child(id);

                const auto larger_x = ((id & 1ULL) == 0) ? 0ULL : 1ULL;
                const auto larger_y = ((id & 2ULL) == 0) ? 0ULL : 1ULL;
                const auto larger_z = ((id & 4ULL) == 0) ? 0ULL : 1ULL;

                const auto offset = Vec3s{ larger_x, larger_y, larger_z };
                const auto pos = Vec3s{ index3d * 2 } + offset;
                stack.emplace_back(child_node, pos);
            }
        }
    }

    /**
     * @brief Updates all local (!) branch nodes and their induced subtrees.
     * @exception Throws a RelearnException if the functor throws
     */
    void update_local_trees() {
        Timers::start(TimerRegion::UPDATE_LOCAL_TREES);

        const auto update_tree = [this](auto* local_tree) {
            update_tree_parallel(local_tree);
        };

        ranges::for_each(branch_nodes | ranges::views::filter(ranges::indirect(&OctreeNode<AdditionalCellAttributes>::is_actual_id)), update_tree);

        Timers::stop_and_add(TimerRegion::UPDATE_LOCAL_TREES);
    }

    /**
     * @brief Synchronizes all (locally) updated branch nodes with all other MPI ranks
     */
    void synchronize_local_trees() {
        Timers::start(TimerRegion::EXCHANGE_BRANCH_NODES);

        const auto number_branch_nodes = branch_nodes.size();

        // Copy local trees' root nodes to correct positions in receive buffer
        auto exchange_branch_nodes = branch_nodes | ranges::views::indirect | ranges::to_vector;

        // All-gather in-place branch nodes from every rank
        const auto number_local_branch_nodes = number_branch_nodes / static_cast<std::size_t>(mpiPP::MPIInfo::get_number_ranks());
        RelearnException::check(number_local_branch_nodes < static_cast<std::size_t>(std::numeric_limits<int>::max()),
                                "Octree::synchronize_local_trees: Too many branch nodes: {}", number_local_branch_nodes);

        mpiPP::MPICollectives::all_gather_inline(std::span{ exchange_branch_nodes.data(), number_branch_nodes }, static_cast<int>(number_local_branch_nodes));

        Timers::stop_and_add(TimerRegion::EXCHANGE_BRANCH_NODES);

        Timers::start(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);
        for (auto i = std::size_t{ 0 }; i < number_branch_nodes; i++) {
            auto& received_node = exchange_branch_nodes[i];
            if (received_node.is_parent()) {
                /*
                 * This part exists for the location-aware Barnes-Hut algorithm.
                 * If the branch node is a leaf, it uses the leaf-case without problems.
                 * Otherwise, we need to store the index of the branch node so that we
                 * can later send it around.
                 */
                // received_node.set_cell_neuron_id(NeuronID::virtual_id(i));
            }

            *branch_nodes[i] = exchange_branch_nodes[i];
        }
        Timers::stop_and_add(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);

        Timers::start(TimerRegion::UPDATE_GLOBAL_TREE);
        if (const auto _level_of_branch_nodes = get_level_of_branch_nodes(); _level_of_branch_nodes > 0) {
            // Only update whenever there are other branches to update
            // The nodes at level_of_branch_nodes are already updated (by other MPI ranks)
            update_tree_parallel(&root, _level_of_branch_nodes - 1);
        }
        Timers::stop_and_add(TimerRegion::UPDATE_GLOBAL_TREE);
    }

    /**
     * @brief Updates the tree induced by local_tree_root until the desired level.
     *      Uses OctreeNode::get_level() to determine the depth. The nodes at that depth are still updated, but not their children.
     *      Potentially updates in parallel based on the depth of the updates.
     * @param local_tree_root The root of the tree from where to update
     * @param max_depth The depth where the updates shall stop
     * @exception Throws a RelearnException if local_tree_root is nullptr or if max_depth is smaller than the depth of local_tree_root
     */
    void update_tree_parallel(OctreeNode<AdditionalCellAttributes>* local_tree_root, const std::uint16_t max_depth = std::numeric_limits<std::uint16_t>::max()) {
        RelearnException::check(local_tree_root != nullptr, "Octree::update_tree_parallel: local_tree_root was nullptr");
        RelearnException::check(local_tree_root->get_level() <= max_depth, "Octree::update_tree_parallel: The root had a larger depth than max_depth.");

        if (const auto update_height = max_depth - local_tree_root->get_level(); update_height < 3) {
            // If the update concerns less than 3 levels, update serially
            OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(local_tree_root, max_depth);
            return;
        }

        // Gather all subtrees two levels down from the current node, update the induced trees in parallel, and then update the upper portion serially

        constexpr auto maximum_number_subtrees = 64;
        auto subtrees = std::vector<OctreeNode<AdditionalCellAttributes>*>{};
        subtrees.reserve(maximum_number_subtrees);

        constexpr auto maximum_number_nodes = 64 + 8 + 1;
        auto tree_upper_part = utility::Stack<OctreeNode<AdditionalCellAttributes>*>{ maximum_number_nodes };
        tree_upper_part.emplace_back(local_tree_root);

        for (const auto& root_child : local_tree_root->get_children()) {
            if (root_child == nullptr) {
                continue;
            }

            tree_upper_part.emplace_back(root_child);

            for (const auto& root_child_child : root_child->get_children()) {
                if (root_child_child == nullptr) {
                    continue;
                }

                tree_upper_part.emplace_back(root_child_child);
                subtrees.emplace_back(root_child_child);
            }
        }

#pragma omp parallel for shared(subtrees, max_depth) default(none)
        for (auto i = 0UL; i < subtrees.size(); i++) {
            auto* other_local_tree_root = subtrees[i];
            OctreeNodeUpdater<AdditionalCellAttributes>::update_tree(other_local_tree_root, max_depth);
        }

        while (!tree_upper_part.empty()) {
            auto* node = tree_upper_part.top();
            tree_upper_part.pop();

            if (node->is_parent()) {
                OctreeNodeUpdater<AdditionalCellAttributes>::update_node(node);
            }
        }
    }

    bounding_box_type simulation_box{};

    std::shared_ptr<SpaceFillingCurve> space_curve{};
    std::uint16_t level_of_branch_nodes{ std::numeric_limits<std::uint16_t>::max() };

    mpiPP::RMAWindow<OctreeNode<AdditionalCellAttributes>> rma_window{};
    MemoryHolder<AdditionalCellAttributes> memory_holder{};
    NodeCache<AdditionalCellAttributes> node_cache{};

    // Root of the tree
    OctreeNode<AdditionalCellAttributes> root{};

    std::vector<OctreeNode<AdditionalCellAttributes>*> branch_nodes{};
    std::vector<OctreeNode<AdditionalCellAttributes>*> all_leaf_nodes{};
};
