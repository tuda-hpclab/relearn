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

#include "algorithm/Internal/octree/OctreeNode.h"
#include "util/RelearnException.h"

#include "cpp-utility/data-structure/SemiStableVector.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"
#include "mpi-wrapper/RMAWindow.h"

#include <fmt/ostream.h>

#include <array>
#include <cstddef>
#include <map>
#include <utility>

class NodeCacheAdapter;

/**
 * This class acts as interface to different cache implementations.
 * @tparam AdditionalCellAttributes The additional cell attributes that are used for the plasticity algorithm
 */
template <typename AdditionalCellAttributes>
class NodeCache {
public:
    using node_type = OctreeNode<AdditionalCellAttributes>;

private:
    friend class NodeCacheAdapter;

    using children_type = std::array<node_type*, Constants::number_oct>;
    using array_type = std::array<node_type, Constants::number_oct>;
    using NodesCacheKey = std::pair<mpiPP::MPIRank, node_type*>;
    using NodesCacheValue = children_type;
    using NodesCache = std::map<NodesCacheKey, NodesCacheValue>;

public:
    NodeCache() = default;

    NodeCache(const NodeCache&) = delete;
    NodeCache(NodeCache&&) = default;

    NodeCache& operator=(const NodeCache&) = delete;
    NodeCache& operator=(NodeCache&&) = default;

    ~NodeCache() = default;

    /**
     * @brief Sets the RMA window that holds the nodes (on each rank)
     * @param win The RMA window that holds the nodes
     */
    void set_rma_window(mpiPP::RMAWindow<node_type>* win) {
        window = win;
    }

    /**
     * @brief Sets the flag that indicates that the cache is already downloaded.
     *      Use only for tests!
     */
    void set_is_already_downloaded() {
        is_already_downloaded = true;
    }

    /**
     * @brief Empties the cache that was built during the connection phase and frees all local copies
     */
    void clear() {
        remote_nodes_cache.clear();
        memory.clear();
    }

    /**
     * @brief Returns the children of the node. Downloads them from another MPI rank if necessary
     * @param node The node, must not be nullptr and not a leaf
     * @exception Throws a RelearnException if node is nullptr or node is a leaf
     * @return The children (perfect copies of the actual children), does not transfer ownership
     */
    [[nodiscard]] std::array<node_type*, Constants::number_oct> get_children(node_type* const node) const {
        RelearnException::check(node != nullptr, "NodeCache::get_children: node is nullptr");
        RelearnException::check(node->is_parent(), "NodeCache::get_children: node is a leaf");

        if (node->is_actual_id() || is_already_downloaded) {
            return node->get_children();
        }

        return download_children(node);
    }

    /**
     * @brief Returns the currently used memory
     * @return The currently used memory
     */
    [[nodiscard]] std::size_t get_memory_size() const noexcept {
        return memory.size();
    }

    /**
     * @brief Returns the current number of cached values
     * @return The current number of cached values
     */
    [[nodiscard]] std::size_t get_cache_size() const noexcept {
        return remote_nodes_cache.size();
    }

private:
    /**
     * @brief Downloads the children of the node (must be on another MPI rank) and returns the children.
     *      Also saves to nodes locally in order to save bandwidth
     * @param node The node for which the children should be downloaded, must be virtual
     * @exception Throws a RelearnException if node is on the current MPI process or if the saved neuron_id is not virtual
     * @return The downloaded children (perfect copies of the actual children), does not transfer ownership
     */
    [[nodiscard]] std::array<node_type*, Constants::number_oct> download_children(node_type* const node) const {
        const auto target_rank = node->get_mpi_rank();
        RelearnException::check(node->get_cell_neuron_id().is_virtual(), "NodeCache::download_children: Tried to download from a non-virtual node");
        RelearnException::check(target_rank != mpiPP::MPIInfo::get_my_rank(), "NodeCache::download_children: Tried to download a local node");

        auto actual_download = [target_rank, this](node_type* const parent_node) {
            auto local_children = children_type{ nullptr };
            const auto rank_address_pair = NodesCacheKey{ target_rank, parent_node };

            const auto& [iterator, inserted] = remote_nodes_cache.insert({ rank_address_pair, local_children });

            if (!inserted) {
                return iterator->second;
            }

            auto& ref = memory.emplace_back();
            auto* where_to_insert = ref.data();

            auto offset = parent_node->get_cell_neuron_id().get_rma_offset();

            window->get(where_to_insert, Constants::number_oct, offset, target_rank);

            for (auto child_index = 0U; child_index < Constants::number_oct; child_index++) {
                if (parent_node->get_child(child_index) == nullptr) {
                    local_children[child_index] = nullptr;
                    continue;
                }

                local_children[child_index] = &(ref[child_index]);
            }

            iterator->second = local_children;

            return local_children;
        };

        auto local_children = children_type{ nullptr };

#pragma omp critical(node_cache_download)
        local_children = actual_download(node);

        return local_children;
    }

    mutable utility::SemiStableVector<array_type> memory{}; // NOLINT
    mutable NodesCache remote_nodes_cache{};

    mpiPP::RMAWindow<node_type>* window{};

    bool is_already_downloaded{ false };
};
