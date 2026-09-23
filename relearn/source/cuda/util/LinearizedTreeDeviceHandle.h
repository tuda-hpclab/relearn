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

#include "cuda/CudaConfig.h"
#include "cuda/CudaTypes.h"
#include "cuda/memory/DeviceArray.h"
#include "util/RelearnException.h"

#include <optional>
#include <vector>

/**
 * Read-only GPU view of a linearized octree's structural arrays (one entry per tree node).
 * Shared between BarnesHutCUDA_CU's target-selection/octree-update kernels and
 * ExchangeAlgorithm's process_calculation_requests_entry_aware, which traverses the same tree.
 */
struct LinearizedTreeDeviceHandle {
    const CudaConfig::bh_index_type* child_index;      ///< Index of the first child of each subtree root.
    const CudaConfig::bh_index_type* parent_index;     ///< Parent index for each first-child entry.
    const CudaConfig::gaussian_type* subdomain_length; ///< Subdomain side length per node.
    const NodeType* node_types;                        ///< Node type (Leaf/VirtualNode/RemoteNode/Placeholder) per node.
    const CudaConfig::number_neurons_type* neuron_ids; ///< Neuron ID for leaf nodes.
    CudaConfig::bh_index_type tree_size;               ///< Total number of nodes in the linearized tree.
};

/**
 * Read-write GPU view of one signal type's per-tree-node population arrays (position, vacant
 * dendrite/axon counts, owning MPI rank). Node-indexed, not neuron-indexed -- distinct from
 * SynapticElementsBaseCudaHandle, which tracks the growth model's per-neuron element counts.
 */
struct NeuronPopulationDeviceHandle {
    SimpleVec3d* positions;                            ///< Position per tree node.
    CudaConfig::synaptic_count_type* vacant_dendrites; ///< Vacant dendritic element count per tree node.
    CudaConfig::synaptic_count_type* vacant_axons;     ///< Vacant axonal element count per tree node.
    CudaConfig::mpi_rank_type* ranks;                  ///< Owning MPI rank per tree node.
};

/**
 * Owns the device buffers backing a LinearizedTreeDeviceHandle. The fields are optional because
 * an instance starts empty (before BarnesHutCUDA::init_octree() first runs) and DeviceArray has
 * no default-constructed/empty state of its own; move-assigning a freshly-built instance over an
 * existing one frees the old device memory automatically (DeviceArray's move assignment), which
 * is what makes rebuilding the tree on every BarnesHutCUDA::init_octree() call safe without a
 * manual cudaFree.
 */
struct LinearizedTreeDeviceStorage {
    std::optional<DeviceArray<CudaConfig::bh_index_type>> child_begin_index;
    std::optional<DeviceArray<CudaConfig::bh_index_type>> parent_index;
    std::optional<DeviceArray<CudaConfig::gaussian_type>> subdomain_length;
    std::optional<DeviceArray<CudaConfig::number_neurons_type>> neuron_ids;
    std::optional<DeviceArray<NodeType>> node_types;
    std::optional<DeviceArray<CudaConfig::bh_index_type>> rma_offset_to_index;

    /**
     * @brief Builds the lightweight, kernel-facing view over the currently-owned buffers.
     * @param tree_size Total number of nodes in the linearized tree.
     */
    [[nodiscard]] LinearizedTreeDeviceHandle handle(const CudaConfig::bh_index_type tree_size) const {
        RelearnException::check(child_begin_index.has_value() && parent_index.has_value() && subdomain_length.has_value() && node_types.has_value() && neuron_ids.has_value(),
                                "LinearizedTreeDeviceStorage::handle: Called before the device buffers were initialized");
        return LinearizedTreeDeviceHandle{
            // NOLINTBEGIN(bugprone-unchecked-optional-access) - has_value() checked above
            child_begin_index->device_ptr(), parent_index->device_ptr(), subdomain_length->device_ptr(),
            node_types->device_ptr(), neuron_ids->device_ptr(), tree_size
            // NOLINTEND(bugprone-unchecked-optional-access)
        };
    }
};

/**
 * Owns the device buffers backing a NeuronPopulationDeviceHandle. See LinearizedTreeDeviceStorage
 * for why the fields are optional.
 */
struct NeuronPopulationDeviceStorage {
    std::optional<DeviceArray<SimpleVec3d>> positions;
    std::optional<DeviceArray<CudaConfig::synaptic_count_type>> vacant_dendrites;
    std::optional<DeviceArray<CudaConfig::synaptic_count_type>> vacant_axons;
    std::optional<DeviceArray<CudaConfig::mpi_rank_type>> ranks;

    /** @brief Builds the lightweight, kernel-facing view over the currently-owned buffers. */
    [[nodiscard]] NeuronPopulationDeviceHandle handle() const {
        RelearnException::check(positions.has_value() && vacant_dendrites.has_value() && vacant_axons.has_value() && ranks.has_value(),
                                "NeuronPopulationDeviceStorage::handle: Called before the device buffers were initialized");
        return NeuronPopulationDeviceHandle{
            // NOLINTBEGIN(bugprone-unchecked-optional-access) - has_value() checked above
            positions->device_ptr(), vacant_dendrites->device_ptr(), vacant_axons->device_ptr(), ranks->device_ptr()
            // NOLINTEND(bugprone-unchecked-optional-access)
        };
    }
};

/** Result of BarnesHutCUDA_CU::init_neurons(): the tree's owning device storage and the total mem usage. */
struct LinearizedTreeInitResult {
    LinearizedTreeDeviceStorage storage;
    std::size_t mem_usage{};
};

/** Result of BarnesHutCUDA_CU::init_neuron_details(): the population's owning device storage and the total mem usage. */
struct NeuronPopulationInitResult {
    NeuronPopulationDeviceStorage storage;
    std::size_t mem_usage{};
};

/**
 * Write-only GPU view of the per-tree-node vacant-element counts recomputed by
 * BarnesHutCUDA_CU::update_leaf_nodes_entry, split by signal type and element kind.
 */
struct TreeVacancyOutputHandle {
    CudaConfig::synaptic_count_type* exc_vacant_axon; ///< Excitatory vacant axon count per tree node.
    CudaConfig::synaptic_count_type* exc_vacant_dend; ///< Excitatory vacant dendrite count per tree node.
    CudaConfig::synaptic_count_type* inh_vacant_axon; ///< Inhibitory vacant axon count per tree node.
    CudaConfig::synaptic_count_type* inh_vacant_dend; ///< Inhibitory vacant dendrite count per tree node.
};

/**
 * Result of a target-neuron search for one signal type: per-rank send counts, source neuron IDs,
 * source positions, and target neuron IDs (all in matching per-request order).
 */
struct TargetNeuronSearchResult {
    std::vector<int> counts_per_rank{};
    DeviceArray<CudaConfig::number_neurons_type> source_ids;
    DeviceArray<SimpleVec3d> source_positions;
    DeviceArray<CudaConfig::number_neurons_type> target_ids;
};

/** Result of BarnesHutCUDA_CU::find_target_neurons_for(): one TargetNeuronSearchResult per signal type. */
struct TargetNeuronSearchResultBothSignalTypes {
    TargetNeuronSearchResult excitatory;
    TargetNeuronSearchResult inhibitory;
};
