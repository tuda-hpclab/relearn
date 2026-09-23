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
#include "cuda/synaptic_elements/SynapticElementsHandle.h"
#include "cuda/util/LinearizedTreeDeviceHandle.h"
#include "neurons/enums/SynapticElementType.h"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace BarnesHutCUDA_CU {

/**
 * @brief Allocates and copies the structural octree arrays to the device.
 * @param h_child_begin_index  Index of the first child of each subtree root (host).
 * @param h_parent_index       Parent index for each first-child entry (host).
 * @param h_subdomain_length   Subdomain side lengths for each node (host).
 * @param neuron_ids           Neuron IDs for leaf nodes (host).
 * @param rma_offset_to_index  Mapping from RMA offset to tree node index (host).
 * @param h_node_types         Node type (Leaf/VirtualNode/RemoteNode/Placeholder) per node (host).
 * @return The tree's owning device storage and the total mem usage. Call .handle(tree_size) on the
 *      storage to get the lightweight view kernels take; the storage itself must outlive the handle.
 */
[[nodiscard]] LinearizedTreeInitResult
init_neurons(
    std::span<const CudaConfig::bh_index_type> h_child_begin_index,
    std::span<const CudaConfig::bh_index_type> h_parent_index,
    std::span<const CudaConfig::gaussian_type> h_subdomain_length,
    std::span<const CudaConfig::number_neurons_type> neuron_ids,
    std::span<const CudaConfig::bh_index_type> rma_offset_to_index,
    std::span<const NodeType> h_node_types);

/**
 * @brief Allocates and copies per-neuron position and element-count arrays to the device.
 * @param h_neuron_positions  Spatial positions of all neurons in the tree (host).
 * @param h_vacant_dendrites  Vacant dendritic element counts per neuron (host).
 * @param h_vacant_axons      Vacant axonal element counts per neuron (host).
 * @param h_ranks             MPI rank of the owning process for each neuron (host).
 * @return The population's owning device storage and the total mem usage. Call .handle() on the
 *      storage to get the lightweight view kernels take; the storage itself must outlive the handle.
 */
[[nodiscard]] NeuronPopulationInitResult
init_neuron_details(
    std::span<const SimpleVec3d> h_neuron_positions,
    std::span<const CudaConfig::synaptic_count_type> h_vacant_dendrites,
    std::span<const CudaConfig::synaptic_count_type> h_vacant_axons,
    std::span<const CudaConfig::mpi_rank_type> h_ranks);

/**
 * @brief Copies updated neuron positions and vacant-dendrite counts from device back to host.
 * @param population                Device handle for positions and vacant-dendrite counts (only
 *                                  those two fields are read; vacant_axons/ranks are ignored).
 * @param size                      Number of entries to copy.
 * @param updated_positions         Host output array for positions.
 * @param updated_vacant_dendrites  Host output array for vacant dendrite counts.
 */
void get_updated_octree(NeuronPopulationDeviceHandle population,
                        std::size_t size,
                        std::vector<SimpleVec3d>& updated_positions,
                        std::vector<CudaConfig::synaptic_count_type>& updated_vacant_dendrites);

/**
 * @brief Computes weighted-average positions and vacant-dendrite sums for virtual nodes level by level.
 *
 * Works bottom-up through h_level_indices so that each virtual node's aggregate reflects
 * the up-to-date values of all its descendants.
 *
 * @param population        Device handle for positions (virtual nodes written in place) and
 *                          vacant-dendrite counts (virtual nodes aggregated); vacant_axons/ranks unused.
 * @param tree              Device handle for the tree structure; only child_index and tree_size are read.
 * @param h_level_indices   Host array of start indices for each octree level (bottom-up order).
 * @param stream            CUDA stream to use.
 */
void calculate_updated_octree_host(NeuronPopulationDeviceHandle population,
                                   LinearizedTreeDeviceHandle tree,
                                   std::span<const CudaConfig::bh_index_type> h_level_indices,
                                   const std::shared_ptr<StreamWrapper>& stream);

/**
 * @brief Tests the Barnes-Hut acceptance criterion for a single (source, target) node pair.
 * Exposed for unit testing of the acceptance formula.
 * @param test_results_size         Number of test cases.
 * @param source_position           Position of the source neuron.
 * @param target_position           Position of the target node.
 * @param vacant_dendritic_elements Vacant dendrite count of the target node.
 * @param subdomain_length          Subdomain side length of the target node.
 * @param acceptance_criterion      Barnes-Hut theta threshold.
 * @param is_leaf                   True if the target is a leaf node.
 * @return One bool per test case.
 */
std::vector<bool> test_acceptance_criterion_host(CudaConfig::number_neurons_type test_results_size,
                                                 const SimpleVec3d& source_position,
                                                 const SimpleVec3d& target_position,
                                                 CudaConfig::synaptic_count_type vacant_dendritic_elements,
                                                 CudaConfig::gaussian_type subdomain_length,
                                                 CudaConfig::gaussian_type acceptance_criterion, bool is_leaf);

/**
 * @brief Atomic-counter variant of find_target_neurons processing excitatory and inhibitory axons concurrently.
 *
 * Launches two independent CUDA streams (exc_stream and inh_stream) for the excitatory and
 * inhibitory populations, then merges the results.  Returns two tuples — one per population —
 * each containing per-rank send counts, source IDs, source positions, and target IDs.
 *
 * @param seed                               RNG seed.
 * @param step                               Current simulation step.
 * @param population_exc                     Device handle (positions + vacant dendrites read;
 *                                            vacant_axons/ranks unused) for the excitatory population.
 * @param node_id_vacant_axons_mapping_exc   Per-axon source IDs for excitatory population (device).
 * @param number_tasks_exc                   Number of excitatory axon tasks.
 * @param population_inh                     Same as population_exc, for the inhibitory population.
 * @param node_id_vacant_axons_mapping_inh   Per-axon source IDs for inhibitory population (device).
 * @param number_tasks_inh                   Number of inhibitory axon tasks.
 * @param neurons_count                      Total number of neurons (population size, distinct from tree.tree_size).
 * @param tree                               Device handle for the tree structure.
 * @param acceptance_criterion               Barnes-Hut theta threshold.
 * @param neuron_ranks                       Device array of MPI ranks per tree node.
 * @param my_rank                            MPI rank of this process.
 * @param squared_sigma_inv                  1/(2σ²) for the Gaussian kernel.
 * @param number_ranks                       Total number of MPI ranks.
 * @param number_neurons                     Total number of local neurons.
 * @param exc_stream                         CUDA stream for the excitatory sub-problem.
 * @param inh_stream                         CUDA stream for the inhibitory sub-problem.
 * @return Pair of (excitatory result tuple, inhibitory result tuple), each containing
 *         per-rank send counts, source IDs, source positions, and target IDs.
 */
[[nodiscard]] TargetNeuronSearchResultBothSignalTypes
find_target_neurons(std::uint64_t seed, std::uint64_t step,

                    NeuronPopulationDeviceHandle population_exc,
                    const CudaConfig::bh_index_type* node_id_vacant_axons_mapping_exc,
                    std::size_t number_tasks_exc,

                    NeuronPopulationDeviceHandle population_inh,
                    const CudaConfig::bh_index_type* node_id_vacant_axons_mapping_inh,
                    std::size_t number_tasks_inh,

                    CudaConfig::number_neurons_type neurons_count,
                    LinearizedTreeDeviceHandle tree,

                    CudaConfig::gaussian_type acceptance_criterion,
                    const CudaConfig::mpi_rank_type* const neuron_ranks,
                    CudaConfig::mpi_rank_type my_rank,
                    CudaConfig::gaussian_type squared_sigma_inv,
                    CudaConfig::mpi_rank_type number_ranks,
                    CudaConfig::number_neurons_type number_neurons,
                    const std::shared_ptr<StreamWrapper>& exc_stream,
                    const std::shared_ptr<StreamWrapper>& inh_stream);

/**
 * @brief Recomputes the vacant-element counts stored in the leaf nodes of the linearized tree.
 *
 * Reads the current synaptic-element arrays and writes updated excitatory/inhibitory
 * vacant axon and vacant dendrite counts into the per-node tree arrays used by the BH traversal.
 *
 * @param tree               Linearized tree's structural arrays (tree_size/node_types/neuron_ids read; child_index/parent_index/subdomain_length unused here).
 * @param signal_types       Signal type (excitatory/inhibitory) for each leaf neuron.
 * @param axon_handle        Read-only GPU handle for axonal elements.
 * @param den_exc_handle     Read-only GPU handle for excitatory dendritic elements.
 * @param den_inh_handle     Read-only GPU handle for inhibitory dendritic elements.
 * @param output             Output: per-tree-node vacant axon/dendrite counts, by signal type.
 * @param stream             CUDA stream to use.
 */
void update_leaf_nodes_entry(LinearizedTreeDeviceHandle tree, const SignalType* signal_types,
                             SynapticElementsBaseCudaHandleConst axon_handle, SynapticElementsBaseCudaHandleConst den_exc_handle, SynapticElementsBaseCudaHandleConst den_inh_handle,
                             TreeVacancyOutputHandle output, const std::shared_ptr<StreamWrapper>& stream);

/**
 * @brief Updates remote-node entries in the tree with data received from other MPI ranks.
 *
 * After each MPI exchange, remote subtree aggregates (positions, vacant dendrite counts,
 * and neuron IDs) need to be written into the corresponding RemoteNode placeholders so
 * that subsequent BH traversals on this rank see up-to-date information.
 *
 * @param tree_size          Total number of nodes in the linearized tree.
 * @param number_ranks       Total number of MPI ranks.
 * @param local_branch_nodes Tree-node indices of the local branch nodes to update.
 * @param population         Device handle (positions + vacant dendrites written; vacant_axons/ranks unused).
 * @param neuron_ids         Device array of neuron IDs to write into remote nodes.
 * @param my_rank            MPI rank of this process (used to skip own-rank nodes).
 * @param stream             CUDA stream to use.
 */
void update_remote_nodes_host(CudaConfig::bh_index_type tree_size, int number_ranks, std::span<const CudaConfig::bh_index_type> local_branch_nodes,
                              NeuronPopulationDeviceHandle population, CudaConfig::number_neurons_type* neuron_ids, CudaConfig::mpi_rank_type my_rank, const std::shared_ptr<StreamWrapper>& stream);

} // namespace BarnesHutCUDA_CU
