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
#include "cuda/synaptic_elements/SynapticElementsHandle.h"
#include "cuda/util/LinearizedTreeDeviceHandle.h"
#include "cuda/util/NeuronsExtraInfoHandle.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseCreationResponse.h"

struct NetworkHandle;

/** Device arrays describing a batch of incoming synapse-creation requests (read-only). */
struct SynapseCreationRequestHandle {
    const CudaConfig::number_neurons_type* source_ids; ///< Source neuron ID for each request.
    const CudaConfig::mpi_rank_type* source_ranks;     ///< Source MPI rank for each request.
    const CudaConfig::number_neurons_type* target_ids; ///< Target (local) neuron ID for each request.
    SignalType dendrite_type_needed;                   ///< Signal type (excitatory/inhibitory) required by the requester.
    std::size_t number_requests;                       ///< Total number of requests in the batch.
};

/** Device arrays identifying the requests a batch of synapse-creation responses belongs to (read-only). */
struct SynapseCreationResponseHandle {
    const CudaConfig::number_neurons_type* source_ids; ///< Source neuron ID for each response.
    const CudaConfig::number_neurons_type* target_ids; ///< Target neuron ID for each response.
    SignalType signal_type;                            ///< Signal type of the synapses being created.
    std::size_t number_requests;                       ///< Total number of responses in the batch.
};

/**
 * @brief Processes synapse-creation requests using the MemoryPoolView network representation.
 *
 * @param request              Read-only request data (source/target ids, source ranks, signal type, count).
 * @param responses            Output: per-request creation response.
 * @param den_exc_handle       GPU handle for excitatory dendritic elements.
 * @param den_inh_handle       GPU handle for inhibitory dendritic elements.
 * @param info_handle          Per-neuron metadata (neuron count, rank, disable flags).
 * @param network_view         GPU view of the network graph for edge insertion.
 * @param seed                 Simulation-wide random seed, used to break dendrite-contention ties
 *                             (multiple requests to the same target with too few vacant dendrites)
 *                             randomly rather than by ascending source neuron id.
 * @param step                 Per-round counter, combined with seed so tie-breaking varies by round.
 * @return Number of successfully created synapses.
 */
[[nodiscard]] std::size_t
process_requests_entry_aware(SynapseCreationRequestHandle request,
                             SynapseCreationResponse* responses,
                             SynapticElementsBaseCudaHandle den_exc_handle,
                             SynapticElementsBaseCudaHandle den_inh_handle,
                             NeuronsExtraInfoGPUHandleConst info_handle,
                             NetworkHandle network_view, std::uint64_t seed, std::uint64_t step);

/**
 * @brief Processes synapse-creation responses, updating the axonal element counts of the source neurons.
 *
 * Called after process_requests_entry_aware() on the source rank to record accepted connections.
 *
 * @param mpi_sizes        Per-rank counts of responses received.
 * @param request          Read-only request data (source/target ids, signal type, count) the responses belong to.
 * @param responses        Responses from the target ranks.
 * @param axon_handle      GPU handle for axonal elements (updated on acceptance).
 * @param info_handle      Per-neuron metadata.
 * @param network_view     GPU view of the network graph for edge insertion.
 * @param target_ranks     Output: MPI rank for each accepted target neuron.
 */
void process_responses_entry_aware(const int* mpi_sizes, SynapseCreationResponseHandle request,
                                   SynapseCreationResponse* responses,
                                   SynapticElementsBaseCudaHandle axon_handle,
                                   NeuronsExtraInfoGPUHandleConst info_handle,
                                   NetworkHandle network_view, CudaConfig::mpi_rank_type* target_ranks);

/** Device arrays for a batch of Barnes-Hut target-selection tasks (in/out) for process_calculation_requests_entry_aware. */
struct BHCalculationRequestHandle {
    const int* counts;                                         ///< Per-rank counts of requests to process.
    const CudaConfig::number_neurons_type* requests_source_id; ///< Source neuron IDs (one per vacant axon).
    SimpleVec3d* requests_source_position;                     ///< Output: source neuron positions, filled by the kernel.
    const CudaConfig::number_neurons_type* requests_target_id; ///< Chosen target neuron IDs (output of a prior stage).
    CudaConfig::number_neurons_type* responses;                ///< Output: selected target IDs.
    CudaConfig::mpi_rank_type* source_ranks;                   ///< Output: MPI rank of the source neuron for each task.
    std::size_t number_requests;                               ///< Total number of source-axon tasks.
};

/** Device arrays mapping remote (RMA) tree nodes back to a rank/index, for process_calculation_requests_entry_aware. */
struct RemoteNodeRankHandle {
    const CudaConfig::mpi_rank_type* neuron_ranks;        ///< Device array mapping tree-node index to MPI rank.
    const CudaConfig::bh_index_type* rma_offset_to_index; ///< RMA-offset to tree-index mapping for remote nodes.
    std::size_t rma_size;                                 ///< Size of the RMA mapping.
};

/**
 * @brief GPU-side Barnes-Hut target-selection kernel (MPI-distributed variant).
 *
 * For each axon-carrying source neuron, traverses the distributed linearized octree and
 * selects a target dendrite using the Barnes-Hut acceptance criterion.
 *
 * @param seed                     RNG seed.
 * @param step                     Current simulation step (used to derive per-step RNG state).
 * @param request                  The batch of tasks to process (counts, source/target ids, source
 *                                 positions, and the responses/source_ranks outputs).
 * @param info_handle              Per-neuron metadata; only .number_neurons/.my_rank/.number_ranks are read.
 * @param population               Device handle (positions + vacant dendrites read; vacant_axons/ranks unused).
 * @param tree                     Device handle for the tree structure.
 * @param acceptance_criterion     Barnes-Hut theta threshold.
 * @param remote_node_ranks        Per-tree-node MPI ranks and the RMA-offset-to-index mapping.
 * @param squared_sigma_inv        1/(2σ²) for the Gaussian target-selection kernel.
 */
void process_calculation_requests_entry_aware(std::uint64_t seed, std::uint64_t step,
                                              BHCalculationRequestHandle request,
                                              NeuronsExtraInfoGPUHandleConst info_handle,
                                              NeuronPopulationDeviceHandle population,
                                              LinearizedTreeDeviceHandle tree,
                                              CudaConfig::gaussian_type acceptance_criterion,
                                              RemoteNodeRankHandle remote_node_ranks,
                                              CudaConfig::gaussian_type squared_sigma_inv);

/** Device arrays for a batch of synapse-deletion requests being located (find_synapses_to_delete_random_entry). */
struct DeletionRequestHandle {
    CudaConfig::synaptic_count_type* to_delete;         ///< Output: per-neuron number of synapses to delete.
    CudaConfig::synaptic_count_type* request_partition; ///< Output: exclusive prefix sum of to_delete (size n+1).
    CudaConfig::mpi_rank_type* request_rank;            ///< Output: target MPI rank for each deletion.
    CudaConfig::number_neurons_type* request_neuron_id; ///< Output: target neuron ID for each deletion.
};

/**
 * @brief Determines how many synapses each neuron should delete and selects which ones.
 *
 * Fills deletion_request.to_delete with per-neuron deletion counts and populates the request
 * arrays with the (rank, neuron_id) pairs of the synapses to remove.
 *
 * @param deletion_request     Per-neuron deletion counts/partition and the target rank/neuron-id outputs.
 * @param info_handle          Per-neuron metadata.
 * @param network              GPU view of the network graph.
 * @param my_element_type      Whether we are deleting axons or dendrites.
 * @param my_signal_type       Signal type (excitatory/inhibitory) to delete.
 * @param random_key           cuRAND stream key for random synapse selection.
 * @param sum_to_delete        Total number of synapses to delete across all neurons.
 */
void find_synapses_to_delete_random_entry(DeletionRequestHandle deletion_request,
                                          NeuronsExtraInfoGPUHandleConst info_handle,
                                          NetworkHandle network,
                                          ElementType my_element_type, SignalType my_signal_type, std::uint32_t random_key, std::uint32_t sum_to_delete);

/** Device arrays describing a batch of synapse deletions to commit (commit_deletions_entry). */
struct DeletionCommitHandle {
    std::size_t* partition;                            ///< Exclusive prefix sum of per-neuron deletion counts.
    CudaConfig::mpi_rank_type* other_ranks;            ///< MPI ranks of the synapse partners to delete.
    CudaConfig::number_neurons_type* other_neuron_ids; ///< Neuron IDs of the synapse partners to delete.
    ElementType* my_element_types;                     ///< Element type (axon/dendrite) for each deletion.
    SignalType* my_signal_types;                       ///< Signal type (excitatory/inhibitory) for each deletion.
};

/**
 * @brief Commits a batch of synapse deletions into the network graph and updates element counts.
 * @param info_handle          Per-neuron metadata.
 * @param axon_handle          GPU handle for axonal elements (updated on deletion).
 * @param den_exc_handle       GPU handle for excitatory dendritic elements (updated on deletion).
 * @param den_inh_handle       GPU handle for inhibitory dendritic elements (updated on deletion).
 * @param network_view         GPU view of the network graph (edges removed in place).
 * @param deletion_commit      The deletions to commit (partition, other rank/neuron-id, element/signal types).
 */
auto commit_deletions_entry(NeuronsExtraInfoGPUHandleConst info_handle,
                            SynapticElementsBaseCudaHandle axon_handle,
                            SynapticElementsBaseCudaHandle den_exc_handle,
                            SynapticElementsBaseCudaHandle den_inh_handle,
                            NetworkHandle network_view,
                            DeletionCommitHandle deletion_commit) -> void;

/**
 * @brief Debug helper: validates the OnlyOutgoingView network representation for consistency.
 * @param network_view   GPU network handle to validate.
 * @param number_neurons Total number of neurons.
 */
void check_only_outgoing_view(NetworkHandle network_view, CudaConfig::number_neurons_type number_neurons);

struct EdgeHandle;

/**
 * @brief Removes edges marked for deletion from the MemoryPoolView edge structure.
 * @param sm_handle    Handle to the MemoryPoolView edge data.
 * @param active_ids   Device array of active neuron ids.
 * @param active_count Number of entries in active_ids.
 */
void remove_deleted_memory_pool(EdgeHandle sm_handle, const CudaConfig::number_neurons_type* active_ids, CudaConfig::number_neurons_type active_count);

/**
 * @brief Computes an exclusive prefix sum (partition) of d_to_delete on the GPU.
 *
 * d_partition must point to device memory of size (n + 1): d_partition[0] = 0,
 * d_partition[i+1] = sum(d_to_delete[0..i]).
 *
 * @param d_to_delete  Device array of per-neuron deletion counts (size n).
 * @param d_partition  Device output array of size (n + 1).
 * @param n            Number of neurons.
 * @return The total sum d_partition[n] (total number of deletions).
 */
[[nodiscard]] std::uint32_t compute_sum_and_partition(const CudaConfig::synaptic_count_type* d_to_delete,
                                                      CudaConfig::synaptic_count_type* d_partition,
                                                      std::size_t n);
