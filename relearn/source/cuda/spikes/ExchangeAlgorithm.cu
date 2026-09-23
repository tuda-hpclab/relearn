/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "ExchangeAlgorithm.h"

#include "network_graph/NetworkHandle.h"
#include "algorithm/BarnesHutInternalCUDA/BarnesHutCUDA_CU.cuh"
#include "cuda/network_graph/NetworkGraph.cuh"
#include "cuda/random/RandomNumber.cuh"
#include "cuda/synaptic_elements/SynapticElements.cuh"
#include "cuda/util/NeuronsExtraInfoHandle.h"
#include "cuda/util/Util.cuh"
#include "util/Timers.h"

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

/**
 * Kernel-facing view of a PartitionedRequestGroups (defined further below): raw device pointers
 * into its sorted_indices/displacements, plus the group count. Passed by value into
 * process_requests_kernel_aware and process_responses_kernel_aware, which otherwise took the same
 * three values as separate parameters.
 */
struct RequestGroupingHandle {
    const std::size_t* indices; ///< Request indices, sorted/grouped by target neuron (and tie-break criteria).
    const int* partition;       ///< Exclusive prefix-sum group boundaries into `indices` (size number_groups + 1).
    int number_groups;          ///< Number of distinct groups (threads with work).
};

template <typename IncomingLocalNet, typename IncomingDistantNet, typename OutgoingLocalNet, typename OutgoingDistantNet>
__global__ void commit_deletions_kernel(NeuronsExtraInfoGPUHandleConst info_handle,
                                        SynapticElementsBaseCudaHandle axon_handle,
                                        SynapticElementsBaseCudaHandle den_exc_handle,
                                        SynapticElementsBaseCudaHandle den_inh_handle,
                                        IncomingLocalNet incoming_local_edges_handle, IncomingDistantNet incoming_distant_edges_handle,
                                        OutgoingLocalNet outgoing_local_edges_handle, OutgoingDistantNet outgoing_distant_edges_handle,
                                        DeletionCommitHandle deletion_commit) {
    const CudaConfig::number_neurons_type neuron_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (neuron_id >= info_handle.number_neurons)
        return;

    const auto begin = deletion_commit.partition[neuron_id];
    const auto end = deletion_commit.partition[neuron_id + 1];

    for (auto i = begin; i < end; i++) {
        const auto other_rank = deletion_commit.other_ranks[i];
        const auto other_neuron_id = deletion_commit.other_neuron_ids[i];
        const auto my_elements_type = deletion_commit.my_element_types[i];

        if (my_elements_type == ElementType::Axon) {
            const bool axon_excitatory = deletion_commit.my_signal_types[i] == SignalType::Excitatory;
            if (info_handle.my_rank == other_rank && outgoing_local_edges_handle.has_local_edges()) {
                outgoing_local_edges_handle.remove_synapse(neuron_id, other_rank, other_neuron_id, axon_excitatory);
            } else {
                outgoing_distant_edges_handle.remove_synapse(neuron_id, other_rank, other_neuron_id, axon_excitatory);
            }
            disconnect_elements(1, neuron_id, axon_handle.connected_elements, axon_handle.vacant_elements, info_handle.number_neurons);
        }
    }

    for (auto i = begin; i < end; i++) {
        const auto other_rank = deletion_commit.other_ranks[i];
        const auto other_neuron_id = deletion_commit.other_neuron_ids[i];
        const auto my_elements_type = deletion_commit.my_element_types[i];
        const auto my_signal_type = deletion_commit.my_signal_types[i];

        if (my_elements_type == ElementType::Dendrite) {
            if (info_handle.my_rank == other_rank && incoming_local_edges_handle.has_local_edges()) {
                incoming_local_edges_handle.remove_synapse(neuron_id, other_rank, other_neuron_id,
                                                           my_signal_type == SignalType::Excitatory);
            } else {
                incoming_distant_edges_handle.remove_synapse(neuron_id, other_rank, other_neuron_id, my_signal_type == SignalType::Excitatory);
            }
            if (my_signal_type == SignalType::Excitatory) {
                disconnect_elements(1, neuron_id, den_exc_handle.connected_elements, den_exc_handle.vacant_elements, info_handle.number_neurons);
            } else {
                disconnect_elements(1, neuron_id, den_inh_handle.connected_elements, den_inh_handle.vacant_elements, info_handle.number_neurons);
            }
        }
    }
}

__global__ void process_calculation_requests_aware_kernel(const std::uint64_t seed, const std::uint32_t number_threads, const std::uint64_t step,
                                                          const BHCalculationRequestHandle request,
                                                          const NeuronsExtraInfoGPUHandleConst info_handle,
                                                          const NeuronPopulationDeviceHandle population,
                                                          const LinearizedTreeDeviceHandle tree,
                                                          const CudaConfig::gaussian_type acceptance_criterion,
                                                          const RemoteNodeRankHandle remote_node_ranks, const CudaConfig::gaussian_type squared_sigma_inv) {
    const auto linear_tree_size = tree.tree_size;
    const auto* const neuron_ids = tree.neuron_ids;

    const uint64_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id >= request.number_requests) {
        return;
    }

    auto source_rank = 0;
    auto offset = 0;

    for (; source_rank < info_handle.number_ranks; ++source_rank) {
        offset += request.counts[source_rank];
        if (thread_id < offset) {
            break;
        }
    }
    RELEARN_DEVICE_CUDA_CHECK(source_rank < info_handle.number_ranks, "process_calculation_requests_aware_kernel: Rank is too high %i < %i", source_rank, info_handle.number_ranks);
    request.source_ranks[thread_id] = source_rank;

    const auto source_neuron_id = request.requests_source_id[thread_id];
    const auto source_position = request.requests_source_position[thread_id];
    const auto target_node = request.requests_target_id[thread_id];

    if (source_rank == info_handle.my_rank && target_node == source_neuron_id) {
        request.responses[thread_id] = info_handle.number_neurons;
        return;
    }
    if (source_rank == info_handle.my_rank) {
        request.responses[thread_id] = target_node;
        return;
    }

    const auto rma_index = target_node;
    RELEARN_DEVICE_CUDA_CHECK(rma_index < remote_node_ranks.rma_size, "process_calculation_requests_aware_kernel: RMA index %u is too large %lu", rma_index, remote_node_ranks.rma_size);
    const auto index = remote_node_ranks.rma_offset_to_index[rma_index];
    RELEARN_DEVICE_CUDA_CHECK(index < linear_tree_size, "process_calculation_requests_aware_kernel: Index %u larger than size of linear tree %u", index, linear_tree_size);

    const auto picked_target = BarnesHutCUDA_CU::find_single_target_neuron(thread_id, number_threads, seed, step, tree.child_index[index], source_position,
                                                                           population, tree, acceptance_criterion,
                                                                           remote_node_ranks.neuron_ranks, info_handle.my_rank, squared_sigma_inv);
    if (picked_target == std::numeric_limits<CudaConfig::bh_index_type>::max()) {
        request.responses[thread_id] = info_handle.number_neurons;
        return;
    }
    const auto target_neuron_id = neuron_ids[picked_target];
    request.responses[thread_id] = target_neuron_id;
}

template <typename IncomingEdgesType, typename IncomingDistantEdgesType>
__global__ void process_requests_kernel_aware(const SynapseCreationRequestHandle request, const RequestGroupingHandle grouping, SynapseCreationResponse* responses, SynapticElementsBaseCudaHandle den_exc_handle, SynapticElementsBaseCudaHandle den_inh_handle, NeuronsExtraInfoGPUHandleConst info_handle,
                                              IncomingEdgesType incoming_local_edges, IncomingDistantEdgesType incoming_distant_edges, std::uint64_t* created_synapses) {
    const auto* source_ids = request.source_ids;
    const auto* source_ranks = request.source_ranks;
    const auto* target_ids = request.target_ids;
    const auto dendrite_type_needed = request.dendrite_type_needed;
    const auto* indices = grouping.indices;
    const auto* partition = grouping.partition;
    const uint64_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id >= grouping.number_groups) {
        return;
    }
    const auto begin = partition[thread_id];
    const auto end = partition[thread_id + 1];

    if (begin == end) {
        return;
    }

    const auto target_neuron_id = target_ids[indices[begin]];

    if (target_neuron_id != info_handle.number_neurons) {
        created_synapses[thread_id] = 0;
    }

    for (auto i = begin; i < end; i++) {
        const auto converted_idx = indices[i];
        const auto source_rank = source_ranks[converted_idx];
        const auto source_neuron_id = source_ids[converted_idx];
        RELEARN_DEVICE_CUDA_CHECK(target_neuron_id == target_ids[converted_idx], "process_requests_kernel_aware: target_neuron_id mismatch within group: %u != %u", target_neuron_id, target_ids[converted_idx]);

        if (target_neuron_id == info_handle.number_neurons || target_neuron_id == std::numeric_limits<CudaConfig::number_neurons_type>::max() || source_rank == info_handle.my_rank && target_neuron_id == source_neuron_id) {
            responses[converted_idx] = SynapseCreationResponse::Failed;
            continue;
        }
        RELEARN_DEVICE_CUDA_CHECK(source_rank != info_handle.my_rank || target_neuron_id != source_neuron_id, "process_requests_kernel_aware: Target and source are equal!");

        auto* d_connected_elements = dendrite_type_needed == SignalType::Excitatory ? den_exc_handle.connected_elements : den_inh_handle.connected_elements;
        auto* d_vacant_elements = dendrite_type_needed == SignalType::Excitatory ? den_exc_handle.vacant_elements : den_inh_handle.vacant_elements;

        RELEARN_DEVICE_CUDA_CHECK(target_neuron_id < info_handle.number_neurons, "process_requests_kernel_aware: target_neuron_id %u exceeds my neurons %u", target_neuron_id, info_handle.number_neurons);

        const auto number_free_elements = d_vacant_elements[target_neuron_id];
        if (number_free_elements == 0) {
            // Other axons were faster and came first
            responses[converted_idx] = SynapseCreationResponse::Failed;
            continue;
        }

        // Increment number of connected dendrites
        connect_elements(1, target_neuron_id, d_connected_elements, d_vacant_elements, info_handle.number_neurons);

        // Set response to "connected" (success)
        responses[converted_idx] = SynapseCreationResponse::Succeeded;

        const auto excitatory = SignalType::Excitatory == dendrite_type_needed;
        created_synapses[target_neuron_id]++;

        if (source_rank == info_handle.my_rank && incoming_local_edges.has_local_edges()) {
            RELEARN_DEVICE_CUDA_CHECK(target_neuron_id != source_neuron_id, "process_requests_kernel_aware: Target and source are equal");
            RELEARN_DEVICE_CUDA_CHECK(source_rank == info_handle.my_rank, "process_requests_kernel_aware: Adding distant edge to local one");
            auto success = incoming_local_edges.add_synapse(target_neuron_id, source_rank, source_neuron_id, excitatory);
            RELEARN_DEVICE_CUDA_CHECK(success, "process_requests_kernel_aware: incoming local edges are full");
        } else {
            RELEARN_DEVICE_CUDA_CHECK(source_rank != info_handle.my_rank, "process_requests_kernel_aware: Adding local edge to distant one");
            const auto success = incoming_distant_edges.add_synapse(target_neuron_id, source_rank, source_neuron_id, excitatory);

            RELEARN_DEVICE_CUDA_CHECK(success, "process_requests_kernel_aware: incoming distant edges are full");
        }
    }
}

template <typename LocalEdges, typename DistantEdges>
__global__ void process_responses_kernel_aware(const RequestGroupingHandle grouping, const SynapseCreationResponseHandle request, SynapseCreationResponse* responses, SynapticElementsBaseCudaHandle axon_handle, NeuronsExtraInfoGPUHandleConst info_handle,
                                               LocalEdges outgoing_local_edges_handle, DistantEdges outgoing_distant_edges_handle, const int* const counts, CudaConfig::mpi_rank_type* target_ranks) {
    const auto* source_ids = request.source_ids;
    const auto* target_ids = request.target_ids;
    const auto dendrite_type_needed = request.signal_type;
    const auto* indices = grouping.indices;
    const auto* partition = grouping.partition;
    const uint64_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (thread_id >= grouping.number_groups) {
        return;
    }
    const auto begin = partition[thread_id];
    const auto end = partition[thread_id + 1];

    if (begin == end) {
        return;
    }

    const auto source_neuron_id = source_ids[indices[begin]];

    for (auto i = begin; i < end; i++) {
        const auto converted_idx = indices[i];
        const auto connected = responses[converted_idx];
        if (connected == SynapseCreationResponse::Failed) {
            continue;
        }

        std::size_t target_rank = 0;
        std::size_t offset = 0;

        for (; target_rank < info_handle.number_ranks; ++target_rank) {
            offset += counts[target_rank];
            if (converted_idx < offset) {
                break;
            }
        }

        RELEARN_DEVICE_CUDA_CHECK(target_rank < info_handle.number_ranks, "process_responses_kernel_aware: Target rank %lu < %i", target_rank, info_handle.number_ranks);

        const auto target_neuron_id = target_ids[converted_idx];
        if (target_neuron_id == info_handle.number_neurons) {
            continue;
        }
        target_ranks[converted_idx] = target_rank;
        RELEARN_DEVICE_CUDA_CHECK(source_neuron_id == source_ids[converted_idx], "process_responses_kernel_aware: source_neuron_id mismatch within group");

        RELEARN_DEVICE_CUDA_CHECK(source_neuron_id < info_handle.number_neurons,
                                  "process_responses_kernel_aware: The source neuron id was too large: %u vs %u", source_neuron_id, info_handle.number_neurons);
        RELEARN_DEVICE_CUDA_CHECK(target_neuron_id < info_handle.number_neurons,
                                  "process_responses_kernel_aware: The target neuron id was too large: %u vs %u", target_neuron_id, info_handle.number_neurons);

        const auto number_free_elements = axon_handle.vacant_elements[source_neuron_id];
        RELEARN_DEVICE_CUDA_CHECK(number_free_elements > 0,
                                  "process_responses_kernel_aware: The source neuron %u did not have a vacant element", source_neuron_id);

        // Increment number of connected axons
        connect_elements(1, source_neuron_id, axon_handle.connected_elements, axon_handle.vacant_elements, info_handle.number_neurons);

        const auto excitatory = SignalType::Excitatory == dendrite_type_needed;
        if (target_rank == info_handle.my_rank && outgoing_local_edges_handle.has_local_edges()) {
            RELEARN_DEVICE_CUDA_CHECK(target_neuron_id != source_neuron_id, "process_responses_kernel_aware: Target and source are equal");
            RELEARN_DEVICE_CUDA_CHECK(target_rank == info_handle.my_rank, "process_responses_kernel_aware: Adding distant edge to local one");
            const auto success = outgoing_local_edges_handle.add_synapse(source_neuron_id, target_rank, target_neuron_id, excitatory);

            RELEARN_DEVICE_CUDA_CHECK(success, "process_responses_kernel_aware: outgoing local edges are full");
            continue;
        }
        RELEARN_DEVICE_CUDA_CHECK(target_rank != info_handle.my_rank, "process_responses_kernel_aware: Adding local edge to distant one");
        const auto success = outgoing_distant_edges_handle.add_synapse(source_neuron_id, target_rank, target_neuron_id, excitatory);
        RELEARN_DEVICE_CUDA_CHECK(success, "process_responses_kernel_aware: outgoing distant edges are full");
    }
}

// Groups requests by target_ids (and, within a target, by source_ranks) so that
// process_requests_kernel_aware/process_responses_kernel_aware can each hand one thread a
// contiguous run of requests for the same neuron.
//
// randomize_ties selects how ties (same target, same source_rank) are broken:
//   - false: an explicit ascending source_ids tiebreak, giving a fully deterministic order.
//     Safe whenever nothing downstream makes an accept/reject decision based on where a request
//     falls within its group (e.g. process_responses_entry_aware, which only uses the grouping
//     to batch axon-side bookkeeping per source neuron -- every request reaching it already
//     succeeded, so there is no contention left to resolve and no ordering-driven bias possible).
//   - true: ties are broken by a seed/step-derived shuffle instead, via a stable sort that
//     otherwise ignores source_ids. Required whenever a downstream consumer accepts requests
//     within a group sequentially until some shared resource is exhausted (e.g.
//     process_requests_kernel_aware's dendrite-vacancy check) -- with the deterministic tiebreak,
//     that resolves contention by ascending source neuron id every round, systematically favoring
//     smaller-numbered neurons instead of choosing among contenders at random.
/** Requests sorted/grouped by target neuron: sorted request indices, per-group displacement offsets, and the number of groups. */
struct PartitionedRequestGroups {
    thrust::device_vector<std::size_t> sorted_indices;
    thrust::device_vector<int> displacements;
    int number_groups;

    [[nodiscard]] RequestGroupingHandle handle() const {
        return RequestGroupingHandle{ sorted_indices.data().get(), displacements.data().get(), number_groups };
    }
};

PartitionedRequestGroups
partition_after_target(const std::size_t number_requests, const CudaConfig::number_neurons_type* const target_ids, const CudaConfig::mpi_rank_type* source_ranks, const CudaConfig::number_neurons_type* source_ids,
                       const bool randomize_ties, const std::uint64_t seed = 0, const std::uint64_t step = 0) {
    Timers::start(TimerRegion::CUDA_PARTITION);
    thrust::device_vector<std::size_t> d_idx(number_requests);
    thrust::sequence(d_idx.begin(), d_idx.end());
    if (randomize_ties) {
        thrust::default_random_engine rng(static_cast<unsigned int>(seed ^ (step * 0x9E3779B97F4A7C15ULL)));
        thrust::shuffle(d_idx.begin(), d_idx.end(), rng);
        thrust::stable_sort(d_idx.begin(), d_idx.end(), [target_ids, source_ranks] __device__(int i, int j) { return target_ids[i] < target_ids[j] || target_ids[i] == target_ids[j] && source_ranks[i] < source_ranks[j]; });
    } else {
        thrust::sort(d_idx.begin(), d_idx.end(), [target_ids, source_ranks, source_ids] __device__(int i, int j) { return target_ids[i] < target_ids[j] || target_ids[i] == target_ids[j] && source_ranks[i] < source_ranks[j] || target_ids[i] == target_ids[j] && source_ranks[i] == source_ranks[j] && source_ids[i] < source_ids[j]; });
    }

    thrust::device_vector<int> flags(number_requests);
    flags[0] = 1;
    thrust::transform(
        d_idx.begin() + 1,
        d_idx.end(),
        d_idx.begin(),
        flags.begin() + 1,
        [target_ids] __host__ __device__(int curr, int prev) {
            return target_ids[curr] != target_ids[prev];
        });
    int num_groups = thrust::reduce(flags.begin(), flags.end());
    thrust::device_vector<int> displ(num_groups + 1);

    auto counting_begin = thrust::make_counting_iterator(0);
    auto counting_end = counting_begin + number_requests;

    auto end = thrust::copy_if(
        counting_begin,
        counting_end,
        flags.begin(),
        displ.begin(),
        [] __host__ __device__(int flag) { return flag != 0; });

    RELEARN_CUDA_CHECK((end - displ.begin()) == num_groups, "partition_after_target: Error while calculating displ");
    displ[num_groups] = number_requests;

    Timers::stop_and_add(TimerRegion::CUDA_PARTITION);

    return PartitionedRequestGroups{ d_idx, displ, num_groups };
}

template <typename LocalNetworkType, typename DistantNetworkType>
std::size_t
process_requests_entry_aware(const SynapseCreationRequestHandle request,
                             SynapseCreationResponse* responses,
                             SynapticElementsBaseCudaHandle den_exc_handle,
                             SynapticElementsBaseCudaHandle den_inh_handle,
                             NeuronsExtraInfoGPUHandleConst info_handle,
                             LocalNetworkType incoming_local_edges, DistantNetworkType incoming_distant_edges,
                             const std::uint64_t seed, const std::uint64_t step) {

    if (request.number_requests == 0) {
        return 0;
    }

    // randomize_ties=true: contention for a target's vacant dendrites must not be resolved by
    // ascending source neuron id (see partition_after_target's comment).
    const auto groups = partition_after_target(request.number_requests, request.target_ids, request.source_ranks, request.source_ids, true, seed, step);
    const auto grouping = groups.handle();

    thrust::device_vector<std::uint64_t> created_synapses(info_handle.number_neurons);

    Timers::start(TimerRegion::CUDA_PROCESS_REQUESTS_KERNEL);
    const auto& [blocks, threads] = get_grid_ands_block_size(groups.number_groups, process_requests_kernel_aware<LocalNetworkType, DistantNetworkType>);
    process_requests_kernel_aware<LocalNetworkType, DistantNetworkType><<<blocks, threads>>>(request, grouping,
                                                                                             responses, den_exc_handle, den_inh_handle,
                                                                                             info_handle,
                                                                                             incoming_local_edges, incoming_distant_edges,
                                                                                             thrust::raw_pointer_cast(created_synapses.data()));
    kernelErrCheck();
    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_PROCESS_REQUESTS_KERNEL);

    Timers::start(TimerRegion::CUDA_REDUCE);
    const auto sum_created_synapses = thrust::reduce(created_synapses.begin(), created_synapses.end(), 0,
                                                     cuda::std::plus<std::size_t>());
    ;

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_REDUCE);
    return sum_created_synapses;
}

template <typename LocalEdges, typename DistantEdges>
void process_responses_entry_aware(const int* mpi_sizes, const SynapseCreationResponseHandle request,
                                   SynapseCreationResponse* responses,
                                   SynapticElementsBaseCudaHandle axon_handle,
                                   NeuronsExtraInfoGPUHandleConst info_handle,
                                   LocalEdges& outgoing_local_edges_handle, DistantEdges& outgoing_distant_edges_handle, CudaConfig::mpi_rank_type* target_ranks) {

    if (request.number_requests == 0) {
        return;
    }

    // randomize_ties=false: every request reaching here already succeeded on the dendrite side,
    // so there is no axon-side contention left to resolve -- this grouping is purely for batching
    // per-source bookkeeping onto one thread each, and needs no random tiebreak (see
    // partition_after_target's comment).
    const auto groups = partition_after_target(request.number_requests, request.source_ids, target_ranks, request.target_ids, false);
    const auto grouping = groups.handle();

    Timers::start(TimerRegion::CUDA_PROCESS_RESPONSES_KERNEL);

    const auto& [blocks, threads] = get_grid_ands_block_size(groups.number_groups, process_responses_kernel_aware<LocalEdges, DistantEdges>);
    process_responses_kernel_aware<LocalEdges, DistantEdges><<<blocks, threads>>>(grouping, request, responses,
                                                                                  axon_handle,
                                                                                  info_handle,
                                                                                  outgoing_local_edges_handle, outgoing_distant_edges_handle, mpi_sizes, target_ranks);

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_PROCESS_RESPONSES_KERNEL);
}

void process_calculation_requests_entry_aware(const std::uint64_t seed, const std::uint64_t step,
                                              const BHCalculationRequestHandle request,
                                              const NeuronsExtraInfoGPUHandleConst info_handle,
                                              const NeuronPopulationDeviceHandle population,
                                              const LinearizedTreeDeviceHandle tree,
                                              const CudaConfig::gaussian_type acceptance_criterion,
                                              const RemoteNodeRankHandle remote_node_ranks,
                                              const CudaConfig::gaussian_type squared_sigma_inv) {
    if (request.number_requests == 0) {
        return;
    }

    const auto& [blocks, threads] = get_grid_ands_block_size(request.number_requests, process_calculation_requests_aware_kernel);
    Timers::start(TimerRegion::CUDA_PROCESS_CALCULATION_REQUESTS_KERNEL);
    process_calculation_requests_aware_kernel<<<blocks, threads>>>(seed, threads, step,
                                                                   request, info_handle,
                                                                   population, tree, acceptance_criterion,
                                                                   remote_node_ranks,
                                                                   squared_sigma_inv);

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_PROCESS_CALCULATION_REQUESTS_KERNEL);
}

template <typename LocalEdges, typename DistantEdges>
__global__ void find_synapses_to_delete_random_kernel(
    DeletionRequestHandle deletion_request, NeuronsExtraInfoGPUHandleConst info_handle,
    LocalEdges local_edges, DistantEdges distant_edges,
    const ElementType my_element_type, const SignalType dendritic_signal_type, const std::uint32_t random_key, std::uint32_t* deleted_indices, bool* request_excitatory) {
    const auto* const to_delete = deletion_request.to_delete;
    const auto* const request_partition = deletion_request.request_partition;
    auto* const request_rank = deletion_request.request_rank;
    auto* const request_neuron_id = deletion_request.request_neuron_id;
    const auto neuron_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (neuron_id >= info_handle.number_neurons)
        return;

    auto count_local = 0U;
    auto count_distant = 0U;

    {
        auto it = local_edges.edge_begin(neuron_id);
        const auto sentinel = local_edges.edge_end(neuron_id);
        while (!(it == sentinel)) {
            const auto [other_neuron_id, other_rank, weight] = it.next();
            if (weight == 0) {
                continue;
            }
            const auto excitatory = weight > 0;
            if constexpr (is_memory_pool_view_v<LocalEdges>) {
                const auto* ranks_ptr = local_edges.other_ranks;
                RELEARN_DEVICE_CUDA_CHECK(other_rank == info_handle.my_rank, "find_synapses_to_delete_random_kernel: Local edges are not local %u:%u -> %u:%u %p", info_handle.my_rank, neuron_id, other_rank, neuron_id, ranks_ptr);
            }

            if (my_element_type == ElementType::Dendrite && (excitatory && dendritic_signal_type == SignalType::Inhibitory || !excitatory && dendritic_signal_type == SignalType::Excitatory)) {
                continue;
            }

            count_local += std::abs(weight);
        }
    }

    // Copy and paste the code from above for distant counting
    {
        auto it = distant_edges.edge_begin(neuron_id);
        const auto sentinel = distant_edges.edge_end(neuron_id);
        while (!(it == sentinel)) {
            const auto [other_neuron_id, other_rank, weight] = it.next();
            if (weight == 0) {
                continue;
            }
            const auto excitatory = weight > 0;
            if (my_element_type == ElementType::Dendrite && (excitatory && dendritic_signal_type == SignalType::Inhibitory || !excitatory && dendritic_signal_type == SignalType::Excitatory)) {
                continue;
            }
            count_distant += std::abs(weight);
        }
    }

    const auto number_to_delete = to_delete[neuron_id];
    const auto request_offset = request_partition[neuron_id];
    RELEARN_DEVICE_CUDA_CHECK(number_to_delete <= count_local + count_distant,
                              "find_synapses_to_delete_random_kernel: More deletions %u than connected elements %u for neuron id %u dend %d exc %d",
                              number_to_delete, count_local + count_distant, neuron_id, my_element_type == ElementType::Dendrite, dendritic_signal_type == SignalType::Excitatory);

    for (auto j = 0U; j < number_to_delete; j++) {
        const auto number_connections = count_local + count_distant;
        std::uint32_t delete_idx{};
        bool already_deleted = true;

        while (already_deleted) {
            delete_idx = RandomNumbers::get_random_value_int(
                number_connections,
                neuron_id,
                random_key);

            already_deleted = false;

            for (auto k = 0U; k < j; k++) {
                if (deleted_indices[request_offset + k] == delete_idx) {
                    already_deleted = true;
                    break;
                }
            }
        }

        auto found = false;
        auto counter = 0U;

        if (delete_idx < count_local) {
            auto it = local_edges.edge_begin(neuron_id);
            const auto sentinel = local_edges.edge_end(neuron_id);
            while (!(it == sentinel)) {
                const auto [other_neuron_id, other_rank, weight] = it.next();
                const auto excitatory = weight > 0;
                if (my_element_type == ElementType::Dendrite && (excitatory && dendritic_signal_type == SignalType::Inhibitory || !excitatory && dendritic_signal_type == SignalType::Excitatory)) {
                    continue;
                }
                if (weight == 0) {
                    continue;
                }
                if (weight == 0) {
                    continue;
                }
                counter += std::abs(weight);

                if (counter > delete_idx) {
                    const auto off = request_offset + j;
                    request_rank[off] = other_rank;
                    request_neuron_id[off] = other_neuron_id;
                    request_excitatory[off] = weight > 0;
                    found = true;
                    deleted_indices[request_offset + j] = delete_idx;
                    break;
                }
            }
            RELEARN_DEVICE_CUDA_CHECK(found,
                                      "find_synapses_to_delete_random_kernel: Could not find a synapse to delete for neuron id %u: // %u %u %u",
                                      neuron_id, delete_idx, count_local, number_connections);
        }
        // Copy and past from above for distant connections
        else {
            counter = count_local;
            auto it = distant_edges.edge_begin(neuron_id);
            const auto sentinel = distant_edges.edge_end(neuron_id);
            while (!(it == sentinel)) {
                // const auto edge_idx = it.current_global_offset;
                const auto [other_neuron_id, other_rank, weight] = it.next();
                if (weight == 0) {
                    continue;
                }
                const auto excitatory = weight > 0;
                if (my_element_type == ElementType::Dendrite && (excitatory && dendritic_signal_type == SignalType::Inhibitory || !excitatory && dendritic_signal_type == SignalType::Excitatory)) {
                    continue;
                }
                counter += std::abs(weight);

                if (counter > delete_idx) {
                    const auto off = request_offset + j;
                    request_rank[off] = other_rank;
                    request_neuron_id[off] = other_neuron_id;
                    request_excitatory[off] = weight > 0;
                    found = true;
                    deleted_indices[request_offset + j] = delete_idx;

                    break;
                }
            }
            RELEARN_DEVICE_CUDA_CHECK(found,
                                      "find_synapses_to_delete_random_kernel: Could not find a synapse to delete for neuron id %u: // %u %u %u",
                                      neuron_id, delete_idx, count_distant, number_connections);
        }
    }

    // Do the deletion
    for (auto j = 0U; j < number_to_delete; j++) {
        const auto off = request_offset + j;
        const auto other_rank = request_rank[off];
        const auto other_neuron_id = request_neuron_id[off];
        std::uint32_t delete_idx = deleted_indices[off];

        if (delete_idx < count_local) {
            local_edges.remove_synapse(neuron_id, other_rank, other_neuron_id,
                                       request_excitatory[off]);
        } else {
            distant_edges.remove_synapse(neuron_id, other_rank, other_neuron_id,
                                         request_excitatory[off]);
        }
    }
}

// Warp-parallel helpers for OnlyOutgoingView incoming-edge deletion (sm3_lane_weight_sum,
// sm3_block_of/sm3_block_start, sm3_distant_sweep, find_delete_locate_sorted_warp_kernel,
// find_delete_locate_bitmap_kernel), the 3-kernel deletion pipeline (find_delete_count_kernel,
// find_delete_draw_kernel, find_delete_apply_kernel, find_delete_apply_by_rank_kernel,
// find_synapses_delete_3kernels), and the OnlyOutgoingView/outgoing-storage consistency checker
// (check_only_outgoing_view) now live in OnlyOutgoingDeletion.h/.cu.

template <typename IncomingNetwork, typename OutgoingNetwork>
void find_synapses_to_delete_random_internal_entry(DeletionRequestHandle deletion_request,
                                                   NeuronsExtraInfoGPUHandleConst info_handle,
                                                   IncomingNetwork& incoming_edges_handle, OutgoingNetwork& outgoing_edges_handle,
                                                   const ElementType my_element_type, const SignalType my_signal_type, const std::uint32_t random_key, const std::uint32_t sum_to_delete) {
    const auto number_neurons = info_handle.number_neurons;
    Timers::start(TimerRegion::CUDA_FIND_SYNAPSES_TO_DELETE_KERNEL);

    DeviceArray<std::uint32_t> deleted_indices(sum_to_delete);
    DeviceArray<bool> deleted_exc(sum_to_delete);

    const auto& [blocks, threads] = get_grid_ands_block_size(number_neurons,
                                                             find_synapses_to_delete_random_kernel<IncomingNetwork, OutgoingNetwork>);

    find_synapses_to_delete_random_kernel<IncomingNetwork, OutgoingNetwork><<<blocks, threads>>>(
        deletion_request, info_handle,
        incoming_edges_handle, outgoing_edges_handle, my_element_type,
        my_signal_type, random_key, deleted_indices.device_ptr(), deleted_exc.device_ptr());

    cudaDeviceSynchronize();

    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_FIND_SYNAPSES_TO_DELETE_KERNEL);
}

template <typename IncomingLocalNet, typename IncomingDistantNet, typename OutgoingLocalNet, typename OutgoingDistantNet>
void commit_deletions_launcher(NeuronsExtraInfoGPUHandleConst info_handle,
                               SynapticElementsBaseCudaHandle axon_handle,
                               SynapticElementsBaseCudaHandle den_exc_handle,
                               SynapticElementsBaseCudaHandle den_inh_handle,
                               IncomingLocalNet incoming_local_edges_handle, IncomingDistantNet incoming_distant_edges_handle,
                               OutgoingLocalNet outgoing_local_edges_handle, OutgoingDistantNet outgoing_distant_edges_handle,
                               DeletionCommitHandle deletion_commit) {
    Timers::start(TimerRegion::CUDA_COMMIT_DELETIONS_KERNEL);
    const auto& [blocks, threads] = get_grid_ands_block_size(info_handle.number_neurons,
                                                             commit_deletions_kernel<IncomingLocalNet, IncomingDistantNet, OutgoingLocalNet, OutgoingDistantNet>);

    commit_deletions_kernel<IncomingLocalNet, IncomingDistantNet, OutgoingLocalNet, OutgoingDistantNet>
        <<<blocks, threads>>>(info_handle,
                              axon_handle, den_exc_handle, den_inh_handle,
                              incoming_local_edges_handle, incoming_distant_edges_handle, outgoing_local_edges_handle, outgoing_distant_edges_handle,
                              deletion_commit);

    cudaDeviceSynchronize();

    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_COMMIT_DELETIONS_KERNEL);
}

template <typename IdT>
void find_synapses_to_delete_random_entry_impl(DeletionRequestHandle deletion_request,
                                               NeuronsExtraInfoGPUHandleConst info_handle,
                                               NetworkHandle network_view,
                                               const ElementType my_element_type, const SignalType my_signal_type, const std::uint32_t random_key, std::uint32_t sum_to_delete) {
    if (my_element_type == ElementType::Axon) {
        if (network_view.outgoing_local_handle.layout_type == LayoutType::MemoryPool) {
            RelearnException::check(network_view.outgoing_distant_handle.layout_type == LayoutType::MemoryPool, "Invalid layout combinations0");
            find_synapses_to_delete_random_internal_entry<MemoryPoolView<IdT>, MemoryPoolView<IdT>>(
                deletion_request, info_handle,
                *static_cast<MemoryPoolView<IdT>*>(network_view.outgoing_local_handle.view_impl),
                *static_cast<MemoryPoolView<IdT>*>(network_view.outgoing_distant_handle.view_impl),
                my_element_type, my_signal_type, random_key, sum_to_delete);
        } else {
            RelearnException::fail("Invalid layout combinations2");
        }
    } else {
        // Incoming edges
        if (network_view.incoming_local_handle.layout_type == LayoutType::MemoryPool) {
            RelearnException::check(network_view.incoming_distant_handle.layout_type == LayoutType::MemoryPool, "Invalid layout combinations4");
            find_synapses_to_delete_random_internal_entry<MemoryPoolView<IdT>, MemoryPoolView<IdT>>(
                deletion_request, info_handle,
                *static_cast<MemoryPoolView<IdT>*>(network_view.incoming_local_handle.view_impl),
                *static_cast<MemoryPoolView<IdT>*>(network_view.incoming_distant_handle.view_impl),
                my_element_type, my_signal_type, random_key, sum_to_delete);
        } else {
            RelearnException::fail("Invalid layout combinations3");
        }
    }
}

void find_synapses_to_delete_random_entry(DeletionRequestHandle deletion_request,
                                          NeuronsExtraInfoGPUHandleConst info_handle,
                                          NetworkHandle network_view,
                                          const ElementType my_element_type, const SignalType my_signal_type, const std::uint32_t random_key, std::uint32_t sum_to_delete) {
    const bool wide_ids = my_element_type == ElementType::Axon
                              ? network_view.outgoing_local_handle.wide_ids
                              : network_view.incoming_local_handle.wide_ids;
    if (wide_ids) {
        find_synapses_to_delete_random_entry_impl<std::uint32_t>(deletion_request, info_handle, network_view,
                                                                 my_element_type, my_signal_type, random_key, sum_to_delete);
    } else {
        find_synapses_to_delete_random_entry_impl<SmallNeuronIdType>(deletion_request, info_handle, network_view,
                                                                     my_element_type, my_signal_type, random_key, sum_to_delete);
    }
}

template <typename IdT>
void commit_deletions_entry_impl(NeuronsExtraInfoGPUHandleConst info_handle,
                                 SynapticElementsBaseCudaHandle axon_handle,
                                 SynapticElementsBaseCudaHandle den_exc_handle,
                                 SynapticElementsBaseCudaHandle den_inh_handle,
                                 NetworkHandle network_view,
                                 DeletionCommitHandle deletion_commit) {

    if (network_view.incoming_local_handle.layout_type == LayoutType::MemoryPool) {
        RelearnException::check(network_view.incoming_distant_handle.layout_type == LayoutType::MemoryPool, "Invalid layout combinations4");
        RelearnException::check(network_view.outgoing_local_handle.layout_type == LayoutType::MemoryPool, "Invalid layout combinations4");
        RelearnException::check(network_view.outgoing_distant_handle.layout_type == LayoutType::MemoryPool, "Invalid layout combinations4");
        commit_deletions_launcher<MemoryPoolView<IdT>, MemoryPoolView<IdT>, MemoryPoolView<IdT>, MemoryPoolView<IdT>>(
            info_handle,
            axon_handle, den_exc_handle, den_inh_handle,
            *static_cast<MemoryPoolView<IdT>*>(network_view.incoming_local_handle.view_impl),
            *static_cast<MemoryPoolView<IdT>*>(network_view.incoming_distant_handle.view_impl),
            *static_cast<MemoryPoolView<IdT>*>(network_view.outgoing_local_handle.view_impl),
            *static_cast<MemoryPoolView<IdT>*>(network_view.outgoing_distant_handle.view_impl),
            deletion_commit);
    } else {
        RelearnException::fail("Invalid layout combinations3");
    }
}

void commit_deletions_entry(NeuronsExtraInfoGPUHandleConst info_handle,
                            SynapticElementsBaseCudaHandle axon_handle,
                            SynapticElementsBaseCudaHandle den_exc_handle,
                            SynapticElementsBaseCudaHandle den_inh_handle,
                            NetworkHandle network_view,
                            DeletionCommitHandle deletion_commit) {
    if (network_view.incoming_local_handle.wide_ids) {
        commit_deletions_entry_impl<std::uint32_t>(info_handle, axon_handle, den_exc_handle, den_inh_handle, network_view,
                                                   deletion_commit);
    } else {
        commit_deletions_entry_impl<SmallNeuronIdType>(info_handle, axon_handle, den_exc_handle, den_inh_handle, network_view,
                                                       deletion_commit);
    }
}

template <typename IdT>
std::size_t
process_requests_entry_aware_impl(const SynapseCreationRequestHandle request,
                                  SynapseCreationResponse* responses,
                                  SynapticElementsBaseCudaHandle den_exc_handle,
                                  SynapticElementsBaseCudaHandle den_inh_handle,
                                  NeuronsExtraInfoGPUHandleConst info_handle,
                                  NetworkHandle network_view, const std::uint64_t seed, const std::uint64_t step) {
    if (network_view.incoming_local_handle.layout_type == LayoutType::MemoryPool) {
        RelearnException::check(network_view.incoming_distant_handle.layout_type == LayoutType::MemoryPool, "Invalid layout combinations4");
        return process_requests_entry_aware<MemoryPoolView<IdT>, MemoryPoolView<IdT>>(request, responses, den_exc_handle, den_inh_handle, info_handle,
                                                                                      *static_cast<MemoryPoolView<IdT>*>(network_view.incoming_local_handle.view_impl),
                                                                                      *static_cast<MemoryPoolView<IdT>*>(network_view.incoming_distant_handle.view_impl), seed, step);
    } else {
        RelearnException::fail("Invalid layout combinations3");
    }
}

std::size_t
process_requests_entry_aware(const SynapseCreationRequestHandle request,
                             SynapseCreationResponse* responses,
                             SynapticElementsBaseCudaHandle den_exc_handle,
                             SynapticElementsBaseCudaHandle den_inh_handle,
                             NeuronsExtraInfoGPUHandleConst info_handle,
                             NetworkHandle network_view, const std::uint64_t seed, const std::uint64_t step) {
    // Each request triggers at most one add_synapse call, and add_synapse touches at most 3
    // DynamicVecVec arrays (neuron-ids, ranks, weights-or-excitatory -- see
    // MemoryPoolView::add_synapse), each consuming at most one new block. This is a hard upper
    // bound on this launch's demand, so the pool can never be exhausted mid-kernel.
    network_view.shared_overflow_pool->ensure_capacity(3U * request.number_requests);
    if (network_view.incoming_local_handle.wide_ids) {
        return process_requests_entry_aware_impl<std::uint32_t>(request,
                                                                responses, den_exc_handle, den_inh_handle, info_handle, network_view, seed, step);
    }
    return process_requests_entry_aware_impl<SmallNeuronIdType>(request,
                                                                responses, den_exc_handle, den_inh_handle, info_handle, network_view, seed, step);
}

template <typename IdT>
void process_responses_entry_aware_impl(const int* mpi_sizes, const SynapseCreationResponseHandle request,
                                        SynapseCreationResponse* responses,
                                        SynapticElementsBaseCudaHandle axon_handle,
                                        NeuronsExtraInfoGPUHandleConst info_handle,
                                        NetworkHandle network_view, CudaConfig::mpi_rank_type* target_ranks) {
    if (network_view.outgoing_local_handle.layout_type == LayoutType::MemoryPool) {
        RelearnException::check(network_view.outgoing_distant_handle.layout_type == LayoutType::MemoryPool, "Invalid layout combinations4");
        process_responses_entry_aware<MemoryPoolView<IdT>, MemoryPoolView<IdT>>(mpi_sizes, request, responses, axon_handle, info_handle,
                                                                                *static_cast<MemoryPoolView<IdT>*>(network_view.outgoing_local_handle.view_impl),
                                                                                *static_cast<MemoryPoolView<IdT>*>(network_view.outgoing_distant_handle.view_impl),
                                                                                target_ranks);
    } else {
        RelearnException::fail("Invalid layout combinations3");
    }
}

void process_responses_entry_aware(const int* mpi_sizes, const SynapseCreationResponseHandle request,
                                   SynapseCreationResponse* responses,
                                   SynapticElementsBaseCudaHandle axon_handle,
                                   NeuronsExtraInfoGPUHandleConst info_handle,
                                   NetworkHandle network_view, CudaConfig::mpi_rank_type* target_ranks) {
    // See the matching comment in process_requests_entry_aware: at most 3 new blocks per response.
    network_view.shared_overflow_pool->ensure_capacity(3U * request.number_requests);
    if (network_view.outgoing_local_handle.wide_ids) {
        process_responses_entry_aware_impl<std::uint32_t>(mpi_sizes, request, responses,
                                                          axon_handle, info_handle, network_view, target_ranks);
    } else {
        process_responses_entry_aware_impl<SmallNeuronIdType>(mpi_sizes, request, responses,
                                                              axon_handle, info_handle, network_view, target_ranks);
    }
}

std::uint32_t compute_sum_and_partition(const CudaConfig::synaptic_count_type* d_to_delete,
                                        CudaConfig::synaptic_count_type* d_partition,
                                        std::size_t n) {
    const auto sum = static_cast<std::uint32_t>(thrust::reduce(
        thrust::device_pointer_cast(d_to_delete),
        thrust::device_pointer_cast(d_to_delete + n),
        CudaConfig::synaptic_count_type{ 0 }));
    cudaMemset(d_partition, 0, sizeof(CudaConfig::synaptic_count_type));
    thrust::inclusive_scan(
        thrust::device_pointer_cast(d_to_delete),
        thrust::device_pointer_cast(d_to_delete + n),
        thrust::device_pointer_cast(d_partition + 1));
    return sum;
}
