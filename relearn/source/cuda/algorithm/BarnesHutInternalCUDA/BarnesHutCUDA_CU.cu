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

#include <cuda_runtime.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "BarnesHutCUDA_CU.cuh"
#include "BarnesHutCUDA_CU.h"

#include "cuda/CudaBaseBridgeFunctions.h"
#include "cuda/CudaTypes.h"
#include "cuda/algorithm/Kernel.cuh"
#include "cuda/mpi/MPICuda.h"
#include "cuda/random/RandomNumber.cuh"
#include "cuda/util/Util.cuh"
#include "cuda/wrapper/StreamWrapper.cuh"
#include "util/Timers.h"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cuda/iterator>

#include <numeric>
#include <stdexcept>

namespace BarnesHutCUDA_CU {
__device__ bool is_leaf(const SimpleVec3d& position, const CudaConfig::bh_index_type child_index) {
    const auto norm = position.x + position.y + position.z;

    // a given neuron is a leaf if 1-norm of position != 0 and child_index == 0
    const auto ret_val = norm != 0 && child_index == 0;
    return ret_val;
}

__device__ bool
is_parent(const SimpleVec3d& position, const CudaConfig::bh_index_type child_index) {
    const auto norm = position.x + position.y + position.z;

    // a given neuron is a parent if 1-norm of position != 0 and child_index != 0
    const auto ret_val = norm != 0 && child_index != 0;
    return ret_val;
}

__device__ bool test_acceptance_criterion(const SimpleVec3d& source_position,
                                          const SimpleVec3d& target_position,
                                          const CudaConfig::synaptic_count_type vacant_dendritic_elements,
                                          const CudaConfig::gaussian_type subdomain_length,
                                          const CudaConfig::gaussian_type acceptance_criterion, const bool is_leaf) {

    // discard neurons with no vacant dendrites
    if (vacant_dendritic_elements == 0) {
        return false;
    }

    const auto distance = calculate_2_norm(target_position, source_position);

    // prevent autapse
    if (distance == 0.0) {
        return false;
    }

    // Always accept a leaf node
    if (is_leaf) {
        return true;
    }

    // Original Barnes-Hut acceptance criterion
    // const auto ret_val = (length / distance) < acceptance_criterion;
    const auto ret_val = subdomain_length < (acceptance_criterion * distance);

    return ret_val;
}

__device__ CudaConfig::bh_index_type
find_single_target_neuron(const std::uint32_t thread_id, const std::uint32_t number_threads, const std::uint64_t seed, const std::uint64_t step,
                          const CudaConfig::bh_index_type start_index,
                          const SimpleVec3d& source_position,
                          const NeuronPopulationDeviceHandle population,
                          const LinearizedTreeDeviceHandle tree,
                          const CudaConfig::gaussian_type acceptance_criterion,
                          const CudaConfig::mpi_rank_type* const neuron_ranks,
                          const CudaConfig::mpi_rank_type my_rank,
                          const CudaConfig::gaussian_type squared_sigma_inv) {
    const auto* const neuron_positions = population.positions;
    const auto* const child_index = tree.child_index;
    const auto* const parent_index = tree.parent_index;
    const auto* const subdomain_length = tree.subdomain_length;
    const auto* const vacant_dendritic_elements = population.vacant_dendrites;
    const auto* const node_types = tree.node_types;
    const auto linear_tree_size = tree.tree_size;

    auto neurons_to_consider_start_index = start_index;

    for (auto it = 0U; true; it++) {
        auto total_sum_probabilities = 0.0;

        // 2 passes
        // 1st pass calculate threshold
        /**
         * ctr ensures the loop has iterated 8 times even if i reaches neurons_to_consider_start_index + 8 early
         * e. g. when neurons_to_consider_start_index = 0 is given, ctr ensures that when the children of any i in [0..7] links start at i = 8, the loop still runs at least 8 times
         *
         */

        for (auto i = neurons_to_consider_start_index, ctr = 0U;
             i != neurons_to_consider_start_index + NUMBER_OCT || ctr < NUMBER_OCT; i++, ctr++) {
            RELEARN_DEVICE_CUDA_CHECK(i < linear_tree_size,
                                      "find_single_target_neuron: Index larger than size of linear tree");
            const auto target_subdomain_length = subdomain_length[i];
            const auto target_child_index = child_index[i];
            const auto target_free_count = vacant_dendritic_elements[i];
            const auto target_position = neuron_positions[i];
            const auto target_is_leaf = is_leaf(target_position, target_child_index);
            const auto target_is_parent = is_parent(target_position, target_child_index);
            const auto node_type = node_types[i];
            const auto target_rank = neuron_ranks[i];

            const auto ac = test_acceptance_criterion(source_position, target_position, target_free_count,
                                                      target_subdomain_length, acceptance_criterion,
                                                      target_is_leaf);

            /**
             * test ac ->
             *   false if dendriten = 0
             *   false if Abstand = 0
             *   true if leaf
             *   true if (length / distance) < acceptance_criterion -> else approximate
             */
            // TODO(NIT) test AC fails in tests -> virtual neuron not unfolded, not all viable target neurons are considered
            if (node_type != NodeType::Placeholder && (ac || target_rank != my_rank)) {
                RELEARN_DEVICE_CUDA_CHECK(target_rank != std::numeric_limits<CudaConfig::mpi_rank_type>::max(),
                                          "find_single_target_neuron: target_rank %d is invalid", target_rank);
                const auto prob = calculate_attractiveness_to_connect(source_position, target_position,
                                                                      target_free_count,
                                                                      false, squared_sigma_inv);
                total_sum_probabilities += prob;

            } else {
                if (target_is_parent) {
                    // jump to child -> we increment i in the next iteration, therefore jump to child index - 1
                    i = target_child_index - 1ULL;
                    // the for-loop's increment clause will also do ctr++ on this continue, but a jump into
                    // a child is not a top-level sibling advance -> cancel it out, otherwise ctr can reach
                    // NUMBER_OCT in lockstep with i reaching the group boundary (happens exactly when the
                    // locally-unfolded sibling is the LAST one in the group, e.g. rank == number_ranks - 1),
                    // causing the loop to exit before the child is ever visited.
                    ctr--;
                    continue;
                }
            }

            while (i % NUMBER_OCT == 7 && i != neurons_to_consider_start_index + 7) {
                // jump back to parent
                const auto pi = parent_index[i - 7];
                ;
                i = pi;
            }
        }

        // only autapses are possible
        if (total_sum_probabilities == 0.0) {
            return std::numeric_limits<CudaConfig::bh_index_type>::max();
        }

        const auto random_number = RandomNumbers::get_stateless_random_number(number_threads * it + thread_id, seed, step);
        const auto threshold = random_number * total_sum_probabilities;

        // 2nd pass, choose neuron
        auto picked_index = 0;
        auto sum_probabilities = 0.0;

        /**
         * ctr ensures the loop has iterated 8 times even if i reaches neurons_to_consider_start_index + 8 early
         * e. g. when neurons_to_consider_start_index = 0 is given, ctr ensures that when the children of any i in [0..7] links start at i = 8, the loop still runs at least 8 times
         */
        for (auto i = neurons_to_consider_start_index, ctr = 0U;
             i != neurons_to_consider_start_index + NUMBER_OCT || ctr < NUMBER_OCT; i++, ctr++) {
            RELEARN_DEVICE_CUDA_CHECK(i < linear_tree_size,
                                      "find_single_target_neuron: Index larger than size of linear tree");
            const auto target_subdomain_length = subdomain_length[i];
            const auto target_child_index = child_index[i];
            const auto target_free_count = vacant_dendritic_elements[i];
            const auto target_position = neuron_positions[i];
            const auto target_is_leaf = is_leaf(target_position, target_child_index);
            const auto target_is_parent = is_parent(target_position, target_child_index);
            const auto node_type = node_types[i];
            const auto target_rank = neuron_ranks[i];

            const auto ac = test_acceptance_criterion(source_position, target_position, target_free_count,
                                                      target_subdomain_length, acceptance_criterion,
                                                      target_is_leaf);

            if (node_type != NodeType::Placeholder && (ac || target_rank != my_rank)) {
                RELEARN_DEVICE_CUDA_CHECK(target_rank != std::numeric_limits<CudaConfig::mpi_rank_type>::max(),
                                          "find_single_target_neuron: target_rank %d is invalid", target_rank);
                const auto prob = calculate_attractiveness_to_connect(source_position, target_position,
                                                                      target_free_count,
                                                                      false, squared_sigma_inv);
                sum_probabilities += prob;

                // update picked_index if prob > 0
                if (prob > 0.0) {
                    picked_index = i;
                }
                /*
                 * return when the sum first exceeds the threshold
                 *
                 * therefore the sum before the picked element is smaller than the random number (threshold)
                 * and the sum before the picked element + the picked element is larger or equal to the random numer (threshold)
                 */
                if (sum_probabilities >= threshold) {
                    break;
                }
            } else {
                if (target_is_parent) {
                    // jump to child -> we increment i in the next iteration, therefore jump to child index - 1
                    i = target_child_index - 1ULL;
                    // see matching comment in the first pass: cancel the implicit ctr++ for this continue.
                    ctr--;
                    continue;
                }
            }

            while (i % NUMBER_OCT == 7 && i != neurons_to_consider_start_index + 7) {
                // jump back to parent
                const auto pi = parent_index[i - 7];
                ;
                i = pi;
            }
        }

        // found_target_dendrites[target_idx] = picked_index;

        const auto target_rank = neuron_ranks[picked_index];
        if (is_leaf(neuron_positions[picked_index], child_index[picked_index]) || (target_rank != std::numeric_limits<CudaConfig::mpi_rank_type>::max() && target_rank != my_rank)) {
            return picked_index;
        }
        neurons_to_consider_start_index = child_index[picked_index];
    }
}

__global__ void update_leaf_nodes_kernel(const LinearizedTreeDeviceHandle tree, const SignalType* signal_types,
                                         SynapticElementsBaseCudaHandleConst axon_handle, SynapticElementsBaseCudaHandleConst den_exc_handle, SynapticElementsBaseCudaHandleConst den_inh_handle,
                                         const TreeVacancyOutputHandle output) {
    const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= tree.tree_size) {
        return;
    }

    const auto node_type = tree.node_types[thread_id];

    if (node_type == NodeType::Leaf) {
        const auto neuron_id = tree.neuron_ids[thread_id];

        const auto signal_type = signal_types[neuron_id];
        const auto num_vacant_axons = axon_handle.vacant_elements[neuron_id];

        // sum_vacant_axons += num_vacant_axons;

        output.exc_vacant_dend[thread_id] = den_exc_handle.vacant_elements[neuron_id];
        output.inh_vacant_dend[thread_id] = den_inh_handle.vacant_elements[neuron_id];
        if (signal_type == SignalType::Excitatory) { // should be the same as child->has_excitatory_dendrite()?
            output.exc_vacant_axon[thread_id] = num_vacant_axons;
            output.inh_vacant_axon[thread_id] = 0;
        } else {
            output.exc_vacant_axon[thread_id] = 0;
            output.inh_vacant_axon[thread_id] = num_vacant_axons;
        }
    } else if (node_type == NodeType::RemoteNode) {
        // Remote nodes are updated separately
        return;
    } else {
        // ++start_index_count;
    }
}

__host__ void update_leaf_nodes_entry(const LinearizedTreeDeviceHandle tree, const SignalType* signal_types,
                                      SynapticElementsBaseCudaHandleConst axon_handle, SynapticElementsBaseCudaHandleConst den_exc_handle, SynapticElementsBaseCudaHandleConst den_inh_handle,
                                      const TreeVacancyOutputHandle output, const std::shared_ptr<StreamWrapper>& stream) {

    const auto& [blocks, threads] = get_grid_ands_block_size(tree.tree_size, update_leaf_nodes_kernel);

    // Launched on `stream` without a following sync here -- the caller (BarnesHutCUDA::
    // prepare_update_connectivity) syncs and resolves GPU timers right after calling this.
    auto* gpu_timer = cuda_start_gpu_timer(TimerRegion::CUDA_UPDATE_LEAF_NODES_KERNEL, *stream);
    update_leaf_nodes_kernel<<<blocks, threads, 0, get_cuda_stream_from_wrapper(*stream)>>>(tree, signal_types,
                                                                                            axon_handle, den_exc_handle, den_inh_handle,
                                                                                            output);
    cuda_stop_gpu_timer(gpu_timer, *stream);
}

__global__ void calculate_updated_octree_dev(NeuronPopulationDeviceHandle population,
                                             const LinearizedTreeDeviceHandle tree,
                                             const CudaConfig::bh_index_type index_max,
                                             const CudaConfig::bh_index_type index_min) {
    auto* const neuron_positions = population.positions;
    const auto* const child_index = tree.child_index;
    auto* const vacant_dendritic_elements = population.vacant_dendrites;
    const auto neurons_count = tree.tree_size;

    const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    const auto index = index_min + thread_id;

    if (index >= index_max) {
        return;
    }

    RELEARN_DEVICE_CUDA_CHECK(index < neurons_count, "calculate_updated_octree_dev: Index %u is too high %u %u %u", index, neurons_count, index_min, index_max);

    // threadID in [min, max)
    const auto child_index_interval_begin = child_index[index];
    // if neuron is leaf or placeholder, return
    if (child_index_interval_begin == 0) {
        return;
    }

    auto weighted_x = 0.0;
    auto weighted_y = 0.0;
    auto weighted_z = 0.0;
    auto sum_dendritic_elements = 0ULL;
#pragma unroll
    for (auto i = 0U; i < 8U; ++i) {
        const auto child_i_index = child_index_interval_begin + i;
        const auto child_vacant_dendritic_elements = vacant_dendritic_elements[child_i_index];
        const auto [x, y, z] = neuron_positions[child_i_index];

        sum_dendritic_elements += child_vacant_dendritic_elements;
        weighted_x += x * child_vacant_dendritic_elements;
        weighted_y += y * child_vacant_dendritic_elements;
        weighted_z += z * child_vacant_dendritic_elements;
    }

    // if there are no vacant dendrites, the parent neuron has 0 attractiveness to connect
    if (sum_dendritic_elements == 0ULL) {
        neuron_positions[index] = SimpleVec3d{ 0, 0, 0 };
        vacant_dendritic_elements[index] = 0;
        return;
    }

    // calculate weighted x y z
    const auto mean_x = weighted_x / sum_dendritic_elements;
    const auto mean_y = weighted_y / sum_dendritic_elements;
    const auto mean_z = weighted_z / sum_dendritic_elements;

    neuron_positions[index] = SimpleVec3d{ mean_x, mean_y, mean_z };
    vacant_dendritic_elements[index] = sum_dendritic_elements;
}

__global__ void find_target_neurons_kernel(const std::uint64_t seed, const std::uint32_t number_threads, const std::uint64_t step,
                                           const NeuronPopulationDeviceHandle population,
                                           const CudaConfig::number_neurons_type neurons_count,
                                           const LinearizedTreeDeviceHandle tree,
                                           const CudaConfig::bh_index_type* node_id_vacant_axons_mapping,
                                           const CudaConfig::gaussian_type acceptance_criterion,
                                           const CudaConfig::number_neurons_type target_size,
                                           const CudaConfig::mpi_rank_type* const neuron_ranks,
                                           const CudaConfig::mpi_rank_type my_rank,
                                           CudaConfig::gaussian_type squared_sigma_inv,
                                           CudaConfig::mpi_rank_type** const out_target_ranks,
                                           CudaConfig::number_neurons_type** const out_target_ids,
                                           CudaConfig::number_neurons_type** const out_source_ids,
                                           SimpleVec3d** const out_source_positions,
                                           const CudaConfig::number_neurons_type number_neurons,
                                           int* const neuron_on_rank_counter,
                                           const int max_number_per_rank) {
    const auto* const neuron_positions = population.positions;
    const auto* const neuron_ids = tree.neuron_ids;
    const auto linear_tree_size = tree.tree_size;

    const auto thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id >= target_size) {
        return;
    }
    const auto source_index = node_id_vacant_axons_mapping[thread_id];
    const auto source_neuron = neuron_ids[source_index];
    RELEARN_DEVICE_CUDA_CHECK(source_neuron < neurons_count, "find_target_neurons_kernel: Source neuron id %u invalid %u", source_neuron, neurons_count);
    const auto source_position = neuron_positions[source_index];

    // iterate over the vacant axonal elements of this neuron
    const auto picked_index = find_single_target_neuron(thread_id, number_threads, seed, step, 0ULL, source_position,
                                                        population, tree, acceptance_criterion,
                                                        neuron_ranks, my_rank,
                                                        squared_sigma_inv);

    if (picked_index != std::numeric_limits<CudaConfig::bh_index_type>::max()) {

        RELEARN_DEVICE_CUDA_CHECK(picked_index < linear_tree_size,
                                  "find_target_neurons_kernel: Invalid target index %u (target_size=%u)", picked_index, target_size);
        const auto target_node = neuron_ids[picked_index];
        const auto target_rank = neuron_ranks[picked_index];

        const auto idx = atomicAdd(&neuron_on_rank_counter[target_rank], 1);
        RELEARN_DEVICE_CUDA_CHECK(idx < max_number_per_rank, "find_target_neurons_kernel: The target vector is already full");

        out_target_ids[target_rank][idx] = target_node;
        RELEARN_DEVICE_CUDA_CHECK(target_node < number_neurons || my_rank != target_rank, "find_target_neurons_kernel: Invalid neuron id %u", target_node);
        out_target_ranks[target_rank][idx] = target_rank;
        out_source_ids[target_rank][idx] = source_neuron;
        out_source_positions[target_rank][idx] = source_position;
    }
}

__global__ void test_acceptance_criterion_dev(bool* test_results, const CudaConfig::number_neurons_type test_count,
                                              const SimpleVec3d source_position,
                                              const SimpleVec3d target_position,
                                              const CudaConfig::synaptic_count_type vacant_dendritic_elements,
                                              const CudaConfig::gaussian_type subdomain_length,
                                              const CudaConfig::gaussian_type acceptance_criterion,
                                              const bool is_leaf) {
    const auto thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_id >= test_count) {
        return;
    }
    const auto result = test_acceptance_criterion(source_position, target_position, vacant_dendritic_elements,
                                                  subdomain_length, acceptance_criterion, is_leaf);

    test_results[thread_id] = result;
}
} // namespace BarnesHutCUDA_CU

LinearizedTreeInitResult
BarnesHutCUDA_CU::init_neurons(
    const std::span<const CudaConfig::bh_index_type> h_child_begin_index,
    const std::span<const CudaConfig::bh_index_type> h_parent_index,
    const std::span<const CudaConfig::gaussian_type> h_subdomain_length, const std::span<const CudaConfig::number_neurons_type> neuron_ids,
    const std::span<const CudaConfig::bh_index_type> rma_offset_to_index, const std::span<const NodeType> h_node_types) {
    Timers::start(TimerRegion::CUDA_BH_INIT_NEURONS);

    auto storage = LinearizedTreeDeviceStorage{};
    storage.child_begin_index.emplace(h_child_begin_index);
    storage.parent_index.emplace(h_parent_index);
    storage.subdomain_length.emplace(h_subdomain_length);
    storage.neuron_ids.emplace(neuron_ids);
    storage.rma_offset_to_index.emplace(rma_offset_to_index);
    storage.node_types.emplace(h_node_types);

    const auto total_mem_usage = h_child_begin_index.size() * sizeof(CudaConfig::bh_index_type)
                                 + h_parent_index.size() * sizeof(CudaConfig::bh_index_type)
                                 + h_subdomain_length.size() * sizeof(CudaConfig::gaussian_type)
                                 + neuron_ids.size() * sizeof(CudaConfig::number_neurons_type)
                                 + rma_offset_to_index.size() * sizeof(CudaConfig::bh_index_type)
                                 + h_node_types.size() * sizeof(NodeType);

    Timers::stop_and_add(TimerRegion::CUDA_BH_INIT_NEURONS);

    return LinearizedTreeInitResult{ std::move(storage), total_mem_usage };
}

NeuronPopulationInitResult
BarnesHutCUDA_CU::init_neuron_details(
    const std::span<const SimpleVec3d> h_neuron_positions,
    const std::span<const CudaConfig::synaptic_count_type> h_vacant_dendrites,
    const std::span<const CudaConfig::synaptic_count_type> h_vacant_axons,
    const std::span<const CudaConfig::mpi_rank_type> h_ranks) {

    Timers::start(TimerRegion::CUDA_BH_INIT_NEURON_DETAILS);

    auto storage = NeuronPopulationDeviceStorage{};
    storage.positions.emplace(h_neuron_positions);
    storage.vacant_dendrites.emplace(h_vacant_dendrites);
    storage.vacant_axons.emplace(h_vacant_axons);
    storage.ranks.emplace(h_ranks);

    const auto total_mem_usage = h_neuron_positions.size() * sizeof(SimpleVec3d)
                                 + h_vacant_dendrites.size() * sizeof(CudaConfig::synaptic_count_type)
                                 + h_vacant_axons.size() * sizeof(CudaConfig::synaptic_count_type)
                                 + h_ranks.size() * sizeof(CudaConfig::mpi_rank_type);

    Timers::stop_and_add(TimerRegion::CUDA_BH_INIT_NEURON_DETAILS);

    return NeuronPopulationInitResult{ std::move(storage), total_mem_usage };
}

__host__ void BarnesHutCUDA_CU::get_updated_octree(const NeuronPopulationDeviceHandle population,
                                                   const size_t size,
                                                   std::vector<SimpleVec3d>& updated_positions,
                                                   std::vector<CudaConfig::synaptic_count_type>& updated_vacant_dendrites) {

    Timers::start(TimerRegion::CUDA_GET_UPDATED_OCTREE);

    CUDA_CHECK(cudaDeviceSynchronize());

    updated_vacant_dendrites.resize(size);
    const auto vacant_dendrites_size = size * sizeof(CudaConfig::synaptic_count_type);
    CUDA_CHECK(cudaMemcpy(updated_vacant_dendrites.data(), population.vacant_dendrites, vacant_dendrites_size,
                          cudaMemcpyDeviceToHost));

    updated_positions.resize(size);

    CUDA_CHECK(
        cudaMemcpy(updated_positions.data(), population.positions, sizeof(SimpleVec3d) * size, cudaMemcpyDeviceToHost));
    Timers::stop_and_add(TimerRegion::CUDA_GET_UPDATED_OCTREE);
}

__host__ void BarnesHutCUDA_CU::calculate_updated_octree_host(const NeuronPopulationDeviceHandle population,
                                                              const LinearizedTreeDeviceHandle tree,
                                                              const std::span<const CudaConfig::bh_index_type> h_level_indices, const std::shared_ptr<StreamWrapper>& stream) {
    // No host Timers wrap here: this only enqueues async work on `stream` (no sync), so a host
    // Timers::start/stop would close long before the GPU kernels actually finish and could never
    // correctly bound the CUDA-event-timed CUDA_BH_OCTREE_UPDATE_KERNEL below. Only the outer
    // UPDATE_LEAF_NODES (in BarnesHutCUDA.cpp, where the device is actually synced and GPU timers
    // resolved) can. It also previously left the timer running forever on the early-return path
    // just below when h_level_indices was empty.

    // Check nullptr
    const auto pos_ok = population.positions != nullptr;
    const auto child_idx_ok = tree.child_index != nullptr;
    const auto dendrites_ok = population.vacant_dendrites != nullptr;
    RELEARN_CUDA_CHECK(pos_ok, "BarnesHutCUDA_CU::calculate_updated_octree_host: Positions array must not be nullptr.");
    RELEARN_CUDA_CHECK(child_idx_ok, "BarnesHutCUDA_CU::calculate_updated_octree_host: Child index array must not be nullptr.");
    RELEARN_CUDA_CHECK(dendrites_ok, "BarnesHutCUDA_CU::calculate_updated_octree_host: Dendrites array must not be nullptr.");

    if (h_level_indices.size() == 0) {
        return;
    }

    /**
     * level_indices holds the index of each first neuron of level l = i + 1
     */
    // Every level's kernel is launched on the same `stream` with no sync in between (correctness
    // is fine -- a stream executes in order -- but a host Timers::start/stop around the loop would
    // only measure the async launches, not the actual GPU execution, which keeps running after this
    // function returns). Time via CUDA events instead, bracketing the whole per-level loop.
    auto* gpu_timer = cuda_start_gpu_timer(TimerRegion::CUDA_BH_OCTREE_UPDATE_KERNEL, *stream);
    for (auto i = h_level_indices.size() - 1; i > 1; i--) {
        const auto level = i - 1;
        const auto min_index_for_level = h_level_indices[level];
        const auto max_index_for_level = h_level_indices[level + 1];

        const auto number_nodes = max_index_for_level - min_index_for_level;
        const auto& [blocks, threads] = get_grid_ands_block_size(number_nodes, calculate_updated_octree_dev);
        calculate_updated_octree_dev<<<blocks, threads, 0, get_cuda_stream_from_wrapper(*stream)>>>(population, tree,
                                                                                                    max_index_for_level, min_index_for_level);
    }
    cuda_stop_gpu_timer(gpu_timer, *stream);
}

__host__ TargetNeuronSearchResultBothSignalTypes
BarnesHutCUDA_CU::find_target_neurons(const std::uint64_t seed, const std::uint64_t step,

                                      NeuronPopulationDeviceHandle population_exc,
                                      const CudaConfig::bh_index_type* node_id_vacant_axons_mapping_exc,
                                      const std::size_t number_tasks_exc,

                                      NeuronPopulationDeviceHandle population_inh,
                                      const CudaConfig::bh_index_type* node_id_vacant_axons_mapping_inh,
                                      const std::size_t number_tasks_inh,

                                      const CudaConfig::number_neurons_type neurons_count,
                                      LinearizedTreeDeviceHandle tree,

                                      const CudaConfig::gaussian_type acceptance_criterion,
                                      const CudaConfig::mpi_rank_type* const neuron_ranks,
                                      const CudaConfig::mpi_rank_type my_rank,
                                      const CudaConfig::gaussian_type squared_sigma_inv,
                                      const CudaConfig::mpi_rank_type number_ranks,
                                      const CudaConfig::number_neurons_type number_neurons,
                                      const std::shared_ptr<StreamWrapper>& exc_stream,
                                      const std::shared_ptr<StreamWrapper>& inh_stream) {

    Timers::start(TimerRegion::CUDA_FIND_TARGET_NEURONS);

    // Check nullptr
    const auto pos_ok = population_exc.positions != nullptr;
    const auto child_idx_ok = tree.child_index != nullptr;
    const auto parent_idx_ok = tree.parent_index != nullptr;
    const auto subdomain_ok = tree.subdomain_length != nullptr;
    const auto dendrites_ok = population_exc.vacant_dendrites != nullptr;
    RELEARN_CUDA_CHECK(pos_ok, "BarnesHutCUDA_CU::find_target_neurons: Positions array must not be nullptr.");
    RELEARN_CUDA_CHECK(child_idx_ok, "BarnesHutCUDA_CU::find_target_neurons: Child index array must not be nullptr.");
    RELEARN_CUDA_CHECK(parent_idx_ok, "BarnesHutCUDA_CU::find_target_neurons: Parent index array must not be nullptr.");
    RELEARN_CUDA_CHECK(subdomain_ok, "BarnesHutCUDA_CU::find_target_neurons: Subdomain array must not be nullptr.");
    RELEARN_CUDA_CHECK(dendrites_ok, "BarnesHutCUDA_CU::find_target_neurons: Dendrites array must not be nullptr.");

    // Test AC
    const auto ac_neg = acceptance_criterion >= 0.0;
    RELEARN_CUDA_CHECK(ac_neg, "BarnesHutCUDA_CU::find_target_neurons: The acceptance criterion was not positive.");
    const auto ac_larger_max = acceptance_criterion <= Constants::bh_max_theta;
    RELEARN_CUDA_CHECK(ac_larger_max, "BarnesHutCUDA_CU::find_target_neurons: The acceptance criterion is too large.");

    Timers::start(TimerRegion::CUDA_BH_FIND_TARGET_NEURONS_KERNEL);

    auto helper = [number_neurons, number_ranks, seed, step, neuron_ranks, tree, acceptance_criterion, neurons_count, my_rank, squared_sigma_inv](const NeuronPopulationDeviceHandle population,
                                                                                                                                                  const CudaConfig::bh_index_type* node_id_vacant_axons_mapping,
                                                                                                                                                  const std::size_t number_tasks, const std::shared_ptr<StreamWrapper>& stream) {
        // call kernel
        const auto& [blocks, threads] = get_grid_ands_block_size(number_tasks,
                                                                 find_target_neurons_kernel);

        Timers::start(TimerRegion::BLOCK1);

        std::vector<int> vv(number_ranks, 0);
        DeviceArray<int> counter_per_rank{ vv, stream };

        std::vector<CudaConfig::mpi_rank_type*> h_target_ranks_ptr(number_ranks);
        std::vector<CudaConfig::number_neurons_type*> h_target_ids_ptr(number_ranks);
        std::vector<SimpleVec3d*> h_source_positions_ptr(number_ranks);
        std::vector<CudaConfig::number_neurons_type*> h_source_ids_ptr(number_ranks);
        std::vector<DeviceArray<CudaConfig::mpi_rank_type>> target_ranks_dev_arr{};
        std::vector<DeviceArray<CudaConfig::number_neurons_type>> target_ids_dev_arr{};
        std::vector<DeviceArray<SimpleVec3d>> source_positions_dev_arr{};
        std::vector<DeviceArray<CudaConfig::number_neurons_type>> source_ids_dev_arr{};

        const auto max_size_per_rank = number_tasks;

        for (auto rank = 0; rank < number_ranks; rank++) {
            target_ranks_dev_arr.emplace_back(max_size_per_rank, stream);
            h_target_ranks_ptr[rank] = target_ranks_dev_arr[rank].device_ptr();
            target_ids_dev_arr.emplace_back(max_size_per_rank, stream);
            h_target_ids_ptr[rank] = target_ids_dev_arr[rank].device_ptr();
            source_ids_dev_arr.emplace_back(max_size_per_rank, stream);
            h_source_ids_ptr[rank] = source_ids_dev_arr[rank].device_ptr();
            source_positions_dev_arr.emplace_back(max_size_per_rank, stream);
            h_source_positions_ptr[rank] = source_positions_dev_arr[rank].device_ptr();
        }

        DeviceArray<CudaConfig::mpi_rank_type*> target_ranks_ptr(h_target_ranks_ptr, stream);
        DeviceArray<CudaConfig::number_neurons_type*> target_ids_ptr(h_target_ids_ptr, stream);
        DeviceArray<SimpleVec3d*> source_positions_ptr(h_source_positions_ptr, stream);
        DeviceArray<CudaConfig::number_neurons_type*> source_ids_ptr(h_source_ids_ptr, stream);

        Timers::stop_and_add(TimerRegion::BLOCK1);

        if (number_tasks != 0) {

            find_target_neurons_kernel<<<blocks, threads, 0, get_cuda_stream_from_wrapper(*stream)>>>(seed, threads, step,
                                                                                                      population,
                                                                                                      neurons_count,
                                                                                                      tree,
                                                                                                      node_id_vacant_axons_mapping,
                                                                                                      acceptance_criterion,
                                                                                                      number_tasks,
                                                                                                      neuron_ranks, my_rank, squared_sigma_inv,
                                                                                                      target_ranks_ptr.device_ptr(), target_ids_ptr.device_ptr(), source_ids_ptr.device_ptr(), source_positions_ptr.device_ptr(), number_neurons, counter_per_rank.device_ptr(), max_size_per_rank);
        }

        return std::make_tuple(std::move(counter_per_rank), std::move(h_target_ids_ptr), std::move(h_source_positions_ptr), std::move(h_source_ids_ptr), std::move(target_ranks_dev_arr),
                               std::move(target_ids_dev_arr), std::move(source_positions_dev_arr), std::move(source_ids_dev_arr));
    };

    // The vectors of  device arrays must be returned and hold in function scope. Otherwise the memory will be freed on the GPU
    auto [counter_per_rank_exc, h_target_ids_ptr_exc, h_source_positions_ptr_exc, h_source_ids_ptr_exc, _0, _1, _2, _3] = helper(population_exc, node_id_vacant_axons_mapping_exc, number_tasks_exc, exc_stream);
    auto [counter_per_rank_inh, h_target_ids_ptr_inh, h_source_positions_ptr_inh, h_source_ids_ptr_inh, _4, _5, _6, _7] = helper(population_inh, node_id_vacant_axons_mapping_inh, number_tasks_inh, inh_stream);

    cudaDeviceSynchronize();
    kernelErrCheck();

    Timers::stop_and_add(TimerRegion::CUDA_BH_FIND_TARGET_NEURONS_KERNEL);

    Timers::start(TimerRegion::BH_POST_PROCESSING);

    auto helper_post = [number_ranks](const DeviceArray<int>& counter_per_rank,
                                      const std::vector<CudaConfig::number_neurons_type*>& h_target_ids_ptr,
                                      const std::vector<SimpleVec3d*>& h_source_positions_ptr,
                                      const std::vector<CudaConfig::number_neurons_type*>& h_source_ids_ptr, const std::shared_ptr<StreamWrapper>& stream) {
        // Copy counts to host
        std::vector<int> h_sizes = counter_per_rank.get_device_data(*stream);
        cudaStreamSynchronize_bride(*stream);

        const auto buffer_size = std::accumulate(h_sizes.begin(), h_sizes.end(), 0);

        DeviceArray<CudaConfig::number_neurons_type> source_ids_out(buffer_size, stream);
        DeviceArray<CudaConfig::number_neurons_type> target_ids_out(buffer_size, stream);
        DeviceArray<SimpleVec3d> source_positions_out(buffer_size, stream);

        auto offset = 0;
        for (auto rank = 0; rank < number_ranks; rank++) {
            cudaMemcpyAsync_on_device_bridge(source_ids_out.device_ptr() + offset, h_source_ids_ptr[rank], h_sizes[rank] * sizeof(CudaConfig::number_neurons_type), *stream);
            cudaMemcpyAsync_on_device_bridge(source_positions_out.device_ptr() + offset, h_source_positions_ptr[rank], h_sizes[rank] * sizeof(SimpleVec3d), *stream);
            cudaMemcpyAsync_on_device_bridge(target_ids_out.device_ptr() + offset, h_target_ids_ptr[rank], h_sizes[rank] * sizeof(CudaConfig::number_neurons_type), *stream);
            offset += h_sizes[rank];
        }

        // find_target_neurons_kernel assigns each rank's compaction slots via an
        // atomicAdd whose order varies between runs (and GPUs) even for the same seed/step, even
        // though the *set* of (source, target) requests it produces is fully deterministic. Sort
        // each rank's segment into a canonical (source, target) order so the result is reproducible
        // run-to-run. h_sizes[rank] is bounded by the number of vacant axons this round, not the
        // neuron count, so this is cheap relative to the tree traversal that dominates this step.
        // Queued on the same stream as the copies above, so they're correctly ordered after them
        // without an extra device-wide sync; the cudaStreamSynchronize_bride() below (only, not a
        // full device sync) is still required so sort_keys isn't freed before its last use.
        const auto exec_policy = thrust::cuda::par.on(get_cuda_stream_from_wrapper(*stream));
        offset = 0;
        for (auto rank = 0; rank < number_ranks; rank++) {
            if (h_sizes[rank] > 0) {
                auto* src_ptr = source_ids_out.device_ptr() + offset;
                auto* tgt_ptr = target_ids_out.device_ptr() + offset;
                auto* pos_ptr = source_positions_out.device_ptr() + offset;

                DeviceArray<std::uint64_t> sort_keys(static_cast<std::size_t>(h_sizes[rank]));
                thrust::transform(exec_policy, thrust::device_pointer_cast(src_ptr), thrust::device_pointer_cast(src_ptr + h_sizes[rank]),
                                  thrust::device_pointer_cast(tgt_ptr), thrust::device_pointer_cast(sort_keys.device_ptr()),
                                  [] __device__(CudaConfig::number_neurons_type s, CudaConfig::number_neurons_type t) {
                                      return (static_cast<std::uint64_t>(s) << 32) | static_cast<std::uint64_t>(t);
                                  });
                auto value_zip = thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_pointer_cast(tgt_ptr), thrust::device_pointer_cast(src_ptr), thrust::device_pointer_cast(pos_ptr)));
                thrust::sort_by_key(exec_policy, thrust::device_pointer_cast(sort_keys.device_ptr()),
                                    thrust::device_pointer_cast(sort_keys.device_ptr() + h_sizes[rank]), value_zip);
            }
            offset += h_sizes[rank];
        }
        cudaStreamSynchronize_bride(*stream);

        return TargetNeuronSearchResult{ h_sizes, std::move(source_ids_out), std::move(source_positions_out), std::move(target_ids_out) };
    };

    auto result_exc = helper_post(counter_per_rank_exc, h_target_ids_ptr_exc, h_source_positions_ptr_exc, h_source_ids_ptr_exc, exc_stream);
    auto result_inh = helper_post(counter_per_rank_inh, h_target_ids_ptr_inh, h_source_positions_ptr_inh, h_source_ids_ptr_inh, inh_stream);

    cudaDeviceSynchronize();
    kernelErrCheck();

    Timers::stop_and_add(TimerRegion::BH_POST_PROCESSING);

    Timers::stop_and_add(TimerRegion::CUDA_FIND_TARGET_NEURONS);

    return TargetNeuronSearchResultBothSignalTypes{ std::move(result_exc), std::move(result_inh) };
}

std::vector<bool> BarnesHutCUDA_CU::test_acceptance_criterion_host(
    const CudaConfig::number_neurons_type test_results_size,
    const SimpleVec3d& source_position,
    const SimpleVec3d& target_position,
    const CudaConfig::synaptic_count_type vacant_dendritic_elements,
    const CudaConfig::gaussian_type subdomain_length,
    const CudaConfig::gaussian_type acceptance_criterion, const bool is_leaf) {
    DeviceArray<bool> d_test_results(test_results_size);

    const auto& [blocks, threads] = get_grid_ands_block_size(test_results_size, test_acceptance_criterion_dev);
    Timers::start(TimerRegion::CUDA_TEST_ACCEPTANCE_CRITERION_KERNEL);
    test_acceptance_criterion_dev<<<blocks, threads>>>(d_test_results.device_ptr(), test_results_size, source_position,
                                                       target_position, vacant_dendritic_elements,
                                                       subdomain_length, acceptance_criterion, is_leaf);

    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_TEST_ACCEPTANCE_CRITERION_KERNEL);
    return d_test_results.get_device_data();
}

__global__ void collect_branch_nodes(const CudaConfig::bh_index_type tree_size, const CudaConfig::bh_index_type* const local_branch_nodes, const std::size_t number_local_branch_nodes,
                                     const NeuronPopulationDeviceHandle population,
                                     SimpleVec3d* const local_branch_positions, CudaConfig::synaptic_count_type* const local_branch_vacant_dends, CudaConfig::number_neurons_type* const local_branch_neuron_ids) {
    const auto* const positions = population.positions;
    const auto* const vacant_dends = population.vacant_dendrites;

    const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= number_local_branch_nodes) {
        return;
    }

    const auto node_idx = local_branch_nodes[thread_id];
    RELEARN_DEVICE_CUDA_CHECK(node_idx < tree_size, "collect_branch_nodes: Node idx %u invalid %u", node_idx, tree_size);
    local_branch_positions[thread_id] = positions[node_idx];
    local_branch_vacant_dends[thread_id] = vacant_dends[node_idx];
    local_branch_neuron_ids[thread_id] = node_idx;
}

__global__ void update_branch_nodes(const std::size_t number_branch_nodes,
                                    const NeuronPopulationDeviceHandle population,
                                    const SimpleVec3d* const recv_branch_positions, const CudaConfig::synaptic_count_type* const recv_branch_vacant_dends, const CudaConfig::number_neurons_type* const recv_branch_neuron_ids,
                                    const int* const recv_sizes, const int number_ranks, const int my_rank) {
    auto* const positions = population.positions;
    auto* const vacant_dends = population.vacant_dendrites;

    const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= number_branch_nodes) {
        return;
    }

    auto rank = 0U;
    auto offset = 0U;
    for (auto i = 0; i < number_ranks; i++) {
        offset += recv_sizes[i];
        rank = i;
        if (thread_id < offset) {
            break;
        }
    }
    if (rank == my_rank) {
        return;
    }
    const auto node_idx = recv_branch_neuron_ids[thread_id];
    positions[node_idx] = recv_branch_positions[thread_id];
    vacant_dends[node_idx] = recv_branch_vacant_dends[thread_id];
}

__host__ void BarnesHutCUDA_CU::update_remote_nodes_host(const CudaConfig::bh_index_type tree_size, const int number_ranks, const std::span<const CudaConfig::bh_index_type> local_branch_nodes,
                                                         const NeuronPopulationDeviceHandle population, CudaConfig::number_neurons_type* const neuron_ids, const CudaConfig::mpi_rank_type my_rank, const std::shared_ptr<StreamWrapper>& stream) {

    const auto number_local_branch_nodes = local_branch_nodes.size();
    const DeviceArray<CudaConfig::bh_index_type> d_local_branch_nodes(local_branch_nodes, stream);
    const DeviceArray<SimpleVec3d> d_local_branch_positions(number_local_branch_nodes, stream);
    const DeviceArray<CudaConfig::synaptic_count_type> d_local_branch_vacant_dends(number_local_branch_nodes, stream);
    const DeviceArray<CudaConfig::number_neurons_type> d_local_branch_ids(number_local_branch_nodes, stream);

    const auto& [blocks, threads] = get_grid_ands_block_size(number_local_branch_nodes, collect_branch_nodes);

    // Launched on `stream` without a following sync here -- the caller (BarnesHutCUDA::
    // update_remote_nodes, called from prepare_update_connectivity under UPDATE_LEAF_NODES) syncs
    // and resolves GPU timers later, once all octree-update rounds have been issued.
    auto* collect_timer = cuda_start_gpu_timer(TimerRegion::CUDA_COLLECT_BRANCH_NODES_KERNEL, *stream);
    collect_branch_nodes<<<blocks, threads, 0, get_cuda_stream_from_wrapper(*stream)>>>(tree_size, d_local_branch_nodes.device_ptr(), number_local_branch_nodes,
                                                                                        population,
                                                                                        d_local_branch_positions.device_ptr(), d_local_branch_vacant_dends.device_ptr(), d_local_branch_ids.device_ptr());
    cuda_stop_gpu_timer(collect_timer, *stream);

    const auto& [recv_branch_positions, recv_sizes, _1] = cuda_allgather(d_local_branch_positions, number_ranks, stream);
    const auto& [recv_branch_vacant_dends, _2, _3] = cuda_allgather(d_local_branch_vacant_dends, number_ranks, stream);
    const auto& [recv_branch_ids, _4, _5] = cuda_allgather(d_local_branch_ids, number_ranks, stream);

    const DeviceArray<int> d_recv_sizes(recv_sizes, stream);
    const auto number_branch_nodes = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0U);
    const auto& [blocks2, threads2] = get_grid_ands_block_size(number_local_branch_nodes, update_branch_nodes);
    auto* update_timer = cuda_start_gpu_timer(TimerRegion::CUDA_UPDATE_BRANCH_NODES_KERNEL, *stream);
    update_branch_nodes<<<blocks2, threads2, 0, get_cuda_stream_from_wrapper(*stream)>>>(number_branch_nodes, population, recv_branch_positions.device_ptr(), recv_branch_vacant_dends.device_ptr(), recv_branch_ids.device_ptr(), d_recv_sizes.device_ptr(), number_ranks, my_rank);
    cuda_stop_gpu_timer(update_timer, *stream);
}

#pragma GCC diagnostic pop