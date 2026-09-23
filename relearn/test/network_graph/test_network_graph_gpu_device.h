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

#ifdef RELEARN_CUDA_ENABLED

#include <cstddef>
#include <cstdint>
#include <vector>

// Plain host-compatible edge tuple returned by cursor-iteration helpers.
struct HostEdge {
    std::uint32_t other_neuron{};
    std::uint16_t other_rank{};
    std::int16_t weight{};
};

// Each function launches a CUDA kernel that exercises MemoryPoolView<SmallNeuronIdType> /
// SortedView / OnlyOutgoingView<SmallNeuronIdType> __device__ methods and returns the results as a
// host-side vector. All functions accept the raw void* returned by
// GPUEdgesBase::get_handle().view_impl (only valid when CudaConfig::use_wide_neuron_ids is
// false, i.e. the default in tests, so the view is instantiated with SmallNeuronIdType).

// ── MemoryPoolView (LayoutType::MemoryPool) device functions ───────────────────────────

// Calls MemoryPoolView::size(i) for every neuron i in [0, number_neurons).
std::vector<std::size_t> device_read_sizes(
    void* view_ptr,
    std::uint32_t number_neurons);

// Calls MemoryPoolView::get_other_neuron(neuron_id, i) for i in [0, num_edges).
std::vector<std::uint32_t> device_read_other_neurons(
    void* view_ptr,
    std::uint32_t neuron_id,
    std::size_t num_edges);

// Calls MemoryPoolView::get_weight(neuron_id, i) for i in [0, num_edges).
std::vector<std::int16_t> device_read_weights(
    void* view_ptr,
    std::uint32_t neuron_id,
    std::size_t num_edges);

// Calls MemoryPoolView::get_other_rank(neuron_id, i) for i in [0, num_edges).
std::vector<std::uint16_t> device_read_other_ranks(
    void* view_ptr,
    std::uint32_t neuron_id,
    std::size_t num_edges);

// Walks MemoryPoolView::edge_begin / edge_end for one neuron and returns every
// EdgeData emitted by the cursor, in order.
std::vector<HostEdge> device_iterate_edges_default(
    void* view_ptr,
    std::uint32_t neuron_id);

// ── SortedView device functions (MemoryPoolView::sorted) ──────────────────────────

// Calls SortedView::get_other_neuron_id_by_rank_and_index(rank, i) for all i
// in the rank's segment.  Returns an empty vector when sorting is disabled or
// the rank has no entries.
std::vector<std::uint32_t> device_read_sorted_ids_for_rank(
    void* view_ptr,
    std::uint16_t rank);

// Calls SortedView::rank_size(r) for every rank r in [0, number_ranks).
std::vector<std::size_t> device_read_rank_sizes(
    void* view_ptr,
    std::uint16_t number_ranks);

// ── MemoryPoolView (LayoutType::MemoryPool) device functions, alternate entry points ───
// Identical semantics to device_read_sizes / device_iterate_edges_default; kept as a separate
// name since older tests exercise the "memory pool view" entry points explicitly.

// Calls MemoryPoolView::size(i) for every neuron i in [0, number_neurons).
std::vector<std::size_t> device_memory_pool_view_sizes(
    void* view_ptr,
    std::uint32_t number_neurons);

// Walks MemoryPoolView::edge_begin / edge_end for one neuron and returns all edges.
std::vector<HostEdge> device_memory_pool_view_iterate_edges(
    void* view_ptr,
    std::uint32_t neuron_id);

// Calls MemoryPoolView::add_synapse(neuron_id, my_rank, other_neuron_id, excitatory)
// `count` times from device code.  Used to test multi-entry weight saturation.
void device_memory_pool_view_add_synapse_n_times(
    void* view_ptr,
    std::uint32_t neuron_id,
    std::uint32_t other_neuron_id,
    std::uint32_t count,
    bool excitatory);

#endif // RELEARN_CUDA_ENABLED
