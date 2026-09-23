/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cuda/network_graph/Views.cuh"
#include "cuda/util/SmallNeuronIdType.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "test_network_graph_gpu_device.h"

// Concrete view types. Tests never set CudaConfig::use_wide_neuron_ids, so GPUEdgesBase always
// builds its handle with IdT = SmallNeuronIdType (see GPUEdgesBase::wide_ids).
using dv_t = MemoryPoolView<SmallNeuronIdType>;
using memory_pool_t = MemoryPoolView<SmallNeuronIdType>;

// ── CUDA kernels ─────────────────────────────────────────────────────────────

__global__ void k_read_sizes(dv_t view, std::size_t* out, neuron_id_type n) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    out[tid] = view.size(tid);
}

__global__ void k_read_other_neurons(dv_t view, neuron_id_type nid,
                                     neuron_id_type* out, std::size_t n) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    out[tid] = view.get_other_neuron(nid, tid);
}

__global__ void k_read_weights(dv_t view, neuron_id_type nid,
                               weight_type* out, std::size_t n) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    out[tid] = view.get_weight(nid, tid);
}

__global__ void k_read_other_ranks(dv_t view, neuron_id_type nid,
                                   mpi_rank_type* out, std::size_t n) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    out[tid] = view.get_other_rank(nid, tid);
}

// ── DefaultView cursor kernel ─────────────────────────────────────────────────

__global__ void k_iterate_edges_default(dv_t view, neuron_id_type nid,
                                        HostEdge* out, std::uint32_t* n_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    auto cur = view.edge_begin(nid);
    const auto end = view.edge_end(nid);
    std::uint32_t count = 0;
    while (!(cur == end)) {
        const auto ed = cur.next();
        out[count++] = { ed.other_neuron_id,
                         static_cast<std::uint16_t>(ed.other_rank),
                         static_cast<std::int16_t>(ed.weight) };
    }
    *n_out = count;
}

// ── MemoryPoolView kernels ─────────────────────────────────────────────────────────

__global__ void k_memory_pool_sizes(memory_pool_t view, std::size_t* out, std::uint32_t N) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    out[tid] = view.size(tid);
}

__global__ void k_memory_pool_iterate_edges(memory_pool_t view, neuron_id_type nid,
                                            HostEdge* out, std::uint32_t* n_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    auto cur = view.edge_begin(nid);
    const auto end = view.edge_end(nid);
    std::uint32_t count = 0;
    while (!(cur == end)) {
        const auto ed = cur.next();
        out[count++] = { ed.other_neuron_id,
                         static_cast<std::uint16_t>(ed.other_rank),
                         static_cast<std::int16_t>(ed.weight) };
    }
    *n_out = count;
}

__global__ void k_memory_pool_add_synapse_n(memory_pool_t view, neuron_id_type nid,
                                            neuron_id_type other_nid,
                                            std::uint32_t count, bool excitatory) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    for (std::uint32_t i = 0; i < count; ++i)
        (void)view.add_synapse(nid, view.my_rank, other_nid, excitatory);
}

// ── Internal helpers ──────────────────────────────────────────────────────────

static const dv_t& extract_view(void* view_ptr) {
    return *static_cast<dv_t*>(view_ptr);
}

static const memory_pool_t& extract_memory_pool_view(void* view_ptr) {
    return *static_cast<memory_pool_t*>(view_ptr);
}

static constexpr unsigned BLOCK = 256;

static unsigned grid(std::size_t n) {
    return static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
}

template <typename T>
static std::vector<T> pull(T* d_ptr, std::size_t n) {
    std::vector<T> host(n);
    cudaMemcpy(host.data(), d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_ptr);
    return host;
}

template <typename T>
static T* alloc_device(std::size_t n) {
    T* ptr = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T));
    return ptr;
}

// ── Public C++ interface ──────────────────────────────────────────────────────

std::vector<std::size_t> device_read_sizes(void* view_ptr, std::uint32_t number_neurons) {
    const dv_t& view = extract_view(view_ptr);
    auto* d_out = alloc_device<std::size_t>(number_neurons);
    k_read_sizes<<<grid(number_neurons), BLOCK>>>(view, d_out, number_neurons);
    cudaDeviceSynchronize();
    return pull(d_out, number_neurons);
}

std::vector<std::uint32_t> device_read_other_neurons(void* view_ptr,
                                                     std::uint32_t neuron_id,
                                                     std::size_t num_edges) {
    if (num_edges == 0)
        return {};
    const dv_t& view = extract_view(view_ptr);
    auto* d_out = alloc_device<neuron_id_type>(num_edges);
    k_read_other_neurons<<<grid(num_edges), BLOCK>>>(view, neuron_id, d_out, num_edges);
    cudaDeviceSynchronize();
    return pull(d_out, num_edges);
}

std::vector<std::int16_t> device_read_weights(void* view_ptr,
                                              std::uint32_t neuron_id,
                                              std::size_t num_edges) {
    if (num_edges == 0)
        return {};
    const dv_t& view = extract_view(view_ptr);
    auto* d_out = alloc_device<weight_type>(num_edges);
    k_read_weights<<<grid(num_edges), BLOCK>>>(view, neuron_id, d_out, num_edges);
    cudaDeviceSynchronize();
    return pull(d_out, num_edges);
}

std::vector<std::uint16_t> device_read_other_ranks(void* view_ptr,
                                                   std::uint32_t neuron_id,
                                                   std::size_t num_edges) {
    if (num_edges == 0)
        return {};
    const dv_t& view = extract_view(view_ptr);
    auto* d_out = alloc_device<mpi_rank_type>(num_edges);
    k_read_other_ranks<<<grid(num_edges), BLOCK>>>(view, neuron_id, d_out, num_edges);
    cudaDeviceSynchronize();
    return pull(d_out, num_edges);
}

std::vector<HostEdge> device_iterate_edges_default(void* view_ptr,
                                                   std::uint32_t neuron_id) {
    const dv_t& view = extract_view(view_ptr);
    constexpr std::uint32_t MAX_EDGES = 1024;
    auto* d_out = alloc_device<HostEdge>(MAX_EDGES);
    auto* d_count = alloc_device<std::uint32_t>(1);
    k_iterate_edges_default<<<1, 1>>>(view, neuron_id, d_out, d_count);
    cudaDeviceSynchronize();
    std::uint32_t h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_count);
    std::vector<HostEdge> result(h_count);
    cudaMemcpy(result.data(), d_out, h_count * sizeof(HostEdge), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    return result;
}

std::vector<std::size_t> device_memory_pool_view_sizes(void* view_ptr,
                                                       std::uint32_t number_neurons) {
    const memory_pool_t& view = extract_memory_pool_view(view_ptr);
    auto* d_out = alloc_device<std::size_t>(number_neurons);
    k_memory_pool_sizes<<<grid(number_neurons), BLOCK>>>(view, d_out, number_neurons);
    cudaDeviceSynchronize();
    return pull(d_out, number_neurons);
}

std::vector<HostEdge> device_memory_pool_view_iterate_edges(void* view_ptr,
                                                            std::uint32_t neuron_id) {
    const memory_pool_t& view = extract_memory_pool_view(view_ptr);
    constexpr std::uint32_t MAX_EDGES = 1024;
    auto* d_out = alloc_device<HostEdge>(MAX_EDGES);
    auto* d_count = alloc_device<std::uint32_t>(1);
    k_memory_pool_iterate_edges<<<1, 1>>>(view, neuron_id, d_out, d_count);
    cudaDeviceSynchronize();
    std::uint32_t h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_count);
    std::vector<HostEdge> result(h_count);
    cudaMemcpy(result.data(), d_out, h_count * sizeof(HostEdge), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    return result;
}

void device_memory_pool_view_add_synapse_n_times(void* view_ptr,
                                                 std::uint32_t neuron_id,
                                                 std::uint32_t other_neuron_id,
                                                 std::uint32_t count,
                                                 bool excitatory) {
    const memory_pool_t& view = extract_memory_pool_view(view_ptr);
    k_memory_pool_add_synapse_n<<<1, 1>>>(view, neuron_id, other_neuron_id, count, excitatory);
    cudaDeviceSynchronize();
}

